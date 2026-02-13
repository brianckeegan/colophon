"""Importers that convert notes exports into Colophon knowledge-graph structure."""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as et
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from .graph import KnowledgeGraph
from .vectors import EmbeddingConfig, InMemoryVectorDB, VectorRecord, create_embedding_client


@dataclass(slots=True)
class NoteDocument:
    """Normalized note document used during notes-to-KG ingestion.

    Parameters
    ----------
    note_id : str
        Stable note identifier.
    title : str
        Note title.
    text : str
        Plain-text note body.
    links : list[str]
        Raw outbound links parsed from note content.
    path : str
        Optional source-relative path for provenance.
    """

    note_id: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)
    path: str = ""


@dataclass(slots=True)
class NotesImportConfig:
    """Configuration for notes ingestion and graph-link generation.

    Parameters
    ----------
    platform : str
        Source platform id (for example ``obsidian`` or ``notion``).
    use_hyperlinks : bool
        Whether hyperlink-based relations should be emitted.
    use_embeddings : bool
        Whether embedding similarity relations should be emitted.
    embedding_config : EmbeddingConfig
        Embedding model/provider settings.
    embedding_top_k : int
        Maximum nearest note neighbors evaluated per note.
    embedding_similarity_threshold : float
        Minimum similarity threshold for ``similar_to`` links.
    vector_db_path : str
        Optional output path for serialized vector records.
    include_external_urls : bool
        Whether external URL links are retained as entities/relations.
    """

    platform: str = "auto"
    use_hyperlinks: bool = True
    use_embeddings: bool = True
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    embedding_top_k: int = 3
    embedding_similarity_threshold: float = 0.35
    vector_db_path: str = ""
    include_external_urls: bool = True


@dataclass(slots=True)
class NotesImportResult:
    """Summary diagnostics emitted by a notes import run.

    Parameters
    ----------
    platform : str
        Resolved source platform.
    notes_loaded : int
        Number of note documents parsed.
    entities_added : int
        Number of entities added to the target graph.
    relations_added : int
        Number of relations added to the target graph.
    hyperlink_relations_added : int
        Number of internal hyperlink relations added.
    embedding_relations_added : int
        Number of embedding similarity relations added.
    url_relations_added : int
        Number of external URL relations added.
    unresolved_links : int
        Count of links that could not be resolved to notes.
    vector_records : int
        Number of vector records generated during embedding linkage.
    """

    platform: str = "unknown"
    notes_loaded: int = 0
    entities_added: int = 0
    relations_added: int = 0
    hyperlink_relations_added: int = 0
    embedding_relations_added: int = 0
    url_relations_added: int = 0
    unresolved_links: int = 0
    vector_records: int = 0

    def to_dict(self) -> dict:
        """To dict.

        Returns
        -------
        dict
            Return value description.
        """
        return {
            "platform": self.platform,
            "notes_loaded": self.notes_loaded,
            "entities_added": self.entities_added,
            "relations_added": self.relations_added,
            "hyperlink_relations_added": self.hyperlink_relations_added,
            "embedding_relations_added": self.embedding_relations_added,
            "url_relations_added": self.url_relations_added,
            "unresolved_links": self.unresolved_links,
            "vector_records": self.vector_records,
        }


@dataclass(slots=True)
class NotesKnowledgeGraphImporter:
    """Build or extend a knowledge graph from exported note corpora.

    Parameters
    ----------
    config : NotesImportConfig
        Notes importer settings for parsing and linking behavior.
    """

    config: NotesImportConfig = field(default_factory=NotesImportConfig)

    def run(
        self,
        source_path: str | Path,
        graph: KnowledgeGraph | None = None,
    ) -> tuple[KnowledgeGraph, NotesImportResult]:
        """Run.

        Parameters
        ----------
        source_path : str | Path
            Parameter description.
        graph : KnowledgeGraph | None
            Parameter description.

        Returns
        -------
        tuple[KnowledgeGraph, NotesImportResult]
            Return value description.
        """
        path = Path(source_path)
        platform = _resolve_platform(path=path, platform=self.config.platform)
        notes = _load_notes(path=path, platform=platform)

        target_graph = graph if graph is not None else KnowledgeGraph()
        start_entities = len(target_graph.entities)
        start_relations = len(target_graph.relations)
        relation_index = {(rel.source, rel.predicate, rel.target) for rel in target_graph.relations}

        by_id, by_title, by_stem = _note_lookup_tables(notes)

        for note in notes:
            note_node = _note_node(note.note_id)
            target_graph.entities.add(note_node)
            if note.title:
                target_graph.entities.add(note.title)
                _add_relation_if_new(target_graph, relation_index, note_node, "has_title", note.title)

        hyperlink_relations_added = 0
        url_relations_added = 0
        unresolved_links = 0
        if self.config.use_hyperlinks:
            for note in notes:
                source_node = _note_node(note.note_id)
                for link in note.links:
                    if _is_external_url(link):
                        if not self.config.include_external_urls:
                            continue
                        target = f"url:{link.strip()}"
                        target_graph.entities.add(target)
                        if _add_relation_if_new(target_graph, relation_index, source_node, "references_url", target):
                            url_relations_added += 1
                        continue

                    resolved = _resolve_internal_link(link=link, by_id=by_id, by_title=by_title, by_stem=by_stem)
                    if resolved is None:
                        unresolved_links += 1
                        continue
                    target_node = _note_node(resolved.note_id)
                    if _add_relation_if_new(target_graph, relation_index, source_node, "links_to", target_node):
                        hyperlink_relations_added += 1

        embedding_relations_added = 0
        vector_records = 0
        if self.config.use_embeddings and len(notes) > 1:
            embedding_relations_added, vector_records = self._add_embedding_links(
                notes=notes,
                target_graph=target_graph,
                relation_index=relation_index,
            )

        result = NotesImportResult(
            platform=platform,
            notes_loaded=len(notes),
            entities_added=max(0, len(target_graph.entities) - start_entities),
            relations_added=max(0, len(target_graph.relations) - start_relations),
            hyperlink_relations_added=hyperlink_relations_added,
            embedding_relations_added=embedding_relations_added,
            url_relations_added=url_relations_added,
            unresolved_links=unresolved_links,
            vector_records=vector_records,
        )
        return target_graph, result

    def _add_embedding_links(
        self,
        notes: list[NoteDocument],
        target_graph: KnowledgeGraph,
        relation_index: set[tuple[str, str, str]],
    ) -> tuple[int, int]:
        """Add embedding links.

        Parameters
        ----------
        notes : list[NoteDocument]
            Parameter description.
        target_graph : KnowledgeGraph
            Parameter description.
        relation_index : set[tuple[str, str, str]]
            Parameter description.

        Returns
        -------
        tuple[int, int]
            Return value description.
        """
        client = create_embedding_client(self.config.embedding_config)
        texts = [_note_embedding_text(note) for note in notes]
        vectors = client.embed(texts)
        if len(vectors) != len(notes):
            raise ValueError("Embedding client returned unexpected number of vectors.")

        vector_db = InMemoryVectorDB()
        records: list[VectorRecord] = []
        for note, text, vector in zip(notes, texts, vectors, strict=True):
            records.append(
                VectorRecord(
                    record_id=note.note_id,
                    text=text,
                    metadata={"title": note.title, "path": note.path},
                    vector=vector,
                )
            )
        vector_db.add_many(records)

        if self.config.vector_db_path.strip():
            vector_db.save_json(self.config.vector_db_path)

        added = 0
        for record in records:
            hits = vector_db.search(
                query_vector=record.vector,
                top_k=max(0, self.config.embedding_top_k),
                exclude_ids={record.record_id},
            )
            source_node = _note_node(record.record_id)
            for other, score in hits:
                if score < self.config.embedding_similarity_threshold:
                    continue
                target_node = _note_node(other.record_id)
                if _add_relation_if_new(target_graph, relation_index, source_node, "similar_to", target_node):
                    added += 1

        return added, len(records)


def _resolve_platform(path: Path, platform: str) -> str:
    """Resolve platform.

    Parameters
    ----------
    path : Path
        Parameter description.
    platform : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    normalized = platform.strip().lower()
    if normalized != "auto":
        if normalized in {"obsidian", "notion", "onenote", "evernote", "markdown"}:
            return normalized
        raise ValueError(f"Unsupported notes platform: {platform}")

    if path.is_dir():
        if (path / ".obsidian").exists():
            return "obsidian"
        return "markdown"

    suffix = path.suffix.lower()
    if suffix == ".enex":
        return "evernote"
    if suffix == ".json":
        return "onenote"
    if suffix == ".md":
        return "markdown"
    raise ValueError(f"Could not infer notes platform for {path}. Set platform explicitly.")


def _load_notes(path: Path, platform: str) -> list[NoteDocument]:
    """Load notes.

    Parameters
    ----------
    path : Path
        Parameter description.
    platform : str
        Parameter description.

    Returns
    -------
    list[NoteDocument]
        Return value description.
    """
    if platform in {"obsidian", "markdown"}:
        return _load_markdown_notes(path=path)
    if platform == "notion":
        return _load_notion_notes(path=path)
    if platform == "onenote":
        return _load_onenote_notes(path=path)
    if platform == "evernote":
        return _load_evernote_notes(path=path)
    raise ValueError(f"Unsupported notes platform: {platform}")


def _load_notion_notes(path: Path) -> list[NoteDocument]:
    """Load notion notes.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    list[NoteDocument]
        Return value description.
    """
    if path.is_dir():
        markdown_files = [candidate for candidate in path.rglob("*.md") if not candidate.name.startswith(".")]
        if markdown_files:
            return _load_markdown_notes(path=path)
        csv_files = sorted(path.rglob("*.csv"))
        if csv_files:
            rows: list[NoteDocument] = []
            for csv_path in csv_files:
                rows.extend(_load_notion_csv_notes(csv_path))
            return rows
        json_files = sorted(path.rglob("*.json"))
        if json_files:
            rows = []
            for json_path in json_files:
                rows.extend(_load_notion_json_notes(json_path))
            return rows
        return []

    suffix = path.suffix.lower()
    if suffix == ".md":
        return _load_markdown_notes(path=path)
    if suffix == ".csv":
        return _load_notion_csv_notes(path)
    if suffix == ".json":
        return _load_notion_json_notes(path)
    raise ValueError(f"Unsupported Notion export file type: {path}")


def _load_notion_csv_notes(path: Path) -> list[NoteDocument]:
    """Load notion csv notes.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    list[NoteDocument]
        Return value description.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    notes: list[NoteDocument] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        title = _first_nonempty(
            row.get("title"),
            row.get("Title"),
            row.get("name"),
            row.get("Name"),
            default=f"Notion Note {index}",
        )
        body = _first_nonempty(
            row.get("content"),
            row.get("Content"),
            row.get("text"),
            row.get("Text"),
            row.get("summary"),
            row.get("Summary"),
            row.get("description"),
            row.get("Description"),
            default="",
        )
        note_id = _normalize_note_id(
            _first_nonempty(
                row.get("id"),
                row.get("ID"),
                row.get("page_id"),
                row.get("Page ID"),
                row.get("url"),
                row.get("URL"),
                default=title,
            )
        )
        links = _extract_links_from_markdown(body)
        url = _first_nonempty(row.get("url"), row.get("URL"), default="")
        if url:
            links.append(url)
        notes.append(
            NoteDocument(
                note_id=note_id or f"notion-{index}",
                title=title,
                text=body,
                links=_unique_nonempty(links),
                path=str(path.name),
            )
        )
    return notes


def _load_notion_json_notes(path: Path) -> list[NoteDocument]:
    """Load notion json notes.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    list[NoteDocument]
        Return value description.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = (
            payload.get("notes")
            or payload.get("pages")
            or payload.get("results")
            or payload.get("items")
            or []
        )
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    notes: list[NoteDocument] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        title = _first_nonempty(
            row.get("title"),
            row.get("name"),
            row.get("page_title"),
            default=f"Notion Note {index}",
        )
        body = _first_nonempty(
            row.get("content"),
            row.get("text"),
            row.get("plain_text"),
            row.get("summary"),
            row.get("description"),
            default="",
        )
        note_id = _normalize_note_id(
            _first_nonempty(
                row.get("id"),
                row.get("page_id"),
                row.get("url"),
                default=title,
            )
        )
        links = _extract_links_from_markdown(body)
        extra_links = row.get("links", row.get("references", []))
        if isinstance(extra_links, list):
            links.extend(_safe_string(link) for link in extra_links if _safe_string(link))
        elif isinstance(extra_links, str) and extra_links.strip():
            links.append(extra_links.strip())
        url = _safe_string(row.get("url"))
        if url:
            links.append(url)
        notes.append(
            NoteDocument(
                note_id=note_id or f"notion-{index}",
                title=title,
                text=body,
                links=_unique_nonempty(links),
                path=str(path.name),
            )
        )
    return notes


def _load_markdown_notes(path: Path) -> list[NoteDocument]:
    """Load markdown notes.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    list[NoteDocument]
        Return value description.
    """
    root = path if path.is_dir() else path.parent
    markdown_files = sorted(root.rglob("*.md")) if path.is_dir() else [path]

    notes: list[NoteDocument] = []
    for index, file_path in enumerate(markdown_files, start=1):
        if file_path.name.startswith("."):
            continue
        if any(part.startswith(".") for part in file_path.relative_to(root).parts):
            continue
        content = file_path.read_text(encoding="utf-8", errors="replace")
        title = _title_from_markdown(content=content, fallback=file_path.stem)
        note_id = _normalize_note_id(str(file_path.relative_to(root).with_suffix("")))
        if not note_id:
            note_id = f"note-{index}"
        links = _extract_links_from_markdown(content)
        notes.append(
            NoteDocument(
                note_id=note_id,
                title=title,
                text=content,
                links=links,
                path=str(file_path.relative_to(root)),
            )
        )
    return notes


def _load_onenote_notes(path: Path) -> list[NoteDocument]:
    """Load onenote notes.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    list[NoteDocument]
        Return value description.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("notes", payload.get("pages", []))
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    notes: list[NoteDocument] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        title = _safe_string(row.get("title", "")) or f"OneNote Note {index}"
        body = _safe_string(row.get("content", "")) or _safe_string(row.get("body", "")) or _safe_string(
            row.get("text", "")
        )
        note_id = _normalize_note_id(_safe_string(row.get("id", "")) or title or f"onenote-{index}")
        links = _extract_links_from_markdown(body)
        extra_links = row.get("links", [])
        if isinstance(extra_links, list):
            links.extend(_safe_string(link) for link in extra_links if _safe_string(link))
        elif isinstance(extra_links, str) and extra_links.strip():
            links.append(extra_links.strip())
        notes.append(
            NoteDocument(
                note_id=note_id or f"onenote-{index}",
                title=title,
                text=body,
                links=_unique_nonempty(links),
                path=str(path.name),
            )
        )
    return notes


def _load_evernote_notes(path: Path) -> list[NoteDocument]:
    """Load evernote notes.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    list[NoteDocument]
        Return value description.
    """
    tree = et.parse(path)
    root = tree.getroot()
    notes: list[NoteDocument] = []
    for index, note_elem in enumerate(root.findall("note"), start=1):
        title = _safe_string(note_elem.findtext("title")) or f"Evernote Note {index}"
        raw_content = _safe_string(note_elem.findtext("content"))
        text = _strip_xml_tags(raw_content)
        links = _extract_links_from_html(raw_content)
        note_id = _normalize_note_id(title) or f"evernote-{index}"
        notes.append(
            NoteDocument(
                note_id=note_id,
                title=title,
                text=text,
                links=links,
                path=str(path.name),
            )
        )
    return notes


def _title_from_markdown(content: str, fallback: str) -> str:
    """Title from markdown.

    Parameters
    ----------
    content : str
        Parameter description.
    fallback : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    match = re.search(r"^\s*#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback.strip() or "Untitled Note"


def _extract_links_from_markdown(content: str) -> list[str]:
    """Extract links from markdown.

    Parameters
    ----------
    content : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    links: list[str] = []
    links.extend(match.group(1).strip() for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", content))
    links.extend(match.group(1).strip() for match in re.finditer(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]", content))
    links.extend(match.group(1).strip() for match in re.finditer(r"<(https?://[^>]+)>", content))
    links.extend(match.group(0).strip() for match in re.finditer(r"(?<![\w(])https?://[^\s)>\]]+", content))
    return _unique_nonempty(links)


def _extract_links_from_html(content: str) -> list[str]:
    """Extract links from html.

    Parameters
    ----------
    content : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    links = [match.group(1).strip() for match in re.finditer(r'href=["\']([^"\']+)["\']', content)]
    return _unique_nonempty(links)


def _strip_xml_tags(content: str) -> str:
    """Strip xml tags.

    Parameters
    ----------
    content : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    stripped = re.sub(r"<[^>]+>", " ", content)
    return " ".join(stripped.split())


def _note_lookup_tables(
    notes: list[NoteDocument],
) -> tuple[dict[str, NoteDocument], dict[str, NoteDocument], dict[str, NoteDocument]]:
    """Note lookup tables.

    Parameters
    ----------
    notes : list[NoteDocument]
        Parameter description.

    Returns
    -------
    tuple[dict[str, NoteDocument], dict[str, NoteDocument], dict[str, NoteDocument]]
        Return value description.
    """
    by_id: dict[str, NoteDocument] = {}
    by_title: dict[str, NoteDocument] = {}
    by_stem: dict[str, NoteDocument] = {}
    for note in notes:
        for key in _lookup_keys(note.note_id):
            by_id.setdefault(key, note)
        for key in _lookup_keys(note.title):
            by_title.setdefault(key, note)
        path_stem = Path(note.path).stem if note.path else note.note_id
        for key in _lookup_keys(path_stem):
            by_stem.setdefault(key, note)
    return by_id, by_title, by_stem


def _resolve_internal_link(
    link: str,
    by_id: dict[str, NoteDocument],
    by_title: dict[str, NoteDocument],
    by_stem: dict[str, NoteDocument],
) -> NoteDocument | None:
    """Resolve internal link.

    Parameters
    ----------
    link : str
        Parameter description.
    by_id : dict[str, NoteDocument]
        Parameter description.
    by_title : dict[str, NoteDocument]
        Parameter description.
    by_stem : dict[str, NoteDocument]
        Parameter description.

    Returns
    -------
    NoteDocument | None
        Return value description.
    """
    cleaned = link.strip()
    if not cleaned:
        return None
    cleaned = cleaned.split("#", 1)[0]
    cleaned = cleaned.split("?", 1)[0]
    cleaned = unquote(cleaned)
    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\\", "/")
    if cleaned.endswith(".md"):
        cleaned = cleaned[:-3]
    lookup_candidates = _lookup_keys(cleaned)
    lookup_candidates.extend(_lookup_keys(Path(cleaned).stem))
    for key in lookup_candidates:
        if key in by_id:
            return by_id[key]
        if key in by_title:
            return by_title[key]
        if key in by_stem:
            return by_stem[key]
    return None


def _note_embedding_text(note: NoteDocument) -> str:
    """Note embedding text.

    Parameters
    ----------
    note : NoteDocument
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    return f"title: {note.title}\ncontent: {note.text}"


def _note_node(note_id: str) -> str:
    """Note node.

    Parameters
    ----------
    note_id : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    return f"note:{note_id}"


def _is_external_url(value: str) -> bool:
    """Is external url.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    bool
        Return value description.
    """
    lowered = value.lower().strip()
    if lowered.startswith(("http://", "https://", "onenote:", "evernote:", "notion://", "obsidian://", "mailto:")):
        return True
    return bool(re.match(r"^[a-z][a-z0-9+.\-]*://", lowered))


def _add_relation_if_new(
    graph: KnowledgeGraph,
    relation_index: set[tuple[str, str, str]],
    source: str,
    predicate: str,
    target: str,
) -> bool:
    """Add relation if new.

    Parameters
    ----------
    graph : KnowledgeGraph
        Parameter description.
    relation_index : set[tuple[str, str, str]]
        Parameter description.
    source : str
        Parameter description.
    predicate : str
        Parameter description.
    target : str
        Parameter description.

    Returns
    -------
    bool
        Return value description.
    """
    key = (source, predicate, target)
    if key in relation_index:
        return False
    graph.add_relation(source=source, predicate=predicate, target=target)
    relation_index.add(key)
    return True


def _normalize_note_id(value: str) -> str:
    """Normalize note id.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    cleaned = value.strip().lower()
    cleaned = cleaned.replace("\\", "/")
    cleaned = cleaned.strip("/")
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"[^a-z0-9/_\-]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    cleaned = cleaned.strip("-")
    return cleaned


def _normalize_key(value: str) -> str:
    """Normalize key.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    normalized = _normalize_note_id(value)
    normalized = normalized.replace("/", "-")
    return normalized


def _lookup_keys(value: str) -> list[str]:
    """Lookup keys.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    keys: list[str] = []
    normalized = _normalize_key(value)
    if normalized:
        keys.append(normalized)

    trimmed = _strip_notion_suffix(value)
    trimmed_normalized = _normalize_key(trimmed)
    if trimmed_normalized and trimmed_normalized not in keys:
        keys.append(trimmed_normalized)
    return keys


def _strip_notion_suffix(value: str) -> str:
    """Strip notion suffix.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    cleaned = value.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("_", " ")
    return re.sub(
        r"[\s\-]+(?:[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()


def _first_nonempty(*values: object, default: str = "") -> str:
    """First nonempty.

    Parameters
    ----------
    *values : object
        Parameter description.
    default : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    for value in values:
        text = _safe_string(value)
        if text:
            return text
    return default


def _safe_string(value: object) -> str:
    """Safe string.

    Parameters
    ----------
    value : object
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _unique_nonempty(values: list[str]) -> list[str]:
    """Unique nonempty.

    Parameters
    ----------
    values : list[str]
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(cleaned)
    return output
