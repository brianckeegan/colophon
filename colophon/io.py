"""I/O loaders and serializers for bibliography, graph, outline, and config files."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

from .graph import KnowledgeGraph, graph_from_dict
from .kg_update import KGUpdateConfig
from .vectors import EmbeddingConfig
from .llm import LLMConfig
from .models import Figure, Source
from .recommendations import RecommendationConfig


def load_json(path: str | Path) -> dict:
    """Load json.

    Parameters
    ----------
    path : str | Path
        Parameter description.

    Returns
    -------
    dict
        Return value description.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_bibliography(path: str | Path) -> list[Source]:
    """Load bibliography.

    Parameters
    ----------
    path : str | Path
        Parameter description.

    Returns
    -------
    list[Source]
        Return value description.
    """
    return load_bibliography_with_format(path=path, bibliography_format="auto")


def load_bibliography_with_format(path: str | Path, bibliography_format: str = "auto") -> list[Source]:
    """Load bibliography with format.

    Parameters
    ----------
    path : str | Path
        Parameter description.
    bibliography_format : str
        Parameter description.

    Returns
    -------
    list[Source]
        Return value description.
    """
    bibliography_path = Path(path)
    resolved_format = _resolve_bibliography_format(path=bibliography_path, bibliography_format=bibliography_format)

    if resolved_format == "json":
        return _load_bibliography_from_json(bibliography_path)
    if resolved_format == "csv":
        return _load_bibliography_from_csv(bibliography_path)
    if resolved_format == "bibtex":
        return _load_bibliography_from_bibtex(bibliography_path)

    raise ValueError(f"Unsupported bibliography format: {resolved_format}")


def load_outline(path: str | Path) -> list[dict]:
    """Load outline.

    Parameters
    ----------
    path : str | Path
        Parameter description.

    Returns
    -------
    list[dict]
        Return value description.
    """
    payload = load_json(path)
    return payload.get("chapters", [])


def _resolve_bibliography_format(path: Path, bibliography_format: str) -> str:
    """Resolve bibliography format.

    Parameters
    ----------
    path : Path
        Parameter description.
    bibliography_format : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    normalized = bibliography_format.strip().lower()
    if normalized != "auto":
        return normalized

    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".bib", ".bibtex"}:
        return "bibtex"
    raise ValueError(
        f"Could not infer bibliography format for {path}. "
        "Use --bibliography-format with one of: json,csv,bibtex."
    )


def _load_bibliography_from_json(path: Path) -> list[Source]:
    """Load bibliography from json.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    list[Source]
        Return value description.
    """
    payload = load_json(path)
    if isinstance(payload, dict):
        rows = payload.get("sources", payload.get("bibliography", []))
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    if not isinstance(rows, list):
        raise ValueError("JSON bibliography must be a list or contain a list under 'sources' or 'bibliography'.")

    return [_source_from_mapping(row, index=idx) for idx, row in enumerate(rows, start=1) if isinstance(row, dict)]


def _load_bibliography_from_csv(path: Path) -> list[Source]:
    """Load bibliography from csv.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    list[Source]
        Return value description.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]

    sources: list[Source] = []
    for idx, row in enumerate(rows, start=1):
        if not row:
            continue
        mapped = {str(key): str(value) if value is not None else "" for key, value in row.items() if key is not None}
        sources.append(_source_from_mapping(mapped, index=idx))
    return sources


def _load_bibliography_from_bibtex(path: Path) -> list[Source]:
    """Load bibliography from bibtex.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    list[Source]
        Return value description.
    """
    content = path.read_text(encoding="utf-8")
    entries = _parse_bibtex_entries(content)

    sources: list[Source] = []
    for idx, entry in enumerate(entries, start=1):
        fields = {key.lower(): value for key, value in entry["fields"].items()}
        publication = (
            fields.get("publication")
            or fields.get("journal")
            or fields.get("booktitle")
            or fields.get("publisher")
            or ""
        )
        mapped = {
            "id": entry.get("key", ""),
            "title": fields.get("title", ""),
            "authors": fields.get("author", ""),
            "year": fields.get("year", ""),
            "text": fields.get("abstract", fields.get("note", "")),
            "publication": publication,
            "entry_type": entry.get("entry_type", ""),
        }
        for key, value in fields.items():
            if key not in {"title", "author", "year", "abstract", "note", "journal", "booktitle", "publisher"}:
                mapped[f"bib_{key}"] = value
        sources.append(_source_from_mapping(mapped, index=idx))
    return sources


def load_prompts(path: str | Path) -> dict[str, str]:
    """Load prompts.

    Parameters
    ----------
    path : str | Path
        Parameter description.

    Returns
    -------
    dict[str, str]
        Return value description.
    """
    payload = load_json(path)
    prompts = payload.get("prompts", payload)
    if not isinstance(prompts, dict):
        raise ValueError("Prompts file must be a JSON object or contain a top-level 'prompts' object.")

    normalized: dict[str, str] = {}
    for key, value in prompts.items():
        if isinstance(value, str):
            normalized[key] = value
    return normalized


def load_llm_config(path: str | Path) -> LLMConfig:
    """Load llm config.

    Parameters
    ----------
    path : str | Path
        Parameter description.

    Returns
    -------
    LLMConfig
        Return value description.
    """
    payload = load_json(path)
    raw = payload.get("llm", payload)
    if not isinstance(raw, dict):
        raise ValueError("LLM config file must be a JSON object or contain a top-level 'llm' object.")

    return LLMConfig(
        provider=_string_or_default(raw.get("provider"), "none"),
        model=_string_or_default(raw.get("model"), ""),
        api_base_url=_string_or_none(raw.get("api_base_url")),
        api_key_env=_string_or_none(raw.get("api_key_env")),
        temperature=_float_or_default(raw.get("temperature"), 0.2),
        max_tokens=_int_or_default(raw.get("max_tokens"), 512),
        timeout_seconds=_float_or_default(raw.get("timeout_seconds"), 30.0),
        system_prompt=_string_or_default(raw.get("system_prompt"), ""),
        extra_headers=_dict_str_str(raw.get("extra_headers")),
    )


def load_recommendation_config(path: str | Path) -> RecommendationConfig:
    """Load recommendation config.

    Parameters
    ----------
    path : str | Path
        Parameter description.

    Returns
    -------
    RecommendationConfig
        Return value description.
    """
    payload = load_json(path)
    raw = payload.get("recommendation", payload)
    if not isinstance(raw, dict):
        raise ValueError(
            "Recommendation config file must be a JSON object or contain a top-level 'recommendation' object."
        )

    return RecommendationConfig(
        provider=_string_or_default(raw.get("provider"), "openalex"),
        api_base_url=_string_or_default(raw.get("api_base_url"), "https://api.openalex.org"),
        timeout_seconds=_float_or_default(raw.get("timeout_seconds"), 20.0),
        per_seed_limit=max(1, _int_or_default(raw.get("per_seed_limit"), 5)),
        top_k=max(0, _int_or_default(raw.get("top_k"), 8)),
        min_score=max(0.0, min(1.0, _float_or_default(raw.get("min_score"), 0.2))),
        mailto=_string_or_default(raw.get("mailto"), ""),
        api_key_env=_string_or_default(raw.get("api_key_env"), ""),
    )


def load_kg_update_config(path: str | Path) -> KGUpdateConfig:
    """Load kg update config.

    Parameters
    ----------
    path : str | Path
        Parameter description.

    Returns
    -------
    KGUpdateConfig
        Return value description.
    """
    payload = load_json(path)
    raw = payload.get("kg_update", payload)
    if not isinstance(raw, dict):
        raise ValueError("KG update config file must be a JSON object or contain a top-level 'kg_update' object.")

    embedding_raw = raw.get("embedding", raw.get("embeddings", {}))
    if not isinstance(embedding_raw, dict):
        embedding_raw = {}

    embedding_config = EmbeddingConfig(
        provider=_string_or_default(embedding_raw.get("provider"), "local"),
        model=_string_or_default(embedding_raw.get("model"), ""),
        api_base_url=_string_or_none(embedding_raw.get("api_base_url")),
        api_key_env=_string_or_none(embedding_raw.get("api_key_env")),
        dimensions=max(8, _int_or_default(embedding_raw.get("dimensions"), 256)),
        timeout_seconds=max(1.0, _float_or_default(embedding_raw.get("timeout_seconds"), 20.0)),
    )

    return KGUpdateConfig(
        embedding_config=embedding_config,
        vector_db_path=_string_or_default(raw.get("vector_db_path"), ""),
        rag_top_k=max(0, _int_or_default(raw.get("rag_top_k"), 3)),
        similarity_threshold=max(
            0.0,
            min(1.0, _float_or_default(raw.get("similarity_threshold"), 0.2)),
        ),
        max_entities_per_doc=max(1, _int_or_default(raw.get("max_entities_per_doc"), 8)),
    )


def load_graph(path: str | Path, graph_format: str = "auto") -> KnowledgeGraph:
    """Load graph.

    Parameters
    ----------
    path : str | Path
        Parameter description.
    graph_format : str
        Parameter description.

    Returns
    -------
    KnowledgeGraph
        Return value description.
    """
    graph_path = Path(path)
    resolved_format = _resolve_graph_format(graph_path=graph_path, graph_format=graph_format)

    if resolved_format == "json":
        payload = load_json(graph_path)
        return graph_from_dict(payload)
    if resolved_format == "csv":
        return _load_graph_from_csv(graph_path)
    if resolved_format in {"sqlite", "sql"}:
        return _load_graph_from_sqlite(graph_path, is_sql_dump=resolved_format == "sql")

    raise ValueError(f"Unsupported graph format: {resolved_format}")


def write_text(path: str | Path, content: str) -> None:
    """Write text.

    Parameters
    ----------
    path : str | Path
        Parameter description.
    content : str
        Parameter description.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _resolve_graph_format(graph_path: Path, graph_format: str) -> str:
    """Resolve graph format.

    Parameters
    ----------
    graph_path : Path
        Parameter description.
    graph_format : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    normalized = graph_format.strip().lower()
    if normalized != "auto":
        return normalized

    suffix = graph_path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        return "sqlite"
    if suffix in {".sql", ".dump"}:
        return "sql"
    raise ValueError(
        f"Could not infer graph format for {graph_path}. "
        "Use --graph-format with one of: json,csv,sqlite,sql."
    )


def _load_graph_from_csv(path: Path) -> KnowledgeGraph:
    """Load graph from csv.

    Parameters
    ----------
    path : Path
        Parameter description.

    Returns
    -------
    KnowledgeGraph
        Return value description.
    """
    graph = KnowledgeGraph()

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return graph

    header = [column.strip().lower() for column in rows[0]]
    has_header = bool(
        _first_present_index(header, ("source", "src", "from", "head")) is not None
        and _first_present_index(header, ("target", "dst", "to", "tail")) is not None
    )

    if has_header:
        source_idx = _first_present_index(header, ("source", "src", "from", "head"))
        target_idx = _first_present_index(header, ("target", "dst", "to", "tail"))
        predicate_idx = _first_present_index(header, ("predicate", "relation", "type", "label"))
        assert source_idx is not None
        assert target_idx is not None

        for row in rows[1:]:
            source = _row_value(row, source_idx)
            target = _row_value(row, target_idx)
            predicate = _row_value(row, predicate_idx) if predicate_idx is not None else None
            if source and target:
                graph.add_relation(source=source, predicate=predicate or "related_to", target=target)
    else:
        for row in rows:
            if len(row) < 2:
                continue
            source = row[0].strip()
            target = row[1].strip()
            predicate = row[2].strip() if len(row) > 2 and row[2].strip() else "related_to"
            if source and target:
                graph.add_relation(source=source, predicate=predicate, target=target)

    return graph


def _load_graph_from_sqlite(path: Path, is_sql_dump: bool) -> KnowledgeGraph:
    """Load graph from sqlite.

    Parameters
    ----------
    path : Path
        Parameter description.
    is_sql_dump : bool
        Parameter description.

    Returns
    -------
    KnowledgeGraph
        Return value description.
    """
    if is_sql_dump:
        connection = sqlite3.connect(":memory:")
        with path.open("r", encoding="utf-8") as handle:
            connection.executescript(handle.read())
    else:
        connection = sqlite3.connect(path)

    try:
        return _graph_from_sqlite_connection(connection)
    finally:
        connection.close()


def _graph_from_sqlite_connection(connection: sqlite3.Connection) -> KnowledgeGraph:
    """Graph from sqlite connection.

    Parameters
    ----------
    connection : sqlite3.Connection
        Parameter description.

    Returns
    -------
    KnowledgeGraph
        Return value description.
    """
    graph = KnowledgeGraph()
    table_rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [row[0] for row in table_rows]

    relation_table = None
    relation_columns: list[str] = []
    for table_name in table_names:
        columns = _table_columns(connection, table_name)
        if _find_column(columns, ("source", "src", "from", "head")) and _find_column(
            columns, ("target", "dst", "to", "tail")
        ):
            relation_table = table_name
            relation_columns = columns
            break

    if relation_table is None:
        return graph

    source_col = _find_column(relation_columns, ("source", "src", "from", "head"))
    target_col = _find_column(relation_columns, ("target", "dst", "to", "tail"))
    predicate_col = _find_column(relation_columns, ("predicate", "relation", "type", "label"))
    assert source_col is not None
    assert target_col is not None

    if predicate_col:
        query = (
            f'SELECT "{source_col}" AS source, "{target_col}" AS target, "{predicate_col}" AS predicate '
            f'FROM "{relation_table}"'
        )
    else:
        query = f'SELECT "{source_col}" AS source, "{target_col}" AS target FROM "{relation_table}"'

    for row in connection.execute(query):
        source = str(row[0]).strip() if row[0] is not None else ""
        target = str(row[1]).strip() if row[1] is not None else ""
        predicate = "related_to"
        if predicate_col and row[2] is not None:
            cleaned = str(row[2]).strip()
            if cleaned:
                predicate = cleaned
        if source and target:
            graph.add_relation(source=source, predicate=predicate, target=target)

    for table_name in table_names:
        columns = _table_columns(connection, table_name)
        entity_col = _find_column(columns, ("entity", "name"))
        if entity_col is None:
            continue
        query = f'SELECT "{entity_col}" FROM "{table_name}"'
        for (entity,) in connection.execute(query):
            if entity is not None and str(entity).strip():
                graph.entities.add(str(entity).strip())

    for table_name in table_names:
        columns = _table_columns(connection, table_name)
        figure_id_col = _find_column(columns, ("figure_id", "id"))
        caption_col = _find_column(columns, ("caption", "title", "label"))
        uri_col = _find_column(columns, ("uri", "url", "path", "file_path", "image_path"))
        if figure_id_col is None or caption_col is None or uri_col is None:
            continue

        related_entities_col = _find_column(columns, ("related_entities", "entities", "entity_tags"))
        alt_text_col = _find_column(columns, ("alt_text", "alt", "description"))

        select_columns = [figure_id_col, caption_col, uri_col]
        if related_entities_col is not None:
            select_columns.append(related_entities_col)
        if alt_text_col is not None:
            select_columns.append(alt_text_col)

        aliased = ", ".join(f'"{column}"' for column in select_columns)
        query = f'SELECT {aliased} FROM "{table_name}"'
        for row in connection.execute(query):
            figure_id = str(row[0]).strip() if row[0] is not None else ""
            caption = str(row[1]).strip() if row[1] is not None else ""
            uri = str(row[2]).strip() if row[2] is not None else ""
            if not figure_id or not caption:
                continue

            offset = 3
            related_entities = []
            if related_entities_col is not None:
                related_value = row[offset]
                if related_value is not None:
                    related_entities = _split_entities(str(related_value))
                offset += 1

            alt_text = ""
            if alt_text_col is not None and offset < len(row) and row[offset] is not None:
                alt_text = str(row[offset]).strip()

            graph.add_figure(
                Figure(
                    id=figure_id,
                    caption=caption,
                    uri=uri,
                    alt_text=alt_text,
                    related_entities=related_entities,
                )
            )

    return graph


def _table_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    """Table columns.

    Parameters
    ----------
    connection : sqlite3.Connection
        Parameter description.
    table_name : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    rows = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return [row[1] for row in rows]


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Find column.

    Parameters
    ----------
    columns : list[str]
        Parameter description.
    candidates : tuple[str, ...]
        Parameter description.

    Returns
    -------
    str | None
        Return value description.
    """
    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    return None


def _first_present_index(columns: list[str], candidates: tuple[str, ...]) -> int | None:
    """First present index.

    Parameters
    ----------
    columns : list[str]
        Parameter description.
    candidates : tuple[str, ...]
        Parameter description.

    Returns
    -------
    int | None
        Return value description.
    """
    for candidate in candidates:
        if candidate in columns:
            return columns.index(candidate)
    return None


def _row_value(row: list[str], index: int) -> str | None:
    """Row value.

    Parameters
    ----------
    row : list[str]
        Parameter description.
    index : int
        Parameter description.

    Returns
    -------
    str | None
        Return value description.
    """
    if index >= len(row):
        return None
    cleaned = row[index].strip()
    return cleaned or None


def _source_from_mapping(row: dict[str, object], index: int) -> Source:
    """Source from mapping.

    Parameters
    ----------
    row : dict[str, object]
        Parameter description.
    index : int
        Parameter description.

    Returns
    -------
    Source
        Return value description.
    """
    normalized = {str(key).lower(): value for key, value in row.items()}

    source_id = _as_nonempty_string(
        _first_value(normalized, ("id", "key", "citation_key", "citationkey"))
    )
    title = _as_nonempty_string(_first_value(normalized, ("title", "name")))
    authors_raw = _first_value(normalized, ("authors", "author"))
    year_raw = _first_value(normalized, ("year", "date", "publication_year"))
    text = _as_nonempty_string(_first_value(normalized, ("text", "abstract", "summary", "content")))
    publication = _as_nonempty_string(
        _first_value(normalized, ("publication", "journal", "booktitle", "venue", "publisher"))
    )

    authors = _parse_authors(authors_raw)
    year = _parse_year(year_raw)

    metadata: dict[str, object] = {}
    if publication:
        metadata["publication"] = publication

    consumed = {
        "id",
        "key",
        "citation_key",
        "citationkey",
        "title",
        "name",
        "authors",
        "author",
        "year",
        "date",
        "publication_year",
        "text",
        "abstract",
        "summary",
        "content",
        "publication",
        "journal",
        "booktitle",
        "venue",
        "publisher",
    }
    for key, value in row.items():
        lowered = str(key).lower()
        if lowered in consumed:
            continue
        if value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            metadata[str(key)] = value_text

    resolved_title = title or f"Untitled Source {index}"
    resolved_id = source_id or _synth_source_id(resolved_title, index=index)

    return Source(
        id=resolved_id,
        title=resolved_title,
        authors=authors,
        year=year,
        text=text,
        metadata=metadata,
    )


def _first_value(payload: dict[str, object], keys: tuple[str, ...]) -> object | None:
    """First value.

    Parameters
    ----------
    payload : dict[str, object]
        Parameter description.
    keys : tuple[str, ...]
        Parameter description.

    Returns
    -------
    object | None
        Return value description.
    """
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _as_nonempty_string(value: object) -> str:
    """As nonempty string.

    Parameters
    ----------
    value : object
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _parse_authors(value: object) -> list[str]:
    """Parse authors.

    Parameters
    ----------
    value : object
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    if " and " in text:
        return [part.strip() for part in text.split(" and ") if part.strip()]
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text]


def _parse_year(value: object) -> int | None:
    """Parse year.

    Parameters
    ----------
    value : object
        Parameter description.

    Returns
    -------
    int | None
        Return value description.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group(0))
    if text.isdigit():
        return int(text)
    return None


def _synth_source_id(title: str, index: int) -> str:
    """Synth source id.

    Parameters
    ----------
    title : str
        Parameter description.
    index : int
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    collapsed = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    base = collapsed or "source"
    return f"{base}-{index}"


def _parse_bibtex_entries(content: str) -> list[dict[str, object]]:
    """Parse bibtex entries.

    Parameters
    ----------
    content : str
        Parameter description.

    Returns
    -------
    list[dict[str, object]]
        Return value description.
    """
    entries: list[dict[str, object]] = []
    cursor = 0
    length = len(content)

    while cursor < length:
        at_index = content.find("@", cursor)
        if at_index == -1:
            break
        brace_index = content.find("{", at_index)
        if brace_index == -1:
            break

        entry_type = content[at_index + 1 : brace_index].strip().lower()
        body, next_cursor = _consume_balanced_block(content, brace_index)
        cursor = next_cursor
        if not body:
            continue

        first_comma = _find_unescaped_top_level_comma(body)
        if first_comma == -1:
            continue

        key = body[:first_comma].strip()
        fields_text = body[first_comma + 1 :]
        fields = _parse_bibtex_fields(fields_text)
        entries.append({"entry_type": entry_type, "key": key, "fields": fields})

    return entries


def _consume_balanced_block(content: str, open_brace_index: int) -> tuple[str, int]:
    """Consume balanced block.

    Parameters
    ----------
    content : str
        Parameter description.
    open_brace_index : int
        Parameter description.

    Returns
    -------
    tuple[str, int]
        Return value description.
    """
    depth = 0
    start = open_brace_index + 1
    cursor = open_brace_index
    while cursor < len(content):
        char = content[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start:cursor], cursor + 1
        cursor += 1
    return "", len(content)


def _find_unescaped_top_level_comma(text: str) -> int:
    """Find unescaped top level comma.

    Parameters
    ----------
    text : str
        Parameter description.

    Returns
    -------
    int
        Return value description.
    """
    depth = 0
    in_quotes = False
    escape = False
    for idx, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            continue
        if in_quotes:
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth = max(0, depth - 1)
            continue
        if char == "," and depth == 0:
            return idx
    return -1


def _parse_bibtex_fields(text: str) -> dict[str, str]:
    """Parse bibtex fields.

    Parameters
    ----------
    text : str
        Parameter description.

    Returns
    -------
    dict[str, str]
        Return value description.
    """
    fields: dict[str, str] = {}
    cursor = 0
    length = len(text)
    while cursor < length:
        while cursor < length and text[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= length:
            break

        name_start = cursor
        while cursor < length and re.match(r"[A-Za-z0-9_\-]", text[cursor]):
            cursor += 1
        field_name = text[name_start:cursor].strip().lower()
        if not field_name:
            break

        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor >= length or text[cursor] != "=":
            break
        cursor += 1
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor >= length:
            break

        field_value, cursor = _parse_bibtex_value(text, cursor)
        cleaned = _clean_bibtex_value(field_value)
        if cleaned:
            fields[field_name] = cleaned
    return fields


def _parse_bibtex_value(text: str, cursor: int) -> tuple[str, int]:
    """Parse bibtex value.

    Parameters
    ----------
    text : str
        Parameter description.
    cursor : int
        Parameter description.

    Returns
    -------
    tuple[str, int]
        Return value description.
    """
    if text[cursor] == "{":
        depth = 0
        start = cursor + 1
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:cursor], cursor + 1
            cursor += 1
        return text[start:], len(text)

    if text[cursor] == '"':
        cursor += 1
        start = cursor
        escaped = False
        while cursor < len(text):
            char = text[cursor]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return text[start:cursor], cursor + 1
            cursor += 1
        return text[start:], len(text)

    start = cursor
    while cursor < len(text) and text[cursor] not in ",\n\r":
        cursor += 1
    return text[start:cursor], cursor


def _clean_bibtex_value(value: str) -> str:
    """Clean bibtex value.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    compact = " ".join(value.replace("\n", " ").replace("\r", " ").split())
    compact = compact.strip().strip(",")
    return compact


def _string_or_none(value: object) -> str | None:
    """String or none.

    Parameters
    ----------
    value : object
        Parameter description.

    Returns
    -------
    str | None
        Return value description.
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _string_or_default(value: object, default: str) -> str:
    """String or default.

    Parameters
    ----------
    value : object
        Parameter description.
    default : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    resolved = _string_or_none(value)
    return resolved if resolved is not None else default


def _float_or_default(value: object, default: float) -> float:
    """Float or default.

    Parameters
    ----------
    value : object
        Parameter description.
    default : float
        Parameter description.

    Returns
    -------
    float
        Return value description.
    """
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _int_or_default(value: object, default: int) -> int:
    """Int or default.

    Parameters
    ----------
    value : object
        Parameter description.
    default : int
        Parameter description.

    Returns
    -------
    int
        Return value description.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _dict_str_str(value: object) -> dict[str, str]:
    """Dict str str.

    Parameters
    ----------
    value : object
        Parameter description.

    Returns
    -------
    dict[str, str]
        Return value description.
    """
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            result[key] = item
    return result


def _split_entities(value: str) -> list[str]:
    """Split entities.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    separators = [",", ";", "|"]
    parts = [value]
    for separator in separators:
        if separator in value:
            parts = [chunk for piece in parts for chunk in piece.split(separator)]
    return [part.strip() for part in parts if part.strip()]
