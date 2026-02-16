"""PDF deconstruction pipeline for bibliography, KG, outline, and prompt artifacts."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse, request


@dataclass(slots=True)
class DeconstructArtifacts:
    """Output artifact paths produced by the deconstruct pipeline."""

    bibliography_path: Path
    knowledge_graph_path: Path
    outline_path: Path
    prompts_path: Path


def run_deconstruct(pdf_path: str | Path, output_dir: str | Path = "", stem: str = "") -> DeconstructArtifacts:
    """Run the end-to-end PDF deconstruction workflow."""
    source_path = Path(pdf_path)
    if not source_path.exists():
        raise ValueError(f"PDF path does not exist: {source_path}")

    cleaned_text = preprocess_pdf_text(source_path)
    body_text, references_section = split_reference_section(cleaned_text)
    raw_references = extract_reference_entries(references_section)
    bibliography = build_bibliography(raw_references)
    knowledge_graph = build_knowledge_graph(body_text, bibliography)
    outline = build_outline(body_text, source_path.stem)
    prompts = build_reverse_prompts(outline=outline, knowledge_graph=knowledge_graph, bibliography=bibliography)

    target_dir = Path(output_dir) if output_dir else source_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    file_stem = stem.strip() or source_path.stem

    bibliography_path = target_dir / f"{file_stem}_bibliography.json"
    knowledge_graph_path = target_dir / f"{file_stem}_kg.json"
    outline_path = target_dir / f"{file_stem}_outline.json"
    prompts_path = target_dir / f"{file_stem}_prompts.json"

    _write_json(bibliography_path, {"bibliography": bibliography})
    _write_json(knowledge_graph_path, knowledge_graph)
    _write_json(outline_path, outline)
    _write_json(prompts_path, prompts)

    return DeconstructArtifacts(
        bibliography_path=bibliography_path,
        knowledge_graph_path=knowledge_graph_path,
        outline_path=outline_path,
        prompts_path=prompts_path,
    )


def preprocess_pdf_text(pdf_path: str | Path) -> str:
    """Extract and normalize PDF text using Kreuzberg, with PyMuPDF fallback."""
    source_path = Path(pdf_path)
    try:
        raw_text = _extract_pdf_text_with_kreuzberg(source_path)
    except Exception:
        try:
            raw_text = _extract_pdf_text_with_pymupdf(source_path)
        except Exception as pymupdf_error:
            raise RuntimeError(
                "PDF extraction failed via Kreuzberg and PyMuPDF fallback. "
                "Install `kreuzberg` (preferred) or `pymupdf`."
            ) from pymupdf_error
    return _clean_text(raw_text)


def _extract_pdf_text_with_kreuzberg(pdf_path: Path) -> str:
    try:
        import kreuzberg  # type: ignore
    except Exception as exc:
        raise RuntimeError("Kreuzberg is not installed.") from exc

    extraction_result = _invoke_kreuzberg_extractor(module=kreuzberg, pdf_path=pdf_path)
    text = _extract_content_from_kreuzberg_result(extraction_result)
    if not text.strip():
        raise RuntimeError("Kreuzberg returned empty content.")
    return text


def _invoke_kreuzberg_extractor(module: object, pdf_path: Path) -> object:
    extractor_names = ("extract_file_sync", "extract_file_content", "extract_file")
    for name in extractor_names:
        extractor = getattr(module, name, None)
        if not callable(extractor):
            continue
        result = extractor(str(pdf_path))
        if inspect.isawaitable(result):
            return _await_kreuzberg_result(result)
        return result
    raise RuntimeError("Could not find a supported Kreuzberg extraction function.")


def _await_kreuzberg_result(awaitable: object) -> object:
    if not inspect.isawaitable(awaitable):
        return awaitable
    try:
        return asyncio.run(awaitable)
    except RuntimeError as exc:
        raise RuntimeError(
            "Kreuzberg returned an async extractor result, but no synchronous event-loop handoff is available."
        ) from exc


def _extract_content_from_kreuzberg_result(result: object) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, str) and content.strip():
            return content

    content_attr = getattr(result, "content", None)
    if isinstance(content_attr, str) and content_attr.strip():
        return content_attr

    to_markdown = getattr(result, "to_markdown", None)
    if callable(to_markdown):
        markdown = to_markdown()
        if isinstance(markdown, str) and markdown.strip():
            return markdown

    pages = getattr(result, "pages", None)
    if isinstance(pages, list):
        page_contents: list[str] = []
        for page in pages:
            if isinstance(page, dict):
                value = page.get("content")
            else:
                value = getattr(page, "content", None)
            if isinstance(value, str) and value.strip():
                page_contents.append(value)
        if page_contents:
            return "\n\n".join(page_contents)

    raise RuntimeError("Kreuzberg extraction result did not expose text content.")


def _extract_pdf_text_with_pymupdf(pdf_path: Path) -> str:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyMuPDF is not installed.") from exc

    document = fitz.open(str(pdf_path))
    page_texts = [page.get_text("text") for page in document]
    document.close()
    return "\n".join(page_texts)


def _clean_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_reference_section(text: str) -> tuple[str, str]:
    """Split article body and references-like section."""
    match = re.search(r"\n\s*(references|bibliography|works cited)\s*\n", text, flags=re.IGNORECASE)
    if not match:
        return text, ""
    idx = match.start()
    return text[:idx].strip(), text[idx:].strip()


def extract_reference_entries(reference_section: str) -> list[str]:
    """Extract individual reference lines from a references section."""
    if not reference_section:
        return []
    lines = [line.strip() for line in reference_section.splitlines() if line.strip()]
    if lines and re.fullmatch(r"(?i)(references|bibliography|works cited)", lines[0]):
        lines = lines[1:]
    entries: list[str] = []
    current: list[str] = []
    for line in lines:
        starts_new = bool(re.match(r"^(\[\d+\]|\d+\.|\(\d+\)|[A-Z][A-Za-z\-']+,)", line))
        if starts_new and current:
            entries.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(" ".join(current).strip())
    return entries


def build_bibliography(raw_references: list[str]) -> list[dict[str, object]]:
    """Resolve raw citation strings into normalized bibliography rows."""
    rows: list[dict[str, object]] = []
    for index, reference in enumerate(raw_references, start=1):
        parsed = _parse_reference(reference)
        enriched = _lookup_openalex(parsed.get("title", ""))
        row: dict[str, object] = {
            "id": f"ref-{index:03d}",
            "raw": reference,
            "title": parsed.get("title", ""),
            "authors": parsed.get("authors", []),
            "year": parsed.get("year"),
            "venue": parsed.get("venue", ""),
            "doi": "",
            "url": "",
            "openalex_id": "",
        }
        row.update(enriched)
        rows.append(row)
    return rows


def _parse_reference(reference: str) -> dict[str, object]:
    year_match = re.search(r"(19|20)\d{2}", reference)
    year = int(year_match.group(0)) if year_match else None
    author_match = re.match(r"^([^\.]+)\.", reference)
    title_match = re.search(r"\.\s+([^\.]+)\.", reference)
    authors = (
        [chunk.strip() for chunk in re.split(r";| and |,", author_match.group(1)) if chunk.strip()]
        if author_match
        else []
    )
    title = title_match.group(1).strip() if title_match else reference[:160]
    venue = ""
    if title_match:
        after_title = reference[title_match.end() :].strip()
        venue = after_title.split(".")[0].strip() if after_title else ""
    return {"title": title, "authors": authors, "year": year, "venue": venue}


def _lookup_openalex(title: str) -> dict[str, object]:
    if not title.strip():
        return {}
    endpoint = "https://api.openalex.org/works?" + parse.urlencode({"search": title, "per-page": "1"})
    req = request.Request(endpoint, headers={"User-Agent": "colophon-deconstruct/0.1"})
    try:
        with request.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}
    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        return {}
    record = results[0] if isinstance(results[0], dict) else {}
    if not record:
        return {}
    authorships = record.get("authorships", [])
    authors: list[str] = []
    if isinstance(authorships, list):
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author", {})
            if isinstance(author, dict) and isinstance(author.get("display_name"), str):
                authors.append(author["display_name"])
    return {
        "title": record.get("display_name", "") or title,
        "authors": authors,
        "year": record.get("publication_year"),
        "venue": ((record.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        if isinstance(record.get("primary_location"), dict)
        else "",
        "doi": str(record.get("doi", "")),
        "url": str(record.get("id", "")),
        "openalex_id": str(record.get("id", "")),
    }


def build_knowledge_graph(body_text: str, bibliography: list[dict[str, object]]) -> dict[str, object]:
    """Build a deterministic claim/reference graph using sift-kg with fallback."""
    try:
        graph = _build_knowledge_graph_with_sift(body_text=body_text, bibliography=bibliography)
    except Exception as exc:
        graph = _build_knowledge_graph_fallback(body_text=body_text, bibliography=bibliography)
        metadata = graph.setdefault("metadata", {})
        metadata["backend"] = "fallback"
        metadata["fallback_reason"] = str(exc)
    return graph


def _build_knowledge_graph_with_sift(body_text: str, bibliography: list[dict[str, object]]) -> dict[str, object]:
    try:
        from sift_kg import KnowledgeGraph as SiftKnowledgeGraph  # type: ignore
    except Exception as exc:
        raise RuntimeError("sift-kg is not installed.") from exc

    graph = SiftKnowledgeGraph()
    reference_nodes: list[str] = []
    relation_counter = 0

    for index, reference in enumerate(bibliography, start=1):
        reference_id = _reference_identifier(reference=reference, index=index)
        paper_node_id = f"paper:{reference_id}"
        reference_nodes.append(paper_node_id)

        graph.add_entity(
            entity_id=paper_node_id,
            entity_type="REFERENCE",
            name=str(reference.get("title", "")).strip() or reference_id,
            confidence=1.0,
            source_documents=[reference_id],
            attributes=_reference_attributes(reference),
        )

        for author in _author_names(reference):
            author_node_id = f"author:{_slugify(author)}"
            graph.add_entity(
                entity_id=author_node_id,
                entity_type="PERSON",
                name=author,
                confidence=1.0,
                source_documents=[reference_id],
            )
            relation_counter = _add_sift_relation(
                graph=graph,
                relation_counter=relation_counter,
                source_id=paper_node_id,
                target_id=author_node_id,
                relation_type="AUTHORED_BY",
                source_document=reference_id,
                evidence="bibliography author field",
            )

        venue = str(reference.get("venue", "")).strip()
        if venue:
            venue_node_id = f"venue:{_slugify(venue)}"
            graph.add_entity(
                entity_id=venue_node_id,
                entity_type="VENUE",
                name=venue,
                confidence=1.0,
                source_documents=[reference_id],
            )
            relation_counter = _add_sift_relation(
                graph=graph,
                relation_counter=relation_counter,
                source_id=paper_node_id,
                target_id=venue_node_id,
                relation_type="PUBLISHED_IN",
                source_document=reference_id,
                evidence="bibliography venue field",
            )

    for claim_index, claim_text in enumerate(_claim_sentences(body_text), start=1):
        claim_id = f"claim:{claim_index:03d}"
        graph.add_entity(
            entity_id=claim_id,
            entity_type="CLAIM",
            name=claim_text,
            confidence=0.9,
            source_documents=["body_text"],
            attributes={"sentence_index": claim_index},
        )
        for marker_position, reference_idx in enumerate(_citation_indices_from_sentence(claim_text), start=1):
            if 1 <= reference_idx <= len(reference_nodes):
                relation_counter = _add_sift_relation(
                    graph=graph,
                    relation_counter=relation_counter,
                    source_id=claim_id,
                    target_id=reference_nodes[reference_idx - 1],
                    relation_type="SUPPORTED_BY",
                    source_document="body_text",
                    evidence=f"[{reference_idx}]",
                    relation_suffix=f"m{marker_position:02d}",
                )

    exported = graph.export() if hasattr(graph, "export") else {}
    return _normalize_sift_export(exported)


def _add_sift_relation(
    graph: object,
    relation_counter: int,
    source_id: str,
    target_id: str,
    relation_type: str,
    source_document: str,
    evidence: str,
    relation_suffix: str = "",
) -> int:
    relation_counter += 1
    relation_id = f"rel:{relation_counter:06d}"
    if relation_suffix:
        relation_id += f":{relation_suffix}"
    graph.add_relation(
        relation_id=relation_id,
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        confidence=1.0,
        evidence=evidence,
        source_document=source_document,
    )
    return relation_counter


def _normalize_sift_export(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected sift-kg export payload.")

    metadata = payload.get("metadata", {})
    normalized_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    normalized_metadata["backend"] = "sift_kg"

    nodes_raw = payload.get("nodes", [])
    links_raw = payload.get("links", payload.get("edges", []))
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []

    if isinstance(nodes_raw, list):
        for node in nodes_raw:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            entity_type = str(node.get("entity_type", "")).strip().upper()
            label = str(node.get("name", node.get("label", node_id))).strip() or node_id
            normalized: dict[str, object] = {
                "id": node_id,
                "type": _node_type_for_entity_type(entity_type),
                "label": label,
                "provenance": {"source": "sift_kg", "method": "entity_import"},
            }
            if entity_type:
                normalized["entity_type"] = entity_type
            attributes = node.get("attributes", {})
            if isinstance(attributes, dict) and attributes:
                normalized["attributes"] = attributes
            nodes.append(normalized)

    if isinstance(links_raw, list):
        for edge in links_raw:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue
            relation = str(edge.get("relation_type", edge.get("relation", "related_to"))).strip().lower() or "related_to"
            evidence = str(edge.get("evidence", "")).strip()
            provenance: dict[str, object]
            if relation == "supported_by" and evidence:
                provenance = {"source": "citation_marker", "method": "regex", "marker": evidence}
            else:
                provenance = {"source": "sift_kg", "method": "relation_import"}
            normalized_edge: dict[str, object] = {
                "source": source,
                "target": target,
                "relation": relation,
                "provenance": provenance,
            }
            if evidence:
                normalized_edge["evidence"] = evidence
            edges.append(normalized_edge)

    nodes.sort(key=lambda node: (str(node.get("type", "")), str(node.get("id", ""))))
    edges.sort(
        key=lambda edge: (
            str(edge.get("source", "")),
            str(edge.get("relation", "")),
            str(edge.get("target", "")),
        )
    )

    return {
        "metadata": normalized_metadata,
        "nodes": nodes,
        "edges": edges,
    }


def _build_knowledge_graph_fallback(body_text: str, bibliography: list[dict[str, object]]) -> dict[str, object]:
    nodes: dict[str, dict[str, object]] = {}
    edges: list[dict[str, object]] = []
    edge_index: set[tuple[str, str, str]] = set()
    reference_nodes: list[str] = []

    for index, reference in enumerate(bibliography, start=1):
        reference_id = _reference_identifier(reference=reference, index=index)
        paper_node_id = f"paper:{reference_id}"
        reference_nodes.append(paper_node_id)

        _upsert_node(
            nodes=nodes,
            node_id=paper_node_id,
            node_type="reference",
            label=str(reference.get("title", "")).strip() or reference_id,
            provenance={"source": "bibliography", "method": "import"},
            attributes=_reference_attributes(reference),
        )

        for author in _author_names(reference):
            author_node_id = f"author:{_slugify(author)}"
            _upsert_node(
                nodes=nodes,
                node_id=author_node_id,
                node_type="entity",
                label=author,
                provenance={"source": "bibliography", "method": "author_parse"},
            )
            _append_edge(
                edges=edges,
                edge_index=edge_index,
                source=paper_node_id,
                target=author_node_id,
                relation="authored_by",
                provenance={"source": "bibliography", "method": "author_parse"},
            )

        venue = str(reference.get("venue", "")).strip()
        if venue:
            venue_node_id = f"venue:{_slugify(venue)}"
            _upsert_node(
                nodes=nodes,
                node_id=venue_node_id,
                node_type="entity",
                label=venue,
                provenance={"source": "bibliography", "method": "venue_parse"},
            )
            _append_edge(
                edges=edges,
                edge_index=edge_index,
                source=paper_node_id,
                target=venue_node_id,
                relation="published_in",
                provenance={"source": "bibliography", "method": "venue_parse"},
            )

    for claim_index, claim_text in enumerate(_claim_sentences(body_text), start=1):
        claim_id = f"claim:{claim_index:03d}"
        _upsert_node(
            nodes=nodes,
            node_id=claim_id,
            node_type="claim",
            label=claim_text,
            provenance={"source": "body_text", "method": "sentence_split", "sentence_index": claim_index},
        )
        for reference_idx in _citation_indices_from_sentence(claim_text):
            if 1 <= reference_idx <= len(reference_nodes):
                marker = f"[{reference_idx}]"
                _append_edge(
                    edges=edges,
                    edge_index=edge_index,
                    source=claim_id,
                    target=reference_nodes[reference_idx - 1],
                    relation="supported_by",
                    provenance={"source": "citation_marker", "method": "regex", "marker": marker},
                    evidence=marker,
                )

    normalized_nodes = sorted(nodes.values(), key=lambda node: (str(node.get("type", "")), str(node.get("id", ""))))
    normalized_edges = sorted(
        edges,
        key=lambda edge: (
            str(edge.get("source", "")),
            str(edge.get("relation", "")),
            str(edge.get("target", "")),
        ),
    )
    return {
        "metadata": {"backend": "fallback"},
        "nodes": normalized_nodes,
        "edges": normalized_edges,
    }


def _upsert_node(
    nodes: dict[str, dict[str, object]],
    node_id: str,
    node_type: str,
    label: str,
    provenance: dict[str, object],
    attributes: dict[str, object] | None = None,
) -> None:
    if node_id in nodes:
        return
    node: dict[str, object] = {
        "id": node_id,
        "type": node_type,
        "label": label,
        "provenance": provenance,
    }
    if attributes:
        node["attributes"] = attributes
    nodes[node_id] = node


def _append_edge(
    edges: list[dict[str, object]],
    edge_index: set[tuple[str, str, str]],
    source: str,
    target: str,
    relation: str,
    provenance: dict[str, object],
    evidence: str = "",
) -> None:
    edge_key = (source, relation, target)
    if edge_key in edge_index:
        return
    edge_index.add(edge_key)
    payload: dict[str, object] = {
        "source": source,
        "target": target,
        "relation": relation,
        "provenance": provenance,
    }
    if evidence:
        payload["evidence"] = evidence
    edges.append(payload)


def _reference_identifier(reference: dict[str, object], index: int) -> str:
    raw_id = str(reference.get("id", "")).strip()
    return raw_id or f"ref-{index:03d}"


def _reference_attributes(reference: dict[str, object]) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for key in ("year", "venue", "doi", "url", "openalex_id"):
        value = reference.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            attributes[key] = text
    return attributes


def _author_names(reference: dict[str, object]) -> list[str]:
    value = reference.get("authors", [])
    if isinstance(value, list):
        return [str(author).strip() for author in value if str(author).strip()]
    if isinstance(value, str):
        return [chunk.strip() for chunk in re.split(r";| and |,", value) if chunk.strip()]
    return []


def _node_type_for_entity_type(entity_type: str) -> str:
    if entity_type == "CLAIM":
        return "claim"
    if entity_type == "REFERENCE":
        return "reference"
    return "entity"


def _claim_sentences(body_text: str, max_claims: int = 40) -> list[str]:
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", body_text) if chunk.strip()]
    return [sentence for sentence in sentences if len(sentence.split()) >= 8][:max_claims]


def _citation_indices_from_sentence(sentence: str) -> list[int]:
    indices: list[int] = []
    for match in re.findall(r"\[(\d+)\]", sentence):
        try:
            indices.append(int(match))
        except ValueError:
            continue
    return indices


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def build_outline(body_text: str, fallback_title: str) -> dict[str, object]:
    """Create a simple article outline object."""
    paragraphs = [chunk.strip() for chunk in body_text.split("\n\n") if chunk.strip()]
    title = paragraphs[0].split("\n")[0].strip() if paragraphs else fallback_title
    intro_summary = paragraphs[1][:300] if len(paragraphs) > 1 else ""
    sections: list[dict[str, str]] = []
    for idx, paragraph in enumerate(paragraphs[2:8], start=1):
        heading = f"Section {idx}"
        sentence = re.split(r"(?<=[.!?])\s+", paragraph)[0]
        sections.append({"title": heading, "summary": sentence[:300]})
    return {
        "title": title,
        "chapters": [
            {
                "id": "chapter-1",
                "title": title,
                "sections": [{"id": "intro", "title": "Introduction", "summary": intro_summary}, *sections],
            }
        ],
    }


def build_reverse_prompts(
    outline: dict[str, object], knowledge_graph: dict[str, object], bibliography: list[dict[str, object]]
) -> dict[str, object]:
    """Generate reverse prompts from extracted ontology-like artifacts."""
    chapter = ((outline.get("chapters") or [{}])[0]) if isinstance(outline.get("chapters"), list) else {}
    chapter_title = chapter.get("title", "the article") if isinstance(chapter, dict) else "the article"
    claim_nodes = [node for node in knowledge_graph.get("nodes", []) if isinstance(node, dict) and node.get("type") == "claim"]
    example_claim = claim_nodes[0].get("label", "") if claim_nodes else ""
    top_sources = [str(item.get("title", "")) for item in bibliography[:3]]
    return {
        "prompts": {
            "recreate_outline": f"Draft an outline for {chapter_title} with clear argument progression and evidence integration.",
            "recreate_claim_style": (
                "Write analytical claims in the same style as this example: "
                f"{example_claim}. Ensure each claim can be tied to a citable source."
            ).strip(),
            "recreate_references": (
                "Use and discuss literature with similar scope to: " + "; ".join([source for source in top_sources if source])
            ).strip(),
        }
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
