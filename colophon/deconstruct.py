"""PDF deconstruction pipeline for bibliography, KG, outline, and prompt artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
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
    """Extract and lightly normalize PDF text using PyMuPDF."""
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency import path
        raise RuntimeError("PyMuPDF is required for deconstruct. Install pymupdf.") from exc

    document = fitz.open(str(pdf_path))
    page_texts = [page.get_text("text") for page in document]
    document.close()
    return _clean_text("\n".join(page_texts))


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
    authors = [chunk.strip() for chunk in re.split(r";| and |,", author_match.group(1)) if chunk.strip()] if author_match else []
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
        for authorhip in authorships:
            if not isinstance(authorhip, dict):
                continue
            author = authorhip.get("author", {})
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
    """Build a lightweight claim/reference knowledge graph."""
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", body_text) if chunk.strip()]
    claims = [s for s in sentences if len(s.split()) >= 8][:40]
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, str]] = []
    for idx, claim in enumerate(claims, start=1):
        claim_id = f"claim-{idx:03d}"
        nodes.append({"id": claim_id, "type": "claim", "label": claim})
        for ref_idx in _citation_indices_from_sentence(claim):
            if 1 <= ref_idx <= len(bibliography):
                ref_id = str(bibliography[ref_idx - 1].get("id", f"ref-{ref_idx:03d}"))
                edges.append({"source": claim_id, "target": ref_id, "relation": "supported_by"})
    for row in bibliography:
        nodes.append({"id": row.get("id", ""), "type": "reference", "label": row.get("title", "")})
    return {"nodes": nodes, "edges": edges}


def _citation_indices_from_sentence(sentence: str) -> list[int]:
    indices: list[int] = []
    for match in re.findall(r"\[(\d+)\]", sentence):
        try:
            indices.append(int(match))
        except ValueError:
            continue
    return indices


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

