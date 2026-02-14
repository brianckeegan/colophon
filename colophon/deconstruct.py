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

@dataclass(slots=True)
class SpacyKGExtractor:
    """Container for deterministic spaCy rule-based extraction components."""

    nlp: object
    entity_ruler: object
    matcher: object
    dependency_matcher: object | None


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
    """Build a deterministic claim/entity/reference graph using spaCy rule matchers."""
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", body_text) if chunk.strip()]
    claims = [sentence for sentence in sentences if len(sentence.split()) >= 8][:40]

    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    node_index: dict[str, dict[str, object]] = {}

    for reference in bibliography:
        ref_id = str(reference.get("id", "")).strip()
        if not ref_id:
            continue
        node = {
            "id": ref_id,
            "type": "reference",
            "label": str(reference.get("title", "")).strip() or ref_id,
            "provenance": {"source": "bibliography", "method": "import"},
        }
        nodes.append(node)
        node_index[ref_id] = node

    extractor = _build_spacy_kg_extractor()

    for claim_index, claim_text in enumerate(claims, start=1):
        claim_id = f"claim-{claim_index:03d}"
        claim_node = {
            "id": claim_id,
            "type": "claim",
            "label": claim_text,
            "provenance": {"source": "body_text", "method": "sentence_split", "sentence_index": claim_index},
        }
        nodes.append(claim_node)
        node_index[claim_id] = claim_node

        entities = _extract_entities_for_claim(extractor=extractor, claim_text=claim_text, claim_id=claim_id)
        for entity in entities:
            entity_id = str(entity["id"])
            if entity_id not in node_index:
                nodes.append(entity)
                node_index[entity_id] = entity
            edges.append(
                {
                    "source": claim_id,
                    "target": entity_id,
                    "relation": "mentions",
                    "provenance": entity.get("provenance", {}),
                }
            )

        edges.extend(_extract_relations_for_claim(extractor=extractor, claim_text=claim_text, claim_id=claim_id, entities=entities))

        for ref_idx in _citation_indices_from_sentence(claim_text):
            if 1 <= ref_idx <= len(bibliography):
                ref_id = str(bibliography[ref_idx - 1].get("id", f"ref-{ref_idx:03d}"))
                edges.append(
                    {
                        "source": claim_id,
                        "target": ref_id,
                        "relation": "supported_by",
                        "provenance": {"source": "citation_marker", "method": "regex", "marker": f"[{ref_idx}]"},
                    }
                )

    return {"nodes": nodes, "edges": edges}


def _build_spacy_kg_extractor() -> SpacyKGExtractor | None:
    """Configure spaCy EntityRuler, Matcher, and DependencyMatcher components."""
    try:
        import spacy
        from spacy.matcher import DependencyMatcher, Matcher
    except Exception:
        return None

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    if "entity_ruler" in nlp.pipe_names:
        entity_ruler = nlp.get_pipe("entity_ruler")
    else:
        entity_ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)

    entity_ruler.add_patterns(
        [
            {"label": "DOMAIN_TERM", "pattern": [{"LOWER": "knowledge"}, {"LOWER": "graph"}]},
            {"label": "DOMAIN_TERM", "pattern": [{"LOWER": "bibliography"}]},
            {"label": "DOMAIN_TERM", "pattern": [{"LOWER": "ontology"}]},
            {"label": "METHOD", "pattern": [{"LOWER": "rule"}, {"LOWER": "based"}]},
            {"label": "METHOD", "pattern": [{"LOWER": "dependency"}, {"LOWER": "matcher"}]},
            {"label": "METHOD", "pattern": [{"LOWER": "entityruler"}]},
            {"label": "CLAIM_VERB", "pattern": [{"LOWER": {"IN": ["supports", "improves", "uses", "causes", "reduces", "increases"]}}]},
        ]
    )

    matcher = Matcher(nlp.vocab)
    matcher.add(
        "RELATION_VERB",
        [[{"LOWER": {"IN": ["supports", "improves", "uses", "causes", "reduces", "increases"]}}]],
    )

    dependency_matcher: DependencyMatcher | None = None
    has_pos = "morphologizer" in nlp.pipe_names or "tagger" in nlp.pipe_names
    has_dep = "parser" in nlp.pipe_names
    if has_pos and has_dep:
        dependency_matcher = DependencyMatcher(nlp.vocab)
        dependency_matcher.add(
            "SUBJECT_VERB_OBJECT",
            [
                [
                    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
                    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
                    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "pobj"]}}},
                ]
            ],
        )

    return SpacyKGExtractor(
        nlp=nlp,
        entity_ruler=entity_ruler,
        matcher=matcher,
        dependency_matcher=dependency_matcher,
    )


def _extract_entities_for_claim(extractor: SpacyKGExtractor | None, claim_text: str, claim_id: str) -> list[dict[str, object]]:
    """Extract entities using EntityRuler + Matcher; produce deterministic node ids."""
    if extractor is None:
        return []

    doc = extractor.nlp(claim_text)
    entity_map: dict[str, dict[str, object]] = {}

    for ent in doc.ents:
        label = ent.text.strip()
        if not label:
            continue
        entity_id = f"entity-{_slugify(label)}"
        entity_map[entity_id] = {
            "id": entity_id,
            "type": "entity",
            "label": label,
            "category": ent.label_,
            "provenance": {
                "source": "body_text",
                "method": "spacy_entityruler",
                "rule": ent.label_,
                "claim_id": claim_id,
                "char_span": [ent.start_char, ent.end_char],
            },
        }

    for _, start, end in extractor.matcher(doc):
        span = doc[start:end]
        label = span.text.strip()
        if not label:
            continue
        entity_id = f"entity-{_slugify(label)}"
        if entity_id in entity_map:
            continue
        entity_map[entity_id] = {
            "id": entity_id,
            "type": "entity",
            "label": label,
            "category": "MATCHED_TERM",
            "provenance": {
                "source": "body_text",
                "method": "spacy_matcher",
                "rule": "RELATION_VERB",
                "claim_id": claim_id,
                "char_span": [span.start_char, span.end_char],
            },
        }

    return list(entity_map.values())


def _extract_relations_for_claim(
    extractor: SpacyKGExtractor | None,
    claim_text: str,
    claim_id: str,
    entities: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Extract deterministic relation edges using Matcher and DependencyMatcher."""
    if extractor is None or len(entities) < 2:
        return []

    doc = extractor.nlp(claim_text)
    sorted_entities = sorted(entities, key=lambda item: str(item.get("label", "")))
    entity_ids = [str(item.get("id", "")) for item in sorted_entities if item.get("id")]
    if len(entity_ids) < 2:
        return []

    edges: list[dict[str, object]] = []

    for _, start, end in extractor.matcher(doc):
        span = doc[start:end]
        relation_label = span.text.lower()
        edges.append(
            {
                "source": entity_ids[0],
                "target": entity_ids[1],
                "relation": relation_label,
                "provenance": {
                    "source": "body_text",
                    "method": "spacy_matcher",
                    "rule": "RELATION_VERB",
                    "claim_id": claim_id,
                    "char_span": [span.start_char, span.end_char],
                },
            }
        )

    if extractor.dependency_matcher is not None:
        for _, token_ids in extractor.dependency_matcher(doc):
            if len(token_ids) < 3:
                continue
            verb = doc[token_ids[0]]
            subject = doc[token_ids[1]]
            obj = doc[token_ids[2]]
            sub_id = _find_entity_id_for_token(entity_ids, sorted_entities, subject)
            obj_id = _find_entity_id_for_token(entity_ids, sorted_entities, obj)
            if not sub_id or not obj_id or sub_id == obj_id:
                continue
            edges.append(
                {
                    "source": sub_id,
                    "target": obj_id,
                    "relation": verb.lemma_.lower() or verb.text.lower(),
                    "provenance": {
                        "source": "body_text",
                        "method": "spacy_dependency_matcher",
                        "rule": "SUBJECT_VERB_OBJECT",
                        "claim_id": claim_id,
                        "token_ids": token_ids,
                    },
                }
            )

    for edge in edges:
        edge.setdefault("context", claim_text)

    return edges


def _find_entity_id_for_token(entity_ids: list[str], entities: list[dict[str, object]], token: object) -> str:
    token_text = getattr(token, "text", "").lower().strip()
    for idx, entity in enumerate(entities):
        label = str(entity.get("label", "")).lower()
        if token_text and token_text in label:
            return entity_ids[idx]
    return ""

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
