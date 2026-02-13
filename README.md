# Colophon

Colophon is a cooperative multi-agent system for long-form writing. It combines a seed knowledge graph and retrieval-augmented evidence over a bibliography to generate chapter/section drafts with citations and diagnostics.

## What this implementation includes

- Core data model for sources, claims, paragraphs, sections, chapters, and manuscripts.
- Knowledge graph primitives for entity/relation context.
- Lightweight retrieval for evidence ranking.
- Multi-agent authoring pipeline:
  - Claim author agent
  - Paragraph synthesis agent
  - Section orchestrator
  - Paragraph/section/chapter/book coordination + editing agents with message passing
  - Citation, figure-reference, and coherence reviewer agents
- CLI for end-to-end generation from JSON inputs.
- Optional LLM API hooks for claim/paragraph generation via provider adapters.
- Optional outline expander agent for transforming draft outlines into detailed plans + prompts.
- Optional KG generator/updater with local or remote embeddings + vector similarity.
- Optional notes importers (Obsidian, Notion, OneNote, Evernote) to construct/extend KGs via hyperlinks and embeddings.
- Narrative customization controls for tone, style, audience, discipline, genre, and language.
- `colophon.genre_ontology` profile support to set defaults/prompts that influence agents, outline expansion, recommendations, and soft validation.
- Optional functional-form soft validation for outline/agents/claims/bibliography/prompts.
- Technical-writing form ontology support (IMRaD, systems tradeoffs, theory/proof, specs, surveys) for coordination, outline expansion, and soft validation.
- Optional Wilson companion writing ontology for background prompts, assumptions, and additional validations.
- Optional scientometric recommendation workflow (OpenAlex/Scholar-style) for bibliography and KG update proposals.
- Sphinx documentation and GitHub Pages deployment workflow.

## Documentation map

- Getting started: `/Users/briankeegan/Documents/New project/docs/getting_started.rst`
- User guide (CLI/options/formats): `/Users/briankeegan/Documents/New project/docs/usage.rst`
- End-to-end tutorial: `/Users/briankeegan/Documents/New project/docs/tutorial.rst`
- Example asset index: `/Users/briankeegan/Documents/New project/docs/examples.rst`
- API reference (modules/classes/functions): `/Users/briankeegan/Documents/New project/docs/api.rst`
- References/citations: `/Users/briankeegan/Documents/New project/docs/references.rst`

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[docs]
```

### 2) Generate a draft from the examples

```bash
colophon \
  --bibliography examples/bibliography.json \
  --bibliography-format json \
  --outline examples/outline_preliminary.json \
  --graph examples/seed_graph.json \
  --prompts examples/prompts.json \
  --enable-outline-expander \
  --outline-expansion-report build/outline_expansion.json \
  --max-figures-per-section 2 \
  --output build/manuscript.md \
  --output-format markdown \
  --report build/diagnostics.json \
  --title "Colophon Demo Draft"
```

### 2a) Update the knowledge graph from bibliography embeddings (optional)

```bash
colophon \
  --bibliography examples/bibliography.json \
  --bibliography-format json \
  --outline examples/outline.json \
  --graph examples/seed_graph.json \
  --prompts examples/prompts.json \
  --enable-kg-updates \
  --kg-update-config examples/kg_update_local.json \
  --kg-vector-db-path build/vectors.json \
  --updated-graph build/updated_graph.json \
  --output build/kg_update_demo.md \
  --report build/kg_update_demo_diagnostics.json \
  --title "KG Update Demo"
```

Inspect:

- `kg_update_result` in `build/kg_update_demo_diagnostics.json`
- `build/vectors.json` for the serialized vector database
- `build/updated_graph.json` for the revised knowledge graph

### 2aa) Expand a preliminary outline into a detailed outline + prompt bundle (optional)

```bash
colophon \
  --bibliography examples/bibliography.json \
  --bibliography-format json \
  --outline examples/outline_preliminary.json \
  --graph examples/seed_graph.json \
  --enable-outline-expander \
  --functional-forms ontology/functional_forms.json \
  --functional-form-id sequential_transformation \
  --outline-max-subsections 4 \
  --outline-expansion-report build/outline_expansion.json \
  --expanded-outline-output build/expanded_outline.json \
  --expanded-prompts-output build/expanded_prompts.json \
  --output build/outline_expander_demo.md \
  --report build/outline_expander_demo_diagnostics.json \
  --title "Outline Expander Demo"
```

Inspect:

- `outline_expansion_result` in `build/outline_expander_demo_diagnostics.json`
- `build/expanded_outline.json` for expanded chapter/section plans
- `build/expanded_prompts.json` for prompt templates
- expanded chapter/section rows include `functional_chapter_type` + `functional_element_id` when form context is loaded

### 2ab) Import a knowledge graph from notes exports (optional)

```bash
# Obsidian vault: hyperlink + embedding links
colophon-import-notes \
  --source examples/notes/obsidian \
  --platform obsidian \
  --seed-graph examples/seed_graph.json \
  --output build/notes_graph_obsidian.json \
  --report build/notes_import_obsidian.json \
  --embedding-provider local \
  --embedding-top-k 2 \
  --embedding-similarity-threshold 0.25

# Notion JSON export: hyperlink-only links
colophon-import-notes \
  --source examples/notes/notion/notion_export.json \
  --platform notion \
  --output build/notes_graph_notion.json \
  --report build/notes_import_notion.json \
  --disable-embeddings
```

Use the imported graph in the main pipeline:

```bash
colophon \
  --bibliography examples/bibliography.json \
  --bibliography-format json \
  --outline examples/outline.json \
  --graph build/notes_graph_obsidian.json \
  --prompts examples/prompts.json \
  --output build/manuscript_from_notes.md \
  --report build/manuscript_from_notes_diagnostics.json \
  --title "Notes-Informed Draft"
```

### 2d) Generate paper recommendations for bibliography + KG proposals

```bash
colophon \
  --bibliography examples/bibliography.bib \
  --bibliography-format bibtex \
  --outline examples/outline.json \
  --graph examples/seed_graph.json \
  --prompts examples/prompts.json \
  --enable-paper-recommendations \
  --recommendation-config examples/recommendation_openalex.json \
  --recommendation-top-k 5 \
  --recommendation-report build/recommendations.json \
  --output build/recommendation_demo.md \
  --report build/recommendation_demo_diagnostics.json \
  --title "Recommendation Demo"
```

Semantic Scholar variant:

```bash
export SEMANTIC_SCHOLAR_API_KEY=...
colophon \
  --bibliography examples/bibliography.bib \
  --bibliography-format bibtex \
  --outline examples/outline.json \
  --graph examples/seed_graph.json \
  --prompts examples/prompts.json \
  --enable-paper-recommendations \
  --recommendation-config examples/recommendation_semantic_scholar.json \
  --recommendation-provider semantic_scholar \
  --recommendation-api-key-env SEMANTIC_SCHOLAR_API_KEY \
  --output build/recommendation_demo_s2.md \
  --report build/recommendation_demo_s2_diagnostics.json \
  --title "Recommendation Demo (Semantic Scholar)"
```

Inspect:

- `recommendation_proposals` in `build/recommendation_demo_diagnostics.json`
- `## Recommended Papers` in `build/recommendation_demo.md`
- `build/recommendations.json` for proposal payloads (bibliography + KG updates)

### 2e) Run functional-form soft validation (optional)

```bash
colophon \
  --bibliography examples/bibliography.json \
  --bibliography-format json \
  --outline examples/outline.json \
  --graph examples/seed_graph.json \
  --prompts examples/prompts.json \
  --narrative-tone formal \
  --narrative-style persuasive \
  --narrative-audience "policy analysts" \
  --narrative-discipline "public policy" \
  --narrative-language Spanish \
  --enable-soft-validation \
  --functional-forms ontology/functional_forms.json \
  --functional-form-id sequential_transformation \
  --functional-validation-report build/functional_validation.json \
  --output build/soft_validation_demo.md \
  --report build/soft_validation_demo_diagnostics.json \
  --title "Soft Validation Demo"
```

Inspect:

- `soft_validation_result` in `build/soft_validation_demo_diagnostics.json`
- `build/functional_validation.json` for standalone findings and coverage summary
- `ontology/functional_forms.json` includes revised `soft_validation_profiles` defaults (`common`, `argumentative`, `exploratory`)
- `narrative_profile` in diagnostics to confirm tone/style/audience/discipline/genre/language settings

### 2f) Run Wilson companion writing ontology checks (optional)

```bash
colophon \
  --bibliography examples/bibliography.json \
  --bibliography-format json \
  --outline examples/outline.json \
  --graph examples/seed_graph.json \
  --prompts examples/prompts.json \
  --functional-forms ontology/functional_forms.json \
  --functional-form-id sequential_transformation \
  --writing-ontology ontology/wilson_academic_writing_ontology.json \
  --writing-ontology-report build/writing_ontology_validation.json \
  --output build/writing_ontology_demo.md \
  --report build/writing_ontology_demo_diagnostics.json \
  --title "Writing Ontology Demo"
```

Inspect:

- `writing_ontology_context` in `build/writing_ontology_demo_diagnostics.json`
- `writing_ontology_validation_result` in `build/writing_ontology_demo_diagnostics.json`
- `build/writing_ontology_validation.json` for standalone companion validation findings

### 2g) Run technical-writing ontology forms (optional)

```bash
colophon \
  --bibliography examples/bibliography.json \
  --bibliography-format json \
  --outline examples/outline_preliminary.json \
  --graph examples/seed_graph.json \
  --enable-outline-expander \
  --enable-soft-validation \
  --functional-forms ontology/technical_forms.json \
  --functional-form-id imrad_contribution \
  --functional-validation-report build/technical_validation.json \
  --output build/technical_form_demo.md \
  --report build/technical_form_demo_diagnostics.json \
  --title "Technical Form Demo"
```

Inspect:

- `outline_expansion_result` for IMRaD-aligned section expansion
- `soft_validation_result` for technical form checks (`imrad_*`, `sys_*`, `theory_*`, `spec_*`, `survey_*`)

### 2c) Run with coordination agents and inspect message hand-offs

```bash
colophon \
  --bibliography examples/bibliography.bib \
  --bibliography-format bibtex \
  --outline examples/outline.json \
  --graph examples/seed_graph.json \
  --prompts examples/prompts.json \
  --functional-forms ontology/functional_forms.json \
  --functional-form-id sequential_transformation \
  --top-k 0 \
  --output build/coordination-demo.md \
  --report build/coordination-demo-diagnostics.json \
  --title "Coordination Demo"
```

This run intentionally limits retrieval (`--top-k 0`) so coordination agents emit gap requests.
Check:

- `coordination_messages` in `build/coordination-demo-diagnostics.json`
- `gap_requests` in `build/coordination-demo-diagnostics.json`
- `## Gap Requests` section in `build/coordination-demo.md`

### 2b) Enable LLM-backed generation hooks (optional)

Use a provider config file:

```bash
export OPENAI_API_KEY=...
colophon \
  --bibliography examples/bibliography.json \
  --outline examples/outline.json \
  --graph examples/seed_graph.json \
  --llm-config examples/llm_openai.json \
  --output build/manuscript.md \
  --report build/diagnostics.json \
  --title "Colophon LLM Draft"
```

Or configure directly via flags:

```bash
export ANTHROPIC_API_KEY=...
colophon \
  --bibliography examples/bibliography.json \
  --outline examples/outline.json \
  --graph examples/seed_graph.json \
  --llm-provider claude \
  --llm-model claude-3-5-sonnet-latest \
  --llm-api-key-env ANTHROPIC_API_KEY \
  --output build/manuscript.md \
  --report build/diagnostics.json \
  --title "Colophon Claude Draft"
```

### 3) Run tests

```bash
python -m unittest discover -s tests -q
```

### 4) Build docs

```bash
sphinx-build -b html docs docs/_build/html
```

### 5) CLI help

```bash
colophon --help
colophon-import-notes --help
```

## Input formats

### Bibliography

#### JSON

```json
{
  "sources": [
    {
      "id": "src-01",
      "title": "...",
      "authors": ["..."],
      "year": 2024,
      "publication": "Journal Name",
      "abstract": "evidence-bearing content"
    }
  ]
}
```

#### CSV

CSV columns can include `title`, `authors`, `publication`, `year`, `abstract` (or `text`), and optional `id`.

```csv
id,title,authors,publication,year,abstract
src-01,Knowledge Graphs for Scholarly Writing,"A. Researcher;B. Collaborator",Journal of Graph Studies,2022,"Knowledge graphs provide explicit structure..."
```

#### BibTeX

```bibtex
@article{researcher2022,
  title = {Knowledge Graphs for Scholarly Writing},
  author = {A. Researcher and B. Collaborator},
  journal = {Journal of Graph Studies},
  year = {2022},
  abstract = {Knowledge graphs provide explicit structure...}
}
```

Reference examples in the repo:

- `/Users/briankeegan/Documents/New project/examples/bibliography.json`
- `/Users/briankeegan/Documents/New project/examples/bibliography.csv`
- `/Users/briankeegan/Documents/New project/examples/bibliography.bib`

### Outline

```json
{
  "chapters": [
    {
      "title": "Chapter Title",
      "sections": ["Section A", "Section B"]
    }
  ]
}
```

### Prompts

Prompt templates are optional overrides for claim and paragraph drafting.

```json
{
  "prompts": {
    "claim_template": "Claim: {lead_entity} appears in {source_title} as evidence for {section_title_lower}.",
    "figure_reference_template": "See Figure {figure_id} ({figure_caption}).",
    "paragraph_template": "{claim_text}\nEvidence: {citations}\nVisuals: {figure_references}",
    "empty_section_template": "Insufficient grounding evidence was retrieved for this section."
  }
}
```

Narrative placeholders available in templates:

- `{narrative_instruction}`
- `{narrative_tone}`
- `{narrative_style}`
- `{narrative_audience}`
- `{narrative_discipline}`
- `{narrative_genre}`
- `{narrative_language}`

### Seed graph

#### JSON

```json
{
  "entities": ["Entity A", "Entity B"],
  "relations": [
    {"source": "Entity A", "predicate": "relates_to", "target": "Entity B"}
  ],
  "figures": [
    {
      "id": "fig-example",
      "caption": "Entity relationship diagram",
      "uri": "figures/entity-relationship.png",
      "related_entities": ["Entity A", "Entity B"]
    }
  ]
}
```

#### CSV edgelist

Headered or headerless CSV is supported.

```csv
source,target,predicate
Entity A,Entity B,relates_to
```

```csv
Entity A,Entity B,relates_to
```

#### SQLite database or SQL dump

Colophon accepts `.sqlite`, `.sqlite3`, `.db`, `.sql`, and `.dump` graph inputs.
Use an edge table with source and target columns (predicate/relation optional). You can also add a
figure table with `id/figure_id`, `caption`, and `uri/path/url`.

```sql
CREATE TABLE edges (source TEXT, target TEXT, relation TEXT);
INSERT INTO edges (source, target, relation) VALUES ('Entity A', 'Entity B', 'relates_to');
CREATE TABLE figures (id TEXT, caption TEXT, path TEXT);
INSERT INTO figures (id, caption, path) VALUES ('fig-example', 'Entity diagram', 'figures/entity.png');
```

If extension-based detection is ambiguous, set `--graph-format` explicitly:

```bash
colophon ... --graph path/to/edges.txt --graph-format csv
```

If bibliography extension is ambiguous, set `--bibliography-format` explicitly:

```bash
colophon ... --bibliography path/to/references.txt --bibliography-format csv
```

### Notes source formats

Use `colophon-import-notes` with `--platform` set to one of:

- `obsidian`: vault directory containing markdown notes (`.obsidian/` marker used for auto-detect).
- `notion`: markdown export directory, JSON page export, or CSV page export (set `--platform notion` for JSON/CSV files).
- `onenote`: JSON export (`notes`/`pages` list with `id`, `title`, `content`).
- `evernote`: `.enex` export file.
- `markdown`: generic markdown directory/file.

Importer options allow:

- hyperlink traversal (`links_to`, `references_url`)
- embedding similarity links (`similar_to`)
- either mode alone or both together

Example note source files in this repo:

- `/Users/briankeegan/Documents/New project/examples/notes/obsidian/Research Questions.md`
- `/Users/briankeegan/Documents/New project/examples/notes/notion/notion_export.json`
- `/Users/briankeegan/Documents/New project/examples/notes/notion/notion_export.csv`
- `/Users/briankeegan/Documents/New project/examples/notes/onenote_export.json`
- `/Users/briankeegan/Documents/New project/examples/notes/evernote_export.enex`
- `/Users/briankeegan/Documents/New project/ontology/functional_forms.json`
- `/Users/briankeegan/Documents/New project/ontology/technical_forms.json`
- `/Users/briankeegan/Documents/New project/ontology/genre_ontology.json`
- `/Users/briankeegan/Documents/New project/ontology/wilson_academic_writing_ontology.json`

## Knowledge graph updater

KG updates are bibliography-driven. For each source, Colophon builds an embedding over title, abstract/text, authors, and publication metadata, stores records in an optional vector DB, retrieves nearest neighbors for RAG context, and revises the graph with paper/entity relations.

CLI controls:

- `--enable-kg-updates`
- `--kg-update-config`
- `--embedding-provider`, `--embedding-model`, `--embedding-api-base-url`, `--embedding-api-key-env`
- `--embedding-dimensions`, `--embedding-timeout-seconds`
- `--kg-vector-db-path`, `--kg-rag-top-k`, `--kg-similarity-threshold`, `--kg-max-entities-per-doc`
- `--updated-graph`

Example config files:

- `/Users/briankeegan/Documents/New project/examples/kg_update_local.json`
- `/Users/briankeegan/Documents/New project/examples/kg_update_openai.json`

## Output formats

Use `--output-format` with one of: `text`, `markdown`, `rst`, `rtf`, `latex` (or `auto`).

```bash
# Plain text
colophon ... --output build/manuscript.txt --output-format text

# Markdown
colophon ... --output build/manuscript.md --output-format markdown

# reStructuredText
colophon ... --output build/manuscript.rst --output-format rst

# Rich Text Format
colophon ... --output build/manuscript.rtf --output-format rtf

# LaTeX
colophon ... --output build/manuscript.tex --output-format latex
```

## Output layout

Use `--output-layout` with:

- `single` / `monolithic` (default): write one manuscript file at `--output`.
- `project`: treat `--output` as a directory and write chapter-level files.

```bash
# Single-file output (default)
colophon ... --output build/manuscript.md --output-format markdown --output-layout single

# Project output with chapter-level files
colophon ... --output build/manuscript_project --output-format markdown --output-layout project
```

Project layout writes:

- `index.<ext>` full manuscript
- `chapter-XX-<slug>.<ext>` per chapter
- `manifest.json` with chapter file mapping

## Coordination and hand-offs

Colophon includes hierarchical coordination agents:

- Paragraph-level coordinator: edits paragraph readability/persuasiveness and requests missing evidence.
- Section-level coordinator: aligns paragraph bundles with outlined section intent.
- Chapter-level coordinator: ensures outlined sections are represented and coherent within each chapter.
- Book-level coordinator: ensures chapter-level coherence against the outline.

Message passing is handled through an internal message bus. Agents send guidance to children and status to parents.
These are surfaced in diagnostics as `coordination_messages`.
When functional forms are loaded, coordinator guidance and gap checks also use
`coordination_ontology`, chapter patterns, and element ontology from `ontology/functional_forms.json`.
When the companion writing ontology is loaded, background guidance from
`ontology/wilson_academic_writing_ontology.json` is also injected into agent/coordinator prompts and validation diagnostics.

When agents detect missing context (bibliography, knowledge graph, outline), they emit structured `gap_requests` and these are also rendered into the manuscript under `Gap Requests`.
Coordinator/editor agents run an iterative revise loop after drafting and stop when content and coordination state converges
or the max pass limit is reached. See `coordination_revision` in diagnostics.

Disable these agents if needed:

```bash
colophon ... --disable-coordination-agents
```

Adjust revise-loop budget:

```bash
colophon ... --coordination-max-iterations 6
```

## Paper Recommendation Workflow

When enabled, the recommender:

1. Queries a scientometric API (OpenAlex-compatible endpoint) using bibliography title/abstract/author seeds.
2. Scores candidates by title/abstract similarity, author overlap, publication match, and citation signal.
3. Filters out papers already present in the bibliography.
4. Produces `RecommendationProposal` objects containing:
   - proposed bibliography entry payload
   - proposed knowledge-graph entity/relation updates
5. Surfaces proposals in manuscript output, diagnostics, and optional recommendation report JSON.

Example recommendation config:

- `/Users/briankeegan/Documents/New project/examples/recommendation_openalex.json`
- `/Users/briankeegan/Documents/New project/examples/recommendation_semantic_scholar.json`

## Tutorial

A full minimum-viable tutorial (bibliography + graph + outline + prompts) is available at:

- `/Users/briankeegan/Documents/New project/docs/tutorial.rst`
- `/Users/briankeegan/Documents/New project/docs/getting_started.rst`
- `/Users/briankeegan/Documents/New project/docs/usage.rst`
- `/Users/briankeegan/Documents/New project/docs/examples.rst`
- `/Users/briankeegan/Documents/New project/docs/api.rst`

## References and citations

- OpenAlex API: https://api.openalex.org/
- Semantic Scholar (AI2) API: https://api.semanticscholar.org/
- BibTeX format reference: http://www.bibtex.org/Format/
- Sphinx documentation system: https://www.sphinx-doc.org/
- reStructuredText specification: https://docutils.sourceforge.io/rst.html
- LaTeX project: https://www.latex-project.org/
- Wilson, Jeffrey R. *Academic Writing* (source for `ontology/wilson_academic_writing_ontology.json`).
- Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.

## LLM hook providers

Built-in provider aliases:

- `openai`, `codex`, `openai_compatible`
- `anthropic`, `claude`
- `github`, `copilot`

Example config files:

- `/Users/briankeegan/Documents/New project/examples/llm_openai.json`
- `/Users/briankeegan/Documents/New project/examples/llm_codex.json`
- `/Users/briankeegan/Documents/New project/examples/llm_claude.json`
- `/Users/briankeegan/Documents/New project/examples/llm_copilot.json`

## Project layout

- `/Users/briankeegan/Documents/New project/colophon`: Source package.
- `/Users/briankeegan/Documents/New project/examples`: Example inputs.
- `/Users/briankeegan/Documents/New project/examples/notes`: Note-export samples for importer workflows.
- `/Users/briankeegan/Documents/New project/docs`: Sphinx documentation.
- `/Users/briankeegan/Documents/New project/.github/workflows/docs.yml`: GitHub Pages docs deployment.

## Current scope and next steps

This implementation is an MVP focused on deterministic orchestration and documentation. Next iterations should add richer retrieval (hybrid BM25+vector), stronger claim/evidence checking, and iterative revise loops per section.
