Architecture
============

Overview
--------

Colophon uses a cooperative workflow with explicit roles:

1. A retrieval layer ranks bibliography sources relevant to each section query.
2. A claim author agent drafts atomic claims grounded in retrieved evidence and graph context, including figure references.
3. A paragraph agent turns claims into prose with inline evidence snippets and figure callouts.
4. Paragraph/section/chapter/book coordination agents exchange messages and perform editing passes, optionally
   guided by functional-form coordination ontology.
5. Reviewer agents validate citations, figure references, and coherence.
6. Optional recommendation workflow proposes related papers and KG updates.
7. Optional KG generator/updater builds embeddings, a vector index, and revises KG entities/relations before drafting.
8. Optional notes importer (Obsidian/Notion/OneNote/Evernote) derives graph relations from hyperlinks and embedding similarity.
9. Optional functional-form soft validator checks outline/agents/claims/bibliography/prompts against genre-aware heuristics.
10. Optional companion writing ontology injects background prompts/assumptions and runs additional process-oriented validations.
11. Optional genre ontology applies audience/discipline/style/genre/language defaults and role prompts across drafting and review.
12. Optional outline expander transforms a preliminary outline into detailed chapter/section plans and prompts,
    optionally aligned to functional-form chapter patterns and element ontology.

Functional-form catalogs may be narrative-oriented (``ontology/functional_forms.json``) or
technical-writing oriented (``ontology/technical_forms.json``). The runtime normalizes pattern
keys and required-element fields so both catalogs drive coordination, outline expansion, and
soft validation through the same pipeline path.

Core components
---------------

- ``colophon.models``: manuscript and evidence data model.
- ``colophon.graph``: seed knowledge graph representation and query-time entity matching.
- ``colophon.io``: bibliography ingestion (JSON/CSV/BibTeX) and graph ingestion (JSON/CSV/SQLite/SQL).
- ``colophon.coordination``: message bus and hierarchical coordination/editing agents.
- ``colophon.llm``: provider-agnostic LLM client hooks and adapters.
- ``colophon.vectors``: local or OpenAI-compatible embedding clients and in-memory vector DB.
- ``colophon.kg_update``: bibliography-driven KG generator/updater using embedding similarity and RAG context.
- ``colophon.recommendations``: scientometric API connectors and recommendation proposal workflow.
- ``colophon.note_import`` and ``colophon.import_cli``: notes-system ingestion into knowledge graph entities and relations.
- ``colophon.functional_forms``: soft structural/rhetorical/genre validation for authoring inputs and outputs.
- ``colophon.writing_ontology``: companion ontology context + validations for background prompts, assumptions, and process checks.
- ``colophon.genre_ontology``: profile ontology for narrative metadata and role prompts with defaults/overrides.
- ``colophon.retrieval``: lexical retrieval over source text.
- ``colophon.agents``: claim/paragraph drafting and review agents.
- ``colophon.agents.OutlineExpanderAgent``: preliminary-outline expansion to detailed planning artifacts.
- ``colophon.pipeline``: orchestration over chapters and sections.
- ``colophon.cli``: command-line interface for batch generation.

Data flow
---------

1. Load bibliography, outline, and seed knowledge graph.
2. Optional outline expander enriches draft outline chapters and generates prompt templates.
3. Optional KG updater embeds bibliography records (title/abstract/authors/publication), builds a vector index, performs similarity retrieval, and adds/revises KG entities/relations.
4. Chapter/section coordinators send guidance messages to child drafting agents.
5. For each section title, retrieve top-k evidence sources and matching graph figure nodes.
6. Generate claims and paragraphs with optional figure references.
7. Paragraph/section/chapter/book coordinators send status messages upward and emit structured gap requests.
8. Aggregate sections into chapters and run citation/figure/coherence review passes.
9. Optionally run paper recommendation workflow (OpenAlex or Semantic Scholar APIs), scoring and deduplicating candidate papers.
10. Optionally run functional-form soft validation and emit findings in diagnostics (and optional standalone report).
11. Optionally run companion writing-ontology validations and emit findings in diagnostics (and optional standalone report).
12. Apply genre-ontology profile metadata/prompts to agent generation, recommendation query/scoring hints, and agent validation checks.
13. Emit manuscript output (single-file or chapter-level project layout) with optional ``Gap Requests`` and ``Recommended Papers`` sections and diagnostics JSON including coordination, outline expansion, KG update, functional validation, companion-ontology validation, genre profile context, and recommendation results. The notes importer emits its own standalone report JSON.

Current limitations
-------------------

- Retrieval is intentionally simple and lexical.
- Claims are template-driven rather than LLM-generated.
- Review checks are deterministic and rule-based.

These choices keep the system transparent and testable while providing a strong base for iterative model-based upgrades.
