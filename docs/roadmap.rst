Roadmap
=======

Implemented in this version
---------------------------

- Baseline data model for sources, claims, and manuscript artifacts.
- Seed knowledge graph ingestion and query-time entity matching.
- Retrieval layer over bibliography content.
- Cooperative section-writing pipeline with distinct drafting and review agents.
- Provider-agnostic LLM API hooks for OpenAI/Codex, Claude/Anthropic, and Copilot-style endpoints.
- Scientometric recommendation workflow for proposing bibliography and knowledge-graph updates.
- CLI and JSON-based interfaces for reproducible batch generation.
- Sphinx documentation with architecture, usage, and API pages.
- GitHub Actions workflow to publish docs to GitHub Pages.

Near-term next steps
--------------------

1. Upgrade retrieval to hybrid lexical + vector search.
2. Add claim-level evidence scoring and contradiction checks.
3. Introduce iterative revise loops (draft -> critique -> rewrite).
4. Add chapter-level style and narrative coherence objectives.
5. Add benchmark datasets and quality regression checks.
