![colophon logo](/docs/_src/logo.png)

Agent + RAG framework for long-form writing.

## Planning

See the implementation plan: [`docs/implementation-plan.md`](docs/implementation-plan.md).
Colophon is for researchers, analysts, and technical writers who need to turn source-heavy notes into coherent long-form drafts. It is designed for people who want transparent, citation-aware generation that combines an outline, bibliography, prompts, and a lightweight knowledge graph into a manuscript they can review and revise.

## What Colophon does

- Ingests bibliographies, outlines, prompts, and graph context.
- Produces chapter/section draft text with citations.
- Generates diagnostics for evidence coverage and coordination behavior.
- Supports optional LLM-backed generation and outline expansion workflows.
- Includes note-import and knowledge-graph update utilities.

## Documentation

- [Getting started](docs/getting_started.rst)
- [Usage and CLI options](docs/usage.rst)
- [Tutorial](docs/tutorial.rst)
- [Examples](docs/examples.rst)
- [API reference](docs/api.rst)
- [Architecture](docs/architecture.rst)
- [Roadmap](docs/roadmap.rst)
- [References](docs/references.rst)

Ontology catalogs used by advanced workflows:

- `ontology/functional_forms.json`
- `ontology/technical_forms.json`
- `ontology/genre_ontology.json`
- `ontology/wilson_academic_writing_ontology.json`

## Minimum getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

colophon \
  --bibliography examples/bibliography.json \
  --bibliography-format json \
  --outline examples/outline.json \
  --graph examples/seed_graph.json \
  --prompts examples/prompts.json \
  --output build/manuscript.md \
  --report build/diagnostics.json \
  --title "Colophon Demo Draft"
```

Upload-first workflow (Codex or Claude Code):

```bash
colophon \
  --runtime codex \
  --artifacts-dir uploads \
  --output build/manuscript.md \
  --report build/diagnostics.json \
  --title "Uploaded Bundle Draft"
```

## Get involved

Contributions are welcome. You can help by opening issues, proposing features from your writing workflow, improving docs/examples, or submitting pull requests for bug fixes and new capabilities. If you want to start small, check the roadmap and docs pages above, then suggest an improvement you would like to use.
