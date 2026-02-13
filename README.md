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
- [Empty-workspace upload tutorial (Claude/Codex)](docs/upload_tutorial.rst)
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
  --request-user-guidance \
  --guidance-output build/user_guidance.json \
  --output build/manuscript.md \
  --report build/diagnostics.json \
  --title "Uploaded Bundle Draft"
```

For a full stand-alone walkthrough from an empty workspace using files uploaded from
`/examples/`, see [docs/upload_tutorial.rst](docs/upload_tutorial.rst).

AskUserQuestion / user_input
----------------------------

Colophon supports interactive planning guidance before drafting:

- CLI: `--request-user-guidance` + `--guidance-output`
- Claude Agent SDK: `colophon.user_input.AgentSDKUserInputHandler`
- OpenAI Codex/OpenAI Responses API: `colophon.user_input.OpenAICodexUserInputHandler`
- Multi-stage support: `planning`, `recommendations`, `outline`, `coordination`

User-input requests are capped to 10 questions per stage/task. If more are provided,
Colophon scores by importance and asks only the top 10.

Python (Claude Agent SDK) example:

```python
import asyncio
from colophon.user_input import request_planning_guidance_via_agent_sdk

async def main() -> None:
    text, guidance = await request_planning_guidance_via_agent_sdk(
        "Plan a draft with recommendation integration and optional outline expansion."
    )
    print(text)
    print(guidance)

asyncio.run(main())
```

Python (OpenAI Codex) example:

```python
import asyncio
from colophon.user_input import request_planning_guidance_via_openai_codex

async def main() -> None:
    text, guidance = await request_planning_guidance_via_openai_codex(
        task_description="Prepare an implementation plan from uploaded writing artifacts.",
        model="gpt-5-codex",
    )
    print(text)
    print(guidance)

asyncio.run(main())
```

## Get involved

Contributions are welcome. You can help by opening issues, proposing features from your writing workflow, improving docs/examples, or submitting pull requests for bug fixes and new capabilities. If you want to start small, check the roadmap and docs pages above, then suggest an improvement you would like to use.
