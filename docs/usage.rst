Usage
=====

Installation
------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[docs]

CLI help
--------

.. code-block:: bash

   colophon --help
   colophon-import-notes --help

Generate a manuscript
---------------------

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline_preliminary.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --enable-outline-expander \
     --outline-expansion-report build/outline_expansion.json \
     --llm-config examples/llm_openai.json \
     --max-figures-per-section 2 \
     --enable-kg-updates \
     --kg-update-config examples/kg_update_local.json \
     --enable-paper-recommendations \
     --output build/manuscript.md \
     --output-format markdown \
     --report build/diagnostics.json \
     --title "Colophon Demo Draft" \
     --top-k 3


Deconstruct a PDF into artifacts
-------------------------------

Use this when you want to bootstrap inputs from a PDF:

.. code-block:: bash

   colophon deconstruct test.pdf

Optional output controls:

.. code-block:: bash

   colophon deconstruct test.pdf --output-dir build/deconstruct --stem test_run

Output files:

- ``*_bibliography.json``
- ``*_kg.json``
- ``*_outline.json``
- ``*_prompts.json``

CLI arguments
-------------

- ``--bibliography``: bibliography input file (JSON, CSV, or BibTeX).
- ``--bibliography-format``: ``auto`` (default), ``json``, ``csv``, or ``bibtex``.
- ``--outline``: outline JSON with ``chapters`` and ``sections``.
- ``--prompts``: optional prompt templates JSON.
- ``--request-user-guidance``: interactively request planning guidance from the user (planning document style, recommendation incorporation, outline expansion, additional notes).
- ``--guidance-output``: optional JSON output path for captured user-guidance responses.
- ``--user-guidance-stages``: comma-separated stage list for guidance capture. Supported values: ``planning``, ``recommendations``, ``outline``, ``coordination``.
- ``--runtime``: runtime target for input resolution (``local`` default, ``codex``, ``claude_code``/``claude-code``).
- ``deconstruct`` subcommand: ``colophon deconstruct <pdf>`` writes bibliography/KG/outline/prompts JSON artifacts from a PDF.
- ``deconstruct --output-dir``: output directory for generated artifacts.
- ``deconstruct --stem``: output filename stem for generated artifacts.
- ``--artifacts-dir``: optional directory containing uploaded artifacts; missing bibliography/outline/graph/prompts paths are auto-discovered.
- ``--llm-config``: optional LLM config JSON.
- ``--llm-provider``: ``none``, ``openai``, ``codex``, ``openai_compatible``, ``anthropic``, ``claude``, ``github``, ``copilot``.
- ``--llm-model``: model name for API calls.
- ``--llm-api-base-url``: API base URL override.
- ``--llm-api-key-env``: environment variable containing API key/token.
- ``--llm-system-prompt``: system prompt for LLM generation.
- ``--llm-temperature``: sampling temperature.
- ``--llm-max-tokens``: maximum response tokens.
- ``--llm-timeout-seconds``: HTTP timeout.
- ``--graph``: knowledge graph input in JSON, CSV edgelist, SQLite DB, or SQL dump.
- ``--graph-format``: ``auto`` (default), ``json``, ``csv``, ``sqlite``, or ``sql``.
- ``--output``: output manuscript path.
- ``--output-format``: ``auto`` (default), ``text``, ``markdown``, ``rst``, ``rtf``, or ``latex``.
- ``--output-layout``: ``single``/``monolithic`` (default) or ``project`` (chapter-level files + manifest).
- ``--report``: diagnostics JSON path.
- ``--title``: manuscript title.
- ``--narrative-tone``: narrative tone (for example: ``formal``, ``conversational``, ``critical``).
- ``--narrative-style``: writing style (for example: ``analytical``, ``persuasive``, ``expository``).
- ``--narrative-audience``: intended readership (for example: ``graduate students``, ``policy analysts``).
- ``--narrative-discipline``: disciplinary framing (for example: ``history``, ``economics``, ``public policy``).
- ``--narrative-genre``: target genre framing (for example: ``scholarly_manuscript``, ``technical_report``, ``policy_brief``).
- ``--narrative-language``: target output language (directly applied in LLM-backed generation prompts).
- ``--genre-ontology``: optional genre ontology JSON for profile defaults/prompts used by agents, recommendations, and validation.
- ``--genre-profile-id``: optional profile id to select from the genre ontology.
- ``--enable-soft-validation``: enable soft checks against functional-form structure, rhetoric, and genre cues.
- ``--functional-forms``: path to functional forms JSON catalog (defaults to ``ontology/functional_forms.json`` when coordination agents, outline expansion, or soft validation are enabled and file exists).
- ``--functional-form-id``: optional functional form id to select from catalog.
- ``--max-soft-validation-findings``: cap number of soft-validation findings in diagnostics.
- ``--functional-validation-report``: optional standalone soft-validation JSON report output path.
- ``--writing-ontology``: optional companion ontology JSON for background prompts, assumptions, and additional validations.
- ``--max-writing-ontology-findings``: cap number of writing-ontology findings in diagnostics.
- ``--writing-ontology-report``: optional standalone writing-ontology validation JSON report output path.
- ``--top-k``: number of sources retrieved per section.
- ``--max-figures-per-section``: max figure nodes attached and referenced per section.
- ``--disable-coordination-agents``: disable paragraph/section/chapter/book coordination and message passing.
- ``--coordination-max-iterations``: maximum iterative revise passes for coordinator/editor convergence.
- ``--enable-paper-recommendations``: enable related-paper recommendation workflow.
- ``--recommendation-config``: optional recommendation config JSON.
- ``--recommendation-provider``: recommendation API provider alias (``openalex``, ``scholar_search``, ``scholar``, ``semantic_scholar``, ``semantic-scholar``, ``ai2``, ``s2``).
- ``--recommendation-api-base-url``: recommendation API base URL.
- ``--recommendation-api-key-env``: environment variable containing recommendation API key (for providers such as Semantic Scholar).
- ``--recommendation-timeout-seconds``: timeout for recommendation API calls.
- ``--recommendation-per-seed``: max candidates fetched per bibliography seed paper.
- ``--recommendation-top-k``: max proposals retained after scoring/deduplication.
- ``--recommendation-min-score``: minimum score threshold for recommendation proposals.
- ``--recommendation-mailto``: optional OpenAlex mailto for polite-pool routing.
- ``--recommendation-report``: optional JSON output for recommendation proposals.
- ``--enable-kg-updates``: enable bibliography-driven KG generation/revision.
- ``--kg-update-config``: optional KG updater config JSON.
- ``--embedding-provider``: embedding provider alias (``local``, ``openai``, ``openai_compatible``, ``remote``).
- ``--embedding-model``: embedding model name for remote providers.
- ``--embedding-api-base-url``: embedding API base URL.
- ``--embedding-api-key-env``: environment variable containing embedding API key/token.
- ``--embedding-dimensions``: vector dimensionality.
- ``--embedding-timeout-seconds``: embedding API timeout.
- ``--kg-vector-db-path``: optional path to write vector DB JSON.
- ``--kg-rag-top-k``: nearest neighbors retrieved for RAG-based KG updates.
- ``--kg-similarity-threshold``: minimum cosine similarity threshold for linking papers.
- ``--kg-max-entities-per-doc``: max extracted entities added per bibliography document.
- ``--updated-graph``: optional path for revised KG JSON output.
- ``--enable-outline-expander``: expand preliminary outline into a more detailed outline before drafting.
- ``--outline-max-subsections``: max subsection titles generated for each section.
- ``--outline-expansion-report``: optional JSON payload containing expanded outline + prompts + diagnostics.
- ``--expanded-outline-output``: optional JSON output containing the expanded ``chapters``.
- ``--expanded-prompts-output``: optional JSON output containing generated ``prompts``.

Codex/Claude Code upload workflow
---------------------------------

If a user uploads bibliography, outline, graph, and prompts files into a workspace folder, run with upload-aware
artifact discovery instead of passing every file path explicitly:

.. code-block:: bash

   colophon \
     --runtime codex \
     --artifacts-dir uploads \
     --output build/manuscript.md \
     --report build/diagnostics.json \
     --title "Upload Bundle Draft"

For Claude Code, use ``--runtime claude-code`` (or ``claude_code``). Explicit flags such as ``--bibliography`` and
``--graph`` still override discovered upload files when both are provided.

For a complete empty-workspace walkthrough (what to upload from ``/examples/``, exact commands, and troubleshooting),
see :doc:`upload_tutorial`.

Interactive planning guidance
-----------------------------

Capture user guidance before drafting:

.. code-block:: bash

   colophon \
     --runtime claude-code \
     --artifacts-dir uploads \
     --request-user-guidance \
     --user-guidance-stages planning,recommendations,outline,coordination \
     --guidance-output build/user_guidance.json \
     --output build/manuscript.md \
     --report build/diagnostics.json

The guidance flow can:

- shape planning-document instructions
- tune recommendation settings (enabled/top-k/min-score strategy)
- tune outline expansion settings (enabled/depth profile/max subsections)
- capture coordination-breakdown remediation priorities (including post-run gap-driven prompts)

Guidance is stored in diagnostics under ``user_guidance``.
User-input prompts are capped to at most 10 per task/stage; if more are available, Colophon ranks by importance and
asks only the top 10.

Agent SDK integrations
----------------------

Claude:

- ``colophon.user_input.AgentSDKUserInputHandler``
- ``colophon.user_input.request_planning_guidance_via_agent_sdk``

OpenAI Codex:

- ``colophon.user_input.OpenAICodexUserInputHandler``
- ``colophon.user_input.build_codex_ask_user_question_tool``
- ``colophon.user_input.request_planning_guidance_via_openai_codex``

Both interfaces enforce the same per-stage input cap (top 10 by importance when overflow occurs).

Claude Agent SDK pattern:

.. code-block:: python

   from claude_agent_sdk import ClaudeAgentOptions, query
   from colophon.user_input import AgentSDKUserInputHandler

   handler = AgentSDKUserInputHandler()
   options = ClaudeAgentOptions(
       tools=["Read", "Grep", "AskUserQuestion"],
       can_use_tool=handler.can_use_tool,
   )

   # prompt should instruct the model to call AskUserQuestion before planning
   async for message in query(prompt="Draft a plan and ask user guidance first.", options=options):
       pass

OpenAI Codex / Responses API pattern:

.. code-block:: python

   from colophon.user_input import (
       OpenAICodexUserInputHandler,
       build_codex_ask_user_question_tool,
   )
   from openai import OpenAI

   client = OpenAI()
   handler = OpenAICodexUserInputHandler()
   tools = [build_codex_ask_user_question_tool()]

   # when a function_call to AskUserQuestion is returned:
   # output = handler.handle_function_call(call["name"], call.get("arguments"))
   # then send function_call_output back into client.responses.create(...)

Graph formats
-------------

JSON format:

.. code-block:: json

   {
     "entities": ["Entity A", "Entity B"],
     "relations": [{"source": "Entity A", "target": "Entity B", "predicate": "relates_to"}],
     "figures": [{"id": "fig-example", "caption": "Entity diagram", "uri": "figures/entity.png"}]
   }

CSV edgelist format:

.. code-block:: text

   source,target,predicate
   Entity A,Entity B,relates_to

SQLite/SQL format:

.. code-block:: sql

   CREATE TABLE edges (source TEXT, target TEXT, relation TEXT);
   INSERT INTO edges (source, target, relation) VALUES ('Entity A', 'Entity B', 'relates_to');
   CREATE TABLE figures (id TEXT, caption TEXT, path TEXT);
   INSERT INTO figures (id, caption, path) VALUES ('fig-example', 'Entity diagram', 'figures/entity.png');

Bibliography formats
--------------------

JSON format:

.. code-block:: json

   {
     "sources": [
       {
         "id": "src-1",
         "title": "Paper title",
         "authors": ["A. Author", "B. Writer"],
         "publication": "Journal Name",
         "year": 2023,
         "abstract": "Evidence text"
       }
     ]
   }

CSV format:

.. code-block:: text

   id,title,authors,publication,year,abstract
   src-1,Paper title,"A. Author;B. Writer",Journal Name,2023,Evidence text

BibTeX format:

.. code-block:: text

   @article{author2023,
     title = {Paper title},
     author = {A. Author and B. Writer},
     journal = {Journal Name},
     year = {2023},
     abstract = {Evidence text}
   }

Notes importers (Obsidian/Notion/OneNote/Evernote)
---------------------------------------------------

Use ``colophon-import-notes`` to build or extend a knowledge graph from notes systems using:

- hyperlink traversal (``links_to`` and ``references_url`` relations)
- embedding similarity (``similar_to`` relations)
- either mechanism alone or both together

Example (Obsidian vault):

.. code-block:: bash

   colophon-import-notes \
     --source examples/notes/obsidian \
     --platform obsidian \
     --seed-graph examples/seed_graph.json \
     --output build/notes_graph_obsidian.json \
     --report build/notes_import_obsidian.json \
     --embedding-provider local \
     --embedding-top-k 2 \
     --embedding-similarity-threshold 0.25

Example (Notion JSON export, hyperlink-only):

.. code-block:: bash

   colophon-import-notes \
     --source examples/notes/notion/notion_export.json \
     --platform notion \
     --output build/notes_graph_notion.json \
     --report build/notes_import_notion.json \
     --disable-embeddings

Example (OneNote JSON export, embedding-only):

.. code-block:: bash

   colophon-import-notes \
     --source examples/notes/onenote_export.json \
     --platform onenote \
     --output build/notes_graph_onenote.json \
     --report build/notes_import_onenote.json \
     --disable-hyperlinks \
     --embedding-provider local

Example (Evernote ENEX export):

.. code-block:: bash

   colophon-import-notes \
     --source examples/notes/evernote_export.enex \
     --platform evernote \
     --output build/notes_graph_evernote.json \
     --report build/notes_import_evernote.json \
     --disable-embeddings

Use the imported graph directly in manuscript generation:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph build/notes_graph_obsidian.json \
     --prompts examples/prompts.json \
     --output build/manuscript_from_notes.md \
     --output-format markdown \
     --report build/manuscript_from_notes_diagnostics.json

Notes source formatting guide
-----------------------------

Obsidian:

- point ``--source`` at the vault root (containing ``.obsidian``)
- store notes as ``.md`` files
- internal links can use ``[[Wiki Links]]`` or markdown links (``[Label](OtherNote.md)``)
- external links (``https://...``) become ``references_url`` relations

Notion:

- use markdown export folders or JSON/CSV page exports
- set ``--platform notion`` for JSON/CSV exports to avoid ``auto`` interpreting ``.json`` as OneNote
- for markdown exports, note filenames with trailing Notion IDs are supported
- include ``title``/``name`` and ``content``/``text`` fields for JSON/CSV rows when available
- include ``url`` or ``links`` fields to preserve outbound references

OneNote:

- provide JSON with a ``notes`` or ``pages`` list
- each item should include ``id``, ``title``, and ``content`` (or ``text``/``body``)
- optional ``links`` list is ingested as explicit references

Evernote:

- provide ``.enex`` export files
- ``<title>`` and ``<content>`` are extracted into note nodes and text
- HTML hyperlinks in ENEX content are ingested as references

Notes importer CLI arguments
----------------------------

- ``--source``: source notes path (directory or file export).
- ``--platform``: ``auto`` (default), ``obsidian``, ``notion``, ``onenote``, ``evernote``, ``markdown``.
- ``--output``: output knowledge graph JSON path.
- ``--report``: output importer diagnostics JSON path.
- ``--seed-graph``: optional existing graph file to merge into before import.
- ``--seed-graph-format``: ``auto`` (default), ``json``, ``csv``, ``sqlite``, ``sql``.
- ``--disable-hyperlinks``: disable hyperlink-based relation extraction.
- ``--disable-embeddings``: disable embedding-based similarity relations.
- ``--embedding-provider``: ``local``, ``hash``, ``offline``, ``openai``, ``openai_compatible``, ``remote``.
- ``--embedding-model``: remote embedding model identifier.
- ``--embedding-api-base-url``: remote embedding API endpoint.
- ``--embedding-api-key-env``: env var with embedding API key/token.
- ``--embedding-dimensions``: embedding dimensionality.
- ``--embedding-timeout-seconds``: embedding HTTP timeout.
- ``--embedding-top-k``: nearest-neighbor links added per note.
- ``--embedding-similarity-threshold``: minimum similarity for ``similar_to`` edges.
- ``--vector-db-path``: optional output for serialized vectors JSON.
- ``--disable-external-urls``: drop URL references during hyperlink ingestion.

KG generator/updater
--------------------

The KG updater reads bibliography fields (title, abstract/text, authors, publication), creates embeddings for each paper, writes an optional vector DB, and revises the knowledge graph using similarity + RAG context.

Local embedding config:

.. code-block:: json

   {
     "kg_update": {
       "embedding": {
         "provider": "local",
         "dimensions": 256
       },
       "vector_db_path": "build/vectors_local.json",
       "rag_top_k": 3,
       "similarity_threshold": 0.2,
       "max_entities_per_doc": 8
     }
   }

OpenAI-compatible embedding config:

.. code-block:: json

   {
     "kg_update": {
       "embedding": {
         "provider": "openai",
         "model": "text-embedding-3-small",
         "api_base_url": "https://api.openai.com/v1",
         "api_key_env": "OPENAI_API_KEY"
       }
     }
   }

Run with KG updates and save revised graph:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.csv \
     --bibliography-format csv \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --enable-kg-updates \
     --kg-update-config examples/kg_update_local.json \
     --kg-vector-db-path build/vectors.json \
     --updated-graph build/updated_graph.json \
     --output build/kg_update_demo.md \
     --report build/kg_update_demo_diagnostics.json \
     --title "KG Update Demo"

Functional-form soft validation
-------------------------------

Colophon can softly validate outline/agents/claims/bibliography/prompts against common structural,
rhetorical, and genre expectations declared in a functional-forms catalog.

Run with soft validation using the bundled example catalog:

.. code-block:: bash

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

Inspect diagnostics keys:

- ``soft_validation_enabled``
- ``soft_validation_form_id``
- ``soft_validation_result`` (findings + coverage summary)
- ``narrative_profile`` (tone/style/audience/discipline/genre/language settings used by agents)

Functional forms catalog shape (excerpt):

.. code-block:: json

   {
     "default_form_id": "sequential_transformation",
     "default_soft_validation_profile": "common",
     "soft_validation_profiles": {
       "common": {
         "outline": {"section_match_threshold": 0.5},
         "bibliography": {"min_text_chars": 60, "max_missing_ratio": 0.35},
         "prompts": {"required_prompt_keys": ["claim_template", "paragraph_template", "empty_section_template"]},
         "claims": {"min_evidence_link_ratio": 0.7}
       }
     },
     "functional_forms": [
       {
         "id": "sequential_transformation",
         "chapter_pattern": [],
         "elements": [],
         "coordination_ontology": {"paragraph": {}, "section": {}, "chapter": {}, "book": {}},
         "outline_expansion": {
           "chapter_goal_templates": {},
           "section_objective_templates": {},
           "subsection_templates": {},
           "prompt_hints": {}
         },
         "validation": {}
       }
     ],
     "companion_ontologies": [
       {
         "id": "wilson_academic_writing_companion",
         "path": "ontology/wilson_academic_writing_ontology.json"
       }
     ]
   }

Wilson companion writing ontology
---------------------------------

The bundled companion ontology summarizes academic-writing process guidance into:

- background prompts for claim/paragraph/outline/coordinator agents
- explicit assumptions for process, evidence, and argument logic
- executable validations that run alongside functional-form soft validation

Run with companion ontology diagnostics:

.. code-block:: bash

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

Inspect diagnostics keys:

- ``writing_ontology_context`` (compatibility + merged role prompts)
- ``writing_ontology_validation_result`` (finding counts + findings)

Genre ontology profiles
-----------------------

Genre ontology profiles define defaults and prompts for:

- audience
- discipline
- style
- genre
- language
- tone

These profile defaults propagate into agent prompts, outline expansion, recommendation query/scoring
hints, and agent-focused soft-validation checks.

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --genre-ontology ontology/genre_ontology.json \
     --genre-profile-id technical_research \
     --enable-soft-validation \
     --functional-forms ontology/technical_forms.json \
     --functional-form-id imrad_contribution \
     --output build/genre_profile_demo.md \
     --report build/genre_profile_demo_diagnostics.json \
     --title "Genre Profile Demo"

Inspect diagnostics keys:

- ``genre_ontology_context``
- ``narrative_profile``

Technical-writing form catalogs
-------------------------------

You can also use a technical-writing ontology catalog (for example IMRaD, systems tradeoffs,
formal theory/proof, spec conformance, survey/taxonomy) with the same flags:

.. code-block:: bash

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

Notes:

- Technical schema keys such as ``chapter_or_section_pattern``/``section_pattern`` and
  ``required_elements`` are normalized automatically for coordination and expansion.
- Diagnostics include technical rule prefixes such as ``imrad_*``, ``sys_*``, ``theory_*``,
  ``spec_*``, and ``survey_*`` when declared by the selected form.

Outline expander
----------------

The outline expander accepts a preliminary chapter/section draft and produces:

- an expanded outline with chapter goals, section objectives, subsection titles, evidence focus, and deliverables
- an accompanying prompt bundle you can feed directly into drafting
- when functional forms are loaded, chapter/section expansion aligns to ``chapter_pattern`` + ``elements`` ontology

Run outline expansion during manuscript generation:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline_preliminary.json \
     --graph examples/seed_graph.json \
     --enable-outline-expander \
     --outline-max-subsections 4 \
     --outline-expansion-report build/outline_expansion.json \
     --expanded-outline-output build/expanded_outline.json \
     --expanded-prompts-output build/expanded_prompts.json \
     --output build/outline_expander_demo.md \
     --report build/outline_expander_demo_diagnostics.json \
     --title "Outline Expander Demo"

Prompt templates
----------------

.. code-block:: json

   {
     "prompts": {
       "claim_template": "Claim: {lead_entity} appears in {source_title} as evidence for {section_title_lower}.",
       "figure_reference_template": "See Figure {figure_id} ({figure_caption}).",
       "paragraph_template": "{claim_text}\\nEvidence: {citations}\\nVisuals: {figure_references}",
       "empty_section_template": "Insufficient grounding evidence was retrieved for this section."
     }
   }

Narrative placeholders available in claim/paragraph templates:

- ``{narrative_instruction}``
- ``{narrative_tone}``
- ``{narrative_style}``
- ``{narrative_audience}``
- ``{narrative_discipline}``
- ``{narrative_genre}``
- ``{narrative_language}``

LLM hooks
---------

OpenAI/Codex style config:

.. code-block:: json

   {
     "llm": {
       "provider": "openai",
       "model": "gpt-5",
       "api_key_env": "OPENAI_API_KEY",
       "api_base_url": "https://api.openai.com/v1",
       "temperature": 0.2,
       "max_tokens": 256
     }
   }

Anthropic/Claude style config:

.. code-block:: json

   {
     "llm": {
       "provider": "claude",
       "model": "claude-3-5-sonnet-latest",
       "api_key_env": "ANTHROPIC_API_KEY",
       "api_base_url": "https://api.anthropic.com/v1"
     }
   }

GitHub/Copilot-style config:

.. code-block:: json

   {
     "llm": {
       "provider": "copilot",
       "model": "gpt-4o-mini",
       "api_key_env": "GITHUB_TOKEN",
       "api_base_url": "https://models.inference.ai.azure.com"
     }
   }

Output formats
--------------

.. code-block:: bash

   # plain text
   colophon ... --output build/manuscript.txt --output-format text

   # markdown
   colophon ... --output build/manuscript.md --output-format markdown

   # reStructuredText
   colophon ... --output build/manuscript.rst --output-format rst

   # Rich Text Format
   colophon ... --output build/manuscript.rtf --output-format rtf

   # LaTeX
   colophon ... --output build/manuscript.tex --output-format latex

Output layout
-------------

Single-file monolith (default):

.. code-block:: bash

   colophon ... --output build/manuscript.md --output-layout single

Chapter-level project output:

.. code-block:: bash

   colophon ... \
     --output build/manuscript_project \
     --output-format markdown \
     --output-layout project

Project mode writes:

- ``index.<ext>`` full manuscript output
- ``chapter-XX-<slug>.<ext>`` per chapter file
- ``manifest.json`` with chapter file mapping

Coordination and message passing
--------------------------------

Hierarchy of coordination/editing agents:

- paragraph coordinator (claim-to-paragraph coherence, readability, persuasion)
- section coordinator (paragraph-to-section coherence vs outline)
- chapter coordinator (section-to-chapter coherence vs outline)
- book coordinator (chapter-to-book coherence)

These agents communicate through a message bus and surface:

- ``coordination_messages`` in diagnostics JSON
- ``gap_requests`` in diagnostics JSON
- ``Gap Requests`` section in rendered manuscript outputs
- When functional forms are loaded, coordinator guidance and gap checks are enriched by
  ``coordination_ontology`` + chapter/element expectations from ``ontology/functional_forms.json`` (or your override).
- When the companion writing ontology is loaded, message-bus guidance and validation diagnostics are additionally
  enriched by ``ontology/wilson_academic_writing_ontology.json``.

After drafting, coordinators run an iterative revise loop until content and cross-agent coordination state stabilizes
or ``--coordination-max-iterations`` is reached. See ``coordination_revision`` in diagnostics for convergence history.

Example gap-surfacing run:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.bib \
     --bibliography-format bibtex \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --top-k 0 \
     --output build/coordination_demo.md \
     --report build/coordination_demo_diagnostics.json \
     --title "Coordination Demo"

Paper recommendation workflow
-----------------------------

Example recommendation run:

.. code-block:: bash

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

Semantic Scholar example:

.. code-block:: bash

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

Outputs:

- ``recommendation_proposals`` in diagnostics JSON.
- ``Recommended Papers`` section in rendered manuscript.
- optional recommendation report with bibliography/KG update proposal payloads.

Run tests
---------

.. code-block:: bash

   python -m unittest discover -s tests -q

Build docs
----------

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

See also
--------

- :doc:`getting_started`
- :doc:`examples`
- :doc:`api`
- :doc:`references`

Selected citations
------------------

- OpenAlex API: https://api.openalex.org/
- Semantic Scholar API: https://api.semanticscholar.org/
- BibTeX format: http://www.bibtex.org/Format/
- Sphinx: https://www.sphinx-doc.org/
