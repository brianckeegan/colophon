Tutorial
========

This tutorial walks through a minimum viable end-to-end run using:

- bibliography
- knowledge graph
- outline
- prompts
- optional LLM API hooks
- optional notes-to-KG import from Obsidian/Notion/OneNote/Evernote exports
- optional KG generator/updater with embeddings + vector similarity
- optional functional-form soft validation for structure/rhetoric/genre diagnostics
- optional Wilson companion writing ontology for background prompts, assumptions, and validations
- optional genre ontology profiles for audience/discipline/style/genre/language defaults and prompts
- optional outline expander for draft-to-detailed chapter planning
- hierarchical coordination/editing agents with message passing
- optional related-paper recommendation proposals

Minimum viable inputs
---------------------

Bibliography (``examples/bibliography.json``):

.. code-block:: json

   {
     "sources": [
       {
         "id": "src-01",
         "title": "Knowledge Graphs for Scholarly Writing",
         "authors": ["A. Researcher"],
         "year": 2022,
         "publication": "Journal of Graph Studies",
         "abstract": "Knowledge graphs provide explicit structure for entities and relations in scientific narratives."
       }
     ]
   }

Knowledge graph (JSON) (``examples/seed_graph.json``):

.. code-block:: json

   {
     "entities": ["Knowledge Graph", "Claim"],
     "relations": [
       {"source": "Knowledge Graph", "predicate": "supports", "target": "Claim"}
     ],
     "figures": [
       {
         "id": "fig-kg-overview",
         "caption": "Knowledge graph schema overview",
         "uri": "figures/kg-overview.png",
         "related_entities": ["Knowledge Graph", "Claim"]
       }
     ]
   }

Outline (``examples/outline.json``):

.. code-block:: json

   {
     "chapters": [
       {
         "title": "Foundations",
         "sections": ["Why Knowledge Graphs Matter"]
       }
     ]
   }

Preliminary outline input for expansion (``examples/outline_preliminary.json``):

.. code-block:: json

   {
     "chapters": [
       {
         "title": "Foundations",
         "sections": ["Why Knowledge Graphs Matter", "Why Retrieval Grounding Matters"]
       }
     ]
   }

Prompts (``examples/prompts.json``):

.. code-block:: json

   {
     "prompts": {
       "claim_template": "Claim: {lead_entity} appears in {source_title} as evidence for {section_title_lower}.",
       "figure_reference_template": "See Figure {figure_id} ({figure_caption}).",
       "paragraph_template": "{claim_text}\\nEvidence: {citations}\\nVisuals: {figure_references}",
       "empty_section_template": "Insufficient grounding evidence was retrieved for this section."
     }
   }

LLM config (OpenAI example) (``examples/llm_openai.json``):

.. code-block:: json

   {
     "llm": {
       "provider": "openai",
       "model": "gpt-5",
       "api_key_env": "OPENAI_API_KEY",
       "api_base_url": "https://api.openai.com/v1"
     }
   }

Optional: import a notes knowledge graph first
----------------------------------------------

Obsidian hyperlink + embedding import:

.. code-block:: bash

   colophon-import-notes \
     --source examples/notes/obsidian \
     --platform obsidian \
     --seed-graph examples/seed_graph.json \
     --output build/tutorial_notes_graph_obsidian.json \
     --report build/tutorial_notes_import_obsidian.json \
     --embedding-provider local \
     --embedding-top-k 2 \
     --embedding-similarity-threshold 0.25

Notion export variants:

.. code-block:: bash

   # markdown export folder
   colophon-import-notes \
     --source examples/notes/notion \
     --platform notion \
     --output build/tutorial_notes_graph_notion_markdown.json \
     --report build/tutorial_notes_import_notion_markdown.json \
     --disable-embeddings

   # JSON export
   colophon-import-notes \
     --source examples/notes/notion/notion_export.json \
     --platform notion \
     --output build/tutorial_notes_graph_notion_json.json \
     --report build/tutorial_notes_import_notion_json.json \
     --disable-embeddings

OneNote and Evernote exports:

.. code-block:: bash

   colophon-import-notes \
     --source examples/notes/onenote_export.json \
     --platform onenote \
     --output build/tutorial_notes_graph_onenote.json \
     --report build/tutorial_notes_import_onenote.json \
     --disable-embeddings

   colophon-import-notes \
     --source examples/notes/evernote_export.enex \
     --platform evernote \
     --output build/tutorial_notes_graph_evernote.json \
     --report build/tutorial_notes_import_evernote.json \
     --disable-embeddings

Notes formatter checklist:

- Obsidian: ``[[Wiki Links]]`` or markdown links between ``.md`` files in the vault.
- Notion: markdown exports, JSON lists with ``title``/``content`` fields, or CSV rows with ``title/content`` columns (use ``--platform notion`` for JSON/CSV).
- OneNote: JSON list under ``notes`` or ``pages`` with ``id/title/content``.
- Evernote: ``.enex`` with note ``title`` and HTML content.

Run the generator
-----------------

If you skipped notes import, replace ``build/tutorial_notes_graph_obsidian.json`` with ``examples/seed_graph.json`` in commands below.

Markdown output:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph build/tutorial_notes_graph_obsidian.json \
     --prompts examples/prompts.json \
     --max-figures-per-section 2 \
     --output build/tutorial.md \
     --output-format markdown \
     --report build/tutorial_diagnostics.json \
     --title "Tutorial Draft"

Enable LLM hooks (optional):

.. code-block:: bash

   export OPENAI_API_KEY=...
   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --llm-config examples/llm_openai.json \
     --max-figures-per-section 2 \
     --output build/tutorial_llm.md \
     --output-format markdown \
     --report build/tutorial_llm_diagnostics.json \
     --title "Tutorial Draft (LLM)"

Enable KG updates with local embeddings (optional):

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --enable-kg-updates \
     --kg-update-config examples/kg_update_local.json \
     --kg-vector-db-path build/tutorial_vectors_local.json \
     --updated-graph build/tutorial_updated_graph_local.json \
     --output build/tutorial_kg_local.md \
     --output-format markdown \
     --report build/tutorial_kg_local_diagnostics.json \
     --title "Tutorial Draft (KG Local)"

Enable KG updates with remote embeddings (OpenAI-compatible):

.. code-block:: bash

   export OPENAI_API_KEY=...
   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --enable-kg-updates \
     --kg-update-config examples/kg_update_openai.json \
     --kg-vector-db-path build/tutorial_vectors_openai.json \
     --updated-graph build/tutorial_updated_graph_openai.json \
     --output build/tutorial_kg_openai.md \
     --output-format markdown \
     --report build/tutorial_kg_openai_diagnostics.json \
     --title "Tutorial Draft (KG Remote)"

Enable outline expansion from a preliminary draft:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline_preliminary.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --enable-outline-expander \
     --functional-forms ontology/functional_forms.json \
     --functional-form-id sequential_transformation \
     --outline-max-subsections 4 \
     --outline-expansion-report build/tutorial_outline_expansion.json \
     --expanded-outline-output build/tutorial_expanded_outline.json \
     --expanded-prompts-output build/tutorial_expanded_prompts.json \
     --output build/tutorial_outline_expander.md \
     --output-format markdown \
     --report build/tutorial_outline_expander_diagnostics.json \
     --title "Tutorial Draft (Outline Expander)"

Enable functional-form soft validation:

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
     --functional-validation-report build/tutorial_functional_validation.json \
     --output build/tutorial_soft_validation.md \
     --output-format markdown \
     --report build/tutorial_soft_validation_diagnostics.json \
     --title "Tutorial Draft (Soft Validation)"

Enable companion writing ontology checks:

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
     --writing-ontology-report build/tutorial_writing_ontology_validation.json \
     --output build/tutorial_writing_ontology.md \
     --output-format markdown \
     --report build/tutorial_writing_ontology_diagnostics.json \
     --title "Tutorial Draft (Writing Ontology)"

Apply genre ontology profile defaults:

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
     --output build/tutorial_genre_profile.md \
     --output-format markdown \
     --report build/tutorial_genre_profile_diagnostics.json \
     --title "Tutorial Draft (Genre Profile)"

Use technical-writing forms (IMRaD example):

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
     --functional-validation-report build/tutorial_technical_validation.json \
     --output build/tutorial_technical_form.md \
     --output-format markdown \
     --report build/tutorial_technical_form_diagnostics.json \
     --title "Tutorial Draft (Technical Form)"

Output options
--------------

Plain text:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --max-figures-per-section 2 \
     --output build/tutorial.txt \
     --output-format text \
     --report build/tutorial_diagnostics_text.json \
     --title "Tutorial Draft"

reStructuredText:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --max-figures-per-section 2 \
     --output build/tutorial.rst \
     --output-format rst \
     --report build/tutorial_diagnostics_rst.json \
     --title "Tutorial Draft"

LaTeX:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --max-figures-per-section 2 \
     --output build/tutorial.tex \
     --output-format latex \
     --report build/tutorial_diagnostics_tex.json \
     --title "Tutorial Draft"

RTF:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --max-figures-per-section 2 \
     --output build/tutorial.rtf \
     --output-format rtf \
     --report build/tutorial_diagnostics_rtf.json \
     --title "Tutorial Draft"

Project output with chapter-level files:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --output build/tutorial_project \
     --output-format markdown \
     --output-layout project \
     --report build/tutorial_project_diagnostics.json \
     --title "Tutorial Draft"

Notes
-----

- ``--output-format auto`` infers format from output extension: ``.md``, ``.txt``, ``.rst``, ``.rtf``, ``.tex``.
- ``--output-layout project`` writes ``index`` + chapter files + ``manifest.json`` in the output directory.
- Bibliography input supports ``.json``, ``.csv``, and ``.bib``/``.bibtex``.
- Bibliography entries should include ``abstract`` (or ``text``) for strongest KG updates.
- Graph input also supports ``.csv``, ``.sqlite``, and ``.sql`` formats.
- KG updater diagnostics appear in ``kg_update_result``.
- ``--updated-graph`` writes the revised graph after KG updates.
- Outline expansion payload appears in ``outline_expansion_result`` diagnostics.
- ``--expanded-outline-output`` and ``--expanded-prompts-output`` write reusable planning artifacts.
- Outline expansion includes ``functional_chapter_type`` and ``functional_element_id`` fields when functional forms are loaded.
- Soft validation payload appears in ``soft_validation_result`` diagnostics.
- Writing-ontology context appears in ``writing_ontology_context`` diagnostics.
- Writing-ontology findings appear in ``writing_ontology_validation_result`` diagnostics.
- Genre-profile context appears in ``genre_ontology_context`` diagnostics.
- Technical form catalogs are supported through ``--functional-forms ontology/technical_forms.json``.
- Narrative controls appear in ``narrative_profile`` diagnostics.

Coordination hand-off demo
--------------------------

To see parent/child hand-offs and gap requests, run with retrieval disabled:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.bib \
     --bibliography-format bibtex \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --functional-forms ontology/functional_forms.json \
     --functional-form-id sequential_transformation \
     --top-k 0 \
     --coordination-max-iterations 6 \
     --output build/tutorial_coordination.md \
     --report build/tutorial_coordination_diagnostics.json \
     --title "Tutorial Coordination Demo"

Inspect:

- ``coordination_messages`` in ``build/tutorial_coordination_diagnostics.json``
- ``gap_requests`` in ``build/tutorial_coordination_diagnostics.json``
- ``coordination_revision`` in ``build/tutorial_coordination_diagnostics.json`` (iteration history + convergence)
- ``Gap Requests`` section in ``build/tutorial_coordination.md``

Paper recommendation demo
-------------------------

Generate recommendations for papers not currently in your bibliography:

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
     --recommendation-report build/tutorial_recommendations.json \
     --output build/tutorial_recommendation.md \
     --report build/tutorial_recommendation_diagnostics.json \
     --title "Tutorial Recommendation Demo"

Semantic Scholar recommendation variant:

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
     --output build/tutorial_recommendation_s2.md \
     --report build/tutorial_recommendation_s2_diagnostics.json \
     --title "Tutorial Recommendation Demo (Semantic Scholar)"

Inspect:

- ``recommendation_proposals`` in ``build/tutorial_recommendation_diagnostics.json``
- ``Recommended Papers`` section in ``build/tutorial_recommendation.md``
- ``build/tutorial_recommendations.json`` for direct bibliography/KG update proposals

See also
--------

- :doc:`getting_started`
- :doc:`usage`
- :doc:`examples`
- :doc:`api`
- :doc:`references`

Selected citations
------------------

- Wilson, Jeffrey R. *Academic Writing*.
- Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- OpenAlex API: https://api.openalex.org/
- Semantic Scholar API: https://api.semanticscholar.org/
