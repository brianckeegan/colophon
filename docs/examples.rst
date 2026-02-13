Examples
========

This page indexes example inputs and common command variants.

Example Inputs
--------------

- Bibliography: ``examples/bibliography.json``, ``examples/bibliography.csv``, ``examples/bibliography.bib``
- Outlines: ``examples/outline.json``, ``examples/outline_preliminary.json``
- Knowledge graph seeds: ``examples/seed_graph.json``, ``examples/graph_edges.csv``, ``examples/graph_dump.sql``
- Prompt templates: ``examples/prompts.json``
- LLM configs: ``examples/llm_openai.json``, ``examples/llm_codex.json``, ``examples/llm_claude.json``, ``examples/llm_copilot.json``
- Recommendation configs: ``examples/recommendation_openalex.json``, ``examples/recommendation_semantic_scholar.json``
- KG update configs: ``examples/kg_update_local.json``, ``examples/kg_update_openai.json``
- Note-import sources: ``examples/notes/``
- Ontology catalogs: ``ontology/functional_forms.json``, ``ontology/technical_forms.json``, ``ontology/genre_ontology.json``, ``ontology/wilson_academic_writing_ontology.json``

Common Runs
-----------

Baseline draft:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --output build/examples_baseline.md \
     --report build/examples_baseline_diagnostics.json

Technical-form + soft-validation draft:

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --outline examples/outline_preliminary.json \
     --graph examples/seed_graph.json \
     --enable-outline-expander \
     --enable-soft-validation \
     --functional-forms ontology/technical_forms.json \
     --functional-form-id imrad_contribution \
     --output build/examples_technical.md \
     --report build/examples_technical_diagnostics.json

Notes-import then draft:

.. code-block:: bash

   colophon-import-notes \
     --source examples/notes/obsidian \
     --platform obsidian \
     --output build/examples_notes_graph.json \
     --report build/examples_notes_report.json

   colophon \
     --bibliography examples/bibliography.json \
     --outline examples/outline.json \
     --graph build/examples_notes_graph.json \
     --output build/examples_notes_draft.md \
     --report build/examples_notes_diagnostics.json
