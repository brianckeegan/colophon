User Guide
==========

This guide focuses on practical Colophon workflows and command patterns.

Notebook Workflow (Python API)
------------------------------

If you want an end-to-end Colophon walkthrough without using CLI commands, use the
Jupyter tutorial notebook based on ``examples/`` inputs:

- Notebook source: ``examples/colophon_python_api_tutorial.ipynb``
- Rendered HTML export: :download:`colophon_python_api_tutorial.html <_static/colophon_python_api_tutorial.html>`

Deconstruct a PDF into input artifacts
--------------------------------------

Use ``deconstruct`` when you have only a PDF and need Colophon-ready artifacts
(bibliography, knowledge graph, outline, prompts):

.. code-block:: bash

   colophon deconstruct test.pdf

By default, Colophon writes artifact files next to the PDF:

- ``test_bibliography.json``
- ``test_kg.json``
- ``test_outline.json``
- ``test_prompts.json``

You can also customize destination and filename stem:

.. code-block:: bash

   colophon deconstruct uploads/test.pdf \
     --output-dir build/deconstruct \
     --stem paper_a

Generate a manuscript from deconstructed artifacts
--------------------------------------------------

.. code-block:: bash

   colophon \
     --bibliography build/deconstruct/paper_a_bibliography.json \
     --outline build/deconstruct/paper_a_outline.json \
     --graph build/deconstruct/paper_a_kg.json \
     --prompts build/deconstruct/paper_a_prompts.json \
     --output build/paper_a_manuscript.md \
     --report build/paper_a_diagnostics.json \
     --title "Paper A Draft"

Notes
-----

- PDF extraction uses ``kreuzberg`` first, with PyMuPDF as a fallback backend.
- Knowledge-graph construction in ``deconstruct`` uses ``sift-kg`` with a deterministic local fallback.
- Bibliography enrichment is best-effort via OpenAlex. When lookups fail, Colophon
  preserves parsed citation data and still emits all artifacts.

For full argument references and advanced flags, see :doc:`usage`.
