Getting Started
===============

This page provides a minimum viable Colophon run and points to deeper guides.

Install
-------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[docs]

Run A Minimal Draft
-------------------

.. code-block:: bash

   colophon \
     --bibliography examples/bibliography.json \
     --bibliography-format json \
     --outline examples/outline.json \
     --graph examples/seed_graph.json \
     --prompts examples/prompts.json \
     --functional-forms ontology/functional_forms.json \
     --genre-ontology ontology/genre_ontology.json \
     --output build/getting_started.md \
     --output-format markdown \
     --report build/getting_started_diagnostics.json \
     --title "Getting Started Draft"

Core Ontology Catalogs
----------------------

- ``ontology/functional_forms.json``: narrative/argumentative form catalog.
- ``ontology/technical_forms.json``: technical-writing form catalog.
- ``ontology/genre_ontology.json``: audience/discipline/style/genre/language profiles.
- ``ontology/wilson_academic_writing_ontology.json``: companion writing prompts and validation checks.

Next Steps
----------

- User guide: :doc:`usage`
- End-to-end tutorial: :doc:`tutorial`
- Example assets and commands: :doc:`examples`
- API reference: :doc:`api`
- References and citations: :doc:`references`
