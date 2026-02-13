# Colophon Ontology Catalogs

This directory contains ontology JSON catalogs used by Colophon runtime components.

## Files

- `functional_forms.json`: primary narrative/argument functional forms catalog.
- `technical_forms.json`: technical writing functional forms catalog.
- `genre_ontology.json`: narrative profile ontology (`audience`, `discipline`, `style`, `genre`, `language`, `tone`).
- `wilson_academic_writing_ontology.json`: companion writing ontology for prompts, assumptions, and validations.

## Shared Top-Level Metadata

Each ontology file includes consistent metadata keys:

- `id` (`str`): stable ontology identifier.
- `name` (`str`): human-readable ontology name.
- `ontology_type` (`str`): ontology category/type.
- `version` (`str`): ontology version.
- `schema_version` (`str`): schema compatibility version.

The `version` and `schema_version` values are intentionally aligned per file.
