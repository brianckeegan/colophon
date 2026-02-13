import unittest
from pathlib import Path


class DocumentationCoverageTests(unittest.TestCase):
    def test_index_includes_core_docs_surfaces(self) -> None:
        index_text = Path("docs/index.rst").read_text(encoding="utf-8")
        for token in ("getting_started", "examples", "usage", "tutorial", "api", "references"):
            self.assertIn(token, index_text)

    def test_api_reference_includes_core_modules(self) -> None:
        api_text = Path("docs/api.rst").read_text(encoding="utf-8")
        modules = (
            "colophon",
            "colophon.models",
            "colophon.graph",
            "colophon.retrieval",
            "colophon.agents",
            "colophon.coordination",
            "colophon.pipeline",
            "colophon.io",
            "colophon.cli",
            "colophon.__main__",
            "colophon.llm",
            "colophon.vectors",
            "colophon.kg_update",
            "colophon.recommendations",
            "colophon.functional_forms",
            "colophon.ontology",
            "colophon.writing_ontology",
            "colophon.genre_ontology",
            "colophon.note_import",
            "colophon.import_cli",
        )
        for module_name in modules:
            self.assertIn(f".. automodule:: {module_name}", api_text)

    def test_readme_links_to_docs_surfaces_and_ontology_catalogs(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        docs_paths = (
            "/Users/briankeegan/Documents/New project/docs/getting_started.rst",
            "/Users/briankeegan/Documents/New project/docs/usage.rst",
            "/Users/briankeegan/Documents/New project/docs/tutorial.rst",
            "/Users/briankeegan/Documents/New project/docs/examples.rst",
            "/Users/briankeegan/Documents/New project/docs/api.rst",
            "/Users/briankeegan/Documents/New project/docs/references.rst",
        )
        for path in docs_paths:
            self.assertIn(path, readme)

        ontology_paths = (
            "ontology/functional_forms.json",
            "ontology/technical_forms.json",
            "ontology/genre_ontology.json",
            "ontology/wilson_academic_writing_ontology.json",
        )
        for ontology_path in ontology_paths:
            self.assertIn(ontology_path, readme)

    def test_references_include_external_citations(self) -> None:
        references_text = Path("docs/references.rst").read_text(encoding="utf-8")
        for token in ("OpenAlex", "Semantic Scholar", "BibTeX", "Sphinx", "Wilson", "Lewis"):
            self.assertIn(token, references_text)


if __name__ == "__main__":
    unittest.main()
