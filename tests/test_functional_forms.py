import json
import unittest
from pathlib import Path

from colophon.functional_forms import run_soft_validation, select_functional_form
from colophon.models import Chapter, Claim, Paragraph, Section, Source


class FunctionalFormsValidationTests(unittest.TestCase):
    def _load_catalog(self) -> dict:
        path = Path("ontology/functional_forms.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_technical_catalog(self) -> dict:
        path = Path("ontology/technical_forms.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_soft_validation_reports_cross_component_findings(self) -> None:
        catalog = self._load_catalog()
        outline = [{"title": "Methods", "sections": ["Data"]}]
        bibliography = [Source(id="s1", title="Paper", authors=[], year=None, text="short")]
        prompts = {"claim_template": "Claim: {lead_entity}"}
        chapters = [
            Chapter(
                id="ch1",
                title="Methods",
                sections=[
                    Section(
                        id="ch1-s1",
                        title="Data",
                        claims=[Claim(id="c1", text="This was inevitable.", evidence_ids=[])],
                        paragraphs=[Paragraph(id="p1", text="This was inevitable.", claim_ids=["c1"])],
                    )
                ],
            )
        ]

        result = run_soft_validation(
            functional_forms_payload=catalog,
            outline=outline,
            bibliography=bibliography,
            prompts=prompts,
            chapters=chapters,
            agent_profile={"top_k": 0, "enable_coordination_agents": False, "llm_enabled": False},
            form_id="sequential_transformation",
            max_findings=32,
        )

        self.assertEqual(result.form_id, "sequential_transformation")
        self.assertTrue(result.findings)
        components = {finding.component for finding in result.findings}
        self.assertIn("outline", components)
        self.assertIn("bibliography", components)
        self.assertIn("prompts", components)
        self.assertIn("agents", components)
        self.assertIn("claims", components)

    def test_soft_validation_selects_requested_form(self) -> None:
        catalog = self._load_catalog()
        result = run_soft_validation(
            functional_forms_payload=catalog,
            outline=[{"title": "Comparison", "sections": ["Case A", "Case B"]}],
            bibliography=[],
            prompts={},
            chapters=[],
            agent_profile={"top_k": 1, "enable_coordination_agents": True, "llm_enabled": False},
            form_id="structured_comparison",
            max_findings=16,
        )

        self.assertEqual(result.form_id, "structured_comparison")
        as_dict = result.to_dict()
        self.assertIn("finding_counts", as_dict)
        self.assertIn("findings", as_dict)

    def test_catalog_includes_coordination_and_outline_expansion_ontology(self) -> None:
        catalog = self._load_catalog()
        forms = catalog.get("functional_forms", [])

        self.assertTrue(forms)
        for form in forms:
            self.assertIn("coordination_ontology", form)
            self.assertIn("outline_expansion", form)
            self.assertIsInstance(form.get("coordination_ontology"), dict)
            self.assertIsInstance(form.get("outline_expansion"), dict)

    def test_catalog_references_companion_writing_ontology(self) -> None:
        catalog = self._load_catalog()
        companions = catalog.get("companion_ontologies", [])

        self.assertTrue(companions)
        self.assertTrue(any(row.get("id") == "wilson_academic_writing_companion" for row in companions if isinstance(row, dict)))

    def test_select_functional_form_normalizes_technical_schema(self) -> None:
        catalog = self._load_technical_catalog()
        selected = select_functional_form(catalog, form_id="imrad_contribution")

        self.assertEqual(selected.get("id"), "imrad_contribution")
        self.assertIn("chapter_pattern", selected)
        self.assertIsInstance(selected.get("chapter_pattern"), list)
        self.assertTrue(selected.get("chapter_pattern"))
        first_required = selected["chapter_pattern"][0].get("required_sections", [])
        self.assertIn("problem_frame", first_required)
        self.assertIn("coordination_ontology", selected)
        self.assertIn("outline_expansion", selected)

    def test_soft_validation_runs_for_technical_forms_catalog(self) -> None:
        catalog = self._load_technical_catalog()
        outline = [{"title": "Methods", "sections": ["Benchmark Design"]}]
        bibliography = [Source(id="s1", title="Paper", authors=[], year=None, text="short")]
        prompts = {"claim_template": "Claim: {lead_entity}"}
        chapters = [
            Chapter(
                id="ch1",
                title="Methods",
                sections=[
                    Section(
                        id="ch1-s1",
                        title="Benchmark Design",
                        claims=[Claim(id="c1", text="This causes better results.", evidence_ids=[])],
                        paragraphs=[Paragraph(id="p1", text="This causes better results.", claim_ids=["c1"])],
                    )
                ],
            )
        ]

        result = run_soft_validation(
            functional_forms_payload=catalog,
            outline=outline,
            bibliography=bibliography,
            prompts=prompts,
            chapters=chapters,
            agent_profile={"top_k": 0, "enable_coordination_agents": False, "llm_enabled": False},
            form_id="imrad_contribution",
            max_findings=32,
        )

        self.assertEqual(result.form_id, "imrad_contribution")
        self.assertTrue(result.findings)
        self.assertIn("imrad_results_without_method", result.rule_checks_declared)


if __name__ == "__main__":
    unittest.main()
