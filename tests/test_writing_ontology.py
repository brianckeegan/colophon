import json
import unittest
from pathlib import Path

from colophon.models import Chapter, Claim, Paragraph, Section, Source
from colophon.writing_ontology import build_writing_ontology_context, run_writing_ontology_validation


class WritingOntologyTests(unittest.TestCase):
    def _load_ontology(self) -> dict:
        return json.loads(Path("ontology/wilson_academic_writing_ontology.json").read_text(encoding="utf-8"))

    def test_context_merges_global_and_agent_prompts(self) -> None:
        ontology = self._load_ontology()
        context = build_writing_ontology_context(ontology_payload=ontology, form_id="sequential_transformation")

        self.assertTrue(context["enabled"])
        self.assertTrue(context["compatible"])
        claim_prompt = context["agent_prompts"]["claim_author_agent"]
        self.assertIn("Treat writing as thinking", claim_prompt)
        self.assertIn("Each claim should be answerable", claim_prompt)
        self.assertIn("question_problem_presence", context["validation_rule_ids"])

    def test_validation_reports_outline_and_grounding_gaps(self) -> None:
        ontology = self._load_ontology()
        outline = [{"title": "Overview", "sections": ["Background"]}]
        bibliography = [Source(id="s1", title="Short Source", authors=[], year=2020, text="tiny")]
        chapters = [
            Chapter(
                id="ch1",
                title="Overview",
                sections=[
                    Section(
                        id="ch1-s1",
                        title="Background",
                        claims=[Claim(id="c1", text="Claim only", evidence_ids=["s1"])],
                        paragraphs=[Paragraph(id="p1", text="This suggests a pattern.")],
                    )
                ],
            )
        ]

        result = run_writing_ontology_validation(
            ontology_payload=ontology,
            outline=outline,
            bibliography=bibliography,
            prompts={},
            chapters=chapters,
            form_id="sequential_transformation",
            functional_form={},
            coordination_revision={"enabled": True, "iterations_run": 1},
            max_findings=16,
        )

        codes = {finding.code for finding in result.findings}
        self.assertIn("question_problem_presence", codes)
        self.assertIn("method_visibility", codes)
        self.assertIn("implication_visibility", codes)
        self.assertIn("bibliography_grounding_density", codes)
        self.assertIn("iterative_revision_process", codes)


if __name__ == "__main__":
    unittest.main()
