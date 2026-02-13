import json
import unittest
from pathlib import Path

from colophon.agents import (
    CitationReviewerAgent,
    ClaimAuthorAgent,
    CoherenceReviewerAgent,
    FigureReviewerAgent,
    OutlineExpanderAgent,
    ParagraphAgent,
    SectionAgent,
)
from colophon.graph import KnowledgeGraph
from colophon.models import Claim, Figure, Paragraph, Section, Source
from colophon.retrieval import RetrievalHit


class _FakeLLMClient:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        return self.output_text


class _OutlineLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls += 1
        if "Return subsection titles" in prompt:
            return "Scope\nMechanisms\nImplications"
        return "Deliver a focused argument backed by evidence."


class AgentTests(unittest.TestCase):
    def test_claim_author_respects_max_claims_and_uses_entity(self) -> None:
        source1 = Source(id="s1", title="Title 1", authors=[], year=None, text="text")
        source2 = Source(id="s2", title="Title 2", authors=[], year=None, text="text")
        hits = [RetrievalHit(source=source1, score=0.9), RetrievalHit(source=source2, score=0.8)]

        claims = ClaimAuthorAgent(max_claims=1).draft(
            section_id="sec1",
            section_title="Methods",
            hits=hits,
            entities=["Knowledge Graph"],
            figures=[],
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].id, "sec1-c1")
        self.assertIn("Knowledge Graph", claims[0].text)
        self.assertEqual(claims[0].evidence_ids, ["s1"])

    def test_claim_author_uses_llm_hook_when_configured(self) -> None:
        fake_llm = _FakeLLMClient("LLM-authored claim")
        source = Source(id="s1", title="Paper", authors=[], year=None, text="Evidence sentence.")
        hits = [RetrievalHit(source=source, score=1.0)]

        claims = ClaimAuthorAgent(max_claims=1, llm_client=fake_llm, llm_system_prompt="system").draft(
            section_id="sec1",
            section_title="Methods",
            hits=hits,
            entities=["Entity"],
            figures=[],
        )

        self.assertEqual(claims[0].text, "LLM-authored claim")
        self.assertEqual(len(fake_llm.calls), 1)
        self.assertEqual(fake_llm.calls[0][1], "system")

    def test_claim_author_includes_narrative_profile_in_llm_prompt(self) -> None:
        fake_llm = _FakeLLMClient("LLM-authored claim")
        source = Source(id="s1", title="Paper", authors=[], year=None, text="Evidence sentence.")
        hits = [RetrievalHit(source=source, score=1.0)]

        ClaimAuthorAgent(
            max_claims=1,
            llm_client=fake_llm,
            tone="formal",
            style="persuasive",
            audience="policy analysts",
            discipline="public policy",
            language="Spanish",
        ).draft(
            section_id="sec1",
            section_title="Methods",
            hits=hits,
            entities=["Entity"],
            figures=[],
        )

        prompt = fake_llm.calls[0][0]
        self.assertIn("Tone: formal", prompt)
        self.assertIn("Style: persuasive", prompt)
        self.assertIn("Audience: policy analysts", prompt)
        self.assertIn("Discipline: public policy", prompt)
        self.assertIn("Language: Spanish", prompt)

    def test_paragraph_agent_handles_empty_claims(self) -> None:
        paragraphs = ParagraphAgent().draft(section_id="s1", claims=[], hits=[])

        self.assertEqual(len(paragraphs), 1)
        self.assertIn("No sufficiently grounded claims", paragraphs[0].text)

    def test_paragraph_agent_adds_citation_excerpt(self) -> None:
        source = Source(id="s1", title="Paper", authors=[], year=None, text="First sentence. Second sentence.")
        claim = Claim(id="c1", text="Claim text", evidence_ids=["s1"])

        paragraphs = ParagraphAgent().draft(
            section_id="sec",
            claims=[claim],
            hits=[RetrievalHit(source=source, score=1.0)],
        )

        self.assertEqual(len(paragraphs), 1)
        self.assertIn("Claim text", paragraphs[0].text)
        self.assertIn("[s1] First sentence.", paragraphs[0].text)

    def test_paragraph_agent_adds_figure_reference_text(self) -> None:
        claim = Claim(id="c1", text="Claim text", evidence_ids=[], figure_ids=["fig-1"])
        figure_lookup = {"fig-1": Figure(id="fig-1", caption="System architecture", uri="figures/system.png")}

        paragraphs = ParagraphAgent().draft_with_figures(
            section_id="sec",
            claims=[claim],
            hits=[],
            figure_lookup=figure_lookup,
        )

        self.assertIn("Figure fig-1: System architecture.", paragraphs[0].text)

    def test_paragraph_agent_uses_llm_hook_when_configured(self) -> None:
        fake_llm = _FakeLLMClient("LLM-authored paragraph")
        source = Source(id="s1", title="Paper", authors=[], year=None, text="Evidence.")
        claim = Claim(id="c1", text="Claim text", evidence_ids=["s1"])

        paragraphs = ParagraphAgent(llm_client=fake_llm, llm_system_prompt="system").draft(
            section_id="sec",
            claims=[claim],
            hits=[RetrievalHit(source=source, score=1.0)],
        )

        self.assertEqual(paragraphs[0].text, "LLM-authored paragraph")
        self.assertEqual(len(fake_llm.calls), 1)
        self.assertEqual(fake_llm.calls[0][1], "system")

    def test_paragraph_agent_applies_narrative_prefix_in_template_mode(self) -> None:
        source = Source(id="s1", title="Paper", authors=[], year=None, text="Evidence.")
        claim = Claim(id="c1", text="Claim text", evidence_ids=["s1"])

        paragraphs = ParagraphAgent(
            tone="conversational",
            style="expository",
            audience="undergraduates",
            discipline="sociology",
            language="French",
        ).draft(
            section_id="sec",
            claims=[claim],
            hits=[RetrievalHit(source=source, score=1.0)],
        )

        self.assertTrue(paragraphs[0].text.startswith("For undergraduates readers in sociology"))

    def test_section_agent_builds_section_ids(self) -> None:
        source = Source(id="s1", title="Paper", authors=[], year=None, text="Graph writing systems")
        hits = [RetrievalHit(source=source, score=1.0)]
        graph = KnowledgeGraph(entities={"Graph"})

        section = SectionAgent(claim_agent=ClaimAuthorAgent(), paragraph_agent=ParagraphAgent()).draft(
            chapter_id="ch1",
            index=2,
            section_title="Graph Systems",
            hits=hits,
            graph=graph,
        )

        self.assertEqual(section.id, "ch1-s2")
        self.assertEqual(section.title, "Graph Systems")
        self.assertTrue(section.claims)
        self.assertTrue(section.paragraphs)

    def test_section_agent_attaches_matching_figures(self) -> None:
        source = Source(id="s1", title="Paper", authors=[], year=None, text="Graph writing systems")
        hits = [RetrievalHit(source=source, score=1.0)]
        graph = KnowledgeGraph(entities={"Graph"})
        graph.add_figure(Figure(id="fig-1", caption="Graph Systems Diagram", uri="figures/graph.png"))

        section = SectionAgent(claim_agent=ClaimAuthorAgent(), paragraph_agent=ParagraphAgent()).draft(
            chapter_id="ch1",
            index=1,
            section_title="Graph Systems",
            hits=hits,
            graph=graph,
            max_figures=2,
        )

        self.assertEqual(len(section.figures), 1)
        self.assertEqual(section.figures[0].id, "fig-1")
        self.assertIn("fig-1", section.claims[0].figure_ids)

    def test_citation_reviewer_flags_unknown_sources(self) -> None:
        section = Section(
            id="sec1",
            title="Title",
            claims=[Claim(id="c1", text="x", evidence_ids=["known", "unknown"])],
            paragraphs=[],
        )

        issues = CitationReviewerAgent().review([section], known_source_ids={"known"})

        self.assertEqual(len(issues), 1)
        self.assertIn("unknown source unknown", issues[0])

    def test_coherence_reviewer_flags_duplicates_normalized(self) -> None:
        section = Section(
            id="sec1",
            title="Title",
            claims=[],
            paragraphs=[
                Paragraph(id="p1", text="Same text"),
                Paragraph(id="p2", text="  same   text  "),
            ],
        )

        issues = CoherenceReviewerAgent().review([section])

        self.assertEqual(len(issues), 1)
        self.assertIn("Paragraph p2 duplicates", issues[0])

    def test_figure_reviewer_flags_unknown_figures(self) -> None:
        section = Section(
            id="sec1",
            title="Title",
            claims=[Claim(id="c1", text="x", evidence_ids=[], figure_ids=["fig-1", "fig-missing"])],
            paragraphs=[],
        )

        issues = FigureReviewerAgent().review([section], known_figure_ids={"fig-1"})

        self.assertEqual(len(issues), 1)
        self.assertIn("unknown figure fig-missing", issues[0])

    def test_outline_expander_generates_details_and_prompts(self) -> None:
        expander = OutlineExpanderAgent(max_subsections_per_section=3)
        preliminary = {
            "chapters": [
                {
                    "title": "Foundations",
                    "sections": ["Why Knowledge Graphs Matter"],
                }
            ]
        }

        result = expander.expand(preliminary)

        self.assertIn("chapters", result)
        self.assertIn("prompts", result)
        self.assertEqual(len(result["chapters"]), 1)
        chapter = result["chapters"][0]
        self.assertEqual(chapter["title"], "Foundations")
        self.assertEqual(chapter["sections"], ["Why Knowledge Graphs Matter"])
        self.assertEqual(len(chapter["section_details"]), 1)
        self.assertLessEqual(len(chapter["section_details"][0]["subsections"]), 3)
        self.assertIn("claim_template", result["prompts"])
        self.assertIn("transition_template", result["prompts"])

    def test_outline_expander_uses_llm_hook(self) -> None:
        expander = OutlineExpanderAgent(
            max_subsections_per_section=3,
            llm_client=_OutlineLLMClient(),
            llm_system_prompt="system",
        )
        preliminary = {
            "chapters": [
                {
                    "title": "System Design",
                    "sections": ["Agent Responsibilities"],
                }
            ]
        }

        result = expander.expand(preliminary)
        section = result["chapters"][0]["section_details"][0]

        self.assertEqual(section["objective"], "Deliver a focused argument backed by evidence.")
        self.assertEqual(section["subsections"], ["Scope", "Mechanisms", "Implications"])

    def test_outline_expander_uses_functional_form_ontology(self) -> None:
        catalog = json.loads(Path("ontology/functional_forms.json").read_text(encoding="utf-8"))
        expander = OutlineExpanderAgent(
            max_subsections_per_section=4,
            functional_forms_payload=catalog,
            functional_form_id="sequential_transformation",
        )
        preliminary = {
            "chapters": [
                {"title": "Foundations", "sections": []},
                {"title": "Phase One", "sections": []},
                {"title": "Synthesis", "sections": []},
                {"title": "Conclusion", "sections": []},
            ]
        }

        result = expander.expand(preliminary)
        first = result["chapters"][0]

        self.assertEqual(first["functional_form_id"], "sequential_transformation")
        self.assertEqual(first["functional_chapter_type"], "introduction")
        self.assertTrue(any(detail["functional_element_id"] == "central_problem" for detail in first["section_details"]))
        self.assertEqual(result["diagnostics"]["functional_form_id"], "sequential_transformation")

    def test_outline_expander_supports_technical_form_schema(self) -> None:
        catalog = json.loads(Path("ontology/technical_forms.json").read_text(encoding="utf-8"))
        expander = OutlineExpanderAgent(
            max_subsections_per_section=4,
            functional_forms_payload=catalog,
            functional_form_id="imrad_contribution",
        )
        preliminary = {
            "chapters": [
                {"title": "Introduction", "sections": []},
                {"title": "Methods", "sections": []},
                {"title": "Experiments", "sections": []},
                {"title": "Discussion", "sections": []},
            ]
        }

        result = expander.expand(preliminary)
        first = result["chapters"][0]

        self.assertEqual(first["functional_form_id"], "imrad_contribution")
        self.assertEqual(first["functional_chapter_type"], "introduction")
        self.assertTrue(any(detail["functional_element_id"] == "problem_frame" for detail in first["section_details"]))
        self.assertEqual(result["diagnostics"]["functional_form_id"], "imrad_contribution")


if __name__ == "__main__":
    unittest.main()
