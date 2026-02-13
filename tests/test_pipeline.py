import unittest
import json
from pathlib import Path

from colophon.graph import graph_from_dict
from colophon.kg_update import KGUpdateConfig
from colophon.models import Source
from colophon.pipeline import ColophonPipeline, PipelineConfig
from colophon.recommendations import RecommendationConfig, RecommendedPaper
from colophon.vectors import EmbeddingConfig


class _FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls += 1
        if "Write one factual claim sentence" in prompt:
            return "Pipeline LLM claim."
        return "Pipeline LLM paragraph."


class _StubRecommendationClient:
    def __init__(self, by_source: dict[str, list[RecommendedPaper]]) -> None:
        self.by_source = by_source

    def find_related(self, seed_source: Source, max_results: int) -> list[RecommendedPaper]:
        return self.by_source.get(seed_source.id, [])[:max_results]


class _StubOutlineExpander:
    def expand(self, preliminary_outline):
        return {
            "chapters": [
                {
                    "title": "Expanded Foundations",
                    "sections": ["Knowledge Graph Overview"],
                    "section_details": [
                        {
                            "title": "Knowledge Graph Overview",
                            "objective": "Expand scope and argument depth.",
                            "subsections": ["Scope", "Mechanisms", "Implications"],
                            "evidence_focus": ["title", "abstract"],
                            "deliverables": ["Claim set"],
                        }
                    ],
                }
            ],
            "prompts": {
                "claim_template": "Expanded claim from {source_title} for {section_title}.",
            },
            "diagnostics": {"chapters": 1, "sections": 1},
        }


class PipelineTests(unittest.TestCase):
    def test_pipeline_emits_soft_validation_diagnostics_when_enabled(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="Seed Literature",
                authors=["Alice"],
                year=2021,
                text="Short text.",
            )
        ]
        outline = [{"title": "Methods", "sections": ["Data"]}]
        graph = graph_from_dict({"entities": [], "relations": []})
        forms = json.loads(Path("ontology/functional_forms.json").read_text(encoding="utf-8"))

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Soft Validation Manuscript",
                top_k=0,
                enable_coordination_agents=False,
                enable_soft_validation=True,
                functional_forms=forms,
                functional_form_id="sequential_transformation",
                max_soft_validation_findings=16,
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        self.assertTrue(manuscript.diagnostics["soft_validation_enabled"])
        self.assertEqual(manuscript.diagnostics["soft_validation_form_id"], "sequential_transformation")
        payload = manuscript.diagnostics["soft_validation_result"]
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload["findings"])

    def test_pipeline_emits_narrative_profile_in_diagnostics(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="KG Paper",
                authors=["A"],
                year=2022,
                text="Knowledge graph methods improve factual grounding.",
            )
        ]
        outline = [{"title": "Intro", "sections": ["Knowledge Graph Overview"]}]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Narrative Profile Manuscript",
                top_k=1,
                narrative_tone="formal",
                narrative_style="persuasive",
                narrative_audience="policy analysts",
                narrative_discipline="public policy",
                narrative_language="Spanish",
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        profile = manuscript.diagnostics["narrative_profile"]
        self.assertEqual(profile["tone"], "formal")
        self.assertEqual(profile["style"], "persuasive")
        self.assertEqual(profile["audience"], "policy analysts")
        self.assertEqual(profile["discipline"], "public policy")
        self.assertEqual(profile["language"], "Spanish")

    def test_pipeline_applies_genre_ontology_profile_defaults(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="KG Paper",
                authors=["A"],
                year=2022,
                text="Knowledge graph methods improve factual grounding.",
            )
        ]
        outline = [{"title": "Intro", "sections": ["Knowledge Graph Overview"]}]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})
        genre_ontology = json.loads(Path("ontology/genre_ontology.json").read_text(encoding="utf-8"))

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Genre Profile Manuscript",
                top_k=1,
                genre_ontology=genre_ontology,
                genre_profile_id="technical_research",
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        profile = manuscript.diagnostics["narrative_profile"]
        self.assertEqual(profile["style"], "technical")
        self.assertEqual(profile["genre"], "technical_report")
        self.assertEqual(profile["audience"], "technical experts")
        genre_context = manuscript.diagnostics["genre_ontology_context"]
        self.assertEqual(genre_context["profile_id"], "technical_research")

    def test_pipeline_generates_sections_and_diagnostics(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="KG Paper",
                authors=["A"],
                year=2022,
                text="Knowledge graph methods improve factual grounding.",
            )
        ]
        outline = [{"title": "Intro", "sections": ["Knowledge Graph Overview"]}]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})

        pipeline = ColophonPipeline(config=PipelineConfig(title="Test Manuscript", top_k=1))
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        self.assertEqual(manuscript.title, "Test Manuscript")
        self.assertEqual(manuscript.diagnostics["chapters_generated"], 1)
        self.assertEqual(manuscript.diagnostics["sections_generated"], 1)
        self.assertEqual(manuscript.diagnostics["citation_issues"], [])
        self.assertEqual(manuscript.diagnostics["figure_issues"], [])
        self.assertEqual(manuscript.diagnostics["figures_available"], 0)
        self.assertFalse(manuscript.diagnostics["recommendations_enabled"])
        self.assertEqual(manuscript.diagnostics["recommendation_provider"], "none")
        self.assertIn("# Test Manuscript", manuscript.to_markdown())

    def test_pipeline_applies_prompt_template_overrides(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="KG Paper",
                authors=["A"],
                year=2022,
                text="Knowledge graph methods improve factual grounding.",
            )
        ]
        outline = [{"title": "Intro", "sections": ["Knowledge Graph Overview"]}]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Prompted Manuscript",
                top_k=1,
                prompt_templates={
                    "claim_template": "Prompted claim via {source_title} for {section_title_lower}.",
                    "paragraph_template": "P::{claim_text} C::{citations}",
                },
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        paragraph_text = manuscript.chapters[0].sections[0].paragraphs[0].text
        self.assertIn("Prompted claim via KG Paper", paragraph_text)
        self.assertIn("P::", paragraph_text)
        self.assertIn("C::", paragraph_text)

    def test_pipeline_uses_llm_hook_when_client_provided(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="KG Paper",
                authors=["A"],
                year=2022,
                text="Knowledge graph methods improve factual grounding.",
            )
        ]
        outline = [{"title": "Intro", "sections": ["Knowledge Graph Overview"]}]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})
        fake_llm = _FakeLLMClient()

        pipeline = ColophonPipeline(
            config=PipelineConfig(title="LLM Manuscript", top_k=1, llm_client=fake_llm, llm_system_prompt="sys")
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        paragraph_text = manuscript.chapters[0].sections[0].paragraphs[0].text
        self.assertIn("Pipeline LLM paragraph.", paragraph_text)
        self.assertGreaterEqual(fake_llm.calls, 2)

    def test_pipeline_references_figures_from_knowledge_graph(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="KG Paper",
                authors=["A"],
                year=2022,
                text="Knowledge graph methods improve factual grounding.",
            )
        ]
        outline = [{"title": "Intro", "sections": ["Knowledge Graph Overview"]}]
        graph = graph_from_dict(
            {
                "entities": ["Knowledge Graph"],
                "relations": [],
                "figures": [
                    {
                        "id": "fig-1",
                        "caption": "Knowledge graph overview",
                        "uri": "figures/kg-overview.png",
                        "related_entities": ["Knowledge Graph"],
                    }
                ],
            }
        )

        pipeline = ColophonPipeline(config=PipelineConfig(title="Figure Manuscript", top_k=1, max_figures_per_section=1))
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        section = manuscript.chapters[0].sections[0]
        self.assertEqual(len(section.figures), 1)
        self.assertEqual(section.figures[0].id, "fig-1")
        self.assertIn("fig-1", section.claims[0].figure_ids)
        self.assertEqual(manuscript.diagnostics["figures_available"], 1)
        self.assertGreaterEqual(manuscript.diagnostics["figures_referenced"], 1)

    def test_pipeline_surfaces_gap_requests_and_coordination_messages(self) -> None:
        bibliography: list[Source] = []
        outline = [{"title": "Intro", "sections": ["Uncovered Topic"]}]
        graph = graph_from_dict({"entities": [], "relations": []})

        pipeline = ColophonPipeline(config=PipelineConfig(title="Gap Manuscript", top_k=1))
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        self.assertGreater(len(manuscript.coordination_messages), 0)
        self.assertGreater(len(manuscript.gap_requests), 0)
        self.assertGreater(len(manuscript.diagnostics["coordination_messages"]), 0)
        self.assertGreater(len(manuscript.diagnostics["gap_requests"]), 0)
        self.assertTrue(any(gap["component"] == "bibliography" for gap in manuscript.diagnostics["gap_requests"]))
        self.assertIn("## Gap Requests", manuscript.to_markdown())
        coordination_revision = manuscript.diagnostics["coordination_revision"]
        self.assertTrue(coordination_revision["enabled"])
        self.assertGreaterEqual(coordination_revision["iterations_run"], 1)
        self.assertIn("history", coordination_revision)
        self.assertTrue(coordination_revision["converged"])

    def test_pipeline_can_disable_coordination_agents(self) -> None:
        bibliography: list[Source] = []
        outline = [{"title": "Intro", "sections": ["Uncovered Topic"]}]
        graph = graph_from_dict({"entities": [], "relations": []})

        pipeline = ColophonPipeline(
            config=PipelineConfig(title="No Coordination Manuscript", top_k=1, enable_coordination_agents=False)
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        self.assertEqual(manuscript.coordination_messages, [])
        self.assertEqual(manuscript.gap_requests, [])
        self.assertEqual(manuscript.diagnostics["coordination_messages"], [])
        self.assertEqual(manuscript.diagnostics["gap_requests"], [])
        coordination_revision = manuscript.diagnostics["coordination_revision"]
        self.assertFalse(coordination_revision["enabled"])
        self.assertEqual(coordination_revision["iterations_run"], 0)

    def test_pipeline_generates_recommendation_proposals(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="Seed Literature",
                authors=["Alice"],
                year=2021,
                text="Knowledge graph retrieval methods for literature discovery.",
                metadata={"publication": "Journal A"},
            )
        ]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})
        outline = [{"title": "Intro", "sections": ["Overview"]}]
        recommendation_client = _StubRecommendationClient(
            {
                "s1": [
                    RecommendedPaper(
                        paper_id="W1",
                        title="Related Discovery Paper",
                        authors=["Alice", "Bob"],
                        publication="Journal B",
                        year=2024,
                        abstract="Knowledge graph retrieval improves scientific literature discovery.",
                        citation_count=120,
                        source_url="https://example.org/w1",
                        doi="https://doi.org/10.1000/w1",
                    )
                ]
            }
        )

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Recommendation Manuscript",
                top_k=1,
                enable_paper_recommendations=True,
                recommendation_config=RecommendationConfig(top_k=5, min_score=0.01),
                recommendation_client=recommendation_client,
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        self.assertEqual(len(manuscript.recommendation_proposals), 1)
        self.assertEqual(manuscript.diagnostics["recommendations_generated"], 1)
        self.assertTrue(manuscript.diagnostics["recommendation_proposals"])
        self.assertIn("## Recommended Papers", manuscript.to_markdown())

    def test_pipeline_adds_gap_when_recommendations_enabled_but_empty(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="Seed Literature",
                authors=["Alice"],
                year=2021,
                text="Knowledge graph retrieval methods for literature discovery.",
            )
        ]
        graph = graph_from_dict({"entities": [], "relations": []})
        outline = [{"title": "Intro", "sections": ["Overview"]}]
        recommendation_client = _StubRecommendationClient({"s1": []})

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Recommendation Gap",
                top_k=1,
                enable_paper_recommendations=True,
                recommendation_config=RecommendationConfig(top_k=5, min_score=0.01),
                recommendation_client=recommendation_client,
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        self.assertEqual(manuscript.diagnostics["recommendations_generated"], 0)
        self.assertTrue(any(gap.component == "bibliography" for gap in manuscript.gap_requests))

    def test_pipeline_runs_kg_updater_when_enabled(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="Graph Grounding",
                authors=["Alice"],
                year=2022,
                text="Knowledge graph grounding improves narrative reliability.",
                metadata={"publication": "Journal A"},
            ),
            Source(
                id="s2",
                title="Retrieval Grounding",
                authors=["Bob"],
                year=2023,
                text="Retrieval augmentation improves narrative reliability.",
                metadata={"publication": "Journal B"},
            ),
        ]
        outline = [{"title": "Intro", "sections": ["Knowledge Graph Overview"]}]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="KG Update Manuscript",
                top_k=1,
                enable_kg_updates=True,
                kg_update_config=KGUpdateConfig(
                    embedding_config=EmbeddingConfig(provider="local", dimensions=64),
                    rag_top_k=1,
                    similarity_threshold=0.0,
                    max_entities_per_doc=4,
                ),
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        self.assertTrue(manuscript.diagnostics["kg_updates_enabled"])
        self.assertIsInstance(manuscript.diagnostics["kg_update_result"], dict)
        self.assertEqual(manuscript.diagnostics["kg_update_result"]["embeddings_indexed"], 2)
        self.assertIn("paper:s1", graph.entities)
        self.assertIn("paper:s2", graph.entities)

    def test_pipeline_uses_outline_expander_output(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="KG Paper",
                authors=["A"],
                year=2022,
                text="Knowledge graph methods improve factual grounding.",
            )
        ]
        outline = [{"title": "Intro", "sections": ["Draft Section"]}]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Expanded Outline Manuscript",
                top_k=1,
                enable_outline_expander=True,
                outline_expander=_StubOutlineExpander(),
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        self.assertEqual(manuscript.chapters[0].title, "Expanded Foundations")
        self.assertEqual(manuscript.chapters[0].sections[0].title, "Knowledge Graph Overview")
        self.assertIn("Expanded claim from KG Paper", manuscript.chapters[0].sections[0].claims[0].text)
        self.assertTrue(manuscript.diagnostics["outline_expander_enabled"])
        self.assertIsInstance(manuscript.diagnostics["outline_expansion_result"], dict)

    def test_pipeline_uses_functional_form_context_for_coordination_and_outline_expansion(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="KG Paper",
                authors=["A"],
                year=2022,
                text="Knowledge graph methods improve factual grounding.",
            )
        ]
        outline = [{"title": "Foundations", "sections": []}]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})
        forms = json.loads(Path("ontology/functional_forms.json").read_text(encoding="utf-8"))

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Form-Aware Manuscript",
                top_k=1,
                enable_outline_expander=True,
                functional_forms=forms,
                functional_form_id="sequential_transformation",
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        context = manuscript.diagnostics["functional_form_context"]
        self.assertEqual(context["form_id"], "sequential_transformation")
        expansion = manuscript.diagnostics["outline_expansion_result"]
        self.assertEqual(expansion["diagnostics"]["functional_form_id"], "sequential_transformation")
        self.assertTrue(
            any(
                "functional-form element" in message["content"].lower()
                for message in manuscript.diagnostics["coordination_messages"]
            )
        )

    def test_pipeline_supports_technical_form_ontology_for_coordination_and_soft_validation(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="Systems Paper",
                authors=["A"],
                year=2024,
                text="This paper reports benchmarks, baselines, and implementation details for reproducible evaluation.",
            )
        ]
        outline = [{"title": "Introduction", "sections": []}]
        graph = graph_from_dict({"entities": ["System"], "relations": []})
        forms = json.loads(Path("ontology/technical_forms.json").read_text(encoding="utf-8"))

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Technical Form Manuscript",
                top_k=1,
                enable_outline_expander=True,
                enable_soft_validation=True,
                functional_forms=forms,
                functional_form_id="imrad_contribution",
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        context = manuscript.diagnostics["functional_form_context"]
        self.assertEqual(context["form_id"], "imrad_contribution")
        expansion = manuscript.diagnostics["outline_expansion_result"]
        self.assertEqual(expansion["diagnostics"]["functional_form_id"], "imrad_contribution")
        self.assertTrue(manuscript.diagnostics["soft_validation_enabled"])
        self.assertEqual(manuscript.diagnostics["soft_validation_result"]["form_id"], "imrad_contribution")
        self.assertIn("imrad_results_without_method", manuscript.diagnostics["soft_validation_result"]["rule_checks_declared"])

    def test_pipeline_emits_writing_ontology_context_and_validation(self) -> None:
        bibliography = [Source(id="s1", title="Seed", authors=["A"], year=2020, text="short")]
        outline = [{"title": "Overview", "sections": ["Background"]}]
        graph = graph_from_dict({"entities": [], "relations": []})
        ontology = json.loads(Path("ontology/wilson_academic_writing_ontology.json").read_text(encoding="utf-8"))

        pipeline = ColophonPipeline(
            config=PipelineConfig(
                title="Writing Ontology Manuscript",
                top_k=0,
                functional_forms=None,
                writing_ontology=ontology,
            )
        )
        manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

        context = manuscript.diagnostics["writing_ontology_context"]
        self.assertTrue(context["enabled"])
        self.assertEqual(context["ontology_id"], "wilson_academic_writing_companion")
        validation = manuscript.diagnostics["writing_ontology_validation_result"]
        self.assertIsInstance(validation, dict)
        self.assertTrue(validation["findings"])
        self.assertTrue(
            any(message["sender"] == "writing_ontology" for message in manuscript.diagnostics["coordination_messages"])
        )


if __name__ == "__main__":
    unittest.main()
