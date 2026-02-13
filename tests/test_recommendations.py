import json
import unittest
from unittest.mock import patch

from colophon.coordination import MessageBus
from colophon.graph import graph_from_dict
from colophon.models import Source
from colophon.recommendations import (
    OpenAlexSearchClient,
    PaperRecommendationWorkflow,
    RecommendationConfig,
    RecommendedPaper,
    SemanticScholarSearchClient,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _StubClient:
    def __init__(self, by_source: dict[str, list[RecommendedPaper]]) -> None:
        self.by_source = by_source

    def find_related(self, seed_source: Source, max_results: int) -> list[RecommendedPaper]:
        return self.by_source.get(seed_source.id, [])[:max_results]


class _RecordingClient:
    def __init__(self, items: list[RecommendedPaper]) -> None:
        self.items = items
        self.last_seed_source: Source | None = None

    def find_related(self, seed_source: Source, max_results: int) -> list[RecommendedPaper]:
        self.last_seed_source = seed_source
        return self.items[:max_results]


class RecommendationTests(unittest.TestCase):
    @patch("colophon.recommendations.request.urlopen")
    def test_openalex_client_parses_response(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "Related Paper",
                        "publication_year": 2023,
                        "cited_by_count": 42,
                        "doi": "https://doi.org/10.1000/test",
                        "primary_location": {
                            "landing_page_url": "https://example.org/paper",
                            "source": {"display_name": "Journal X"},
                        },
                        "authorships": [{"author": {"display_name": "Alice"}}],
                        "abstract_inverted_index": {"related": [0], "paper": [1]},
                    }
                ]
            }
        )

        client = OpenAlexSearchClient(config=RecommendationConfig(provider="openalex"))
        seed = Source(id="s1", title="Seed", authors=["Author"], year=2020, text="Seed abstract")

        papers = client.find_related(seed_source=seed, max_results=3)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].title, "Related Paper")
        self.assertEqual(papers[0].publication, "Journal X")
        self.assertEqual(papers[0].citation_count, 42)
        self.assertEqual(papers[0].abstract, "related paper")

    @patch("colophon.recommendations.request.urlopen")
    def test_semantic_scholar_client_parses_response(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(
            {
                "data": [
                    {
                        "paperId": "abc123",
                        "title": "Semantic Scholar Related Paper",
                        "year": 2021,
                        "citationCount": 17,
                        "abstract": "A recommendation-focused abstract.",
                        "url": "https://www.semanticscholar.org/paper/abc123",
                        "venue": "AI2 Venue",
                        "authors": [{"name": "Alice"}, {"name": "Bob"}],
                        "externalIds": {"DOI": "10.1000/s2paper"},
                    }
                ]
            }
        )

        client = SemanticScholarSearchClient(
            config=RecommendationConfig(
                provider="semantic_scholar",
                api_base_url="https://api.semanticscholar.org",
            )
        )
        seed = Source(id="s1", title="Seed", authors=["Author"], year=2020, text="Seed abstract")

        papers = client.find_related(seed_source=seed, max_results=3)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].paper_id, "abc123")
        self.assertEqual(papers[0].title, "Semantic Scholar Related Paper")
        self.assertEqual(papers[0].publication, "AI2 Venue")
        self.assertEqual(papers[0].authors, ["Alice", "Bob"])
        self.assertEqual(papers[0].doi, "https://doi.org/10.1000/s2paper")

    def test_workflow_default_client_supports_semantic_scholar_aliases(self) -> None:
        for provider in ("semantic_scholar", "semantic-scholar", "ai2", "s2"):
            workflow = PaperRecommendationWorkflow(
                config=RecommendationConfig(provider=provider),
                client=None,
            )
            client = workflow._default_client()
            self.assertIsInstance(client, SemanticScholarSearchClient)

    def test_workflow_generates_proposals_and_dedupes_existing_title(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="Seed Paper",
                authors=["Alice"],
                year=2022,
                text="Knowledge graph retrieval methods for literature discovery.",
                metadata={"publication": "Journal A"},
            )
        ]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})

        duplicate_candidate = RecommendedPaper(
            paper_id="W-dup",
            title="Seed Paper",
            authors=["Alice"],
            publication="Journal A",
            year=2023,
            abstract="Duplicate",
            citation_count=5,
            source_url="https://example.org/dup",
            doi="",
        )
        new_candidate = RecommendedPaper(
            paper_id="W-new",
            title="Graph Retrieval for Scientific Discovery",
            authors=["Alice", "Bob"],
            publication="Journal B",
            year=2024,
            abstract="Knowledge graph retrieval improves scientific discovery workflows.",
            citation_count=80,
            source_url="https://example.org/new",
            doi="https://doi.org/10.1000/new",
        )

        workflow = PaperRecommendationWorkflow(
            config=RecommendationConfig(top_k=5, min_score=0.05),
            client=_StubClient({"s1": [duplicate_candidate, new_candidate]}),
        )

        proposals = workflow.generate_proposals(
            bibliography=bibliography,
            graph=graph,
            outline=[{"title": "Intro", "sections": ["Overview"]}],
            message_bus=None,
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].title, "Graph Retrieval for Scientific Discovery")
        self.assertIn("id", proposals[0].bibliography_entry)
        self.assertIn("entities", proposals[0].knowledge_graph_update)
        self.assertIn("relations", proposals[0].knowledge_graph_update)

    def test_workflow_sends_messages_and_handles_empty_bibliography(self) -> None:
        workflow = PaperRecommendationWorkflow(config=RecommendationConfig(), client=_StubClient({}))
        bus = MessageBus()

        proposals = workflow.generate_proposals(
            bibliography=[],
            graph=graph_from_dict({"entities": [], "relations": []}),
            outline=[],
            message_bus=bus,
        )

        self.assertEqual(proposals, [])
        self.assertTrue(any(message.receiver == "user" for message in bus.messages))

    def test_workflow_applies_genre_query_terms_to_seed_metadata(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="Seed Paper",
                authors=["Alice"],
                year=2022,
                text="Graph retrieval methods.",
            )
        ]
        graph = graph_from_dict({"entities": [], "relations": []})
        candidate = RecommendedPaper(
            paper_id="W-new",
            title="Benchmarking Retrieval Systems",
            authors=["Bob"],
            publication="Venue",
            year=2024,
            abstract="Baseline and benchmark details.",
            citation_count=5,
            source_url="https://example.org/new",
            doi="",
        )
        client = _RecordingClient([candidate])
        workflow = PaperRecommendationWorkflow(
            config=RecommendationConfig(top_k=3, min_score=0.0),
            client=client,
        )
        workflow.generate_proposals(
            bibliography=bibliography,
            graph=graph,
            outline=[{"title": "Intro", "sections": []}],
            message_bus=None,
            genre_context={"recommendation": {"query_terms": ["benchmark", "ablation"], "keyword_weight": 0.1}},
        )

        self.assertIsNotNone(client.last_seed_source)
        self.assertIn("genre_query_terms", client.last_seed_source.metadata)
        self.assertIn("benchmark", client.last_seed_source.metadata["genre_query_terms"])


if __name__ == "__main__":
    unittest.main()
