import unittest

from colophon.graph import KnowledgeGraph, graph_from_dict, graph_to_dict
from colophon.models import Figure


class GraphTests(unittest.TestCase):
    def test_add_relation_tracks_entities_and_neighbors(self) -> None:
        graph = KnowledgeGraph()

        graph.add_relation(source="A", predicate="links", target="B")

        self.assertIn("A", graph.entities)
        self.assertIn("B", graph.entities)
        self.assertEqual(graph.neighbors("A"), {"B"})
        self.assertEqual(graph.neighbors("unknown"), set())

    def test_entities_for_query_prefers_larger_token_overlap(self) -> None:
        graph = KnowledgeGraph(entities={"Knowledge Graph", "Graph Database", "Retrieval"})

        entities = graph.entities_for_query("knowledge graph methods")

        self.assertEqual(entities[0], "Knowledge Graph")
        self.assertIn("Graph Database", entities)

    def test_graph_from_dict_uses_default_predicate(self) -> None:
        graph = graph_from_dict(
            {
                "relations": [{"source": "Node1", "target": "Node2"}],
                "entities": ["Standalone"],
            }
        )

        self.assertEqual(len(graph.relations), 1)
        self.assertEqual(graph.relations[0].predicate, "related_to")
        self.assertIn("Standalone", graph.entities)

    def test_figures_for_query_returns_matching_figures(self) -> None:
        graph = KnowledgeGraph()
        graph.add_figure(Figure(id="fig-1", caption="Knowledge Graph Pipeline", uri="figures/pipeline.png"))
        graph.add_figure(Figure(id="fig-2", caption="Retrieval Diagram", uri="figures/retrieval.png"))

        matches = graph.figures_for_query("knowledge graph", max_items=2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].id, "fig-1")

    def test_graph_from_dict_loads_figures(self) -> None:
        graph = graph_from_dict(
            {
                "entities": [],
                "relations": [],
                "figures": [
                    {
                        "id": "fig-1",
                        "caption": "System Design",
                        "uri": "figures/system.png",
                        "related_entities": ["Knowledge Graph"],
                    }
                ],
            }
        )

        self.assertIn("fig-1", graph.figures)
        self.assertEqual(graph.figures["fig-1"].caption, "System Design")
        self.assertIn("Knowledge Graph", graph.entities)

    def test_graph_to_dict_serializes_relations_and_figures(self) -> None:
        graph = KnowledgeGraph()
        graph.add_relation(source="A", predicate="links", target="B")
        graph.add_figure(
            Figure(
                id="fig-1",
                caption="A links to B",
                uri="figures/a-b.png",
                related_entities=["A", "B"],
                metadata={"kind": "diagram"},
            )
        )

        payload = graph_to_dict(graph)

        self.assertIn("entities", payload)
        self.assertIn("relations", payload)
        self.assertIn("figures", payload)
        self.assertEqual(payload["relations"][0]["predicate"], "links")
        self.assertEqual(payload["figures"][0]["id"], "fig-1")


if __name__ == "__main__":
    unittest.main()
