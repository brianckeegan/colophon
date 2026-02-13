import tempfile
import unittest
from pathlib import Path

from colophon.coordination import MessageBus
from colophon.graph import graph_from_dict
from colophon.kg_update import KGUpdateConfig, KnowledgeGraphGeneratorUpdater
from colophon.models import Source
from colophon.vectors import EmbeddingConfig


class KGUpdateTests(unittest.TestCase):
    def test_updater_indexes_embeddings_and_adds_graph_links(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="Knowledge Graph Grounding",
                authors=["Alice"],
                year=2023,
                text="Knowledge graph retrieval supports grounding and coherence.",
                metadata={"publication": "Journal A"},
            ),
            Source(
                id="s2",
                title="Grounded Retrieval Workflows",
                authors=["Bob"],
                year=2024,
                text="Grounded retrieval systems improve coherence in long-form writing.",
                metadata={"publication": "Journal B"},
            ),
        ]
        graph = graph_from_dict({"entities": ["Knowledge Graph"], "relations": []})
        bus = MessageBus()

        with tempfile.TemporaryDirectory() as tmp_dir:
            vector_path = Path(tmp_dir) / "vectors.json"
            updater = KnowledgeGraphGeneratorUpdater(
                config=KGUpdateConfig(
                    embedding_config=EmbeddingConfig(provider="local", dimensions=64),
                    vector_db_path=str(vector_path),
                    rag_top_k=2,
                    similarity_threshold=0.0,
                    max_entities_per_doc=5,
                )
            )
            result = updater.run(bibliography=bibliography, graph=graph, message_bus=bus)

            self.assertTrue(vector_path.exists())

        self.assertEqual(result.embeddings_indexed, 2)
        self.assertEqual(result.vector_records, 2)
        self.assertGreater(result.entities_added, 0)
        self.assertGreater(result.relations_added, 0)
        self.assertIn("paper:s1", graph.entities)
        self.assertIn("paper:s2", graph.entities)
        self.assertTrue(any(relation.predicate == "discusses" for relation in graph.relations))
        self.assertTrue(any(message.message_type == "kg_update_summary" for message in bus.messages))

    def test_updater_reports_missing_abstracts_and_is_idempotent(self) -> None:
        bibliography = [
            Source(
                id="s1",
                title="Paper One",
                authors=["Alice"],
                year=2023,
                text="A study on graph-aware writing.",
                metadata={"publication": "Journal A"},
            ),
            Source(
                id="s2",
                title="Paper Two",
                authors=["Bob"],
                year=2024,
                text="",
                metadata={"publication": "Journal B"},
            ),
        ]
        graph = graph_from_dict({"entities": [], "relations": []})
        bus = MessageBus()
        updater = KnowledgeGraphGeneratorUpdater(
            config=KGUpdateConfig(
                embedding_config=EmbeddingConfig(provider="local", dimensions=32),
                rag_top_k=1,
                similarity_threshold=0.0,
                max_entities_per_doc=3,
            )
        )

        first = updater.run(bibliography=bibliography, graph=graph, message_bus=bus)
        second = updater.run(bibliography=bibliography, graph=graph, message_bus=bus)

        self.assertIn("s2", first.missing_abstract_source_ids)
        self.assertGreater(first.relations_added, 0)
        self.assertEqual(second.entities_added, 0)
        self.assertEqual(second.relations_added, 0)
        self.assertTrue(any(message.message_type == "kg_update_gap" for message in bus.messages))


if __name__ == "__main__":
    unittest.main()
