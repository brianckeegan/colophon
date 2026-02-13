import sqlite3
import tempfile
import unittest
from pathlib import Path

from colophon.io import load_graph


class GraphLoadingTests(unittest.TestCase):
    def test_load_graph_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "graph.json"
            path.write_text(
                (
                    '{"entities":["A"],"relations":[{"source":"A","predicate":"links","target":"B"}],'
                    '"figures":[{"id":"fig-1","caption":"A to B","uri":"figures/a-b.png","related_entities":["A"]}]}'
                ),
                encoding="utf-8",
            )

            graph = load_graph(path)

            self.assertIn("A", graph.entities)
            self.assertIn("B", graph.entities)
            self.assertEqual(len(graph.relations), 1)
            self.assertEqual(graph.relations[0].predicate, "links")
            self.assertIn("fig-1", graph.figures)

    def test_load_graph_from_csv_edgelist_with_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "graph.csv"
            path.write_text("source,target,predicate\nX,Y,causes\n", encoding="utf-8")

            graph = load_graph(path)

            self.assertIn("X", graph.entities)
            self.assertIn("Y", graph.entities)
            self.assertEqual(graph.relations[0].predicate, "causes")

    def test_load_graph_from_csv_without_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "edges.csv"
            path.write_text("M,N\n", encoding="utf-8")

            graph = load_graph(path)

            self.assertIn("M", graph.entities)
            self.assertIn("N", graph.entities)
            self.assertEqual(graph.relations[0].predicate, "related_to")

    def test_load_graph_from_sqlite_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "graph.sqlite"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE TABLE edges (source TEXT, target TEXT, predicate TEXT)"
                )
                connection.execute(
                    "INSERT INTO edges (source, target, predicate) VALUES ('Node1', 'Node2', 'supports')"
                )
                connection.execute(
                    "CREATE TABLE figures (figure_id TEXT, caption TEXT, uri TEXT, related_entities TEXT)"
                )
                connection.execute(
                    "INSERT INTO figures (figure_id, caption, uri, related_entities) VALUES "
                    "('fig-1', 'Node relationship', 'figures/node.png', 'Node1,Node2')"
                )
                connection.commit()
            finally:
                connection.close()

            graph = load_graph(path)

            self.assertIn("Node1", graph.entities)
            self.assertIn("Node2", graph.entities)
            self.assertEqual(graph.relations[0].predicate, "supports")
            self.assertIn("fig-1", graph.figures)

    def test_load_graph_from_sql_dump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "graph.sql"
            path.write_text(
                "\n".join(
                    [
                        "CREATE TABLE edges (source TEXT, target TEXT, relation TEXT);",
                        "INSERT INTO edges (source, target, relation) VALUES ('P', 'Q', 'connects');",
                        "CREATE TABLE figures (id TEXT, caption TEXT, path TEXT);",
                        "INSERT INTO figures (id, caption, path) VALUES ('fig-2', 'P to Q', 'figures/p-q.png');",
                    ]
                ),
                encoding="utf-8",
            )

            graph = load_graph(path)

            self.assertIn("P", graph.entities)
            self.assertIn("Q", graph.entities)
            self.assertEqual(graph.relations[0].predicate, "connects")
            self.assertIn("fig-2", graph.figures)

    def test_load_graph_with_explicit_format_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "edges.txt"
            path.write_text("source,target\nA,B\n", encoding="utf-8")

            graph = load_graph(path, graph_format="csv")

            self.assertIn("A", graph.entities)
            self.assertIn("B", graph.entities)


if __name__ == "__main__":
    unittest.main()
