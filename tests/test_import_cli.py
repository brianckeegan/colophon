import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from colophon.import_cli import build_parser, main


class NotesImportCLITests(unittest.TestCase):
    def test_help_output_contains_expected_flags(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()

        self.assertIn("Import notes exports into a Colophon knowledge graph.", help_text)
        self.assertIn("--source", help_text)
        self.assertIn("--platform", help_text)
        self.assertIn("--disable-embeddings", help_text)

    def test_main_help_exits_cleanly(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            with self.assertRaises(SystemExit) as exc:
                main(["--help"])
        self.assertEqual(exc.exception.code, 0)

    def test_parser_accepts_notes_import_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--source",
                "examples/notes/obsidian",
                "--platform",
                "obsidian",
                "--output",
                "build/notes_graph.json",
                "--report",
                "build/notes_report.json",
                "--embedding-provider",
                "local",
                "--embedding-top-k",
                "2",
            ]
        )

        self.assertEqual(args.platform, "obsidian")
        self.assertEqual(args.embedding_provider, "local")
        self.assertEqual(args.embedding_top_k, 2)

    def test_main_writes_graph_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            notes_dir = workspace / "notes"
            notes_dir.mkdir(parents=True)
            (notes_dir / "A.md").write_text("# A\nSee [[B]].\n", encoding="utf-8")
            (notes_dir / "B.md").write_text("# B\nLinked back to [A](A.md).\n", encoding="utf-8")

            seed_graph_path = workspace / "seed_graph.json"
            seed_graph_path.write_text(
                json.dumps(
                    {
                        "entities": ["Existing Entity"],
                        "relations": [
                            {
                                "source": "Existing Entity",
                                "predicate": "related_to",
                                "target": "A",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output_path = workspace / "output_graph.json"
            report_path = workspace / "output_report.json"

            exit_code = main(
                [
                    "--source",
                    str(notes_dir),
                    "--platform",
                    "markdown",
                    "--seed-graph",
                    str(seed_graph_path),
                    "--output",
                    str(output_path),
                    "--report",
                    str(report_path),
                    "--disable-embeddings",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertTrue(report_path.exists())

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

            relation_keys = {
                (row["source"], row["predicate"], row["target"])
                for row in payload.get("relations", [])
            }
            self.assertIn(("note:a", "links_to", "note:b"), relation_keys)
            self.assertIn(("Existing Entity", "related_to", "A"), relation_keys)

            self.assertEqual(report["platform"], "markdown")
            self.assertEqual(report["notes_loaded"], 2)
            self.assertGreaterEqual(report["hyperlink_relations_added"], 2)


if __name__ == "__main__":
    unittest.main()
