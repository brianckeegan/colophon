import json
import tempfile
import unittest
from pathlib import Path

from colophon.note_import import NotesImportConfig, NotesKnowledgeGraphImporter
from colophon.vectors import EmbeddingConfig


class NotesImportTests(unittest.TestCase):
    def test_obsidian_hyperlink_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault = Path(tmp_dir) / "vault"
            (vault / ".obsidian").mkdir(parents=True)
            (vault / "A.md").write_text(
                "# A\nSee [[B]] and [Website](https://example.org).\n",
                encoding="utf-8",
            )
            (vault / "B.md").write_text("# B\nBacklink to [A](A.md).\n", encoding="utf-8")

            importer = NotesKnowledgeGraphImporter(
                config=NotesImportConfig(
                    platform="obsidian",
                    use_hyperlinks=True,
                    use_embeddings=False,
                )
            )
            graph, result = importer.run(vault)

        self.assertEqual(result.notes_loaded, 2)
        self.assertGreaterEqual(result.hyperlink_relations_added, 2)
        self.assertGreaterEqual(result.url_relations_added, 1)
        self.assertTrue(any(rel.predicate == "links_to" for rel in graph.relations))
        self.assertTrue(any(rel.predicate == "references_url" for rel in graph.relations))

    def test_embedding_similarity_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            notes_dir = Path(tmp_dir) / "notes"
            notes_dir.mkdir(parents=True)
            (notes_dir / "one.md").write_text("# One\nShared embedding content.\n", encoding="utf-8")
            (notes_dir / "two.md").write_text("# Two\nShared embedding content.\n", encoding="utf-8")
            vector_path = Path(tmp_dir) / "vectors.json"

            importer = NotesKnowledgeGraphImporter(
                config=NotesImportConfig(
                    platform="markdown",
                    use_hyperlinks=False,
                    use_embeddings=True,
                    embedding_config=EmbeddingConfig(provider="local", dimensions=64),
                    embedding_top_k=1,
                    embedding_similarity_threshold=0.1,
                    vector_db_path=str(vector_path),
                )
            )
            graph, result = importer.run(notes_dir)

            self.assertTrue(vector_path.exists())

        self.assertEqual(result.notes_loaded, 2)
        self.assertEqual(result.vector_records, 2)
        self.assertGreaterEqual(result.embedding_relations_added, 1)
        self.assertTrue(any(rel.predicate == "similar_to" for rel in graph.relations))

    def test_onenote_and_evernote_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            onenote_path = Path(tmp_dir) / "onenote.json"
            onenote_path.write_text(
                json.dumps(
                    {
                        "notes": [
                            {
                                "id": "note-1",
                                "title": "OneNote A",
                                "content": "See [OneNote B](note-2).",
                                "links": ["https://example.com/reference"],
                            },
                            {
                                "id": "note-2",
                                "title": "OneNote B",
                                "content": "Supporting details.",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            evernote_path = Path(tmp_dir) / "evernote.enex"
            evernote_path.write_text(
                "\n".join(
                    [
                        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                        "<en-export>",
                        "  <note>",
                        "    <title>Evernote Note</title>",
                        "    <content><![CDATA[",
                        "      <en-note>External <a href=\"https://example.org/evernote\">link</a>.</en-note>",
                        "    ]]></content>",
                        "  </note>",
                        "</en-export>",
                    ]
                ),
                encoding="utf-8",
            )

            onenote_importer = NotesKnowledgeGraphImporter(
                config=NotesImportConfig(platform="onenote", use_hyperlinks=True, use_embeddings=False)
            )
            _, onenote_result = onenote_importer.run(onenote_path)

            evernote_importer = NotesKnowledgeGraphImporter(
                config=NotesImportConfig(platform="evernote", use_hyperlinks=True, use_embeddings=False)
            )
            _, evernote_result = evernote_importer.run(evernote_path)

        self.assertEqual(onenote_result.notes_loaded, 2)
        self.assertGreaterEqual(onenote_result.hyperlink_relations_added, 1)
        self.assertGreaterEqual(onenote_result.url_relations_added, 1)
        self.assertEqual(evernote_result.notes_loaded, 1)
        self.assertGreaterEqual(evernote_result.url_relations_added, 1)

    def test_notion_json_import_with_internal_and_external_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            notion_path = Path(tmp_dir) / "notion_export.json"
            notion_path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                "title": "Research Plan aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                "content": (
                                    "See [Methods](Methods%20bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.md) "
                                    "and https://www.notion.so/workspace/ref."
                                ),
                            },
                            {
                                "id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                                "title": "Methods bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                                "content": "Method details",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            importer = NotesKnowledgeGraphImporter(
                config=NotesImportConfig(platform="notion", use_hyperlinks=True, use_embeddings=False)
            )
            graph, result = importer.run(notion_path)

        self.assertEqual(result.notes_loaded, 2)
        self.assertGreaterEqual(result.hyperlink_relations_added, 1)
        self.assertGreaterEqual(result.url_relations_added, 1)
        self.assertTrue(any(rel.predicate == "links_to" for rel in graph.relations))
        self.assertTrue(any(rel.predicate == "references_url" for rel in graph.relations))

    def test_notion_csv_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            notion_csv = Path(tmp_dir) / "notion.csv"
            notion_csv.write_text(
                "\n".join(
                    [
                        "id,title,content,url",
                        (
                            "11111111111111111111111111111111,Plan,"
                            "\"See [Methods](Methods%2022222222222222222222222222222222.md)\","
                            "https://www.notion.so/workspace/plan"
                        ),
                        (
                            "22222222222222222222222222222222,Methods,"
                            "\"Method details\",https://www.notion.so/workspace/methods"
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            importer = NotesKnowledgeGraphImporter(
                config=NotesImportConfig(platform="notion", use_hyperlinks=True, use_embeddings=False)
            )
            graph, result = importer.run(notion_csv)

        self.assertEqual(result.notes_loaded, 2)
        self.assertGreaterEqual(result.hyperlink_relations_added, 1)
        self.assertGreaterEqual(result.url_relations_added, 2)
        self.assertTrue(any(rel.predicate == "links_to" for rel in graph.relations))

    def test_onenote_protocol_url_is_treated_as_external_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            onenote_path = Path(tmp_dir) / "onenote.json"
            onenote_path.write_text(
                json.dumps(
                    {
                        "notes": [
                            {
                                "id": "note-1",
                                "title": "OneNote Root",
                                "content": "See onenote:https://d.docs.live.net/abc/page",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            importer = NotesKnowledgeGraphImporter(
                config=NotesImportConfig(platform="onenote", use_hyperlinks=True, use_embeddings=False)
            )
            _, result = importer.run(onenote_path)

        self.assertEqual(result.notes_loaded, 1)
        self.assertGreaterEqual(result.url_relations_added, 1)


if __name__ == "__main__":
    unittest.main()
