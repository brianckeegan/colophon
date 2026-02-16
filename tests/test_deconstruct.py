import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colophon.deconstruct import (
    build_bibliography,
    build_knowledge_graph,
    build_outline,
    build_reverse_prompts,
    extract_reference_entries,
    preprocess_pdf_text,
    run_deconstruct,
    split_reference_section,
)


class DeconstructTests(unittest.TestCase):
    def test_split_reference_section(self) -> None:
        text = "Intro text\n\nReferences\n[1] Example citation"
        body, refs = split_reference_section(text)
        self.assertIn("Intro text", body)
        self.assertIn("Example citation", refs)

    def test_extract_reference_entries(self) -> None:
        refs = "References\n[1] First citation.\n[2] Second citation."
        entries = extract_reference_entries(refs)
        self.assertEqual(len(entries), 2)

    def test_build_knowledge_graph_links_claims_to_refs(self) -> None:
        bibliography = [{"id": "ref-001", "title": "Paper A"}]
        graph = build_knowledge_graph(
            "Knowledge graph methods improve drafting quality in this sufficiently long claim sentence [1].",
            bibliography,
        )
        self.assertTrue(graph["nodes"])
        self.assertTrue(graph["edges"])
        self.assertTrue(any(edge.get("relation") == "supported_by" for edge in graph["edges"]))
        self.assertTrue(any("provenance" in edge for edge in graph["edges"]))
        self.assertTrue(any(node.get("type") == "claim" for node in graph["nodes"]))
        self.assertTrue(any(node.get("type") == "reference" for node in graph["nodes"]))

    def test_build_knowledge_graph_edges_are_deterministic(self) -> None:
        bibliography = [{"id": "ref-001", "title": "Paper A"}]
        text = "The knowledge graph supports bibliography alignment and improves synthesis quality for researchers."
        graph_first = build_knowledge_graph(text, bibliography)
        graph_second = build_knowledge_graph(text, bibliography)

        self.assertEqual(graph_first["nodes"], graph_second["nodes"])
        self.assertEqual(graph_first["edges"], graph_second["edges"])

    @patch("colophon.deconstruct._extract_pdf_text_with_pymupdf")
    @patch("colophon.deconstruct._extract_pdf_text_with_kreuzberg")
    def test_preprocess_pdf_text_prefers_kreuzberg(self, mock_kreuzberg, mock_pymupdf) -> None:
        mock_kreuzberg.return_value = "Line one.\n\nLine two."
        text = preprocess_pdf_text("fake.pdf")
        self.assertEqual(text, "Line one.\n\nLine two.")
        mock_pymupdf.assert_not_called()

    @patch("colophon.deconstruct._extract_pdf_text_with_pymupdf")
    @patch("colophon.deconstruct._extract_pdf_text_with_kreuzberg")
    def test_preprocess_pdf_text_falls_back_to_pymupdf(self, mock_kreuzberg, mock_pymupdf) -> None:
        mock_kreuzberg.side_effect = RuntimeError("kreuzberg unavailable")
        mock_pymupdf.return_value = "Fallback content."
        text = preprocess_pdf_text("fake.pdf")
        self.assertEqual(text, "Fallback content.")

    @patch("colophon.deconstruct._build_knowledge_graph_with_sift")
    def test_build_knowledge_graph_uses_fallback_when_sift_fails(self, mock_sift_build) -> None:
        mock_sift_build.side_effect = RuntimeError("sift missing")
        graph = build_knowledge_graph("A sufficiently long sentence to be treated as a claim [1].", [{"id": "ref-001"}])
        self.assertEqual(graph["metadata"]["backend"], "fallback")
        self.assertIn("fallback_reason", graph["metadata"])

    def test_build_outline_and_prompts(self) -> None:
        outline = build_outline("Paper title\n\nIntro paragraph.\n\nSecond paragraph.", "fallback")
        prompts = build_reverse_prompts(outline, {"nodes": []}, [{"title": "Source A"}])
        self.assertIn("chapters", outline)
        self.assertIn("prompts", prompts)

    @patch("colophon.deconstruct.build_bibliography")
    @patch("colophon.deconstruct.preprocess_pdf_text")
    def test_run_deconstruct_writes_all_files(self, mock_preprocess, mock_bibliography) -> None:
        mock_preprocess.return_value = "Title\n\nBody sentence [1].\n\nReferences\n[1] Ref one."
        mock_bibliography.return_value = [{"id": "ref-001", "title": "Ref one"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")
            artifacts = run_deconstruct(pdf_path, output_dir=tmp_dir)

            for output in [
                artifacts.bibliography_path,
                artifacts.knowledge_graph_path,
                artifacts.outline_path,
                artifacts.prompts_path,
            ]:
                self.assertTrue(output.exists())
                self.assertTrue(json.loads(output.read_text(encoding="utf-8")))

    @patch("colophon.deconstruct._lookup_openalex")
    def test_build_bibliography(self, mock_lookup) -> None:
        mock_lookup.return_value = {"doi": "10.1234/test"}
        bibliography = build_bibliography(["Smith, J.. Sample title. Journal. 2020."])
        self.assertEqual(len(bibliography), 1)
        self.assertEqual(bibliography[0]["doi"], "10.1234/test")

if __name__ == "__main__":
    unittest.main()
