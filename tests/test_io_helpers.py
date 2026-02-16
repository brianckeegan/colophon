import json
import tempfile
import unittest
from pathlib import Path

from colophon.io import (
    load_bibliography,
    load_bibliography_with_format,
    load_graph,
    load_kg_update_config,
    load_llm_config,
    load_outline,
    load_prompts,
    load_recommendation_config,
    resolve_input_artifacts,
    write_text,
)


class IOHelperTests(unittest.TestCase):
    def test_load_bibliography_and_outline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bibliography_path = Path(tmp_dir) / "bibliography.json"
            outline_path = Path(tmp_dir) / "outline.json"

            bibliography_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "id": "s1",
                                "title": "Paper",
                                "authors": ["A"],
                                "year": 2024,
                                "text": "evidence",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            outline_path.write_text(
                json.dumps({"chapters": [{"title": "Ch", "sections": ["Sec"]}]}),
                encoding="utf-8",
            )

            bibliography = load_bibliography(bibliography_path)
            outline = load_outline(outline_path)

            self.assertEqual(len(bibliography), 1)
            self.assertEqual(bibliography[0].id, "s1")
            self.assertEqual(outline[0]["title"], "Ch")

    def test_load_bibliography_from_json_list_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bibliography_path = Path(tmp_dir) / "bibliography.json"
            bibliography_path.write_text(
                json.dumps(
                    [
                        {
                            "title": "Paper",
                            "authors": "A. Author and B. Writer",
                            "publication": "Journal of Tests",
                            "year": "2024",
                            "abstract": "Summary text",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            bibliography = load_bibliography(bibliography_path)

            self.assertEqual(len(bibliography), 1)
            self.assertEqual(bibliography[0].title, "Paper")
            self.assertEqual(bibliography[0].authors, ["A. Author", "B. Writer"])
            self.assertEqual(bibliography[0].year, 2024)
            self.assertEqual(bibliography[0].metadata.get("publication"), "Journal of Tests")

    def test_load_bibliography_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bibliography_path = Path(tmp_dir) / "bibliography.csv"
            bibliography_path.write_text(
                "\n".join(
                    [
                        "id,title,authors,publication,year,abstract",
                        "src-1,CSV Paper,A. Author;B. Writer,CSV Journal,2023,CSV abstract",
                    ]
                ),
                encoding="utf-8",
            )

            bibliography = load_bibliography(bibliography_path)

            self.assertEqual(len(bibliography), 1)
            self.assertEqual(bibliography[0].id, "src-1")
            self.assertEqual(bibliography[0].title, "CSV Paper")
            self.assertEqual(bibliography[0].authors, ["A. Author", "B. Writer"])
            self.assertEqual(bibliography[0].year, 2023)
            self.assertEqual(bibliography[0].text, "CSV abstract")
            self.assertEqual(bibliography[0].metadata.get("publication"), "CSV Journal")

    def test_load_bibliography_from_bibtex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bibliography_path = Path(tmp_dir) / "references.bib"
            bibliography_path.write_text(
                "\n".join(
                    [
                        "@article{smith2022,",
                        "  title = {A BibTeX Paper},",
                        "  author = {Alice Smith and Bob Jones},",
                        "  journal = {Journal of BibTeX},",
                        "  year = {2022},",
                        "  abstract = {BibTeX abstract text.}",
                        "}",
                    ]
                ),
                encoding="utf-8",
            )

            bibliography = load_bibliography(bibliography_path)

            self.assertEqual(len(bibliography), 1)
            self.assertEqual(bibliography[0].id, "smith2022")
            self.assertEqual(bibliography[0].title, "A BibTeX Paper")
            self.assertEqual(bibliography[0].authors, ["Alice Smith", "Bob Jones"])
            self.assertEqual(bibliography[0].year, 2022)
            self.assertEqual(bibliography[0].metadata.get("publication"), "Journal of BibTeX")

    def test_load_bibliography_with_explicit_format_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bibliography_path = Path(tmp_dir) / "references.txt"
            bibliography_path.write_text(
                "\n".join(
                    [
                        "title,authors,publication,year",
                        "Override Paper,A. Author,Override Journal,2021",
                    ]
                ),
                encoding="utf-8",
            )

            bibliography = load_bibliography_with_format(bibliography_path, bibliography_format="csv")

            self.assertEqual(len(bibliography), 1)
            self.assertEqual(bibliography[0].title, "Override Paper")

    def test_write_text_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "dir" / "out.txt"

            write_text(output_path, "hello")

            self.assertEqual(output_path.read_text(encoding="utf-8"), "hello")

    def test_resolve_input_artifacts_uses_explicit_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            bibliography_path = base / "refs.csv"
            outline_path = base / "outline_payload.json"
            graph_path = base / "kg.sqlite"
            prompts_path = base / "prompt_bundle.json"

            bibliography_path.write_text("id,title\ns1,Paper\n", encoding="utf-8")
            outline_path.write_text(json.dumps({"chapters": []}), encoding="utf-8")
            graph_path.write_text("", encoding="utf-8")
            prompts_path.write_text(json.dumps({"prompts": {}}), encoding="utf-8")

            resolved = resolve_input_artifacts(
                bibliography=str(bibliography_path),
                bibliography_format="auto",
                outline=str(outline_path),
                graph=str(graph_path),
                graph_format="auto",
                prompts=str(prompts_path),
                artifacts_dir="",
                runtime="local",
            )

            self.assertEqual(resolved.bibliography, str(bibliography_path))
            self.assertEqual(resolved.bibliography_format, "csv")
            self.assertEqual(resolved.outline, str(outline_path))
            self.assertEqual(resolved.graph, str(graph_path))
            self.assertEqual(resolved.graph_format, "sqlite")
            self.assertEqual(resolved.prompts, str(prompts_path))

    def test_resolve_input_artifacts_discovers_uploaded_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            uploads = Path(tmp_dir) / "uploads"
            uploads.mkdir(parents=True, exist_ok=True)

            bibliography_path = uploads / "project_bibliography.json"
            outline_path = uploads / "project_outline.json"
            graph_path = uploads / "knowledge_graph.json"
            prompts_path = uploads / "prompts.json"

            bibliography_path.write_text(json.dumps({"sources": []}), encoding="utf-8")
            outline_path.write_text(json.dumps({"chapters": []}), encoding="utf-8")
            graph_path.write_text(json.dumps({"entities": [], "relations": []}), encoding="utf-8")
            prompts_path.write_text(json.dumps({"prompts": {"claim_template": "x"}}), encoding="utf-8")

            resolved = resolve_input_artifacts(
                bibliography="",
                bibliography_format="auto",
                outline="",
                graph="",
                graph_format="auto",
                prompts="",
                artifacts_dir=tmp_dir,
                runtime="codex",
            )

            self.assertEqual(resolved.runtime, "codex")
            self.assertEqual(resolved.artifacts_dir, tmp_dir)
            self.assertEqual(resolved.bibliography, str(bibliography_path))
            self.assertEqual(resolved.outline, str(outline_path))
            self.assertEqual(resolved.graph, str(graph_path))
            self.assertEqual(resolved.prompts, str(prompts_path))

    def test_resolve_input_artifacts_raises_when_required_upload_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            uploads = Path(tmp_dir) / "uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            (uploads / "bibliography.json").write_text(json.dumps({"sources": []}), encoding="utf-8")
            (uploads / "graph.json").write_text(json.dumps({"entities": [], "relations": []}), encoding="utf-8")

            with self.assertRaises(ValueError):
                resolve_input_artifacts(
                    bibliography="",
                    bibliography_format="auto",
                    outline="",
                    graph="",
                    graph_format="auto",
                    prompts="",
                    artifacts_dir=tmp_dir,
                    runtime="claude-code",
                )

    def test_load_graph_auto_raises_for_unknown_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "graph.unknown"
            graph_path.write_text("irrelevant", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_graph(graph_path)

    def test_load_bibliography_auto_raises_for_unknown_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bibliography_path = Path(tmp_dir) / "bibliography.unknown"
            bibliography_path.write_text("irrelevant", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_bibliography(bibliography_path)

    def test_load_prompts_from_wrapped_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts_path = Path(tmp_dir) / "prompts.json"
            prompts_path.write_text(
                json.dumps(
                    {
                        "prompts": {
                            "claim_template": "Claim: {source_title}",
                            "paragraph_template": "{claim_text}",
                            "ignored_non_string": 7,
                        }
                    }
                ),
                encoding="utf-8",
            )

            prompts = load_prompts(prompts_path)

            self.assertIn("claim_template", prompts)
            self.assertIn("paragraph_template", prompts)
            self.assertNotIn("ignored_non_string", prompts)

    def test_load_llm_config_from_wrapped_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            llm_path = Path(tmp_dir) / "llm.json"
            llm_path.write_text(
                json.dumps(
                    {
                        "llm": {
                            "provider": "openai",
                            "model": "gpt-5",
                            "api_base_url": "https://api.openai.com/v1",
                            "api_key_env": "OPENAI_API_KEY",
                            "temperature": 0.1,
                            "max_tokens": 256,
                            "timeout_seconds": 10.0,
                            "system_prompt": "You are a test assistant.",
                            "extra_headers": {"X-Test": "true"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_llm_config(llm_path)

            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.model, "gpt-5")
            self.assertEqual(config.api_key_env, "OPENAI_API_KEY")
            self.assertEqual(config.max_tokens, 256)
            self.assertEqual(config.extra_headers["X-Test"], "true")

    def test_load_llm_config_supports_pi_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            llm_path = Path(tmp_dir) / "llm_pi.json"
            llm_path.write_text(
                json.dumps(
                    {
                        "llm": {
                            "provider": "pi",
                            "model": "anthropic/claude-test",
                            "pi_binary": "pi",
                            "pi_provider": "anthropic",
                            "pi_no_session": False,
                            "pi_coordination_memory": 42,
                            "pi_extra_args": ["--session-dir", "/tmp/pi-session"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_llm_config(llm_path)

            self.assertEqual(config.provider, "pi")
            self.assertEqual(config.pi_binary, "pi")
            self.assertEqual(config.pi_provider, "anthropic")
            self.assertFalse(config.pi_no_session)
            self.assertEqual(config.pi_coordination_memory, 42)
            self.assertEqual(config.pi_extra_args, ["--session-dir", "/tmp/pi-session"])

    def test_load_recommendation_config_from_wrapped_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "recommendation.json"
            config_path.write_text(
                json.dumps(
                    {
                        "recommendation": {
                            "provider": "openalex",
                            "api_base_url": "https://api.openalex.org",
                            "api_key_env": "SEMANTIC_SCHOLAR_API_KEY",
                            "timeout_seconds": 12.0,
                            "per_seed_limit": 4,
                            "top_k": 6,
                            "min_score": 0.3,
                            "mailto": "test@example.com",
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_recommendation_config(config_path)

            self.assertEqual(config.provider, "openalex")
            self.assertEqual(config.timeout_seconds, 12.0)
            self.assertEqual(config.per_seed_limit, 4)
            self.assertEqual(config.top_k, 6)
            self.assertEqual(config.min_score, 0.3)
            self.assertEqual(config.mailto, "test@example.com")
            self.assertEqual(config.api_key_env, "SEMANTIC_SCHOLAR_API_KEY")

    def test_load_kg_update_config_from_wrapped_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "kg_update.json"
            config_path.write_text(
                json.dumps(
                    {
                        "kg_update": {
                            "embedding": {
                                "provider": "openai",
                                "model": "text-embedding-3-small",
                                "api_base_url": "https://api.openai.com/v1",
                                "api_key_env": "OPENAI_API_KEY",
                                "dimensions": 512,
                                "timeout_seconds": 12.0,
                            },
                            "vector_db_path": "build/vectors.json",
                            "rag_top_k": 4,
                            "similarity_threshold": 0.4,
                            "max_entities_per_doc": 7,
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_kg_update_config(config_path)

            self.assertEqual(config.embedding_config.provider, "openai")
            self.assertEqual(config.embedding_config.model, "text-embedding-3-small")
            self.assertEqual(config.embedding_config.api_key_env, "OPENAI_API_KEY")
            self.assertEqual(config.embedding_config.dimensions, 512)
            self.assertEqual(config.vector_db_path, "build/vectors.json")
            self.assertEqual(config.rag_top_k, 4)
            self.assertEqual(config.similarity_threshold, 0.4)
            self.assertEqual(config.max_entities_per_doc, 7)


if __name__ == "__main__":
    unittest.main()
