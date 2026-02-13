import json
import tempfile
import unittest
import unittest.mock
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from colophon.cli import (
    _resolve_kg_update_config,
    _resolve_llm_config,
    _resolve_output_layout,
    _resolve_output_format,
    _resolve_recommendation_config,
    _write_manuscript_output,
    build_parser,
)
from colophon.models import Chapter, Manuscript, Paragraph, Section


class CLITests(unittest.TestCase):
    def test_parser_help_contains_key_sections(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()

        self.assertIn("Generate long-form drafts with Colophon.", help_text)
        self.assertIn("--bibliography", help_text)
        self.assertIn("--output-format", help_text)
        self.assertIn("--enable-paper-recommendations", help_text)
        self.assertIn("--coordination-max-iterations", help_text)

    def test_main_help_exits_cleanly(self) -> None:
        from colophon.cli import main

        buffer = StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            with self.assertRaises(SystemExit) as exc:
                main(["--help"])
        self.assertEqual(exc.exception.code, 0)

    def test_resolve_output_format_auto_from_extension(self) -> None:
        self.assertEqual(_resolve_output_format("auto", "out.md"), "markdown")
        self.assertEqual(_resolve_output_format("auto", "out.txt"), "text")
        self.assertEqual(_resolve_output_format("auto", "out.rst"), "rst")
        self.assertEqual(_resolve_output_format("auto", "out.rtf"), "rtf")
        self.assertEqual(_resolve_output_format("auto", "out.tex"), "latex")

    def test_resolve_output_format_auto_default_to_markdown(self) -> None:
        self.assertEqual(_resolve_output_format("auto", "out.unknown"), "markdown")

    def test_resolve_output_format_explicit(self) -> None:
        self.assertEqual(_resolve_output_format("latex", "out.md"), "latex")

    def test_resolve_output_layout(self) -> None:
        self.assertEqual(_resolve_output_layout("single"), "single")
        self.assertEqual(_resolve_output_layout("monolithic"), "single")
        self.assertEqual(_resolve_output_layout("project"), "project")

    def test_parser_accepts_outline_expander_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--bibliography",
                "examples/bibliography.json",
                "--outline",
                "examples/outline.json",
                "--graph",
                "examples/seed_graph.json",
                "--output",
                "build/out.md",
                "--enable-outline-expander",
                "--outline-max-subsections",
                "4",
                "--outline-expansion-report",
                "build/outline_expansion.json",
                "--expanded-outline-output",
                "build/expanded_outline.json",
                "--expanded-prompts-output",
                "build/expanded_prompts.json",
            ]
        )

        self.assertTrue(args.enable_outline_expander)
        self.assertEqual(args.outline_max_subsections, 4)

    def test_parser_accepts_upload_runtime_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--artifacts-dir",
                "uploads",
                "--runtime",
                "claude-code",
                "--output",
                "build/out.md",
            ]
        )

        self.assertEqual(args.artifacts_dir, "uploads")
        self.assertEqual(args.runtime, "claude-code")

    def test_parser_accepts_user_guidance_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--artifacts-dir",
                "uploads",
                "--output",
                "build/out.md",
                "--request-user-guidance",
                "--user-guidance-stages",
                "planning,recommendations,outline,coordination",
                "--guidance-output",
                "build/guidance.json",
            ]
        )

        self.assertTrue(args.request_user_guidance)
        self.assertEqual(args.user_guidance_stages, "planning,recommendations,outline,coordination")
        self.assertEqual(args.guidance_output, "build/guidance.json")

    def test_parser_accepts_soft_validation_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--bibliography",
                "examples/bibliography.json",
                "--outline",
                "examples/outline.json",
                "--graph",
                "examples/seed_graph.json",
                "--output",
                "build/out.md",
                "--enable-soft-validation",
                "--functional-forms",
                "ontology/functional_forms.json",
                "--functional-form-id",
                "sequential_transformation",
                "--functional-validation-report",
                "build/functional_validation.json",
            ]
        )

        self.assertTrue(args.enable_soft_validation)
        self.assertEqual(args.functional_forms, "ontology/functional_forms.json")
        self.assertEqual(args.functional_form_id, "sequential_transformation")

    def test_parser_accepts_writing_ontology_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--bibliography",
                "examples/bibliography.json",
                "--outline",
                "examples/outline.json",
                "--graph",
                "examples/seed_graph.json",
                "--output",
                "build/out.md",
                "--writing-ontology",
                "ontology/wilson_academic_writing_ontology.json",
                "--max-writing-ontology-findings",
                "24",
                "--writing-ontology-report",
                "build/writing_ontology_validation.json",
            ]
        )

        self.assertEqual(args.writing_ontology, "ontology/wilson_academic_writing_ontology.json")
        self.assertEqual(args.max_writing_ontology_findings, 24)
        self.assertEqual(args.writing_ontology_report, "build/writing_ontology_validation.json")

    def test_parser_accepts_genre_ontology_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--bibliography",
                "examples/bibliography.json",
                "--outline",
                "examples/outline.json",
                "--graph",
                "examples/seed_graph.json",
                "--output",
                "build/out.md",
                "--genre-ontology",
                "ontology/genre_ontology.json",
                "--genre-profile-id",
                "technical_research",
            ]
        )

        self.assertEqual(args.genre_ontology, "ontology/genre_ontology.json")
        self.assertEqual(args.genre_profile_id, "technical_research")

    def test_parser_accepts_narrative_customization_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--bibliography",
                "examples/bibliography.json",
                "--outline",
                "examples/outline.json",
                "--graph",
                "examples/seed_graph.json",
                "--output",
                "build/out.md",
                "--narrative-tone",
                "formal",
                "--narrative-style",
                "persuasive",
                "--narrative-audience",
                "policy analysts",
                "--narrative-discipline",
                "public policy",
                "--narrative-genre",
                "policy_brief",
                "--narrative-language",
                "Spanish",
            ]
        )

        self.assertEqual(args.narrative_tone, "formal")
        self.assertEqual(args.narrative_style, "persuasive")
        self.assertEqual(args.narrative_audience, "policy analysts")
        self.assertEqual(args.narrative_discipline, "public policy")
        self.assertEqual(args.narrative_genre, "policy_brief")
        self.assertEqual(args.narrative_language, "Spanish")

    def test_parser_accepts_coordination_iteration_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--bibliography",
                "examples/bibliography.json",
                "--outline",
                "examples/outline.json",
                "--graph",
                "examples/seed_graph.json",
                "--output",
                "build/out.md",
                "--coordination-max-iterations",
                "6",
            ]
        )

        self.assertEqual(args.coordination_max_iterations, 6)


    def test_main_dispatches_deconstruct_command(self) -> None:
        from colophon.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")
            with unittest.mock.patch("colophon.cli.run_deconstruct") as mock_run:
                mock_run.return_value = unittest.mock.Mock(
                    bibliography_path=Path(tmp_dir) / "b.json",
                    knowledge_graph_path=Path(tmp_dir) / "k.json",
                    outline_path=Path(tmp_dir) / "o.json",
                    prompts_path=Path(tmp_dir) / "p.json",
                )
                code = main(["deconstruct", str(pdf_path)])

        self.assertEqual(code, 0)
        mock_run.assert_called_once()

    def test_write_manuscript_output_single_layout(self) -> None:
        manuscript = Manuscript(
            title="Single Output",
            chapters=[
                Chapter(
                    id="ch1",
                    title="Intro",
                    sections=[Section(id="s1", title="Setup", paragraphs=[Paragraph(id="p1", text="Body")])],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "single.md"
            _write_manuscript_output(
                output_path=str(output_path),
                manuscript=manuscript,
                output_format="markdown",
                output_layout="single",
            )

            self.assertTrue(output_path.exists())
            self.assertIn("# Single Output", output_path.read_text(encoding="utf-8"))

    def test_write_manuscript_output_project_layout(self) -> None:
        manuscript = Manuscript(
            title="Project Output",
            chapters=[
                Chapter(
                    id="ch1",
                    title="Intro",
                    sections=[Section(id="s1", title="Setup", paragraphs=[Paragraph(id="p1", text="Body")])],
                ),
                Chapter(
                    id="ch2",
                    title="Methods",
                    sections=[Section(id="s2", title="Approach", paragraphs=[Paragraph(id="p2", text="Body")])],
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir) / "project_output"
            _write_manuscript_output(
                output_path=str(project_dir),
                manuscript=manuscript,
                output_format="markdown",
                output_layout="project",
            )

            index_path = project_dir / "index.md"
            manifest_path = project_dir / "manifest.json"
            chapter_files = sorted(project_dir.glob("chapter-*.md"))

            self.assertTrue(index_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(len(chapter_files), 2)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["layout"], "project")
            self.assertEqual(manifest["output_format"], "markdown")
            self.assertEqual(len(manifest["chapters"]), 2)

    def test_resolve_llm_config_prefers_cli_over_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "llm.json"
            config_path.write_text(
                json.dumps(
                    {
                        "llm": {
                            "provider": "openai",
                            "model": "gpt-file",
                            "api_key_env": "OPENAI_API_KEY",
                        }
                    }
                ),
                encoding="utf-8",
            )

            args = Namespace(
                llm_config=str(config_path),
                llm_provider="anthropic",
                llm_model="claude-cli",
                llm_api_base_url=None,
                llm_api_key_env=None,
                llm_system_prompt="system",
                llm_temperature=0.3,
                llm_max_tokens=333,
                llm_timeout_seconds=5.0,
            )
            config = _resolve_llm_config(args)

            self.assertEqual(config.provider, "anthropic")
            self.assertEqual(config.model, "claude-cli")
            self.assertEqual(config.system_prompt, "system")
            self.assertEqual(config.max_tokens, 333)

    def test_resolve_recommendation_config(self) -> None:
        args = Namespace(
            recommendation_config="",
            recommendation_provider="openalex",
            recommendation_api_base_url="https://api.openalex.org",
            recommendation_api_key_env="SEMANTIC_SCHOLAR_API_KEY",
            recommendation_timeout_seconds=15.0,
            recommendation_per_seed=6,
            recommendation_top_k=10,
            recommendation_min_score=0.35,
            recommendation_mailto="user@example.com",
        )
        config = _resolve_recommendation_config(args)

        self.assertEqual(config.provider, "openalex")
        self.assertEqual(config.per_seed_limit, 6)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.min_score, 0.35)
        self.assertEqual(config.mailto, "user@example.com")
        self.assertEqual(config.api_key_env, "SEMANTIC_SCHOLAR_API_KEY")

    def test_resolve_recommendation_config_prefers_file_when_flags_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "recommendation.json"
            config_path.write_text(
                json.dumps(
                    {
                        "recommendation": {
                            "provider": "openalex",
                            "api_base_url": "https://api.openalex.org",
                            "timeout_seconds": 11.0,
                            "per_seed_limit": 3,
                            "top_k": 4,
                            "min_score": 0.4,
                            "mailto": "x@example.com",
                        }
                    }
                ),
                encoding="utf-8",
            )

            args = Namespace(
                recommendation_config=str(config_path),
                recommendation_provider=None,
                recommendation_api_base_url=None,
                recommendation_api_key_env=None,
                recommendation_timeout_seconds=None,
                recommendation_per_seed=None,
                recommendation_top_k=None,
                recommendation_min_score=None,
                recommendation_mailto=None,
            )
            config = _resolve_recommendation_config(args)

            self.assertEqual(config.per_seed_limit, 3)
            self.assertEqual(config.top_k, 4)
            self.assertEqual(config.min_score, 0.4)
            self.assertEqual(config.mailto, "x@example.com")

    def test_resolve_kg_update_config(self) -> None:
        args = Namespace(
            kg_update_config="",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_api_base_url="https://api.openai.com/v1",
            embedding_api_key_env="OPENAI_API_KEY",
            embedding_dimensions=512,
            embedding_timeout_seconds=12.0,
            kg_vector_db_path="build/vectors.json",
            kg_rag_top_k=4,
            kg_similarity_threshold=0.45,
            kg_max_entities_per_doc=6,
        )
        config = _resolve_kg_update_config(args)

        self.assertEqual(config.embedding_config.provider, "openai")
        self.assertEqual(config.embedding_config.model, "text-embedding-3-small")
        self.assertEqual(config.embedding_config.dimensions, 512)
        self.assertEqual(config.vector_db_path, "build/vectors.json")
        self.assertEqual(config.rag_top_k, 4)
        self.assertEqual(config.similarity_threshold, 0.45)
        self.assertEqual(config.max_entities_per_doc, 6)

    def test_resolve_kg_update_config_prefers_file_when_flags_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "kg_update.json"
            config_path.write_text(
                json.dumps(
                    {
                        "kg_update": {
                            "embedding": {
                                "provider": "local",
                                "dimensions": 128,
                                "timeout_seconds": 8.0,
                            },
                            "vector_db_path": "build/file_vectors.json",
                            "rag_top_k": 2,
                            "similarity_threshold": 0.33,
                            "max_entities_per_doc": 5,
                        }
                    }
                ),
                encoding="utf-8",
            )

            args = Namespace(
                kg_update_config=str(config_path),
                embedding_provider=None,
                embedding_model=None,
                embedding_api_base_url=None,
                embedding_api_key_env=None,
                embedding_dimensions=None,
                embedding_timeout_seconds=None,
                kg_vector_db_path=None,
                kg_rag_top_k=None,
                kg_similarity_threshold=None,
                kg_max_entities_per_doc=None,
            )
            config = _resolve_kg_update_config(args)

            self.assertEqual(config.embedding_config.provider, "local")
            self.assertEqual(config.embedding_config.dimensions, 128)
            self.assertEqual(config.vector_db_path, "build/file_vectors.json")
            self.assertEqual(config.rag_top_k, 2)
            self.assertEqual(config.similarity_threshold, 0.33)
            self.assertEqual(config.max_entities_per_doc, 5)


if __name__ == "__main__":
    unittest.main()
