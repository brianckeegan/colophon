"""Command-line interface for running Colophon manuscript generation workflows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .io import (
    load_bibliography_with_format,
    load_graph,
    load_json,
    load_kg_update_config,
    load_llm_config,
    load_outline,
    load_prompts,
    load_recommendation_config,
    write_text,
)
from .kg_update import KGUpdateConfig
from .graph import graph_to_dict
from .agents import OutlineExpanderAgent
from .llm import LLMConfig, create_llm_client
from .models import Manuscript
from .ontology import (
    normalize_functional_forms_catalog,
    normalize_genre_ontology,
    normalize_writing_companion_ontology,
)
from .pipeline import ColophonPipeline, PipelineConfig
from .recommendations import RecommendationConfig
from .vectors import EmbeddingConfig


def build_parser() -> argparse.ArgumentParser:
    """Build parser.

    Returns
    -------
    argparse.ArgumentParser
        Return value description.
    """
    parser = argparse.ArgumentParser(description="Generate long-form drafts with Colophon.")
    parser.add_argument("--bibliography", required=True, help="Path to bibliography input file.")
    parser.add_argument(
        "--bibliography-format",
        default="auto",
        choices=("auto", "json", "csv", "bibtex"),
        help="Bibliography input format. Defaults to auto-detect from extension.",
    )
    parser.add_argument("--outline", required=True, help="Path to outline JSON file.")
    parser.add_argument(
        "--prompts",
        help="Optional prompts JSON file for template overrides (claim/paragraph/empty section).",
    )
    parser.add_argument("--llm-config", help="Optional LLM config JSON file.")
    parser.add_argument(
        "--llm-provider",
        choices=("none", "openai", "codex", "openai_compatible", "anthropic", "claude", "github", "copilot"),
        help="LLM provider alias. If omitted, deterministic templates are used.",
    )
    parser.add_argument("--llm-model", help="LLM model name for API calls.")
    parser.add_argument("--llm-api-base-url", help="Override API base URL.")
    parser.add_argument("--llm-api-key-env", help="Environment variable containing API key/token.")
    parser.add_argument("--llm-system-prompt", help="System prompt applied to LLM calls.")
    parser.add_argument("--llm-temperature", type=float, help="Sampling temperature for LLM calls.")
    parser.add_argument("--llm-max-tokens", type=int, help="Max tokens for LLM calls.")
    parser.add_argument("--llm-timeout-seconds", type=float, help="HTTP timeout for each LLM API request.")
    parser.add_argument(
        "--graph",
        required=True,
        help="Path to seed knowledge graph input (JSON, CSV edgelist, SQLite DB, or SQL dump).",
    )
    parser.add_argument(
        "--graph-format",
        default="auto",
        choices=("auto", "json", "csv", "sqlite", "sql"),
        help="Graph input format. Defaults to auto-detect from file extension.",
    )
    parser.add_argument("--output", required=True, help="Output manuscript path.")
    parser.add_argument(
        "--output-format",
        default="auto",
        choices=("auto", "text", "markdown", "rst", "rtf", "latex"),
        help="Output format. Defaults to auto-detect from output extension.",
    )
    parser.add_argument(
        "--output-layout",
        default="single",
        choices=("single", "monolithic", "project"),
        help="Output layout: one monolithic file or a project folder with chapter-level files.",
    )
    parser.add_argument("--report", default="build/diagnostics.json", help="Output diagnostics report path.")
    parser.add_argument("--title", default="Colophon Draft", help="Manuscript title.")
    parser.add_argument("--narrative-tone", default="neutral", help="Narrative tone (e.g., formal, conversational).")
    parser.add_argument(
        "--narrative-style",
        default="analytical",
        help="Narrative style (e.g., technical, persuasive, expository).",
    )
    parser.add_argument(
        "--narrative-audience",
        default="general",
        help="Target audience (e.g., policymakers, graduate students, clinicians).",
    )
    parser.add_argument(
        "--narrative-discipline",
        default="interdisciplinary",
        help="Primary disciplinary framing for language and argument conventions.",
    )
    parser.add_argument(
        "--narrative-genre",
        default="scholarly_manuscript",
        help="Narrative genre (for example: scholarly_manuscript, technical_report, policy_brief).",
    )
    parser.add_argument(
        "--narrative-language",
        default="English",
        help="Target output language (applied directly by LLM-backed generation).",
    )
    parser.add_argument(
        "--genre-ontology",
        default="",
        help=(
            "Path to genre ontology JSON defining defaults/prompts for audience, discipline, "
            "style, genre, and language."
        ),
    )
    parser.add_argument(
        "--genre-profile-id",
        default="",
        help="Optional profile id to select from the genre ontology.",
    )
    parser.add_argument(
        "--enable-soft-validation",
        action="store_true",
        help="Run soft structural/rhetorical/genre validation using functional forms config.",
    )
    parser.add_argument(
        "--functional-forms",
        default="",
        help="Path to functional forms JSON profile (for coordination, outline expansion, and soft validation).",
    )
    parser.add_argument(
        "--functional-form-id",
        default="",
        help="Optional functional form id to select within the functional forms catalog.",
    )
    parser.add_argument(
        "--max-soft-validation-findings",
        type=int,
        default=64,
        help="Maximum number of soft-validation findings retained in diagnostics.",
    )
    parser.add_argument(
        "--functional-validation-report",
        default="",
        help="Optional path for standalone soft-validation report JSON.",
    )
    parser.add_argument(
        "--writing-ontology",
        default="",
        help=(
            "Path to companion writing ontology JSON (background prompts, assumptions, and validations) "
            "that runs alongside functional forms."
        ),
    )
    parser.add_argument(
        "--max-writing-ontology-findings",
        type=int,
        default=32,
        help="Maximum number of writing-ontology validation findings retained in diagnostics.",
    )
    parser.add_argument(
        "--writing-ontology-report",
        default="",
        help="Optional path for standalone writing-ontology validation report JSON.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Retriever top-k evidence sources per section.")
    parser.add_argument(
        "--max-figures-per-section",
        type=int,
        default=2,
        help="Maximum number of graph figure nodes attached to each section.",
    )
    parser.add_argument(
        "--disable-coordination-agents",
        action="store_true",
        help="Disable paragraph/section/chapter/book coordination and message-passing agents.",
    )
    parser.add_argument(
        "--coordination-max-iterations",
        type=int,
        default=4,
        help="Maximum iterative revision passes for coordinator/editor convergence.",
    )
    parser.add_argument(
        "--enable-paper-recommendations",
        action="store_true",
        help="Enable external related-paper recommendation proposals for bibliography and KG updates.",
    )
    parser.add_argument(
        "--recommendation-config",
        default="",
        help="Optional recommendation config JSON file.",
    )
    parser.add_argument(
        "--recommendation-provider",
        choices=("openalex", "scholar_search", "scholar", "semantic_scholar", "semantic-scholar", "ai2", "s2"),
        default=None,
        help="Scientometric API provider for paper recommendations.",
    )
    parser.add_argument(
        "--recommendation-api-base-url",
        default=None,
        help="Base URL for recommendation provider API.",
    )
    parser.add_argument(
        "--recommendation-api-key-env",
        default=None,
        help="Environment variable containing recommendation API key (used by providers like Semantic Scholar).",
    )
    parser.add_argument(
        "--recommendation-timeout-seconds",
        type=float,
        default=None,
        help="Timeout for recommendation API calls.",
    )
    parser.add_argument(
        "--recommendation-per-seed",
        type=int,
        default=None,
        help="Max API candidates fetched per seed bibliography entry.",
    )
    parser.add_argument(
        "--recommendation-top-k",
        type=int,
        default=None,
        help="Max recommendation proposals kept after scoring/deduplication.",
    )
    parser.add_argument(
        "--recommendation-min-score",
        type=float,
        default=None,
        help="Minimum recommendation score to keep a proposal.",
    )
    parser.add_argument(
        "--recommendation-mailto",
        default=None,
        help="Optional contact email for OpenAlex polite pool requests.",
    )
    parser.add_argument(
        "--recommendation-report",
        default="",
        help="Optional output path for recommendation proposals JSON.",
    )
    parser.add_argument(
        "--enable-kg-updates",
        action="store_true",
        help="Enable bibliography-driven KG generation/update with embeddings + vector similarity.",
    )
    parser.add_argument(
        "--kg-update-config",
        default="",
        help="Optional KG updater config JSON.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("local", "hash", "offline", "openai", "openai_compatible", "remote"),
        default=None,
        help="Embedding provider alias for KG updater.",
    )
    parser.add_argument("--embedding-model", default=None, help="Embedding model for remote provider.")
    parser.add_argument("--embedding-api-base-url", default=None, help="Embedding API base URL.")
    parser.add_argument("--embedding-api-key-env", default=None, help="Environment variable containing embedding API key.")
    parser.add_argument("--embedding-dimensions", type=int, default=None, help="Embedding vector dimensionality.")
    parser.add_argument(
        "--embedding-timeout-seconds",
        type=float,
        default=None,
        help="Embedding API timeout.",
    )
    parser.add_argument("--kg-vector-db-path", default=None, help="Optional path to write KG updater vector DB JSON.")
    parser.add_argument("--kg-rag-top-k", type=int, default=None, help="Nearest-neighbor count for KG updater RAG context.")
    parser.add_argument(
        "--kg-similarity-threshold",
        type=float,
        default=None,
        help="Minimum similarity score for linking papers in KG updater.",
    )
    parser.add_argument(
        "--kg-max-entities-per-doc",
        type=int,
        default=None,
        help="Maximum extracted entity candidates per bibliography document.",
    )
    parser.add_argument(
        "--updated-graph",
        default="",
        help="Optional output path for revised knowledge graph JSON after KG updater runs.",
    )
    parser.add_argument(
        "--enable-outline-expander",
        action="store_true",
        help="Expand a preliminary outline into a detailed outline plus prompt bundle before drafting.",
    )
    parser.add_argument(
        "--outline-max-subsections",
        type=int,
        default=3,
        help="Maximum subsection titles generated per section by the outline expander.",
    )
    parser.add_argument(
        "--outline-expansion-report",
        default="",
        help="Optional output path for full outline expansion payload JSON.",
    )
    parser.add_argument(
        "--expanded-outline-output",
        default="",
        help="Optional output path for expanded outline JSON (chapters only).",
    )
    parser.add_argument(
        "--expanded-prompts-output",
        default="",
        help="Optional output path for expander-generated prompt templates JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main.

    Parameters
    ----------
    argv : list[str] | None
        Parameter description.

    Returns
    -------
    int
        Return value description.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    bibliography = load_bibliography_with_format(args.bibliography, bibliography_format=args.bibliography_format)
    outline = load_outline(args.outline)
    graph = load_graph(args.graph, graph_format=args.graph_format)
    prompts = load_prompts(args.prompts) if args.prompts else {}
    llm_config = _resolve_llm_config(args)
    llm_client = create_llm_client(llm_config)
    recommendation_config = _resolve_recommendation_config(args)
    kg_update_config = _resolve_kg_update_config(args)
    functional_forms_payload = None
    functional_forms_path = args.functional_forms.strip()
    needs_functional_forms = args.enable_soft_validation or args.enable_outline_expander or not args.disable_coordination_agents
    if functional_forms_path:
        functional_forms_payload = load_json(functional_forms_path)
    elif needs_functional_forms:
        for default_forms_path in (Path("ontology/functional_forms.json"), Path("examples/functional_forms.json")):
            if default_forms_path.exists():
                functional_forms_payload = load_json(default_forms_path)
                break
    if functional_forms_payload is not None:
        functional_forms_payload = normalize_functional_forms_catalog(functional_forms_payload)
    writing_ontology_payload = None
    writing_ontology_path = args.writing_ontology.strip()
    if writing_ontology_path:
        writing_ontology_payload = load_json(writing_ontology_path)
        writing_ontology_payload = normalize_writing_companion_ontology(writing_ontology_payload)
    genre_ontology_payload = None
    genre_ontology_path = args.genre_ontology.strip()
    if genre_ontology_path:
        genre_ontology_payload = load_json(genre_ontology_path)
        genre_ontology_payload = normalize_genre_ontology(genre_ontology_payload)
    else:
        for default_genre_ontology_path in (Path("ontology/genre_ontology.json"), Path("examples/genre_ontology.json")):
            if default_genre_ontology_path.exists():
                genre_ontology_payload = load_json(default_genre_ontology_path)
                genre_ontology_payload = normalize_genre_ontology(genre_ontology_payload)
                break

    soft_validation_enabled = args.enable_soft_validation

    pipeline = ColophonPipeline(
        config=PipelineConfig(
            top_k=args.top_k,
            max_figures_per_section=args.max_figures_per_section,
            title=args.title,
            prompt_templates=prompts,
            llm_client=llm_client,
            llm_system_prompt=llm_config.system_prompt,
            narrative_tone=args.narrative_tone.strip() or "neutral",
            narrative_style=args.narrative_style.strip() or "analytical",
            narrative_audience=args.narrative_audience.strip() or "general",
            narrative_discipline=args.narrative_discipline.strip() or "interdisciplinary",
            narrative_genre=args.narrative_genre.strip() or "scholarly_manuscript",
            narrative_language=args.narrative_language.strip() or "English",
            genre_ontology=genre_ontology_payload,
            genre_profile_id=args.genre_profile_id.strip(),
            enable_coordination_agents=not args.disable_coordination_agents,
            coordination_max_revision_iterations=max(1, args.coordination_max_iterations),
            enable_paper_recommendations=args.enable_paper_recommendations,
            recommendation_config=recommendation_config,
            enable_kg_updates=args.enable_kg_updates,
            kg_update_config=kg_update_config,
            enable_outline_expander=args.enable_outline_expander,
            outline_expander=OutlineExpanderAgent(
                max_subsections_per_section=max(1, args.outline_max_subsections),
                llm_client=llm_client,
                llm_system_prompt=llm_config.system_prompt or None,
                tone=args.narrative_tone.strip() or "neutral",
                style=args.narrative_style.strip() or "analytical",
                audience=args.narrative_audience.strip() or "general",
                discipline=args.narrative_discipline.strip() or "interdisciplinary",
                genre=args.narrative_genre.strip() or "scholarly_manuscript",
                language=args.narrative_language.strip() or "English",
            ),
            enable_soft_validation=soft_validation_enabled,
            functional_forms=functional_forms_payload,
            functional_form_id=args.functional_form_id.strip(),
            max_soft_validation_findings=max(1, args.max_soft_validation_findings),
            writing_ontology=writing_ontology_payload,
            max_writing_ontology_findings=max(1, args.max_writing_ontology_findings),
        )
    )
    manuscript = pipeline.run(bibliography=bibliography, outline=outline, graph=graph)

    output_format = _resolve_output_format(args.output_format, args.output)
    output_layout = _resolve_output_layout(args.output_layout)
    _write_manuscript_output(
        output_path=args.output,
        manuscript=manuscript,
        output_format=output_format,
        output_layout=output_layout,
    )
    write_text(args.report, json.dumps(manuscript.diagnostics, indent=2) + "\n")
    if args.recommendation_report:
        proposals = manuscript.diagnostics.get("recommendation_proposals", [])
        write_text(args.recommendation_report, json.dumps(proposals, indent=2) + "\n")
    if args.updated_graph:
        write_text(args.updated_graph, json.dumps(graph_to_dict(graph), indent=2) + "\n")
    if args.outline_expansion_report:
        write_text(
            args.outline_expansion_report,
            json.dumps(manuscript.diagnostics.get("outline_expansion_result", {}), indent=2) + "\n",
        )
    if args.expanded_outline_output:
        expanded = manuscript.diagnostics.get("outline_expansion_result", {})
        if isinstance(expanded, dict):
            write_text(args.expanded_outline_output, json.dumps({"chapters": expanded.get("chapters", [])}, indent=2) + "\n")
    if args.expanded_prompts_output:
        expanded = manuscript.diagnostics.get("outline_expansion_result", {})
        if isinstance(expanded, dict):
            write_text(args.expanded_prompts_output, json.dumps({"prompts": expanded.get("prompts", {})}, indent=2) + "\n")
    if args.functional_validation_report:
        write_text(
            args.functional_validation_report,
            json.dumps(manuscript.diagnostics.get("soft_validation_result", {}), indent=2) + "\n",
        )
    if args.writing_ontology_report:
        write_text(
            args.writing_ontology_report,
            json.dumps(manuscript.diagnostics.get("writing_ontology_validation_result", {}), indent=2) + "\n",
        )

    if manuscript.diagnostics["citation_issues"]:
        print("Completed with citation issues; see diagnostics report.", file=sys.stderr)
    else:
        print("Generation complete.")
    return 0


def _resolve_output_format(output_format: str, output_path: str) -> str:
    """Resolve output format.

    Parameters
    ----------
    output_format : str
        Parameter description.
    output_path : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    normalized = output_format.strip().lower()
    if normalized != "auto":
        return normalized

    suffix = Path(output_path).suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix == ".txt":
        return "text"
    if suffix == ".rst":
        return "rst"
    if suffix == ".rtf":
        return "rtf"
    if suffix in {".tex", ".latex"}:
        return "latex"
    return "markdown"


def _resolve_output_layout(layout: str) -> str:
    """Resolve output layout.

    Parameters
    ----------
    layout : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    normalized = layout.strip().lower()
    if normalized in {"single", "monolithic"}:
        return "single"
    if normalized == "project":
        return "project"
    raise ValueError(f"Unsupported output layout: {layout}")


def _write_manuscript_output(
    output_path: str,
    manuscript: Manuscript,
    output_format: str,
    output_layout: str,
) -> None:
    """Write manuscript output.

    Parameters
    ----------
    output_path : str
        Parameter description.
    manuscript : Manuscript
        Parameter description.
    output_format : str
        Parameter description.
    output_layout : str
        Parameter description.
    """
    if output_layout == "single":
        write_text(output_path, manuscript.render(output_format))
        return

    project_dir = Path(output_path)
    project_dir.mkdir(parents=True, exist_ok=True)
    extension = _output_extension(output_format)
    index_file = f"index.{extension}"
    write_text(project_dir / index_file, manuscript.render(output_format))

    chapter_entries: list[dict[str, str]] = []
    for index, chapter in enumerate(manuscript.chapters, start=1):
        slug = _slugify(chapter.title) or f"chapter-{index:02d}"
        filename = f"chapter-{index:02d}-{slug}.{extension}"
        chapter_manuscript = Manuscript(
            title=f"{manuscript.title}: {chapter.title}",
            chapters=[chapter],
        )
        write_text(project_dir / filename, chapter_manuscript.render(output_format))
        chapter_entries.append({"id": chapter.id, "title": chapter.title, "file": filename})

    manifest = {
        "title": manuscript.title,
        "output_format": output_format,
        "layout": "project",
        "index_file": index_file,
        "chapters": chapter_entries,
    }
    write_text(project_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def _output_extension(output_format: str) -> str:
    """Output extension.

    Parameters
    ----------
    output_format : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    normalized = output_format.strip().lower()
    if normalized == "text":
        return "txt"
    if normalized == "markdown":
        return "md"
    if normalized == "rst":
        return "rst"
    if normalized == "rtf":
        return "rtf"
    if normalized == "latex":
        return "tex"
    raise ValueError(f"Unsupported output format: {output_format}")


def _slugify(value: str) -> str:
    """Slugify.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    collapsed = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return collapsed


def _resolve_llm_config(args: argparse.Namespace) -> LLMConfig:
    """Resolve llm config.

    Parameters
    ----------
    args : argparse.Namespace
        Parameter description.

    Returns
    -------
    LLMConfig
        Return value description.
    """
    base = load_llm_config(args.llm_config) if args.llm_config else LLMConfig()

    provider = args.llm_provider or base.provider
    model = args.llm_model if args.llm_model is not None else base.model
    api_base_url = args.llm_api_base_url if args.llm_api_base_url is not None else base.api_base_url
    api_key_env = args.llm_api_key_env if args.llm_api_key_env is not None else base.api_key_env
    system_prompt = args.llm_system_prompt if args.llm_system_prompt is not None else base.system_prompt
    temperature = args.llm_temperature if args.llm_temperature is not None else base.temperature
    max_tokens = args.llm_max_tokens if args.llm_max_tokens is not None else base.max_tokens
    timeout_seconds = args.llm_timeout_seconds if args.llm_timeout_seconds is not None else base.timeout_seconds

    return LLMConfig(
        provider=provider,
        model=model,
        api_base_url=api_base_url,
        api_key_env=api_key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        system_prompt=system_prompt,
        extra_headers=base.extra_headers,
    )


def _resolve_recommendation_config(args: argparse.Namespace) -> RecommendationConfig:
    """Resolve recommendation config.

    Parameters
    ----------
    args : argparse.Namespace
        Parameter description.

    Returns
    -------
    RecommendationConfig
        Return value description.
    """
    base = load_recommendation_config(args.recommendation_config) if args.recommendation_config else RecommendationConfig()

    return RecommendationConfig(
        provider=args.recommendation_provider if args.recommendation_provider is not None else base.provider,
        api_base_url=(
            args.recommendation_api_base_url if args.recommendation_api_base_url is not None else base.api_base_url
        ),
        timeout_seconds=(
            args.recommendation_timeout_seconds
            if args.recommendation_timeout_seconds is not None
            else base.timeout_seconds
        ),
        per_seed_limit=max(
            1,
            args.recommendation_per_seed if args.recommendation_per_seed is not None else base.per_seed_limit,
        ),
        top_k=max(0, args.recommendation_top_k if args.recommendation_top_k is not None else base.top_k),
        min_score=max(
            0.0,
            min(
                1.0,
                args.recommendation_min_score if args.recommendation_min_score is not None else base.min_score,
            ),
        ),
        mailto=(args.recommendation_mailto if args.recommendation_mailto is not None else base.mailto).strip(),
        api_key_env=(args.recommendation_api_key_env if args.recommendation_api_key_env is not None else base.api_key_env).strip(),
    )


def _resolve_kg_update_config(args: argparse.Namespace) -> KGUpdateConfig:
    """Resolve kg update config.

    Parameters
    ----------
    args : argparse.Namespace
        Parameter description.

    Returns
    -------
    KGUpdateConfig
        Return value description.
    """
    base = load_kg_update_config(args.kg_update_config) if args.kg_update_config else KGUpdateConfig()

    embedding_base = base.embedding_config
    embedding_config = EmbeddingConfig(
        provider=args.embedding_provider if args.embedding_provider is not None else embedding_base.provider,
        model=args.embedding_model if args.embedding_model is not None else embedding_base.model,
        api_base_url=args.embedding_api_base_url if args.embedding_api_base_url is not None else embedding_base.api_base_url,
        api_key_env=args.embedding_api_key_env if args.embedding_api_key_env is not None else embedding_base.api_key_env,
        dimensions=max(8, args.embedding_dimensions if args.embedding_dimensions is not None else embedding_base.dimensions),
        timeout_seconds=max(
            1.0,
            args.embedding_timeout_seconds
            if args.embedding_timeout_seconds is not None
            else embedding_base.timeout_seconds,
        ),
    )

    return KGUpdateConfig(
        embedding_config=embedding_config,
        vector_db_path=args.kg_vector_db_path if args.kg_vector_db_path is not None else base.vector_db_path,
        rag_top_k=max(0, args.kg_rag_top_k if args.kg_rag_top_k is not None else base.rag_top_k),
        similarity_threshold=max(
            0.0,
            min(
                1.0,
                args.kg_similarity_threshold if args.kg_similarity_threshold is not None else base.similarity_threshold,
            ),
        ),
        max_entities_per_doc=max(
            1,
            args.kg_max_entities_per_doc if args.kg_max_entities_per_doc is not None else base.max_entities_per_doc,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
