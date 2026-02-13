"""Command-line interface for importing notes exports into knowledge graphs."""

from __future__ import annotations

import argparse
import json
import sys

from .graph import KnowledgeGraph, graph_to_dict
from .io import load_graph, write_text
from .note_import import NotesImportConfig, NotesKnowledgeGraphImporter
from .vectors import EmbeddingConfig


def build_parser() -> argparse.ArgumentParser:
    """Build parser.

    Returns
    -------
    argparse.ArgumentParser
        Return value description.
    """
    parser = argparse.ArgumentParser(description="Import notes exports into a Colophon knowledge graph.")
    parser.add_argument("--source", required=True, help="Path to note export source (vault/folder/file).")
    parser.add_argument(
        "--platform",
        default="auto",
        choices=("auto", "obsidian", "notion", "onenote", "evernote", "markdown"),
        help="Source platform. Defaults to auto-detection.",
    )
    parser.add_argument("--output", required=True, help="Output graph JSON path.")
    parser.add_argument("--report", default="build/notes_import_report.json", help="Importer report JSON path.")
    parser.add_argument("--seed-graph", default="", help="Optional existing graph file to merge into.")
    parser.add_argument(
        "--seed-graph-format",
        default="auto",
        choices=("auto", "json", "csv", "sqlite", "sql"),
        help="Seed graph format.",
    )
    parser.add_argument("--disable-hyperlinks", action="store_true", help="Disable hyperlink-based graph linking.")
    parser.add_argument("--disable-embeddings", action="store_true", help="Disable embedding-based similarity links.")
    parser.add_argument(
        "--embedding-provider",
        choices=("local", "hash", "offline", "openai", "openai_compatible", "remote"),
        default="local",
        help="Embedding provider alias.",
    )
    parser.add_argument("--embedding-model", default="", help="Embedding model for remote providers.")
    parser.add_argument("--embedding-api-base-url", default="", help="Embedding API base URL.")
    parser.add_argument("--embedding-api-key-env", default="", help="Embedding API key environment variable.")
    parser.add_argument("--embedding-dimensions", type=int, default=256, help="Embedding vector dimensionality.")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=20.0, help="Embedding API timeout.")
    parser.add_argument("--embedding-top-k", type=int, default=3, help="Nearest neighbors per note.")
    parser.add_argument(
        "--embedding-similarity-threshold",
        type=float,
        default=0.35,
        help="Minimum cosine similarity for similarity links.",
    )
    parser.add_argument("--vector-db-path", default="", help="Optional vector DB output path.")
    parser.add_argument(
        "--disable-external-urls",
        action="store_true",
        help="Ignore external URL links when building graph relations.",
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

    graph = load_graph(args.seed_graph, graph_format=args.seed_graph_format) if args.seed_graph else KnowledgeGraph()

    config = NotesImportConfig(
        platform=args.platform,
        use_hyperlinks=not args.disable_hyperlinks,
        use_embeddings=not args.disable_embeddings,
        embedding_config=EmbeddingConfig(
            provider=args.embedding_provider,
            model=args.embedding_model,
            api_base_url=args.embedding_api_base_url or None,
            api_key_env=args.embedding_api_key_env or None,
            dimensions=max(8, args.embedding_dimensions),
            timeout_seconds=max(1.0, args.embedding_timeout_seconds),
        ),
        embedding_top_k=max(0, args.embedding_top_k),
        embedding_similarity_threshold=max(0.0, min(1.0, args.embedding_similarity_threshold)),
        vector_db_path=args.vector_db_path,
        include_external_urls=not args.disable_external_urls,
    )

    importer = NotesKnowledgeGraphImporter(config=config)
    graph, result = importer.run(source_path=args.source, graph=graph)

    write_text(args.output, json.dumps(graph_to_dict(graph), indent=2) + "\n")
    write_text(args.report, json.dumps(result.to_dict(), indent=2) + "\n")

    print(
        (
            f"Imported {result.notes_loaded} notes from {result.platform}; "
            f"added {result.entities_added} entities and {result.relations_added} relations."
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
