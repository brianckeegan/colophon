"""Bibliography-driven knowledge-graph generation and update routines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .coordination import MessageBus
from .graph import KnowledgeGraph
from .models import Source
from .vectors import EmbeddingClient, EmbeddingConfig, InMemoryVectorDB, VectorRecord, create_embedding_client

_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "been",
    "between",
    "both",
    "from",
    "into",
    "more",
    "most",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "using",
    "with",
    "which",
}


@dataclass(slots=True)
class KGUpdateConfig:
    """Configuration for embedding-backed bibliography-to-KG updates.

    Parameters
    ----------
    embedding_config : EmbeddingConfig
        Embedding provider/model settings used for document vectors.
    vector_db_path : str
        Optional output path for persisted vector records.
    rag_top_k : int
        Number of nearest-neighbor documents used for local RAG context.
    similarity_threshold : float
        Minimum cosine similarity for creating ``similar_to`` links.
    max_entities_per_doc : int
        Maximum entity phrases extracted per bibliography document.
    """

    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db_path: str = ""
    rag_top_k: int = 3
    similarity_threshold: float = 0.2
    max_entities_per_doc: int = 8


@dataclass(slots=True)
class KGUpdateResult:
    """Summary counts and diagnostics produced by a KG update run.

    Parameters
    ----------
    embeddings_indexed : int
        Number of bibliography records embedded.
    vector_records : int
        Number of vector records stored in memory/output.
    entities_added : int
        Number of new entities inserted into the KG.
    relations_added : int
        Number of new relations inserted into the KG.
    similar_links_added : int
        Number of ``similar_to`` links added from embedding neighbors.
    missing_abstract_source_ids : list[str]
        Source ids missing usable abstract/text content.
    """

    embeddings_indexed: int = 0
    vector_records: int = 0
    entities_added: int = 0
    relations_added: int = 0
    similar_links_added: int = 0
    missing_abstract_source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """To dict.

        Returns
        -------
        dict
            Return value description.
        """
        return {
            "embeddings_indexed": self.embeddings_indexed,
            "vector_records": self.vector_records,
            "entities_added": self.entities_added,
            "relations_added": self.relations_added,
            "similar_links_added": self.similar_links_added,
            "missing_abstract_source_ids": self.missing_abstract_source_ids,
        }


@dataclass(slots=True)
class KnowledgeGraphGeneratorUpdater:
    """Generate and revise KG structure from bibliography metadata and text.

    Parameters
    ----------
    config : KGUpdateConfig
        Runtime update policy and thresholds.
    embedding_client : EmbeddingClient | None
        Optional injected embedding client; default is provider-resolved.
    """

    config: KGUpdateConfig = field(default_factory=KGUpdateConfig)
    embedding_client: EmbeddingClient | None = None

    def run(
        self,
        bibliography: list[Source],
        graph: KnowledgeGraph,
        message_bus: MessageBus | None = None,
    ) -> KGUpdateResult:
        """Run.

        Parameters
        ----------
        bibliography : list[Source]
            Parameter description.
        graph : KnowledgeGraph
            Parameter description.
        message_bus : MessageBus | None
            Parameter description.

        Returns
        -------
        KGUpdateResult
            Return value description.
        """
        if not bibliography:
            if message_bus is not None:
                message_bus.send(
                    sender="kg_updater",
                    receiver="user",
                    message_type="kg_update_skipped",
                    content="Knowledge graph updater skipped because bibliography is empty.",
                    priority="high",
                )
            return KGUpdateResult()

        existing_entities = set(graph.entities)
        initial_relation_count = len(graph.relations)
        existing_relations = {(relation.source, relation.predicate, relation.target) for relation in graph.relations}

        records = self._embed_sources(bibliography)
        vector_db = InMemoryVectorDB()
        vector_db.add_many(records)

        if self.config.vector_db_path.strip():
            vector_db.save_json(self.config.vector_db_path)

        missing_abstract = [source.id for source in bibliography if not source.text.strip()]

        for source in bibliography:
            paper_node = f"paper:{source.id}"
            graph.entities.add(paper_node)
            if source.title:
                graph.entities.add(source.title)
                self._add_relation_if_new(graph, existing_relations, paper_node, "has_title", source.title)

            for author in source.authors:
                author_name = author.strip()
                if not author_name:
                    continue
                graph.entities.add(author_name)
                self._add_relation_if_new(graph, existing_relations, paper_node, "authored_by", author_name)

            publication = _source_publication(source)
            if publication:
                graph.entities.add(publication)
                self._add_relation_if_new(graph, existing_relations, paper_node, "published_in", publication)

        similar_links_added = 0
        for source in bibliography:
            source_record = next((record for record in records if record.record_id == source.id), None)
            if source_record is None:
                continue

            paper_node = f"paper:{source.id}"

            neighbors = vector_db.search(
                query_vector=source_record.vector,
                top_k=max(0, self.config.rag_top_k),
                exclude_ids={source.id},
            )

            rag_context = [source_record.text]
            for neighbor_record, score in neighbors:
                if score < self.config.similarity_threshold:
                    continue
                rag_context.append(neighbor_record.text)
                if self._add_relation_if_new(
                    graph,
                    existing_relations,
                    paper_node,
                    "similar_to",
                    f"paper:{neighbor_record.record_id}",
                ):
                    similar_links_added += 1

            entities = _extract_entities_from_context(
                text="\n".join(rag_context),
                max_entities=max(1, self.config.max_entities_per_doc),
            )
            for entity in entities:
                graph.entities.add(entity)
                self._add_relation_if_new(graph, existing_relations, paper_node, "discusses", entity)

            for match in graph.entities_for_query(f"{source.title} {source.text}")[:3]:
                if match != paper_node:
                    self._add_relation_if_new(graph, existing_relations, paper_node, "grounded_by", match)

        result = KGUpdateResult(
            embeddings_indexed=len(records),
            vector_records=len(vector_db.records),
            entities_added=max(0, len(graph.entities) - len(existing_entities)),
            relations_added=max(0, len(graph.relations) - initial_relation_count),
            similar_links_added=similar_links_added,
            missing_abstract_source_ids=missing_abstract,
        )

        if message_bus is not None:
            message_bus.send(
                sender="kg_updater",
                receiver="book_coordinator",
                message_type="kg_update_summary",
                content=(
                    f"KG updater indexed {result.embeddings_indexed} documents and added "
                    f"{result.entities_added} entities / {result.relations_added} relations."
                ),
            )
            if result.missing_abstract_source_ids:
                message_bus.send(
                    sender="kg_updater",
                    receiver="user",
                    message_type="kg_update_gap",
                    content=(
                        "Some bibliography entries are missing abstract/text fields: "
                        + ", ".join(result.missing_abstract_source_ids)
                    ),
                    priority="high",
                )

        return result

    def _embed_sources(self, bibliography: list[Source]) -> list[VectorRecord]:
        """Embed sources.

        Parameters
        ----------
        bibliography : list[Source]
            Parameter description.

        Returns
        -------
        list[VectorRecord]
            Return value description.
        """
        client = self.embedding_client or create_embedding_client(self.config.embedding_config)

        texts = [_source_profile_text(source) for source in bibliography]
        vectors = client.embed(texts)
        if len(vectors) != len(bibliography):
            raise ValueError("Embedding client returned an unexpected number of vectors.")

        records: list[VectorRecord] = []
        for source, text, vector in zip(bibliography, texts, vectors, strict=True):
            records.append(
                VectorRecord(
                    record_id=source.id,
                    text=text,
                    metadata={
                        "title": source.title,
                        "authors": source.authors,
                        "publication": _source_publication(source),
                    },
                    vector=vector,
                )
            )
        return records

    @staticmethod
    def _add_relation_if_new(
        graph: KnowledgeGraph,
        relation_index: set[tuple[str, str, str]],
        source: str,
        predicate: str,
        target: str,
    ) -> bool:
        """Add relation if new.

        Parameters
        ----------
        graph : KnowledgeGraph
            Parameter description.
        relation_index : set[tuple[str, str, str]]
            Parameter description.
        source : str
            Parameter description.
        predicate : str
            Parameter description.
        target : str
            Parameter description.

        Returns
        -------
        bool
            Return value description.
        """
        key = (source, predicate, target)
        if key in relation_index:
            return False
        graph.add_relation(source=source, predicate=predicate, target=target)
        relation_index.add(key)
        return True


def _source_profile_text(source: Source) -> str:
    """Source profile text.

    Parameters
    ----------
    source : Source
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    publication = _source_publication(source)
    authors = ", ".join(source.authors)
    parts = [
        f"title: {source.title}",
        f"authors: {authors}",
        f"publication: {publication}",
        f"abstract: {source.text}",
    ]
    return "\n".join(part for part in parts if part.strip())


def _source_publication(source: Source) -> str:
    """Source publication.

    Parameters
    ----------
    source : Source
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    if not isinstance(source.metadata, dict):
        return ""
    return str(
        source.metadata.get("publication")
        or source.metadata.get("journal")
        or source.metadata.get("venue")
        or source.metadata.get("publisher")
        or ""
    ).strip()


def _extract_entities_from_context(text: str, max_entities: int) -> list[str]:
    """Extract entities from context.

    Parameters
    ----------
    text : str
        Parameter description.
    max_entities : int
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())]
    frequencies: dict[str, int] = {}
    for token in tokens:
        if token in _STOPWORDS:
            continue
        frequencies[token] = frequencies.get(token, 0) + 1

    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    selected = [token.title() for token, _ in ranked[: max_entities]]
    return selected
