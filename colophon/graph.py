"""Knowledge graph primitives and query helpers used by drafting agents."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .models import Figure


@dataclass(slots=True)
class Relation:
    """Directed semantic edge between two entities in the knowledge graph.

    Parameters
    ----------
    source : str
        Source entity identifier.
    predicate : str
        Relation label between source and target.
    target : str
        Target entity identifier.
    """

    source: str
    predicate: str
    target: str


@dataclass(slots=True)
class KnowledgeGraph:
    """In-memory knowledge graph with entities, relations, and figure nodes.

    Parameters
    ----------
    entities : set[str]
        Distinct entity labels currently present in the graph.
    relations : list[Relation]
        Directed semantic edges linking entities.
    adjacency : dict[str, set[str]]
        Source-to-neighbors index for fast lookup.
    figures : dict[str, Figure]
        Figure registry keyed by figure id.
    """

    entities: set[str] = field(default_factory=set)
    relations: list[Relation] = field(default_factory=list)
    adjacency: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    figures: dict[str, Figure] = field(default_factory=dict)

    def add_relation(self, source: str, predicate: str, target: str) -> None:
        """Add relation.

        Parameters
        ----------
        source : str
            Parameter description.
        predicate : str
            Parameter description.
        target : str
            Parameter description.
        """
        self.entities.update({source, target})
        self.relations.append(Relation(source=source, predicate=predicate, target=target))
        self.adjacency[source].add(target)

    def neighbors(self, entity: str) -> set[str]:
        """Neighbors.

        Parameters
        ----------
        entity : str
            Parameter description.

        Returns
        -------
        set[str]
            Return value description.
        """
        return set(self.adjacency.get(entity, set()))

    def add_figure(self, figure: Figure) -> None:
        """Add figure.

        Parameters
        ----------
        figure : Figure
            Parameter description.
        """
        self.figures[figure.id] = figure
        self.entities.update(figure.related_entities)

    def entities_for_query(self, query: str) -> list[str]:
        """Entities for query.

        Parameters
        ----------
        query : str
            Parameter description.

        Returns
        -------
        list[str]
            Return value description.
        """
        tokens = _tokenize(query)
        matches = []
        for entity in sorted(self.entities):
            entity_tokens = _tokenize(entity)
            overlap = len(tokens.intersection(entity_tokens))
            if overlap:
                matches.append((overlap, entity))
        matches.sort(reverse=True)
        return [entity for _, entity in matches]

    def figures_for_query(self, query: str, max_items: int = 3) -> list[Figure]:
        """Figures for query.

        Parameters
        ----------
        query : str
            Parameter description.
        max_items : int
            Parameter description.

        Returns
        -------
        list[Figure]
            Return value description.
        """
        tokens = _tokenize(query)
        matches: list[tuple[int, str]] = []
        for figure in self.figures.values():
            figure_tokens = _figure_tokens(figure)
            overlap = len(tokens.intersection(figure_tokens))
            if overlap:
                matches.append((overlap, figure.id))
        matches.sort(reverse=True)
        return [self.figures[figure_id] for _, figure_id in matches[:max_items]]


# Light tokenizer keeps dependencies minimal and deterministic.
def _tokenize(value: str) -> set[str]:
    """Tokenize.

    Parameters
    ----------
    value : str
        Parameter description.

    Returns
    -------
    set[str]
        Return value description.
    """
    return {chunk.strip(".,;:!?()[]{}\"'\n\t ").lower() for chunk in value.split() if chunk.strip()}


def _figure_tokens(figure: Figure) -> set[str]:
    """Figure tokens.

    Parameters
    ----------
    figure : Figure
        Parameter description.

    Returns
    -------
    set[str]
        Return value description.
    """
    tokens = _tokenize(figure.id)
    tokens.update(_tokenize(figure.caption))
    tokens.update(_tokenize(figure.alt_text))
    for entity in figure.related_entities:
        tokens.update(_tokenize(entity))
    return tokens


def graph_from_dict(payload: dict) -> KnowledgeGraph:
    """Graph from dict.

    Parameters
    ----------
    payload : dict
        Parameter description.

    Returns
    -------
    KnowledgeGraph
        Return value description.
    """
    graph = KnowledgeGraph()
    for relation in payload.get("relations", []):
        graph.add_relation(
            source=relation["source"],
            predicate=relation.get("predicate", "related_to"),
            target=relation["target"],
        )
    for entity in payload.get("entities", []):
        graph.entities.add(entity)
    for figure in payload.get("figures", []):
        graph.add_figure(
            Figure(
                id=figure["id"],
                caption=figure.get("caption", figure["id"]),
                uri=figure.get("uri", figure.get("path", "")),
                alt_text=figure.get("alt_text", ""),
                related_entities=_string_list(figure.get("related_entities", [])),
                metadata=dict(figure.get("metadata", {})),
            )
        )
    return graph


def graph_to_dict(graph: KnowledgeGraph) -> dict:
    """Graph to dict.

    Parameters
    ----------
    graph : KnowledgeGraph
        Parameter description.

    Returns
    -------
    dict
        Return value description.
    """
    return {
        "entities": sorted(graph.entities),
        "relations": [
            {"source": relation.source, "predicate": relation.predicate, "target": relation.target}
            for relation in graph.relations
        ],
        "figures": [
            {
                "id": figure.id,
                "caption": figure.caption,
                "uri": figure.uri,
                "alt_text": figure.alt_text,
                "related_entities": list(figure.related_entities),
                "metadata": dict(figure.metadata),
            }
            for figure in graph.figures.values()
        ],
    }


def _string_list(value: object) -> list[str]:
    """String list.

    Parameters
    ----------
    value : object
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []
