"""Scientometric paper recommendation clients and proposal workflow."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Protocol
from urllib import parse, request

from .coordination import MessageBus
from .graph import KnowledgeGraph
from .models import RecommendationProposal, Source


class PaperSearchClient(Protocol):
    """External scientometric paper search interface."""

    def find_related(self, seed_source: Source, max_results: int) -> list["RecommendedPaper"]:
        """Return related paper candidates for one seed source."""


@dataclass(slots=True)
class RecommendationConfig:
    """Configuration for external paper-recommendation retrieval and scoring.

    Parameters
    ----------
    provider : str
        Search provider alias (for example ``openalex`` or ``semantic_scholar``).
    api_base_url : str
        Provider API base URL.
    timeout_seconds : float
        HTTP timeout in seconds.
    per_seed_limit : int
        Maximum candidates fetched per bibliography seed source.
    top_k : int
        Maximum scored recommendations kept after ranking.
    min_score : float
        Minimum aggregate score required to keep a recommendation.
    mailto : str
        Optional OpenAlex polite-pool contact email.
    api_key_env : str
        Optional API key environment variable for authenticated providers.
    """

    provider: str = "openalex"
    api_base_url: str = "https://api.openalex.org"
    timeout_seconds: float = 20.0
    per_seed_limit: int = 5
    top_k: int = 8
    min_score: float = 0.2
    mailto: str = ""
    api_key_env: str = ""


@dataclass(slots=True)
class RecommendedPaper:
    """Normalized paper metadata returned by recommendation APIs.

    Parameters
    ----------
    paper_id : str
        Provider-specific paper identifier.
    title : str
        Candidate paper title.
    authors : list[str]
        Candidate author names.
    publication : str
        Publication venue/journal name.
    year : int | None
        Publication year.
    abstract : str
        Abstract text used for scoring and downstream KG updates.
    citation_count : int
        Citation count proxy for influence.
    source_url : str
        Canonical paper URL.
    doi : str
        DOI string when available.
    """

    paper_id: str
    title: str
    authors: list[str]
    publication: str
    year: int | None
    abstract: str
    citation_count: int
    source_url: str
    doi: str


@dataclass(slots=True)
class ScoredRecommendation:
    """Recommendation candidate enriched with score and provenance metadata.

    Parameters
    ----------
    paper : RecommendedPaper
        Candidate paper metadata.
    score : float
        Final aggregate recommendation score.
    reasons : list[str]
        Human-readable reasons explaining the score.
    based_on_source_ids : list[str]
        Bibliography source ids that contributed to recommendation ranking.
    """

    paper: RecommendedPaper
    score: float
    reasons: list[str]
    based_on_source_ids: list[str]


@dataclass(slots=True)
class OpenAlexSearchClient:
    """OpenAlex API client for related paper lookup.

    Parameters
    ----------
    config : RecommendationConfig
        Search configuration including endpoint and request limits.
    """

    config: RecommendationConfig

    def find_related(self, seed_source: Source, max_results: int) -> list[RecommendedPaper]:
        """Find related.

        Parameters
        ----------
        seed_source : Source
            Parameter description.
        max_results : int
            Parameter description.

        Returns
        -------
        list[RecommendedPaper]
            Return value description.
        """
        query = self._build_query(seed_source)
        endpoint = self._build_endpoint(query=query, per_page=max_results)

        payload = _fetch_json(url=endpoint, timeout_seconds=self.config.timeout_seconds)
        results = payload.get("results", [])
        if not isinstance(results, list):
            return []

        papers: list[RecommendedPaper] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            paper = _paper_from_openalex_result(result)
            if paper.title:
                papers.append(paper)
        return papers

    def _build_query(self, source: Source) -> str:
        """Build query.

        Parameters
        ----------
        source : Source
            Parameter description.

        Returns
        -------
        str
            Return value description.
        """
        parts = [source.title]
        if source.authors:
            parts.append(source.authors[0])
        if source.text:
            parts.append(" ".join(source.text.split()[:24]))
        if isinstance(source.metadata, dict):
            query_terms = source.metadata.get("genre_query_terms", [])
            if isinstance(query_terms, list):
                parts.extend([term for term in query_terms if isinstance(term, str) and term.strip()])
        return " ".join(part for part in parts if part).strip()

    def _build_endpoint(self, query: str, per_page: int) -> str:
        """Build endpoint.

        Parameters
        ----------
        query : str
            Parameter description.
        per_page : int
            Parameter description.

        Returns
        -------
        str
            Return value description.
        """
        params = {
            "search": query,
            "per-page": str(max(1, per_page)),
        }
        if self.config.mailto.strip():
            params["mailto"] = self.config.mailto.strip()
        encoded = parse.urlencode(params)
        return f"{self.config.api_base_url.rstrip('/')}/works?{encoded}"


@dataclass(slots=True)
class SemanticScholarSearchClient:
    """Semantic Scholar API client for related paper lookup.

    Parameters
    ----------
    config : RecommendationConfig
        Search configuration including endpoint and auth env var.
    """

    config: RecommendationConfig

    def find_related(self, seed_source: Source, max_results: int) -> list[RecommendedPaper]:
        """Find related.

        Parameters
        ----------
        seed_source : Source
            Parameter description.
        max_results : int
            Parameter description.

        Returns
        -------
        list[RecommendedPaper]
            Return value description.
        """
        query = self._build_query(seed_source)
        endpoint = self._build_endpoint(query=query, limit=max_results)
        headers = self._request_headers()

        payload = _fetch_json(
            url=endpoint,
            timeout_seconds=self.config.timeout_seconds,
            headers=headers,
        )
        results = payload.get("data", [])
        if not isinstance(results, list):
            return []

        papers: list[RecommendedPaper] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            paper = _paper_from_semantic_scholar_result(result)
            if paper.title:
                papers.append(paper)
        return papers

    def _build_query(self, source: Source) -> str:
        """Build query.

        Parameters
        ----------
        source : Source
            Parameter description.

        Returns
        -------
        str
            Return value description.
        """
        parts = [source.title]
        if source.authors:
            parts.append(source.authors[0])
        if source.text:
            parts.append(" ".join(source.text.split()[:24]))
        if isinstance(source.metadata, dict):
            query_terms = source.metadata.get("genre_query_terms", [])
            if isinstance(query_terms, list):
                parts.extend([term for term in query_terms if isinstance(term, str) and term.strip()])
        return " ".join(part for part in parts if part).strip()

    def _build_endpoint(self, query: str, limit: int) -> str:
        """Build endpoint.

        Parameters
        ----------
        query : str
            Parameter description.
        limit : int
            Parameter description.

        Returns
        -------
        str
            Return value description.
        """
        params = {
            "query": query,
            "limit": str(max(1, limit)),
            "fields": "paperId,title,authors,year,abstract,citationCount,url,venue,externalIds",
        }
        encoded = parse.urlencode(params)
        return f"{self.config.api_base_url.rstrip('/')}/graph/v1/paper/search?{encoded}"

    def _request_headers(self) -> dict[str, str]:
        """Request headers.

        Returns
        -------
        dict[str, str]
            Return value description.
        """
        env_name = self.config.api_key_env.strip()
        if not env_name:
            return {}
        api_key = os.getenv(env_name, "").strip()
        if not api_key:
            return {}
        return {"x-api-key": api_key}


@dataclass(slots=True)
class PaperRecommendationWorkflow:
    """End-to-end proposal generator for bibliography and KG recommendation updates.

    Parameters
    ----------
    config : RecommendationConfig
        Retrieval and ranking policy.
    client : PaperSearchClient | None
        Optional injected search client; defaults to provider-specific client.
    """

    config: RecommendationConfig
    client: PaperSearchClient | None = None

    def generate_proposals(
        self,
        bibliography: list[Source],
        graph: KnowledgeGraph,
        outline: list[dict],
        message_bus: MessageBus | None = None,
        genre_context: dict[str, object] | None = None,
    ) -> list[RecommendationProposal]:
        """Generate proposals.

        Parameters
        ----------
        bibliography : list[Source]
            Parameter description.
        graph : KnowledgeGraph
            Parameter description.
        outline : list[dict]
            Parameter description.
        message_bus : MessageBus | None
            Parameter description.
        genre_context : dict[str, object] | None
            Optional genre ontology context to influence query terms and ranking cues.

        Returns
        -------
        list[RecommendationProposal]
            Return value description.
        """
        if not bibliography:
            if message_bus is not None:
                message_bus.send(
                    sender="paper_recommender",
                    receiver="user",
                    message_type="recommendation_skipped",
                    content="No bibliography entries available to seed recommendations.",
                    priority="high",
                )
            return []

        search_client = self.client or self._default_client()
        existing_titles = {_normalize_text(source.title) for source in bibliography if source.title}
        existing_dois = {
            _normalize_text(str(source.metadata.get("doi", "")))
            for source in bibliography
            if isinstance(source.metadata, dict)
        }

        aggregate: dict[str, ScoredRecommendation] = {}

        for source in bibliography:
            seed_source = _source_with_genre_context(source=source, genre_context=genre_context)
            try:
                candidates = search_client.find_related(seed_source, max_results=self.config.per_seed_limit)
            except Exception as exc:  # pragma: no cover - API failures are environment-specific.
                if message_bus is not None:
                    message_bus.send(
                        sender="paper_recommender",
                        receiver="user",
                        message_type="recommendation_error",
                        content=f"Failed to retrieve recommendations for '{source.title}': {exc}",
                        related_id=source.id,
                        priority="high",
                    )
                continue

            for candidate in candidates:
                if _normalize_text(candidate.title) in existing_titles:
                    continue
                if candidate.doi and _normalize_text(candidate.doi) in existing_dois:
                    continue

                score, reasons = _score_candidate(seed=source, candidate=candidate, genre_context=genre_context)
                if score < self.config.min_score:
                    continue

                key = _candidate_key(candidate)
                existing = aggregate.get(key)
                if existing is None:
                    aggregate[key] = ScoredRecommendation(
                        paper=candidate,
                        score=score,
                        reasons=reasons,
                        based_on_source_ids=[source.id],
                    )
                else:
                    if score > existing.score:
                        aggregate[key] = ScoredRecommendation(
                            paper=candidate,
                            score=score,
                            reasons=reasons,
                            based_on_source_ids=sorted(set(existing.based_on_source_ids + [source.id])),
                        )
                    else:
                        existing.based_on_source_ids = sorted(set(existing.based_on_source_ids + [source.id]))
                        existing.reasons = sorted(set(existing.reasons + reasons))

        ranked = sorted(aggregate.values(), key=lambda item: item.score, reverse=True)
        trimmed = ranked[: max(0, self.config.top_k)]

        proposals = [
            _proposal_from_recommendation(index=idx, recommendation=item, graph=graph)
            for idx, item in enumerate(trimmed, start=1)
        ]

        if message_bus is not None:
            if proposals:
                message_bus.send(
                    sender="paper_recommender",
                    receiver="user",
                    message_type="recommendation_summary",
                    content=(
                        f"Generated {len(proposals)} recommendation proposal(s) for bibliography and KG updates."
                    ),
                )
                for proposal in proposals:
                    message_bus.send(
                        sender="paper_recommender",
                        receiver="book_coordinator",
                        message_type="recommendation_proposal",
                        content=f"Proposed paper '{proposal.title}' (score={proposal.score:.3f}).",
                        related_id=proposal.proposal_id,
                    )
            else:
                message_bus.send(
                    sender="paper_recommender",
                    receiver="user",
                    message_type="recommendation_summary",
                    content="No recommendation proposals met quality thresholds.",
                    priority="high",
                )

        return proposals

    def _default_client(self) -> PaperSearchClient:
        """Default client.

        Returns
        -------
        PaperSearchClient
            Return value description.
        """
        provider = self.config.provider.strip().lower()
        if provider in {"openalex", "scholar_search", "scholar"}:
            return OpenAlexSearchClient(config=self.config)
        if provider in {"semantic_scholar", "semantic-scholar", "ai2", "s2"}:
            configured_base = self.config.api_base_url.rstrip("/")
            if configured_base in {"", "https://api.openalex.org"}:
                adjusted = RecommendationConfig(
                    provider=self.config.provider,
                    api_base_url="https://api.semanticscholar.org",
                    timeout_seconds=self.config.timeout_seconds,
                    per_seed_limit=self.config.per_seed_limit,
                    top_k=self.config.top_k,
                    min_score=self.config.min_score,
                    mailto=self.config.mailto,
                    api_key_env=self.config.api_key_env,
                )
            else:
                adjusted = self.config
            return SemanticScholarSearchClient(config=adjusted)
        raise ValueError(f"Unsupported recommendation provider: {self.config.provider}")


def _paper_from_openalex_result(result: dict) -> RecommendedPaper:
    """Paper from openalex result.

    Parameters
    ----------
    result : dict
        Parameter description.

    Returns
    -------
    RecommendedPaper
        Return value description.
    """
    title = _safe_string(result.get("display_name", ""))
    paper_id = _safe_string(result.get("id", ""))
    year = result.get("publication_year") if isinstance(result.get("publication_year"), int) else None
    citation_count = result.get("cited_by_count") if isinstance(result.get("cited_by_count"), int) else 0

    authors: list[str] = []
    authorships = result.get("authorships", [])
    if isinstance(authorships, list):
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author_info = authorship.get("author", {})
            if isinstance(author_info, dict):
                author_name = _safe_string(author_info.get("display_name", ""))
                if author_name:
                    authors.append(author_name)

    publication = ""
    source = result.get("primary_location", {})
    if isinstance(source, dict):
        source_info = source.get("source", {})
        if isinstance(source_info, dict):
            publication = _safe_string(source_info.get("display_name", ""))

    doi = _safe_string(result.get("doi", ""))
    source_url = ""
    if isinstance(source, dict):
        source_url = _safe_string(source.get("landing_page_url", ""))
    if not source_url:
        source_url = _safe_string(result.get("id", ""))

    abstract = _openalex_abstract_to_text(result.get("abstract_inverted_index"))

    return RecommendedPaper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        publication=publication,
        year=year,
        abstract=abstract,
        citation_count=max(0, citation_count),
        source_url=source_url,
        doi=doi,
    )


def _paper_from_semantic_scholar_result(result: dict) -> RecommendedPaper:
    """Paper from semantic scholar result.

    Parameters
    ----------
    result : dict
        Parameter description.

    Returns
    -------
    RecommendedPaper
        Return value description.
    """
    title = _safe_string(result.get("title", ""))
    paper_id = _safe_string(result.get("paperId", ""))
    year = result.get("year") if isinstance(result.get("year"), int) else None
    citation_count = result.get("citationCount") if isinstance(result.get("citationCount"), int) else 0

    authors: list[str] = []
    raw_authors = result.get("authors", [])
    if isinstance(raw_authors, list):
        for author in raw_authors:
            if not isinstance(author, dict):
                continue
            name = _safe_string(author.get("name", ""))
            if name:
                authors.append(name)

    publication = _safe_string(result.get("venue", ""))
    abstract = _safe_string(result.get("abstract", ""))
    source_url = _safe_string(result.get("url", ""))
    if not source_url and paper_id:
        source_url = f"https://www.semanticscholar.org/paper/{paper_id}"

    doi = ""
    external_ids = result.get("externalIds", {})
    if isinstance(external_ids, dict):
        doi = _safe_string(external_ids.get("DOI", ""))
    if doi and not doi.lower().startswith("http"):
        doi = f"https://doi.org/{doi}"

    return RecommendedPaper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        publication=publication,
        year=year,
        abstract=abstract,
        citation_count=max(0, citation_count),
        source_url=source_url,
        doi=doi,
    )


def _openalex_abstract_to_text(value: object) -> str:
    """Openalex abstract to text.

    Parameters
    ----------
    value : object
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    if not isinstance(value, dict):
        return ""
    indexed: list[tuple[int, str]] = []
    for token, positions in value.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                indexed.append((pos, token))
    indexed.sort(key=lambda item: item[0])
    if not indexed:
        return ""
    return " ".join(token for _, token in indexed)


def _score_candidate(
    seed: Source,
    candidate: RecommendedPaper,
    genre_context: dict[str, object] | None = None,
) -> tuple[float, list[str]]:
    """Score candidate.

    Parameters
    ----------
    seed : Source
        Parameter description.
    candidate : RecommendedPaper
        Parameter description.
    genre_context : dict[str, object] | None
        Optional genre context for additional ranking cues.

    Returns
    -------
    tuple[float, list[str]]
        Return value description.
    """
    reasons: list[str] = []

    seed_title_tokens = _tokenize(seed.title)
    candidate_title_tokens = _tokenize(candidate.title)
    title_overlap = _jaccard(seed_title_tokens, candidate_title_tokens)
    if title_overlap > 0:
        reasons.append(f"title overlap {title_overlap:.2f}")

    seed_text_tokens = _tokenize(seed.text)
    candidate_text_tokens = _tokenize(candidate.abstract)
    abstract_overlap = _jaccard(seed_text_tokens, candidate_text_tokens)
    if abstract_overlap > 0:
        reasons.append(f"abstract overlap {abstract_overlap:.2f}")

    seed_authors = {_normalize_text(author) for author in seed.authors}
    candidate_authors = {_normalize_text(author) for author in candidate.authors}
    author_overlap = _jaccard(seed_authors, candidate_authors)
    if author_overlap > 0:
        reasons.append("author overlap")

    publication_match = 0.0
    seed_publication = _normalize_text(str(seed.metadata.get("publication", ""))) if isinstance(seed.metadata, dict) else ""
    if seed_publication and seed_publication == _normalize_text(candidate.publication):
        publication_match = 1.0
        reasons.append("same publication venue")

    citation_signal = min(1.0, math.log1p(max(0, candidate.citation_count)) / math.log(101))
    if candidate.citation_count > 0:
        reasons.append(f"{candidate.citation_count} citations")

    genre_keyword_overlap = _genre_keyword_overlap(candidate=candidate, genre_context=genre_context)
    if genre_keyword_overlap > 0:
        reasons.append(f"genre keyword overlap {genre_keyword_overlap:.2f}")
    keyword_weight = _genre_keyword_weight(genre_context=genre_context)

    score = (
        0.35 * title_overlap
        + 0.25 * abstract_overlap
        + 0.2 * author_overlap
        + 0.1 * publication_match
        + 0.1 * citation_signal
        + keyword_weight * genre_keyword_overlap
    )

    return score, reasons


def _source_with_genre_context(source: Source, genre_context: dict[str, object] | None) -> Source:
    """Attach genre query terms to seed source metadata for API search.

    Parameters
    ----------
    source : Source
        Original bibliography source.
    genre_context : dict[str, object] | None
        Optional genre context.

    Returns
    -------
    Source
        Source copy enriched with genre query hints.
    """
    if not isinstance(genre_context, dict):
        return source
    recommendation = genre_context.get("recommendation", {})
    if not isinstance(recommendation, dict):
        return source
    terms = recommendation.get("query_terms", [])
    if not isinstance(terms, list):
        return source
    query_terms = [term for term in terms if isinstance(term, str) and term.strip()]
    if not query_terms:
        return source
    metadata = dict(source.metadata) if isinstance(source.metadata, dict) else {}
    existing = metadata.get("genre_query_terms", [])
    merged: list[str] = []
    if isinstance(existing, list):
        merged.extend([term for term in existing if isinstance(term, str) and term.strip()])
    merged.extend(query_terms)
    metadata["genre_query_terms"] = sorted(set(merged))
    return Source(
        id=source.id,
        title=source.title,
        authors=list(source.authors),
        year=source.year,
        text=source.text,
        metadata=metadata,
    )


def _genre_keyword_overlap(candidate: RecommendedPaper, genre_context: dict[str, object] | None) -> float:
    """Compute overlap between candidate text and genre query terms.

    Parameters
    ----------
    candidate : RecommendedPaper
        Candidate paper.
    genre_context : dict[str, object] | None
        Optional genre context.

    Returns
    -------
    float
        Jaccard overlap score in [0, 1].
    """
    if not isinstance(genre_context, dict):
        return 0.0
    recommendation = genre_context.get("recommendation", {})
    if not isinstance(recommendation, dict):
        return 0.0
    terms = recommendation.get("query_terms", [])
    if not isinstance(terms, list):
        return 0.0
    term_tokens = _tokenize(" ".join([term for term in terms if isinstance(term, str)]))
    if not term_tokens:
        return 0.0
    candidate_tokens = _tokenize(f"{candidate.title} {candidate.abstract} {candidate.publication}")
    return _jaccard(term_tokens, candidate_tokens)


def _genre_keyword_weight(genre_context: dict[str, object] | None) -> float:
    """Return bounded genre-keyword ranking weight.

    Parameters
    ----------
    genre_context : dict[str, object] | None
        Optional genre context.

    Returns
    -------
    float
        Weight in [0, 0.2].
    """
    if not isinstance(genre_context, dict):
        return 0.0
    recommendation = genre_context.get("recommendation", {})
    if not isinstance(recommendation, dict):
        return 0.0
    value = recommendation.get("keyword_weight", 0.08)
    if isinstance(value, (int, float)):
        return max(0.0, min(0.2, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(0.2, float(value.strip())))
        except ValueError:
            return 0.0
    return 0.0


def _proposal_from_recommendation(index: int, recommendation: ScoredRecommendation, graph: KnowledgeGraph) -> RecommendationProposal:
    """Proposal from recommendation.

    Parameters
    ----------
    index : int
        Parameter description.
    recommendation : ScoredRecommendation
        Parameter description.
    graph : KnowledgeGraph
        Parameter description.

    Returns
    -------
    RecommendationProposal
        Return value description.
    """
    paper = recommendation.paper
    proposal_id = f"rec-{index:03d}"

    bibliography_entry_id = _synth_source_id(title=paper.title, index=index)
    bibliography_entry = {
        "id": bibliography_entry_id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "text": paper.abstract,
        "metadata": {
            "publication": paper.publication,
            "doi": paper.doi,
            "source_url": paper.source_url,
            "citation_count": paper.citation_count,
            "recommended_via": "paper_recommendation",
            "based_on_source_ids": recommendation.based_on_source_ids,
        },
    }

    seed_entities = graph.entities_for_query(paper.title)
    candidate_entities = _extract_candidate_entities(title=paper.title, abstract=paper.abstract)
    entities = sorted(set(candidate_entities + seed_entities[:3]))

    relations = []
    for seed_id in recommendation.based_on_source_ids:
        relations.append(
            {
                "source": seed_id,
                "predicate": "related_literature",
                "target": bibliography_entry_id,
            }
        )

    knowledge_graph_update = {
        "entities": entities,
        "relations": relations,
    }

    return RecommendationProposal(
        proposal_id=proposal_id,
        title=paper.title,
        authors=paper.authors,
        publication=paper.publication,
        year=paper.year,
        abstract=paper.abstract,
        citation_count=paper.citation_count,
        source_url=paper.source_url,
        doi=paper.doi,
        score=recommendation.score,
        reasons=recommendation.reasons,
        based_on_source_ids=recommendation.based_on_source_ids,
        bibliography_entry=bibliography_entry,
        knowledge_graph_update=knowledge_graph_update,
    )


def _fetch_json(url: str, timeout_seconds: float, headers: dict[str, str] | None = None) -> dict:
    """Fetch json.

    Parameters
    ----------
    url : str
        Parameter description.
    timeout_seconds : float
        Parameter description.
    headers : dict[str, str] | None
        Parameter description.

    Returns
    -------
    dict
        Return value description.
    """
    req = request.Request(url=url, method="GET", headers=headers or {})
    with request.urlopen(req, timeout=timeout_seconds) as response:  # pragma: no cover - mocked in tests
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def _extract_candidate_entities(title: str, abstract: str) -> list[str]:
    """Extract candidate entities.

    Parameters
    ----------
    title : str
        Parameter description.
    abstract : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    tokens = _tokenize(f"{title} {abstract}")
    ranked = sorted(tokens, key=lambda item: (-len(item), item))
    selected = [token.title() for token in ranked if len(token) >= 6][:6]
    return selected


def _candidate_key(candidate: RecommendedPaper) -> str:
    """Candidate key.

    Parameters
    ----------
    candidate : RecommendedPaper
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    if candidate.doi:
        return f"doi:{_normalize_text(candidate.doi)}"
    if candidate.paper_id:
        return f"id:{_normalize_text(candidate.paper_id)}"
    return f"title:{_normalize_text(candidate.title)}"


def _jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard.

    Parameters
    ----------
    left : set[str]
        Parameter description.
    right : set[str]
        Parameter description.

    Returns
    -------
    float
        Return value description.
    """
    if not left or not right:
        return 0.0
    union = left.union(right)
    if not union:
        return 0.0
    return len(left.intersection(right)) / len(union)


def _tokenize(text: str) -> set[str]:
    """Tokenize.

    Parameters
    ----------
    text : str
        Parameter description.

    Returns
    -------
    set[str]
        Return value description.
    """
    if not text:
        return set()
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) >= 3
    }


def _normalize_text(text: str) -> str:
    """Normalize text.

    Parameters
    ----------
    text : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _safe_string(value: object) -> str:
    """Safe string.

    Parameters
    ----------
    value : object
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    if isinstance(value, str):
        return value.strip()
    return ""


def _synth_source_id(title: str, index: int) -> str:
    """Synth source id.

    Parameters
    ----------
    title : str
        Parameter description.
    index : int
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    collapsed = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{collapsed or 'recommended-paper'}-{index}"
