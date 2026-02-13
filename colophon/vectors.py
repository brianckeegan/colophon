"""Embedding clients and lightweight in-memory vector database utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class EmbeddingClient(Protocol):
    """Embedding interface for local or remote providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


@dataclass(slots=True)
class EmbeddingConfig:
    """Embedding provider configuration for KG updates and note import.

    Parameters
    ----------
    provider : str
        Provider alias (for example ``local`` or ``openai``).
    model : str
        Model name used by remote embedding providers.
    api_base_url : str | None
        Optional base URL for remote embedding endpoint.
    api_key_env : str | None
        Optional environment variable containing provider token.
    dimensions : int
        Embedding vector dimensionality.
    timeout_seconds : float
        Request timeout for remote embedding APIs.
    """

    provider: str = "local"
    model: str = ""
    api_base_url: str | None = None
    api_key_env: str | None = None
    dimensions: int = 256
    timeout_seconds: float = 20.0


@dataclass(slots=True)
class VectorRecord:
    """Single vectorized document entry stored in the vector database.

    Parameters
    ----------
    record_id : str
        Stable record identifier.
    text : str
        Source text used to generate the vector.
    metadata : dict[str, Any]
        Additional metadata attached to the vector record.
    vector : list[float]
        Numeric embedding vector.
    """

    record_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class InMemoryVectorDB:
    """Simple vector store supporting add/search/load/save operations.

    Parameters
    ----------
    records : list[VectorRecord]
        Existing records to seed the vector database.
    """

    records: list[VectorRecord] = field(default_factory=list)

    def add(self, record: VectorRecord) -> None:
        """Add.

        Parameters
        ----------
        record : VectorRecord
            Parameter description.
        """
        self.records.append(record)

    def add_many(self, records: list[VectorRecord]) -> None:
        """Add many.

        Parameters
        ----------
        records : list[VectorRecord]
            Parameter description.
        """
        self.records.extend(records)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[VectorRecord, float]]:
        """Search.

        Parameters
        ----------
        query_vector : list[float]
            Parameter description.
        top_k : int
            Parameter description.
        exclude_ids : set[str] | None
            Parameter description.

        Returns
        -------
        list[tuple[VectorRecord, float]]
            Return value description.
        """
        blocked = exclude_ids or set()
        scored: list[tuple[VectorRecord, float]] = []

        for record in self.records:
            if record.record_id in blocked:
                continue
            score = _cosine_similarity(query_vector, record.vector)
            if score > 0:
                scored.append((record, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(0, top_k)]

    def save_json(self, path: str | Path) -> None:
        """Save json.

        Parameters
        ----------
        path : str | Path
            Parameter description.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "record_id": record.record_id,
                "text": record.text,
                "metadata": record.metadata,
                "vector": record.vector,
            }
            for record in self.records
        ]
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "InMemoryVectorDB":
        """Load json.

        Parameters
        ----------
        path : str | Path
            Parameter description.

        Returns
        -------
        "InMemoryVectorDB"
            Return value description.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Vector DB JSON must be a list of records.")

        records: list[VectorRecord] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("record_id", "")).strip()
            if not record_id:
                continue
            text = str(item.get("text", ""))
            metadata = item.get("metadata", {})
            vector = item.get("vector", [])
            if not isinstance(metadata, dict) or not isinstance(vector, list):
                continue
            if not all(isinstance(value, (int, float)) for value in vector):
                continue
            records.append(
                VectorRecord(
                    record_id=record_id,
                    text=text,
                    metadata=metadata,
                    vector=[float(value) for value in vector],
                )
            )

        return cls(records=records)


@dataclass(slots=True)
class LocalHashEmbeddingClient:
    """Dependency-free deterministic embedding model for local use."""

    config: EmbeddingConfig

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed.

        Parameters
        ----------
        texts : list[str]
            Parameter description.

        Returns
        -------
        list[list[float]]
            Return value description.
        """
        dims = max(8, int(self.config.dimensions))
        return [_hash_embedding(text=text, dims=dims) for text in texts]


@dataclass(slots=True)
class OpenAIEmbeddingClient:
    """OpenAI-compatible embeddings client."""

    config: EmbeddingConfig

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed.

        Parameters
        ----------
        texts : list[str]
            Parameter description.

        Returns
        -------
        list[list[float]]
            Return value description.
        """
        if not texts:
            return []

        api_key = _require_api_key(self.config.api_key_env)
        base_url = _require_value(self.config.api_base_url, "api_base_url")
        model = _require_value(self.config.model, "model")

        payload = {
            "model": model,
            "input": texts,
        }
        req = request.Request(
            url=f"{base_url.rstrip('/')}/embeddings",
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload).encode("utf-8"),
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:  # pragma: no cover
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:  # pragma: no cover
            details = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingError(f"Embedding API HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:  # pragma: no cover
            raise EmbeddingError(f"Embedding API connection failed: {exc.reason}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EmbeddingError("Embedding API returned non-JSON response") from exc

        data = payload.get("data", [])
        if not isinstance(data, list):
            raise EmbeddingError("Embedding API response missing 'data' list")

        vectors: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            embedding = item.get("embedding", [])
            if isinstance(embedding, list) and all(isinstance(value, (int, float)) for value in embedding):
                vectors.append([float(value) for value in embedding])

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs."
            )

        return vectors


def create_embedding_client(config: EmbeddingConfig) -> EmbeddingClient:
    """Create embedding client.

    Parameters
    ----------
    config : EmbeddingConfig
        Parameter description.

    Returns
    -------
    EmbeddingClient
        Return value description.
    """
    provider = config.provider.strip().lower()

    if provider in {"local", "hash", "offline"}:
        return LocalHashEmbeddingClient(config=config)

    if provider in {"openai", "openai_compatible", "remote"}:
        resolved = _apply_embedding_provider_defaults(config)
        return OpenAIEmbeddingClient(config=resolved)

    raise ValueError(f"Unsupported embedding provider: {config.provider}")


def _apply_embedding_provider_defaults(config: EmbeddingConfig) -> EmbeddingConfig:
    """Apply embedding provider defaults.

    Parameters
    ----------
    config : EmbeddingConfig
        Parameter description.

    Returns
    -------
    EmbeddingConfig
        Return value description.
    """
    provider = config.provider.strip().lower()

    api_base_url = config.api_base_url
    api_key_env = config.api_key_env

    if provider in {"openai", "openai_compatible", "remote"}:
        api_base_url = api_base_url or "https://api.openai.com/v1"
        api_key_env = api_key_env or "OPENAI_API_KEY"

    return EmbeddingConfig(
        provider=config.provider,
        model=config.model,
        api_base_url=api_base_url,
        api_key_env=api_key_env,
        dimensions=config.dimensions,
        timeout_seconds=config.timeout_seconds,
    )


def _hash_embedding(text: str, dims: int) -> list[float]:
    """Hash embedding.

    Parameters
    ----------
    text : str
        Parameter description.
    dims : int
        Parameter description.

    Returns
    -------
    list[float]
        Return value description.
    """
    vector = [0.0] * dims
    tokens = _tokenize(text)

    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            chunk = digest[offset : offset + 4]
            if len(chunk) < 4:
                continue
            value = int.from_bytes(chunk, "big", signed=False)
            index = value % dims
            sign = 1.0 if ((value >> 3) & 1) == 0 else -1.0
            vector[index] += sign

    return _normalize(vector)


def _tokenize(text: str) -> list[str]:
    """Tokenize.

    Parameters
    ----------
    text : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    return [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token]


def _normalize(vector: list[float]) -> list[float]:
    """Normalize.

    Parameters
    ----------
    vector : list[float]
        Parameter description.

    Returns
    -------
    list[float]
        Return value description.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity.

    Parameters
    ----------
    left : list[float]
        Parameter description.
    right : list[float]
        Parameter description.

    Returns
    -------
    float
        Return value description.
    """
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        # Graceful fallback for mismatched vectors.
        size = min(len(left), len(right))
        left = left[:size]
        right = right[:size]
    return sum(a * b for a, b in zip(left, right, strict=True))


def _require_api_key(env_name: str | None) -> str:
    """Require api key.

    Parameters
    ----------
    env_name : str | None
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    key_name = _require_value(env_name, "api_key_env")
    value = os.getenv(key_name, "").strip()
    if not value:
        raise EmbeddingError(f"Missing embedding API key in environment variable: {key_name}")
    return value


def _require_value(value: str | None, field: str) -> str:
    """Require value.

    Parameters
    ----------
    value : str | None
        Parameter description.
    field : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    if value is None or not value.strip():
        raise EmbeddingError(f"Missing embedding config field: {field}")
    return value.strip()
