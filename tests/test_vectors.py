import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colophon.vectors import (
    EmbeddingConfig,
    InMemoryVectorDB,
    OpenAIEmbeddingClient,
    VectorRecord,
    create_embedding_client,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class VectorTests(unittest.TestCase):
    def test_local_embedding_client_is_deterministic(self) -> None:
        client = create_embedding_client(
            EmbeddingConfig(provider="local", dimensions=32),
        )
        vectors = client.embed(["Deterministic text", "Deterministic text"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 32)
        self.assertEqual(vectors[0], vectors[1])

    def test_in_memory_vector_db_search_and_persistence(self) -> None:
        db = InMemoryVectorDB()
        db.add_many(
            [
                VectorRecord(record_id="a", text="A", vector=[1.0, 0.0, 0.0]),
                VectorRecord(record_id="b", text="B", vector=[0.0, 1.0, 0.0]),
                VectorRecord(record_id="c", text="C", vector=[0.8, 0.2, 0.0]),
            ]
        )

        hits = db.search([1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0][0].record_id, "a")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vectors.json"
            db.save_json(path)
            restored = InMemoryVectorDB.load_json(path)

        self.assertEqual(len(restored.records), 3)
        self.assertEqual(restored.records[0].record_id, "a")

    @patch("colophon.vectors.request.urlopen")
    def test_openai_embedding_client_parses_response(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(
            {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ]
            }
        )

        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )
        client = OpenAIEmbeddingClient(config=config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            vectors = client.embed(["first", "second"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0], [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
