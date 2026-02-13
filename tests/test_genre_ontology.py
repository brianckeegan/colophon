import json
import unittest
from pathlib import Path

from colophon.genre_ontology import build_genre_ontology_context


class GenreOntologyTests(unittest.TestCase):
    def test_build_context_uses_defaults_when_payload_missing(self) -> None:
        context = build_genre_ontology_context(
            genre_ontology_payload=None,
            profile_id="",
            overrides=None,
        )

        self.assertTrue(context["enabled"])
        self.assertEqual(context["profile_id"], "general_academic")
        self.assertEqual(context["metadata"]["style"], "analytical")
        self.assertIn("claim_author_agent", context["role_prompts"])

    def test_build_context_selects_profile_and_applies_overrides(self) -> None:
        payload = json.loads(Path("ontology/genre_ontology.json").read_text(encoding="utf-8"))
        context = build_genre_ontology_context(
            genre_ontology_payload=payload,
            profile_id="technical_research",
            overrides={"language": "Spanish", "genre": "white_paper"},
        )

        self.assertEqual(context["profile_id"], "technical_research")
        self.assertEqual(context["metadata"]["style"], "technical")
        self.assertEqual(context["metadata"]["language"], "Spanish")
        self.assertEqual(context["metadata"]["genre"], "white_paper")
        self.assertGreaterEqual(context["validation"]["min_top_k"], 1)


if __name__ == "__main__":
    unittest.main()
