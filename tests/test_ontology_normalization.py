import unittest

from colophon.ontology import (
    normalize_functional_forms_catalog,
    normalize_genre_ontology,
    normalize_writing_companion_ontology,
)


class OntologyNormalizationTests(unittest.TestCase):
    def test_normalize_functional_forms_catalog_sets_metadata_and_form_ids(self) -> None:
        payload = {"functional_forms": [{"name": "Unnamed Form"}, "bad-row"]}
        normalized = normalize_functional_forms_catalog(payload)

        self.assertEqual(normalized["ontology_type"], "functional_forms_catalog")
        self.assertIn("schema_version", normalized)
        self.assertIn("version", normalized)
        self.assertEqual(len(normalized["functional_forms"]), 1)
        self.assertEqual(normalized["functional_forms"][0]["id"], "form_1")

    def test_normalize_genre_ontology_sets_profile_defaults(self) -> None:
        payload = {"profiles": [{"id": "p1", "name": "Profile One"}]}
        normalized = normalize_genre_ontology(payload)

        self.assertEqual(normalized["ontology_type"], "genre_profile_ontology")
        self.assertEqual(normalized["default_profile_id"], "p1")
        profile = normalized["profiles"][0]
        self.assertEqual(profile["audience"], "general")
        self.assertEqual(profile["discipline"], "interdisciplinary")
        self.assertEqual(profile["style"], "analytical")
        self.assertEqual(profile["genre"], "scholarly_manuscript")
        self.assertEqual(profile["language"], "English")
        self.assertEqual(profile["tone"], "neutral")
        self.assertIsInstance(profile["agent_prompts"], dict)
        self.assertIsInstance(profile["recommendation"], dict)
        self.assertIsInstance(profile["validation"], dict)

    def test_normalize_writing_companion_ontology_sets_rule_defaults(self) -> None:
        payload = {
            "assumptions": [{"id": "a1", "statement": "x"}, {"statement": "missing-id"}],
            "validations": {"rules": [{"id": "r1"}, "bad-row"]},
        }
        normalized = normalize_writing_companion_ontology(payload)

        self.assertEqual(normalized["ontology_type"], "writing_companion_ontology")
        self.assertEqual(len(normalized["assumptions"]), 1)
        self.assertEqual(normalized["assumptions"][0]["id"], "a1")
        self.assertEqual(len(normalized["validations"]["rules"]), 1)
        rule = normalized["validations"]["rules"][0]
        self.assertEqual(rule["id"], "r1")
        self.assertEqual(rule["severity"], "info")
        self.assertEqual(rule["category"], "structure")
        self.assertEqual(rule["component"], "outline")


if __name__ == "__main__":
    unittest.main()
