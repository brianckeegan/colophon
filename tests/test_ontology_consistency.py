import json
import unittest
from pathlib import Path


class OntologyConsistencyTests(unittest.TestCase):
    def _load(self, path: str) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_shared_metadata_fields_are_present_and_typed(self) -> None:
        catalogs = {
            "ontology/functional_forms.json": "functional_forms_catalog",
            "ontology/technical_forms.json": "functional_forms_catalog",
            "ontology/genre_ontology.json": "genre_profile_ontology",
            "ontology/wilson_academic_writing_ontology.json": "writing_companion_ontology",
        }
        for path, expected_type in catalogs.items():
            payload = self._load(path)
            for key in ("id", "name", "ontology_type", "version", "schema_version"):
                self.assertIn(key, payload, msg=f"{path} missing '{key}'")
                self.assertIsInstance(payload[key], str, msg=f"{path} '{key}' must be string")
                self.assertTrue(payload[key].strip(), msg=f"{path} '{key}' must be non-empty")
            self.assertEqual(payload["ontology_type"], expected_type, msg=f"{path} ontology_type mismatch")
            self.assertEqual(payload["version"], payload["schema_version"], msg=f"{path} version/schema mismatch")

    def test_functional_catalogs_have_consistent_form_structure(self) -> None:
        for path in ("ontology/functional_forms.json", "ontology/technical_forms.json"):
            payload = self._load(path)
            forms = payload.get("functional_forms")
            self.assertIsInstance(forms, list, msg=f"{path} functional_forms must be list")
            self.assertTrue(forms, msg=f"{path} functional_forms cannot be empty")

            form_ids: list[str] = []
            for row in forms:
                self.assertIsInstance(row, dict, msg=f"{path} form rows must be objects")
                self.assertIsInstance(row.get("id"), str, msg=f"{path} form id must be string")
                self.assertTrue(str(row.get("id", "")).strip(), msg=f"{path} form id cannot be empty")
                form_ids.append(str(row.get("id")))

            self.assertEqual(len(form_ids), len(set(form_ids)), msg=f"{path} form ids must be unique")

    def test_functional_and_writing_cross_references_exist(self) -> None:
        functional = self._load("ontology/functional_forms.json")
        writing = self._load("ontology/wilson_academic_writing_ontology.json")

        companions = functional.get("companion_ontologies")
        self.assertIsInstance(companions, list)
        self.assertTrue(companions)
        companion_paths = [
            row.get("path")
            for row in companions
            if isinstance(row, dict) and isinstance(row.get("path"), str) and row.get("path")
        ]
        self.assertIn("ontology/wilson_academic_writing_ontology.json", companion_paths)
        for companion_path in companion_paths:
            self.assertTrue(Path(companion_path).exists(), msg=f"missing companion ontology {companion_path}")

        compatibility = writing.get("compatibility", {})
        self.assertIsInstance(compatibility, dict)
        forms_path = compatibility.get("functional_forms_catalog")
        self.assertEqual(forms_path, "ontology/functional_forms.json")
        self.assertTrue(Path(forms_path).exists(), msg="writing ontology points to missing functional catalog")

    def test_genre_profile_structure_is_consistent(self) -> None:
        payload = self._load("ontology/genre_ontology.json")
        self.assertIsInstance(payload.get("default_profile_id"), str)
        profiles = payload.get("profiles")
        self.assertIsInstance(profiles, list)
        self.assertTrue(profiles)

        profile_ids: list[str] = []
        for profile in profiles:
            self.assertIsInstance(profile, dict)
            for key in ("id", "name", "audience", "discipline", "style", "genre", "language", "tone"):
                self.assertIsInstance(profile.get(key), str, msg=f"profile field '{key}' must be string")
                self.assertTrue(str(profile.get(key, "")).strip(), msg=f"profile field '{key}' cannot be empty")
            profile_ids.append(str(profile.get("id")))
            self.assertIsInstance(profile.get("agent_prompts", {}), dict)
            self.assertIsInstance(profile.get("recommendation", {}), dict)
            self.assertIsInstance(profile.get("validation", {}), dict)

        self.assertEqual(len(profile_ids), len(set(profile_ids)), msg="genre profile ids must be unique")
        self.assertIn(payload["default_profile_id"], profile_ids, msg="default_profile_id must reference a profile")

    def test_writing_ontology_structure_is_consistent(self) -> None:
        payload = self._load("ontology/wilson_academic_writing_ontology.json")
        assumptions = payload.get("assumptions")
        self.assertIsInstance(assumptions, list)
        self.assertTrue(assumptions)
        for assumption in assumptions:
            self.assertIsInstance(assumption, dict)
            self.assertIsInstance(assumption.get("id"), str)
            self.assertIsInstance(assumption.get("statement"), str)

        validations = payload.get("validations", {})
        self.assertIsInstance(validations, dict)
        rules = validations.get("rules")
        self.assertIsInstance(rules, list)
        self.assertTrue(rules)
        for rule in rules:
            self.assertIsInstance(rule, dict)
            self.assertIsInstance(rule.get("id"), str)
            self.assertIsInstance(rule.get("severity"), str)
            self.assertIsInstance(rule.get("category"), str)
            self.assertIsInstance(rule.get("component"), str)
            extra_keys = set(rule.keys()) - {"id", "severity", "category", "component"}
            self.assertTrue(extra_keys, msg=f"rule '{rule.get('id')}' should define rule-specific parameters")


if __name__ == "__main__":
    unittest.main()
