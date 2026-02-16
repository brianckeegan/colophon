import tempfile
import unittest
from pathlib import Path

from colophon.agent_skills import AgentSkillsRuntime


class AgentSkillsTests(unittest.TestCase):
    def test_runtime_discovers_valid_skill_and_generates_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            skill_dir = root / "knowledge-graph-guidance"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: knowledge-graph-guidance",
                        "description: Guidance for knowledge graph sections and graph-grounded claims.",
                        "metadata:",
                        "  author: test-suite",
                        "---",
                        "# Skill",
                        "Always include graph-specific terminology.",
                    ]
                ),
                encoding="utf-8",
            )

            runtime = AgentSkillsRuntime.from_directories([str(root)])

            self.assertEqual(len(runtime.skills), 1)
            self.assertEqual(runtime.skills[0].name, "knowledge-graph-guidance")
            self.assertEqual(runtime.invalid_skills, [])
            self.assertIn("<available_skills>", runtime.available_skills_xml)
            self.assertIn("knowledge-graph-guidance", runtime.available_skills_xml)
            self.assertIn(str(skill_dir / "SKILL.md"), runtime.available_skills_xml)

    def test_runtime_reports_invalid_skill_name_directory_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            skill_dir = root / "mismatched-directory"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: wrong-name",
                        "description: Invalid because the name does not match the directory.",
                        "---",
                        "Body",
                    ]
                ),
                encoding="utf-8",
            )

            runtime = AgentSkillsRuntime.from_directories([str(root)])

            self.assertEqual(runtime.skills, [])
            self.assertEqual(len(runtime.invalid_skills), 1)
            self.assertTrue(any("must match frontmatter name" in error for error in runtime.invalid_skills[0].errors))

    def test_runtime_matches_and_activates_skill_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            skill_dir = root / "methods-writing"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: methods-writing",
                        "description: Help draft methods sections with reproducibility details and implementation steps.",
                        "---",
                        "Use reproducibility language and cite implementation details explicitly.",
                    ]
                ),
                encoding="utf-8",
            )

            runtime = AgentSkillsRuntime.from_directories([str(root)])
            activations = runtime.match_and_activate(
                task="Draft the methods section with implementation details.",
                max_matches=2,
                min_token_overlap=1,
            )

            self.assertEqual(len(activations), 1)
            self.assertEqual(activations[0].skill.name, "methods-writing")
            self.assertIn("reproducibility language", activations[0].instructions)
            self.assertIn("methods", activations[0].matched_tokens)


if __name__ == "__main__":
    unittest.main()
