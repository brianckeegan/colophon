import json
import unittest
from pathlib import Path

from colophon.coordination import (
    BookCoordinationAgent,
    ChapterCoordinationAgent,
    MessageBus,
    ParagraphCoordinationAgent,
    SectionCoordinationAgent,
)
from colophon.functional_forms import select_functional_form
from colophon.models import Chapter, Paragraph, Section


class CoordinationTests(unittest.TestCase):
    def _load_form(self) -> dict:
        payload = json.loads(Path("ontology/functional_forms.json").read_text(encoding="utf-8"))
        return select_functional_form(payload, form_id="sequential_transformation")

    def _load_technical_form(self) -> dict:
        payload = json.loads(Path("ontology/technical_forms.json").read_text(encoding="utf-8"))
        return select_functional_form(payload, form_id="imrad_contribution")

    def test_message_bus_send_and_filter(self) -> None:
        bus = MessageBus()
        bus.send("section", "paragraph", "guidance", "Write clearly")
        bus.send("chapter", "section", "guidance", "Follow outline")

        paragraph_messages = bus.messages_for("paragraph")

        self.assertEqual(len(paragraph_messages), 1)
        self.assertEqual(paragraph_messages[0].content, "Write clearly")

    def test_message_bus_deduplicates_identical_messages(self) -> None:
        bus = MessageBus()
        bus.send("section", "paragraph", "guidance", "Write clearly")
        bus.send("section", "paragraph", "guidance", "Write clearly")

        self.assertEqual(len(bus.messages), 1)

    def test_paragraph_coordinator_emits_gap_for_fallback_text(self) -> None:
        bus = MessageBus()
        section = Section(
            id="ch1-s1",
            title="Methods",
            paragraphs=[Paragraph(id="p1", text="No sufficiently grounded claims were generated for this section.")],
            claims=[],
        )

        gaps = ParagraphCoordinationAgent().coordinate(section=section, bus=bus)

        self.assertGreaterEqual(len(gaps), 1)
        self.assertTrue(any(gap.component == "bibliography" for gap in gaps))
        self.assertTrue(any(message.receiver == "section_coordinator" for message in bus.messages))

    def test_section_coordinator_fallback_suffix_is_idempotent(self) -> None:
        bus = MessageBus()
        section = Section(
            id="ch1-s1",
            title="Methods",
            paragraphs=[Paragraph(id="p1", text="No sufficiently grounded claims were generated for this section.")],
            claims=[],
        )
        agent = SectionCoordinationAgent()

        agent.coordinate(section=section, expected_title="Methods", bus=bus)
        once = section.paragraphs[0].text
        agent.coordinate(section=section, expected_title="Methods", bus=bus)
        twice = section.paragraphs[0].text

        self.assertEqual(once, twice)

    def test_section_chapter_book_coordinators_identify_outline_gaps(self) -> None:
        bus = MessageBus()
        section = Section(id="ch1-s1", title="Unexpected Section", paragraphs=[], claims=[])
        chapter = Chapter(id="ch1", title="Chapter One", sections=[section])

        section_gaps = SectionCoordinationAgent().coordinate(section=section, expected_title="Expected Section", bus=bus)
        chapter_gaps = ChapterCoordinationAgent().coordinate(
            chapter=chapter,
            expected_sections=["Expected Section", "Missing Section"],
            bus=bus,
        )
        book_gaps = BookCoordinationAgent().coordinate(
            chapters=[chapter],
            outline=[{"title": "Chapter One"}, {"title": "Missing Chapter"}],
            bus=bus,
        )

        self.assertTrue(any(gap.component == "outline" for gap in section_gaps))
        self.assertTrue(any("Missing Section" in gap.request for gap in chapter_gaps))
        self.assertTrue(any("Missing Chapter" in gap.request for gap in book_gaps))

    def test_section_coordinator_emits_functional_form_guidance(self) -> None:
        bus = MessageBus()
        form = self._load_form()
        agent = SectionCoordinationAgent(functional_form=form)

        agent.prepare_child_guidance(
            section_title="Central Problem",
            bus=bus,
            chapter_id="ch1",
            chapter_type="introduction",
        )

        claim_messages = bus.messages_for("claim_author_agent")
        self.assertTrue(claim_messages)
        self.assertTrue(any("functional-form element" in msg.content.lower() for msg in claim_messages))

    def test_chapter_coordinator_flags_missing_expected_functional_elements(self) -> None:
        bus = MessageBus()
        form = self._load_form()
        chapter = Chapter(id="ch1", title="Foundations", sections=[Section(id="ch1-s1", title="Background")])
        gaps = ChapterCoordinationAgent(functional_form=form).coordinate(
            chapter=chapter,
            expected_sections=["Background"],
            bus=bus,
            chapter_type="introduction",
            expected_elements=["central_problem"],
        )

        self.assertTrue(any("functional-form element 'central_problem'" in gap.request for gap in gaps))

    def test_technical_form_synthesizes_coordination_policy_targets(self) -> None:
        bus = MessageBus()
        form = self._load_technical_form()
        section = Section(
            id="ch1-s1",
            title="Problem Frame",
            paragraphs=[Paragraph(id="p1", text="Claim without citation")],
            claims=[],
        )
        gaps = ParagraphCoordinationAgent(functional_form=form).coordinate(section=section, bus=bus)

        self.assertTrue(any(gap.component == "bibliography" for gap in gaps))
        summaries = [message.content for message in bus.messages_for("section_coordinator")]
        self.assertTrue(
            any(
                ("evidence linkage" in content.lower()) or ("problem frame" in content.lower())
                for content in summaries
            )
        )


if __name__ == "__main__":
    unittest.main()
