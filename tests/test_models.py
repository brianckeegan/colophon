import unittest

from colophon.models import Chapter, Figure, GapRequest, Manuscript, Paragraph, RecommendationProposal, Section


class ModelTests(unittest.TestCase):
    def test_manuscript_to_markdown_renders_hierarchy(self) -> None:
        manuscript = Manuscript(
            title="My Draft",
            chapters=[
                Chapter(
                    id="ch1",
                    title="Chapter One",
                    sections=[
                        Section(
                            id="ch1-s1",
                            title="Section A",
                            paragraphs=[
                                Paragraph(id="p1", text="First paragraph."),
                                Paragraph(id="p2", text="Second paragraph."),
                            ],
                        )
                    ],
                )
            ],
        )

        markdown = manuscript.to_markdown()

        self.assertIn("# My Draft", markdown)
        self.assertIn("## Chapter One", markdown)
        self.assertIn("### Section A", markdown)
        self.assertIn("First paragraph.", markdown)
        self.assertIn("Second paragraph.", markdown)
        self.assertTrue(markdown.endswith("\n"))

    def test_manuscript_to_markdown_handles_empty_body(self) -> None:
        manuscript = Manuscript(title="Title Only")
        self.assertEqual(manuscript.to_markdown(), "# Title Only\n")

    def test_manuscript_render_supports_all_output_formats(self) -> None:
        manuscript = Manuscript(
            title="Format Draft",
            chapters=[
                Chapter(
                    id="ch1",
                    title="Intro",
                    sections=[
                        Section(
                            id="s1",
                            title="Setup",
                            paragraphs=[Paragraph(id="p1", text="Body text")],
                            figures=[Figure(id="fig-1", caption="Setup diagram", uri="figures/setup.png")],
                        )
                    ],
                )
            ],
        )

        plain_text = manuscript.render("text")
        markdown = manuscript.render("markdown")
        rst = manuscript.render("rst")
        rtf = manuscript.render("rtf")
        latex = manuscript.render("latex")

        self.assertIn("Chapter: Intro", plain_text)
        self.assertIn("# Format Draft", markdown)
        self.assertIn("Format Draft\n============", rst)
        self.assertIn("{\\rtf1\\ansi", rtf)
        self.assertIn("Chapter: Intro", rtf)
        self.assertIn("\\section{Intro}", latex)
        self.assertIn("\\subsection{Setup}", latex)
        self.assertIn("Figure fig-1: Setup diagram", markdown)
        self.assertIn("Figure fig-1: Setup diagram", plain_text)

    def test_render_raises_for_unknown_output_format(self) -> None:
        manuscript = Manuscript(title="X")
        with self.assertRaises(ValueError):
            manuscript.render("docx")

    def test_gap_requests_are_rendered_in_markdown(self) -> None:
        manuscript = Manuscript(
            title="Needs Input",
            gap_requests=[
                GapRequest(
                    level="section",
                    component="bibliography",
                    request="Add sources for methods section",
                    rationale="Current evidence is missing.",
                    related_id="ch1-s2",
                )
            ],
        )

        markdown = manuscript.to_markdown()

        self.assertIn("## Gap Requests", markdown)
        self.assertIn("Add sources for methods section", markdown)

    def test_recommendations_are_rendered_in_markdown(self) -> None:
        manuscript = Manuscript(
            title="Recommendations",
            recommendation_proposals=[
                RecommendationProposal(
                    proposal_id="rec-001",
                    title="Related Paper",
                    authors=["Alice"],
                    publication="Journal",
                    year=2024,
                    abstract="Abstract",
                    citation_count=10,
                    source_url="https://example.org/paper",
                    doi="https://doi.org/10.1000/paper",
                    score=0.8,
                )
            ],
        )

        markdown = manuscript.to_markdown()

        self.assertIn("## Recommended Papers", markdown)
        self.assertIn("Related Paper", markdown)


if __name__ == "__main__":
    unittest.main()
