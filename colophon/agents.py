"""Drafting, reviewing, and outline-expansion agents for Colophon pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .coordination import MessageBus
from .functional_forms import select_functional_form
from .graph import KnowledgeGraph
from .llm import LLMClient
from .models import Claim, CoordinationMessage, Figure, Paragraph, Section
from .retrieval import RetrievalHit

CLAIM_TEMPLATE_DEFAULT = (
    "{narrative_instruction}{lead_entity} is discussed in {source_title}, "
    "which contributes evidence for {section_title_lower}."
)
PARAGRAPH_TEMPLATE_DEFAULT = "{narrative_instruction}{claim_text} {citations} {figure_references}"
EMPTY_SECTION_TEMPLATE_DEFAULT = "No sufficiently grounded claims were generated for this section."
FIGURE_REFERENCE_TEMPLATE_DEFAULT = "See Figure {figure_id} ({figure_caption})."
OUTLINE_EXPANSION_PROMPTS_DEFAULT = {
    "claim_template": (
        "{lead_entity} is discussed in {source_title}, "
        "which contributes evidence for {section_title_lower}."
    ),
    "paragraph_template": "{claim_text} {citations} {figure_references}",
    "figure_reference_template": "See Figure {figure_id} ({figure_caption}).",
    "empty_section_template": "No sufficiently grounded claims were generated for this section.",
    "transition_template": "Transition from {previous_section} to {next_section} by emphasizing causal linkage.",
    "counterargument_template": (
        "A plausible counterargument to {claim_text} is {counterargument}; "
        "address it with evidence from {citations}."
    ),
}


@dataclass(slots=True)
class ClaimAuthorAgent:
    """Creates supported, scoped claims for a section."""

    max_claims: int = 3
    claim_template: str = CLAIM_TEMPLATE_DEFAULT
    figure_reference_template: str = FIGURE_REFERENCE_TEMPLATE_DEFAULT
    llm_client: LLMClient | None = None
    llm_system_prompt: str | None = None
    tone: str = "neutral"
    style: str = "analytical"
    audience: str = "general"
    discipline: str = "interdisciplinary"
    genre: str = "scholarly_manuscript"
    language: str = "English"
    background_prompt: str = ""

    def draft(
        self,
        section_id: str,
        section_title: str,
        hits: list[RetrievalHit],
        entities: list[str],
        figures: list[Figure],
        guidance_messages: list[CoordinationMessage] | None = None,
    ) -> list[Claim]:
        """Draft.

        Parameters
        ----------
        section_id : str
            Parameter description.
        section_title : str
            Parameter description.
        hits : list[RetrievalHit]
            Parameter description.
        entities : list[str]
            Parameter description.
        figures : list[Figure]
            Parameter description.
        guidance_messages : list[CoordinationMessage] | None
            Parameter description.

        Returns
        -------
        list[Claim]
            Return value description.
        """
        claims: list[Claim] = []
        for idx, hit in enumerate(hits[: self.max_claims], start=1):
            lead_entity = entities[0] if entities else section_title
            figure = figures[(idx - 1) % len(figures)] if figures else None
            claim_text = self._draft_claim_text(
                section_title=section_title,
                lead_entity=lead_entity,
                hit=hit,
                figure=figure,
                guidance_messages=guidance_messages or [],
            )
            figure_ids = [figure.id] if figure is not None else []
            claims.append(
                Claim(
                    id=f"{section_id}-c{idx}",
                    text=claim_text,
                    evidence_ids=[hit.source.id],
                    figure_ids=figure_ids,
                )
            )
        return claims

    def _draft_claim_text(
        self,
        section_title: str,
        lead_entity: str,
        hit: RetrievalHit,
        figure: Figure | None,
        guidance_messages: list[CoordinationMessage],
    ) -> str:
        """Draft claim text.

        Parameters
        ----------
        section_title : str
            Parameter description.
        lead_entity : str
            Parameter description.
        hit : RetrievalHit
            Parameter description.
        figure : Figure | None
            Parameter description.
        guidance_messages : list[CoordinationMessage]
            Parameter description.

        Returns
        -------
        str
            Return value description.
        """
        if self.llm_client is not None:
            excerpt = hit.source.text.split(".")[0].strip()
            narrative_profile = _narrative_profile_block(
                tone=self.tone,
                style=self.style,
                audience=self.audience,
                discipline=self.discipline,
                genre=self.genre,
                language=self.language,
            )
            figure_context = ""
            if figure is not None:
                figure_context = (
                    f"\nFigure id: {figure.id}\n"
                    f"Figure caption: {figure.caption}\n"
                    "Include an explicit figure reference (e.g., 'Figure <id>')."
                )
            guidance_context = ""
            if guidance_messages:
                guidance_lines = [f"- {message.content}" for message in guidance_messages]
                guidance_context = "\nCoordination guidance:\n" + "\n".join(guidance_lines)
            background_context = ""
            if self.background_prompt:
                background_context = "\nBackground guidance:\n- " + self.background_prompt
            prompt = (
                "Write one factual claim sentence for a section in a long-form manuscript. "
                "Ground it in the evidence excerpt and keep it concise.\n\n"
                f"{narrative_profile}\n"
                f"Section title: {section_title}\n"
                f"Lead entity: {lead_entity}\n"
                f"Source title: {hit.source.title}\n"
                f"Source id: {hit.source.id}\n"
                f"Evidence excerpt: {excerpt}"
                f"{figure_context}"
                f"{background_context}"
                f"{guidance_context}"
            )
            text = self.llm_client.generate(prompt=prompt, system_prompt=self.llm_system_prompt).strip()
            if text:
                return text

        narrative_instruction = _narrative_instruction_prefix(
            tone=self.tone,
            style=self.style,
            audience=self.audience,
            discipline=self.discipline,
            genre=self.genre,
            language=self.language,
        )
        claim_text = _safe_format(
            self.claim_template,
            narrative_instruction=narrative_instruction,
            narrative_tone=self.tone,
            narrative_style=self.style,
            narrative_audience=self.audience,
            narrative_discipline=self.discipline,
            narrative_genre=self.genre,
            narrative_language=self.language,
            lead_entity=lead_entity,
            source_title=hit.source.title,
            section_title=section_title,
            section_title_lower=section_title.lower(),
            source_id=hit.source.id,
        )
        if narrative_instruction and not claim_text.startswith(narrative_instruction):
            claim_text = narrative_instruction + claim_text
        if figure is not None:
            claim_text += " " + _safe_format(
                self.figure_reference_template,
                figure_id=figure.id,
                figure_caption=figure.caption,
                figure_uri=figure.uri,
            )
        return claim_text.strip()


@dataclass(slots=True)
class ParagraphAgent:
    """Composes prose from claim and evidence bundles."""

    paragraph_template: str = PARAGRAPH_TEMPLATE_DEFAULT
    empty_section_template: str = EMPTY_SECTION_TEMPLATE_DEFAULT
    llm_client: LLMClient | None = None
    llm_system_prompt: str | None = None
    tone: str = "neutral"
    style: str = "analytical"
    audience: str = "general"
    discipline: str = "interdisciplinary"
    genre: str = "scholarly_manuscript"
    language: str = "English"
    background_prompt: str = ""

    def draft(self, section_id: str, claims: list[Claim], hits: list[RetrievalHit]) -> list[Paragraph]:
        """Draft.

        Parameters
        ----------
        section_id : str
            Parameter description.
        claims : list[Claim]
            Parameter description.
        hits : list[RetrievalHit]
            Parameter description.

        Returns
        -------
        list[Paragraph]
            Return value description.
        """
        return self.draft_with_figures(
            section_id=section_id,
            claims=claims,
            hits=hits,
            figure_lookup={},
            guidance_messages=None,
        )

    def draft_with_figures(
        self,
        section_id: str,
        claims: list[Claim],
        hits: list[RetrievalHit],
        figure_lookup: dict[str, Figure],
        guidance_messages: list[CoordinationMessage] | None = None,
    ) -> list[Paragraph]:
        """Draft with figures.

        Parameters
        ----------
        section_id : str
            Parameter description.
        claims : list[Claim]
            Parameter description.
        hits : list[RetrievalHit]
            Parameter description.
        figure_lookup : dict[str, Figure]
            Parameter description.
        guidance_messages : list[CoordinationMessage] | None
            Parameter description.

        Returns
        -------
        list[Paragraph]
            Return value description.
        """
        if not claims:
            return [
                Paragraph(
                    id=f"{section_id}-p1",
                    text=self.empty_section_template,
                    claim_ids=[],
                )
            ]

        evidence_map = {hit.source.id: hit.source for hit in hits}
        paragraphs: list[Paragraph] = []

        for idx, claim in enumerate(claims, start=1):
            citation_text = []
            for evidence_id in claim.evidence_ids:
                source = evidence_map.get(evidence_id)
                if source is not None:
                    excerpt = source.text.split(".")[0].strip()
                    citation_text.append(f"[{source.id}] {excerpt}.")

            figure_references = []
            for figure_id in claim.figure_ids:
                figure = figure_lookup.get(figure_id)
                if figure is not None:
                    figure_references.append(f"Figure {figure.id}: {figure.caption}.")
                else:
                    figure_references.append(f"Figure {figure_id}.")

            paragraph_text = self._draft_paragraph_text(
                claim=claim,
                citations=citation_text,
                figure_references=figure_references,
                guidance_messages=guidance_messages or [],
            )

            paragraphs.append(Paragraph(id=f"{section_id}-p{idx}", text=paragraph_text, claim_ids=[claim.id]))

        return paragraphs

    def _draft_paragraph_text(
        self,
        claim: Claim,
        citations: list[str],
        figure_references: list[str],
        guidance_messages: list[CoordinationMessage],
    ) -> str:
        """Draft paragraph text.

        Parameters
        ----------
        claim : Claim
            Parameter description.
        citations : list[str]
            Parameter description.
        figure_references : list[str]
            Parameter description.
        guidance_messages : list[CoordinationMessage]
            Parameter description.

        Returns
        -------
        str
            Return value description.
        """
        if self.llm_client is not None:
            narrative_profile = _narrative_profile_block(
                tone=self.tone,
                style=self.style,
                audience=self.audience,
                discipline=self.discipline,
                genre=self.genre,
                language=self.language,
            )
            guidance_context = ""
            if guidance_messages:
                guidance_lines = [f"- {message.content}" for message in guidance_messages]
                guidance_context = "\nCoordination guidance:\n" + "\n".join(guidance_lines)
            background_context = ""
            if self.background_prompt:
                background_context = "\nBackground guidance:\n- " + self.background_prompt
            prompt = (
                "Write one concise paragraph from this claim. "
                "Preserve citation brackets exactly as provided.\n\n"
                f"{narrative_profile}\n"
                f"Claim: {claim.text}\n"
                f"Citations: {' '.join(citations)}\n"
                f"Figure references: {' '.join(figure_references)}"
                f"{background_context}"
                f"{guidance_context}"
            )
            text = self.llm_client.generate(prompt=prompt, system_prompt=self.llm_system_prompt).strip()
            if text:
                return text

        narrative_instruction = _narrative_instruction_prefix(
            tone=self.tone,
            style=self.style,
            audience=self.audience,
            discipline=self.discipline,
            genre=self.genre,
            language=self.language,
        )
        paragraph_text = _safe_format(
            self.paragraph_template,
            narrative_instruction=narrative_instruction,
            narrative_tone=self.tone,
            narrative_style=self.style,
            narrative_audience=self.audience,
            narrative_discipline=self.discipline,
            narrative_genre=self.genre,
            narrative_language=self.language,
            claim_text=claim.text,
            citations=" ".join(citations),
            figure_references=" ".join(figure_references),
        ).strip()
        if narrative_instruction and not paragraph_text.startswith(narrative_instruction):
            return (narrative_instruction + paragraph_text).strip()
        return paragraph_text


@dataclass(slots=True)
class OutlineExpanderAgent:
    """Expands a preliminary outline and proposes prompt templates."""

    max_subsections_per_section: int = 3
    functional_forms_payload: dict[str, object] | None = None
    functional_form_id: str = ""
    llm_client: LLMClient | None = None
    llm_system_prompt: str | None = None
    tone: str = "neutral"
    style: str = "analytical"
    audience: str = "general"
    discipline: str = "interdisciplinary"
    genre: str = "scholarly_manuscript"
    language: str = "English"
    background_prompt: str = ""

    def expand(self, preliminary_outline: dict | list[dict]) -> dict:
        """Expand.

        Parameters
        ----------
        preliminary_outline : dict | list[dict]
            Parameter description.

        Returns
        -------
        dict
            Return value description.
        """
        chapters = _normalize_outline_chapters(preliminary_outline)
        selected_form = select_functional_form(
            functional_forms_payload=_coerce_mapping(self.functional_forms_payload),
            form_id=self.functional_form_id,
        )
        element_map = _functional_form_element_map(selected_form)
        chapter_pattern = _chapter_pattern_rows(selected_form)
        form_prompt_hints = _outline_prompt_hints(selected_form)

        expanded_chapters = []

        for chapter_idx, chapter in enumerate(chapters, start=1):
            chapter_title = str(chapter.get("title", "")).strip() or f"Chapter {chapter_idx}"
            chapter_type, chapter_pattern_row = _resolve_chapter_pattern_row(
                chapter=chapter,
                chapter_idx=chapter_idx,
                chapter_count=len(chapters),
                chapter_pattern=chapter_pattern,
            )
            section_expectations = _section_expectations_for_chapter(
                chapter=chapter,
                chapter_idx=chapter_idx,
                chapter_pattern_row=chapter_pattern_row,
                element_map=element_map,
            )
            section_titles = [row["title"] for row in section_expectations]
            expected_elements = [row["element_id"] for row in section_expectations if row["element_id"]]
            section_details = []
            for section_idx, section_row in enumerate(section_expectations, start=1):
                section_title = section_row["title"]
                element_id = section_row["element_id"]
                element_definition = section_row["definition"]
                objective = self._section_objective(
                    chapter_title=chapter_title,
                    section_title=section_title,
                    section_idx=section_idx,
                    chapter_type=chapter_type,
                    selected_form=selected_form,
                    element_id=element_id,
                    element_definition=element_definition,
                )
                subsections = self._section_subsections(
                    chapter_title=chapter_title,
                    section_title=section_title,
                    selected_form=selected_form,
                    element_id=element_id,
                )
                section_details.append(
                    {
                        "title": section_title,
                        "functional_element_id": element_id,
                        "objective": objective,
                        "subsections": subsections,
                        "evidence_focus": _evidence_focus_for_section(section_title),
                        "deliverables": _deliverables_for_section(section_title),
                    }
                )

            expanded_chapters.append(
                {
                    "title": chapter_title,
                    "functional_form_id": _string(selected_form.get("id")),
                    "functional_chapter_type": chapter_type,
                    "goal": _chapter_goal(chapter_title, chapter_type=chapter_type, selected_form=selected_form),
                    "expected_elements": expected_elements,
                    "sections": section_titles,
                    "section_details": section_details,
                }
            )

        prompts = dict(OUTLINE_EXPANSION_PROMPTS_DEFAULT)
        prompts.update(self._expanded_prompt_hints(expanded_chapters))
        prompts.update(form_prompt_hints)

        return {
            "chapters": expanded_chapters,
            "prompts": prompts,
            "diagnostics": {
                "chapters": len(expanded_chapters),
                "sections": sum(len(chapter.get("sections", [])) for chapter in expanded_chapters),
                "functional_form_id": _string(selected_form.get("id")),
                "functional_form_name": _string(selected_form.get("name")),
            },
        }

    def _section_objective(
        self,
        chapter_title: str,
        section_title: str,
        section_idx: int,
        chapter_type: str,
        selected_form: dict[str, Any],
        element_id: str,
        element_definition: str,
    ) -> str:
        """Section objective.

        Parameters
        ----------
        chapter_title : str
            Parameter description.
        section_title : str
            Parameter description.
        section_idx : int
            Parameter description.
        chapter_type : str
            Functional chapter role label.
        selected_form : dict[str, Any]
            Selected functional form object.
        element_id : str
            Functional-form element id expected for this section.
        element_definition : str
            Functional-form element definition.

        Returns
        -------
        str
            Return value description.
        """
        template = _section_objective_template(selected_form, element_id)
        if template:
            fallback = _safe_format(
                template,
                chapter_title=chapter_title,
                section_title=section_title,
                element_id=element_id,
                element_definition=element_definition,
                chapter_type=chapter_type,
            )
        elif element_definition:
            fallback = (
                f"Develop '{section_title}' for chapter '{chapter_title}' by covering: {element_definition}."
            )
        else:
            fallback = (
                f"Establish the argument for '{section_title}' in the context of '{chapter_title}', "
                "with clear definitions, evidence, and implications."
            )
        if self.llm_client is None:
            return fallback

        narrative_profile = _narrative_profile_block(
            tone=self.tone,
            style=self.style,
            audience=self.audience,
            discipline=self.discipline,
            genre=self.genre,
            language=self.language,
        )
        prompt = (
            "Write one concise objective sentence for a section in a technical long-form outline.\n"
            f"{narrative_profile}\n"
            f"Chapter: {chapter_title}\n"
            f"Section: {section_title}\n"
            f"Section index: {section_idx}\n"
            f"Functional chapter type: {chapter_type or 'unspecified'}\n"
            f"Functional element id: {element_id or 'unspecified'}\n"
            f"Functional element definition: {element_definition or 'unspecified'}\n"
            "Constraints: one sentence, action-oriented, no bullet points."
        )
        if self.background_prompt:
            prompt += f"\nBackground guidance: {self.background_prompt}\n"
        response = self.llm_client.generate(prompt=prompt, system_prompt=self.llm_system_prompt).strip()
        return response or fallback

    def _section_subsections(
        self,
        chapter_title: str,
        section_title: str,
        selected_form: dict[str, Any],
        element_id: str,
    ) -> list[str]:
        """Section subsections.

        Parameters
        ----------
        chapter_title : str
            Parameter description.
        section_title : str
            Parameter description.
        selected_form : dict[str, Any]
            Selected functional form object.
        element_id : str
            Functional-form element id expected for the section.

        Returns
        -------
        list[str]
            Return value description.
        """
        fallback = _form_subsection_templates(
            selected_form=selected_form,
            element_id=element_id,
            max_items=self.max_subsections_per_section,
        )
        if not fallback:
            fallback = _default_subsections_for_section(section_title, max_items=self.max_subsections_per_section)
        if self.llm_client is None:
            return fallback

        narrative_profile = _narrative_profile_block(
            tone=self.tone,
            style=self.style,
            audience=self.audience,
            discipline=self.discipline,
            genre=self.genre,
            language=self.language,
        )
        prompt = (
            "Return subsection titles for this section, one per line.\n"
            f"{narrative_profile}\n"
            f"Chapter: {chapter_title}\n"
            f"Section: {section_title}\n"
            f"Functional element id: {element_id or 'unspecified'}\n"
            f"Max subsection count: {self.max_subsections_per_section}\n"
            "Constraints: title case, no numbering."
        )
        if self.background_prompt:
            prompt += f"\nBackground guidance: {self.background_prompt}\n"
        response = self.llm_client.generate(prompt=prompt, system_prompt=self.llm_system_prompt).strip()
        parsed = _parse_lines(response, limit=self.max_subsections_per_section)
        return parsed or fallback

    def _expanded_prompt_hints(self, chapters: list[dict]) -> dict[str, str]:
        """Expanded prompt hints.

        Parameters
        ----------
        chapters : list[dict]
            Parameter description.

        Returns
        -------
        dict[str, str]
            Return value description.
        """
        first_section = ""
        for chapter in chapters:
            sections = chapter.get("sections", [])
            if sections:
                first_section = str(sections[0])
                break
        if not first_section:
            first_section = "the section"
        return {
            "section_planning_template": (
                f"Use the section objective and subsection plan to draft claims for {first_section}."
            ),
            "synthesis_template": "Synthesize key claims into a coherent transition to the next section.",
        }


@dataclass(slots=True)
class SectionAgent:
    """Compose a section by coordinating claim and paragraph agents.

    Parameters
    ----------
    claim_agent : ClaimAuthorAgent
        Agent that drafts section claims.
    paragraph_agent : ParagraphAgent
        Agent that synthesizes section paragraphs from claims/evidence.
    """

    claim_agent: ClaimAuthorAgent
    paragraph_agent: ParagraphAgent

    def draft(
        self,
        chapter_id: str,
        index: int,
        section_title: str,
        hits: list[RetrievalHit],
        graph: KnowledgeGraph,
        max_figures: int = 3,
        message_bus: MessageBus | None = None,
    ) -> Section:
        """Draft.

        Parameters
        ----------
        chapter_id : str
            Parameter description.
        index : int
            Parameter description.
        section_title : str
            Parameter description.
        hits : list[RetrievalHit]
            Parameter description.
        graph : KnowledgeGraph
            Parameter description.
        max_figures : int
            Parameter description.
        message_bus : MessageBus | None
            Parameter description.

        Returns
        -------
        Section
            Return value description.
        """
        section_id = f"{chapter_id}-s{index}"
        entities = graph.entities_for_query(section_title)
        figures = graph.figures_for_query(query=section_title, max_items=max(0, max_figures))
        guidance_messages: list[CoordinationMessage] = []
        if message_bus is not None:
            guidance_messages = message_bus.messages_for("claim_author_agent") + message_bus.messages_for("paragraph_agent")
        claims = self.claim_agent.draft(
            section_id=section_id,
            section_title=section_title,
            hits=hits,
            entities=entities,
            figures=figures,
            guidance_messages=guidance_messages,
        )
        paragraphs = self.paragraph_agent.draft_with_figures(
            section_id=section_id,
            claims=claims,
            hits=hits,
            figure_lookup={figure.id: figure for figure in figures},
            guidance_messages=guidance_messages,
        )
        return Section(id=section_id, title=section_title, paragraphs=paragraphs, claims=claims, figures=figures)


@dataclass(slots=True)
class CitationReviewerAgent:
    """Ensures every claim references a known evidence source."""

    def review(self, sections: list[Section], known_source_ids: set[str]) -> list[str]:
        """Review.

        Parameters
        ----------
        sections : list[Section]
            Parameter description.
        known_source_ids : set[str]
            Parameter description.

        Returns
        -------
        list[str]
            Return value description.
        """
        issues: list[str] = []
        for section in sections:
            for claim in section.claims:
                for evidence_id in claim.evidence_ids:
                    if evidence_id not in known_source_ids:
                        issues.append(
                            f"Claim {claim.id} in section {section.id} references unknown source {evidence_id}."
                        )
        return issues


@dataclass(slots=True)
class CoherenceReviewerAgent:
    """Flags low-variety prose that often indicates degenerate generation."""

    def review(self, sections: list[Section]) -> list[str]:
        """Review.

        Parameters
        ----------
        sections : list[Section]
            Parameter description.

        Returns
        -------
        list[str]
            Return value description.
        """
        issues: list[str] = []
        seen: set[str] = set()
        for section in sections:
            for paragraph in section.paragraphs:
                normalized = " ".join(paragraph.text.lower().split())
                if normalized in seen:
                    issues.append(f"Paragraph {paragraph.id} duplicates existing text.")
                seen.add(normalized)
        return issues


@dataclass(slots=True)
class FigureReviewerAgent:
    """Ensures figure references point to known figure nodes."""

    def review(self, sections: list[Section], known_figure_ids: set[str]) -> list[str]:
        """Review.

        Parameters
        ----------
        sections : list[Section]
            Parameter description.
        known_figure_ids : set[str]
            Parameter description.

        Returns
        -------
        list[str]
            Return value description.
        """
        issues: list[str] = []
        for section in sections:
            for claim in section.claims:
                for figure_id in claim.figure_ids:
                    if figure_id not in known_figure_ids:
                        issues.append(
                            f"Claim {claim.id} in section {section.id} references unknown figure {figure_id}."
                        )
        return issues


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        """Missing.

        Parameters
        ----------
        key : str
            Parameter description.

        Returns
        -------
        str
            Return value description.
        """
        return ""


def _safe_format(template: str, **values: str) -> str:
    """Safe format.

    Parameters
    ----------
    template : str
        Parameter description.
    **values : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    return template.format_map(_SafeFormatDict(values))


def _narrative_profile_block(
    tone: str,
    style: str,
    audience: str,
    discipline: str,
    genre: str,
    language: str,
) -> str:
    """Build a narrative-profile block for generation prompts.

    Parameters
    ----------
    tone : str
        Requested narrative tone.
    style : str
        Requested narrative style.
    audience : str
        Intended audience.
    discipline : str
        Target discipline.
    genre : str
        Target narrative genre.
    language : str
        Requested output language.

    Returns
    -------
    str
        Prompt block encoding narrative preferences.
    """
    return (
        "Narrative profile:\n"
        f"- Tone: {tone}\n"
        f"- Style: {style}\n"
        f"- Audience: {audience}\n"
        f"- Discipline: {discipline}\n"
        f"- Genre: {genre}\n"
        f"- Language: {language}\n"
        "Apply this profile while keeping claims evidence-grounded."
    )


def _narrative_instruction_prefix(
    tone: str,
    style: str,
    audience: str,
    discipline: str,
    genre: str,
    language: str,
) -> str:
    """Create a deterministic narrative prefix for template-based outputs.

    Parameters
    ----------
    tone : str
        Requested narrative tone.
    style : str
        Requested narrative style.
    audience : str
        Intended audience.
    discipline : str
        Target discipline.
    genre : str
        Target narrative genre.
    language : str
        Requested output language.

    Returns
    -------
    str
        Prefix text, or an empty string for default profile values.
    """
    defaults = ("neutral", "analytical", "general", "interdisciplinary", "scholarly_manuscript", "English")
    current = (tone, style, audience, discipline, genre, language)
    if current == defaults:
        return ""
    return (
        f"For {audience} readers in {discipline} writing a {genre}, using a {tone} tone and {style} style in {language}: "
    )


def _normalize_outline_chapters(preliminary_outline: dict | list[dict]) -> list[dict]:
    """Normalize outline chapters.

    Parameters
    ----------
    preliminary_outline : dict | list[dict]
        Parameter description.

    Returns
    -------
    list[dict]
        Return value description.
    """
    if isinstance(preliminary_outline, dict):
        chapters = preliminary_outline.get("chapters", [])
        if isinstance(chapters, list):
            return [chapter for chapter in chapters if isinstance(chapter, dict)]
        return []
    if isinstance(preliminary_outline, list):
        return [chapter for chapter in preliminary_outline if isinstance(chapter, dict)]
    return []


def _section_titles_for_chapter(chapter: dict, chapter_idx: int) -> list[str]:
    """Section titles for chapter.

    Parameters
    ----------
    chapter : dict
        Parameter description.
    chapter_idx : int
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    raw_sections = chapter.get("sections", [])
    if not isinstance(raw_sections, list):
        raw_sections = []

    titles: list[str] = []
    for section in raw_sections:
        if isinstance(section, str):
            title = section.strip()
            if title:
                titles.append(title)
            continue
        if isinstance(section, dict):
            title = str(section.get("title", "")).strip()
            if title:
                titles.append(title)

    if titles:
        return titles
    return [f"Section {chapter_idx}.1", f"Section {chapter_idx}.2"]


def _chapter_pattern_rows(selected_form: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized chapter-pattern rows for a selected functional form.

    Parameters
    ----------
    selected_form : dict[str, Any]
        Functional-form configuration.

    Returns
    -------
    list[dict[str, Any]]
        Chapter pattern row mappings.
    """
    rows = selected_form.get("chapter_pattern", [])
    if not isinstance(rows, list):
        rows = selected_form.get("chapter_or_section_pattern", selected_form.get("section_pattern", []))
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mapped = dict(row)
        required = mapped.get("required_sections")
        if not isinstance(required, list):
            required = mapped.get("required_elements")
        if isinstance(required, list):
            mapped["required_sections"] = [_string(value) for value in required if _string(value)]
        else:
            mapped["required_sections"] = []
        normalized.append(mapped)
    return normalized


def _functional_form_element_map(selected_form: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build an element-id lookup from functional-form ontology definitions.

    Parameters
    ----------
    selected_form : dict[str, Any]
        Functional-form configuration.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping from element id to normalized metadata.
    """
    elements = selected_form.get("elements", [])
    if not isinstance(elements, list):
        return {}
    mapped: dict[str, dict[str, str]] = {}
    for row in elements:
        if not isinstance(row, dict):
            continue
        element_id = _string(row.get("id"))
        if not element_id:
            continue
        mapped[element_id] = {
            "id": element_id,
            "label": _string(row.get("label")) or _humanize_id(element_id),
            "definition": _string(row.get("definition")),
        }
    return mapped


def _resolve_chapter_pattern_row(
    chapter: dict[str, Any],
    chapter_idx: int,
    chapter_count: int,
    chapter_pattern: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Resolve functional chapter type/pattern row for a chapter index and title.

    Parameters
    ----------
    chapter : dict[str, Any]
        Chapter outline spec.
    chapter_idx : int
        1-based chapter index.
    chapter_count : int
        Number of chapters in outline.
    chapter_pattern : list[dict[str, Any]]
        Chapter pattern rows from selected form.

    Returns
    -------
    tuple[str, dict[str, Any]]
        Resolved chapter type label and corresponding pattern row.
    """
    if not chapter_pattern:
        return "", {}

    explicit_type = _string(chapter.get("functional_chapter_type"))
    if explicit_type:
        for row in chapter_pattern:
            if _string(row.get("type")) == explicit_type:
                return explicit_type, row

    chapter_title = _string(chapter.get("title")).lower()
    if chapter_title:
        for row in chapter_pattern:
            chapter_type = _string(row.get("type"))
            chapter_tokens = _tokenize(chapter_title)
            type_tokens = _tokenize(chapter_type.replace("_", " "))
            if chapter_tokens and type_tokens and len(chapter_tokens & type_tokens) / max(1, len(type_tokens)) >= 0.5:
                return chapter_type, row

    first_row = chapter_pattern[0]
    last_row = chapter_pattern[-1]
    repeatable_rows = [row for row in chapter_pattern if bool(row.get("repeatable", False))]

    if chapter_idx == 1:
        return _string(first_row.get("type")), first_row
    if chapter_idx == chapter_count and chapter_count > 1:
        return _string(last_row.get("type")), last_row
    if repeatable_rows:
        row = repeatable_rows[0]
        return _string(row.get("type")), row

    fallback_index = min(max(chapter_idx - 1, 0), len(chapter_pattern) - 1)
    row = chapter_pattern[fallback_index]
    return _string(row.get("type")), row


def _section_expectations_for_chapter(
    chapter: dict[str, Any],
    chapter_idx: int,
    chapter_pattern_row: dict[str, Any],
    element_map: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Resolve section title + ontology-element expectations for an expanded chapter.

    Parameters
    ----------
    chapter : dict[str, Any]
        Chapter outline spec.
    chapter_idx : int
        1-based chapter index.
    chapter_pattern_row : dict[str, Any]
        Resolved chapter-pattern row for this chapter.
    element_map : dict[str, dict[str, str]]
        Functional-form element lookup.

    Returns
    -------
    list[dict[str, str]]
        Ordered section expectation rows with ``title``, ``element_id``, and ``definition``.
    """
    required_ids = _string_list(chapter_pattern_row.get("required_sections"))
    if not required_ids:
        required_ids = _string_list(chapter_pattern_row.get("required_elements"))
    explicit_rows: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    seen_ids: set[str] = set()

    section_details = chapter.get("section_details", [])
    if isinstance(section_details, list):
        for row in section_details:
            if not isinstance(row, dict):
                continue
            title = _string(row.get("title"))
            if not title:
                continue
            element_id = _string(row.get("functional_element_id")) or _match_title_to_element(title, element_map)
            normalized = title.lower()
            if normalized in seen_titles:
                continue
            seen_titles.add(normalized)
            if element_id:
                seen_ids.add(element_id)
            explicit_rows.append(
                {
                    "title": title,
                    "element_id": element_id,
                    "definition": _string(element_map.get(element_id, {}).get("definition")),
                }
            )

    raw_sections = chapter.get("sections", [])
    if isinstance(raw_sections, list):
        for row in raw_sections:
            title = ""
            element_id = ""
            if isinstance(row, str):
                title = _string(row)
            elif isinstance(row, dict):
                title = _string(row.get("title"))
                element_id = _string(row.get("functional_element_id"))
            if not title:
                continue
            normalized = title.lower()
            if normalized in seen_titles:
                continue
            element_id = element_id or _match_title_to_element(title, element_map)
            seen_titles.add(normalized)
            if element_id:
                seen_ids.add(element_id)
            explicit_rows.append(
                {
                    "title": title,
                    "element_id": element_id,
                    "definition": _string(element_map.get(element_id, {}).get("definition")),
                }
            )

    if not explicit_rows:
        for required_id in required_ids:
            title = _string(element_map.get(required_id, {}).get("label")) or _humanize_id(required_id)
            explicit_rows.append(
                {
                    "title": title,
                    "element_id": required_id,
                    "definition": _string(element_map.get(required_id, {}).get("definition")),
                }
            )

    for required_id in required_ids:
        if required_id in seen_ids:
            continue
        title = _string(element_map.get(required_id, {}).get("label")) or _humanize_id(required_id)
        normalized = title.lower()
        if normalized in seen_titles:
            continue
        seen_titles.add(normalized)
        explicit_rows.append(
            {
                "title": title,
                "element_id": required_id,
                "definition": _string(element_map.get(required_id, {}).get("definition")),
            }
        )

    if explicit_rows:
        return explicit_rows
    return [
        {"title": f"Section {chapter_idx}.1", "element_id": "", "definition": ""},
        {"title": f"Section {chapter_idx}.2", "element_id": "", "definition": ""},
    ]


def _outline_prompt_hints(selected_form: dict[str, Any]) -> dict[str, str]:
    """Extract form-specific prompt-hint overrides for outline expansion.

    Parameters
    ----------
    selected_form : dict[str, Any]
        Functional-form configuration.

    Returns
    -------
    dict[str, str]
        Prompt-hint overrides.
    """
    expansion = _coerce_mapping(selected_form.get("outline_expansion"))
    hints = _coerce_mapping(expansion.get("prompt_hints"))
    output: dict[str, str] = {}
    for key, value in hints.items():
        key_text = _string(key)
        value_text = _string(value)
        if key_text and value_text:
            output[key_text] = value_text
    return output


def _section_objective_template(selected_form: dict[str, Any], element_id: str) -> str:
    """Return an objective template for a specific ontology element id.

    Parameters
    ----------
    selected_form : dict[str, Any]
        Functional-form configuration.
    element_id : str
        Element id for the section.

    Returns
    -------
    str
        Objective template string, if present.
    """
    if not element_id:
        return ""
    expansion = _coerce_mapping(selected_form.get("outline_expansion"))
    templates = _coerce_mapping(expansion.get("section_objective_templates"))
    return _string(templates.get(element_id))


def _form_subsection_templates(
    selected_form: dict[str, Any],
    element_id: str,
    max_items: int,
) -> list[str]:
    """Return subsection templates for a form element id.

    Parameters
    ----------
    selected_form : dict[str, Any]
        Functional-form configuration.
    element_id : str
        Element id for the section.
    max_items : int
        Maximum items to return.

    Returns
    -------
    list[str]
        Subsection template titles.
    """
    if not element_id:
        return []
    expansion = _coerce_mapping(selected_form.get("outline_expansion"))
    templates = expansion.get("subsection_templates", {})
    if not isinstance(templates, dict):
        return []
    raw_items = templates.get(element_id, [])
    if not isinstance(raw_items, list):
        return []
    parsed = [_string(item) for item in raw_items if _string(item)]
    return parsed[: max(1, max_items)]


def _match_title_to_element(title: str, element_map: dict[str, dict[str, str]]) -> str:
    """Match a section title to the most likely ontology element id.

    Parameters
    ----------
    title : str
        Section title.
    element_map : dict[str, dict[str, str]]
        Functional-form element lookup.

    Returns
    -------
    str
        Matched element id, if similarity is high enough.
    """
    title_tokens = _tokenize(title)
    if not title_tokens:
        return ""
    best_id = ""
    best_score = 0.0
    for element_id, element in element_map.items():
        for candidate in (element_id, _string(element.get("label")), _string(element.get("definition"))):
            candidate_tokens = _tokenize(candidate)
            if not candidate_tokens:
                continue
            score = len(title_tokens & candidate_tokens) / max(1, len(candidate_tokens))
            if score > best_score:
                best_id = element_id
                best_score = score
    return best_id if best_score >= 0.5 else ""


def _chapter_goal(chapter_title: str, chapter_type: str = "", selected_form: dict[str, Any] | None = None) -> str:
    """Chapter goal.

    Parameters
    ----------
    chapter_title : str
        Parameter description.
    chapter_type : str, optional
        Functional chapter role label.
    selected_form : dict[str, Any] | None, optional
        Selected functional form object.

    Returns
    -------
    str
        Return value description.
    """
    form = _coerce_mapping(selected_form)
    expansion = _coerce_mapping(form.get("outline_expansion"))
    templates = _coerce_mapping(expansion.get("chapter_goal_templates"))
    if chapter_type:
        template = _string(templates.get(chapter_type))
        if template:
            return _safe_format(
                template,
                chapter_title=chapter_title,
                chapter_type=chapter_type,
                form_name=_string(form.get("name")),
            )
    return f"Advance the chapter-level argument for '{chapter_title}' with evidence-backed sections."


def _default_subsections_for_section(section_title: str, max_items: int) -> list[str]:
    """Default subsections for section.

    Parameters
    ----------
    section_title : str
        Parameter description.
    max_items : int
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    patterns: list[str]
    lowered = section_title.lower()
    if lowered.startswith("why "):
        patterns = ["Problem Framing", "Evidence Base", "Practical Implications", "Open Questions"]
    elif "evaluation" in lowered or "review" in lowered:
        patterns = ["Criteria", "Methods", "Results", "Limitations"]
    elif "design" in lowered or "architecture" in lowered:
        patterns = ["Requirements", "Component Design", "Integration", "Tradeoffs"]
    else:
        patterns = ["Scope and Definitions", "Core Mechanisms", "Evidence and Examples", "Implications"]
    return patterns[: max(1, max_items)]


def _evidence_focus_for_section(section_title: str) -> list[str]:
    """Evidence focus for section.

    Parameters
    ----------
    section_title : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    lowered = section_title.lower()
    focus = ["title", "abstract", "authors", "publication"]
    if "evaluation" in lowered:
        focus.append("quantitative results")
    if "design" in lowered or "architecture" in lowered:
        focus.append("system diagrams")
    return focus


def _deliverables_for_section(section_title: str) -> list[str]:
    """Deliverables for section.

    Parameters
    ----------
    section_title : str
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    return [
        f"One thesis claim for {section_title}",
        "Two supporting claims with citations",
        "One transition sentence to adjacent section",
    ]


def _coerce_mapping(value: object) -> dict[str, Any]:
    """Return a mapping value or an empty mapping.

    Parameters
    ----------
    value : object
        Candidate value.

    Returns
    -------
    dict[str, Any]
        Mapping value if input is dict, else empty mapping.
    """
    if isinstance(value, dict):
        return value
    return {}


def _string(value: object) -> str:
    """Convert text-like inputs to stripped strings.

    Parameters
    ----------
    value : object
        Candidate text value.

    Returns
    -------
    str
        Stripped string for text-like values, else empty string.
    """
    if isinstance(value, str):
        return value.strip()
    return ""


def _string_list(value: object) -> list[str]:
    """Normalize a list-like value into non-empty strings.

    Parameters
    ----------
    value : object
        Candidate list value.

    Returns
    -------
    list[str]
        Non-empty normalized strings.
    """
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for row in value:
        text = _string(row)
        if text:
            output.append(text)
    return output


def _tokenize(value: str) -> set[str]:
    """Tokenize free text into lowercase alphanumeric tokens.

    Parameters
    ----------
    value : str
        Input text.

    Returns
    -------
    set[str]
        Token set.
    """
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return {token for token in cleaned.split() if token}


def _humanize_id(value: str) -> str:
    """Convert snake-case identifiers to title-cased labels.

    Parameters
    ----------
    value : str
        Identifier string.

    Returns
    -------
    str
        Human-readable label.
    """
    return " ".join(part.capitalize() for part in value.replace("-", "_").split("_") if part)


def _parse_lines(value: str, limit: int) -> list[str]:
    """Parse lines.

    Parameters
    ----------
    value : str
        Parameter description.
    limit : int
        Parameter description.

    Returns
    -------
    list[str]
        Return value description.
    """
    lines = []
    for raw in value.splitlines():
        cleaned = raw.strip().lstrip("-*0123456789. ").strip()
        if cleaned:
            lines.append(cleaned)
    unique = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(line)
    return unique[: max(1, limit)]
