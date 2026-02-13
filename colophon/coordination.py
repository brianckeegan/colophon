"""Hierarchical coordination agents and message bus for draft hand-offs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Chapter, CoordinationMessage, GapRequest, Section


@dataclass(slots=True)
class MessageBus:
    """In-memory message channel for parent/child hand-offs between agents."""

    messages: list[CoordinationMessage] = field(default_factory=list)

    def send(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        content: str,
        related_id: str = "",
        priority: str = "normal",
    ) -> CoordinationMessage:
        """Send.

        Parameters
        ----------
        sender : str
            Parameter description.
        receiver : str
            Parameter description.
        message_type : str
            Parameter description.
        content : str
            Parameter description.
        related_id : str
            Parameter description.
        priority : str
            Parameter description.

        Returns
        -------
        CoordinationMessage
            Return value description.
        """
        existing = self._find_existing(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content,
            related_id=related_id,
            priority=priority,
        )
        if existing is not None:
            return existing

        message = CoordinationMessage(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content,
            related_id=related_id,
            priority=priority,
        )
        self.messages.append(message)
        return message

    def messages_for(self, receiver: str) -> list[CoordinationMessage]:
        """Messages for.

        Parameters
        ----------
        receiver : str
            Parameter description.

        Returns
        -------
        list[CoordinationMessage]
            Return value description.
        """
        return [message for message in self.messages if message.receiver == receiver]

    def _find_existing(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        content: str,
        related_id: str,
        priority: str,
    ) -> CoordinationMessage | None:
        """Find an existing identical message in the bus.

        Parameters
        ----------
        sender : str
            Message sender id.
        receiver : str
            Message receiver id.
        message_type : str
            Message type label.
        content : str
            Message content.
        related_id : str
            Related entity identifier.
        priority : str
            Priority label.

        Returns
        -------
        CoordinationMessage | None
            Existing message if present; otherwise ``None``.
        """
        for message in self.messages:
            if (
                message.sender == sender
                and message.receiver == receiver
                and message.message_type == message_type
                and message.content == content
                and message.related_id == related_id
                and message.priority == priority
            ):
                return message
        return None


@dataclass(slots=True)
class ParagraphCoordinationAgent:
    """Ensures claim bundles become readable, authoritative, and persuasive paragraphs."""

    functional_form: dict[str, Any] | None = None
    background_prompt: str = ""

    def coordinate(self, section: Section, bus: MessageBus) -> list[GapRequest]:
        """Coordinate.

        Parameters
        ----------
        section : Section
            Parameter description.
        bus : MessageBus
            Parameter description.

        Returns
        -------
        list[GapRequest]
            Return value description.
        """
        gaps: list[GapRequest] = []
        policy = _coordination_policy(self.functional_form, "paragraph")
        concise_suffix = _string(policy.get("concise_suffix")) or " This supports the section's argument with concise evidence."
        min_words = _int(policy.get("min_words"), 12)
        require_citation_markers = bool(policy.get("require_citation_markers", True))
        quality_targets = _string_list(policy.get("quality_targets"))

        for paragraph in section.paragraphs:
            text = paragraph.text.strip()
            if not text:
                paragraph.text = "This paragraph is currently empty and needs sourced claims."
                gaps.append(
                    GapRequest(
                        level="paragraph",
                        component="bibliography",
                        request="Add evidence-bearing sources",
                        rationale="Empty paragraph indicates no grounded content was generated.",
                        related_id=paragraph.id,
                    )
                )
                continue

            if "No sufficiently grounded claims" in text or "Insufficient grounding evidence" in text:
                gaps.append(
                    GapRequest(
                        level="paragraph",
                        component="bibliography",
                        request="Add sources relevant to section",
                        rationale="Paragraph fallback text indicates retrieval could not ground claims.",
                        related_id=section.id,
                    )
                )
                gaps.append(
                    GapRequest(
                        level="paragraph",
                        component="knowledge_graph",
                        request="Add entities/relations aligned to section topics",
                        rationale="Missing graph context limits claim targeting and evidence linkage.",
                        related_id=section.id,
                    )
                )

            if not text.endswith((".", "!", "?")):
                paragraph.text = f"{text}."

            if (
                len(paragraph.text.split()) < min_words
                and "No sufficiently grounded claims" not in paragraph.text
                and concise_suffix not in paragraph.text
            ):
                paragraph.text += concise_suffix

            if require_citation_markers and "[" not in paragraph.text and "No sufficiently grounded claims" not in paragraph.text:
                gaps.append(
                    GapRequest(
                        level="paragraph",
                        component="bibliography",
                        request="Strengthen citation coverage",
                        rationale="Generated paragraph lacks explicit citation markers.",
                        related_id=paragraph.id,
                    )
                )

        status = f"Section '{section.title}' has {len(section.paragraphs)} paragraph(s) after readability pass."
        if quality_targets:
            status += " Targets: " + ", ".join(quality_targets)
        if self.background_prompt:
            status += " Background guidance: " + self.background_prompt
        bus.send(
            sender="paragraph_coordinator",
            receiver="section_coordinator",
            message_type="paragraph_quality_summary",
            content=status,
            related_id=section.id,
        )

        return gaps


@dataclass(slots=True)
class SectionCoordinationAgent:
    """Ensures section paragraphs cohere and align with the outline's intended section."""

    functional_form: dict[str, Any] | None = None
    background_prompt: str = ""

    def prepare_child_guidance(
        self,
        section_title: str,
        bus: MessageBus,
        chapter_id: str,
        chapter_type: str = "",
        expected_element_id: str = "",
    ) -> None:
        """Prepare child guidance.

        Parameters
        ----------
        section_title : str
            Parameter description.
        bus : MessageBus
            Parameter description.
        chapter_id : str
            Parameter description.
        chapter_type : str, optional
            Functional chapter type for this section's parent chapter.
        expected_element_id : str, optional
            Functional-form element id expected for this section.
        """
        guidance_parts = [
            f"Prioritize claims that explicitly support section '{section_title}', "
            "include evidence citations, and reference relevant figures when available."
        ]
        policy = _coordination_policy(self.functional_form, "section")
        quality_targets = _string_list(policy.get("quality_targets"))
        if quality_targets:
            guidance_parts.append("Satisfy section quality targets: " + ", ".join(quality_targets) + ".")
        if chapter_type:
            guidance_parts.append(f"This section belongs to a '{chapter_type}' chapter role.")

        element_map = _element_definition_map(self.functional_form)
        matched_element_id = expected_element_id or _match_title_to_element_id(section_title, element_map)
        if matched_element_id:
            element = element_map.get(matched_element_id, {})
            definition = _string(element.get("definition"))
            if definition:
                guidance_parts.append(
                    f"Ensure claims cover functional-form element '{matched_element_id}': {definition}"
                )
        if self.background_prompt:
            guidance_parts.append("Background guidance: " + self.background_prompt)

        guidance = " ".join(guidance_parts)
        bus.send(
            sender="section_coordinator",
            receiver="claim_author_agent",
            message_type="claim_guidance",
            content=guidance,
            related_id=chapter_id,
        )
        bus.send(
            sender="section_coordinator",
            receiver="paragraph_agent",
            message_type="paragraph_guidance",
            content=(
                "Compose persuasive, readable paragraphs that integrate citations and visual references. "
                + (
                    "Preserve section-level rhetorical coherence from the selected functional form."
                    if self.functional_form
                    else ""
                )
            ).strip(),
            related_id=chapter_id,
        )

    def coordinate(
        self,
        section: Section,
        expected_title: str,
        bus: MessageBus,
        expected_element_id: str = "",
        chapter_type: str = "",
    ) -> list[GapRequest]:
        """Coordinate.

        Parameters
        ----------
        section : Section
            Parameter description.
        expected_title : str
            Parameter description.
        bus : MessageBus
            Parameter description.
        expected_element_id : str, optional
            Functional-form element expected for the section.
        chapter_type : str, optional
            Functional chapter type for the section's parent chapter.

        Returns
        -------
        list[GapRequest]
            Return value description.
        """
        gaps: list[GapRequest] = []
        missing_coverage_suffix = " Additional bibliography and graph coverage is required for this section."

        if section.title != expected_title:
            gaps.append(
                GapRequest(
                    level="section",
                    component="outline",
                    request="Align generated section title with outline",
                    rationale=f"Expected '{expected_title}' but generated '{section.title}'.",
                    related_id=section.id,
                )
            )

        if not section.paragraphs:
            gaps.append(
                GapRequest(
                    level="section",
                    component="bibliography",
                    request="Add sources for section",
                    rationale="No paragraphs were generated for this outlined section.",
                    related_id=section.id,
                )
            )

        if (
            section.paragraphs
            and section.paragraphs[0].text.startswith("No sufficiently grounded claims")
            and missing_coverage_suffix not in section.paragraphs[0].text
        ):
            section.paragraphs[0].text += missing_coverage_suffix

        if expected_element_id:
            element_map = _element_definition_map(self.functional_form)
            element = element_map.get(expected_element_id, {})
            if not _section_mentions_element(section, expected_element_id=expected_element_id, element=element):
                gaps.append(
                    GapRequest(
                        level="section",
                        component="claims",
                        request=f"Cover functional-form element '{expected_element_id}' in section claims.",
                        rationale="Section content does not strongly signal the expected functional-form element.",
                        related_id=section.id,
                    )
                )

        bus.send(
            sender="section_coordinator",
            receiver="chapter_coordinator",
            message_type="section_status",
            content=(
                f"Section '{section.title}' contains {len(section.claims)} claim(s) and {len(section.figures)} figure(s)."
                + (f" Chapter role: {chapter_type}." if chapter_type else "")
            ),
            related_id=section.id,
        )

        return gaps


@dataclass(slots=True)
class ChapterCoordinationAgent:
    """Ensures sections cohere into chapter structure defined by outline."""

    functional_form: dict[str, Any] | None = None
    background_prompt: str = ""

    def prepare_child_guidance(
        self,
        chapter_title: str,
        expected_sections: list[str],
        bus: MessageBus,
        chapter_id: str,
        chapter_type: str = "",
        expected_elements: list[str] | None = None,
    ) -> None:
        """Prepare child guidance.

        Parameters
        ----------
        chapter_title : str
            Parameter description.
        expected_sections : list[str]
            Parameter description.
        bus : MessageBus
            Parameter description.
        chapter_id : str
            Parameter description.
        chapter_type : str, optional
            Functional chapter type for this chapter.
        expected_elements : list[str] | None, optional
            Expected functional-form element ids for this chapter's sections.
        """
        chapter_policy = _coordination_policy(self.functional_form, "chapter")
        quality_targets = _string_list(chapter_policy.get("quality_targets"))
        guidance = f"Maintain chapter '{chapter_title}' scope across sections: " + ", ".join(expected_sections)
        if chapter_type:
            guidance += f". Chapter type: {chapter_type}"
        if expected_elements:
            guidance += ". Expected functional-form elements: " + ", ".join(expected_elements)
        if quality_targets:
            guidance += ". Chapter quality targets: " + ", ".join(quality_targets)
        if self.background_prompt:
            guidance += ". Background guidance: " + self.background_prompt
        bus.send(
            sender="chapter_coordinator",
            receiver="section_coordinator",
            message_type="chapter_outline_guidance",
            content=guidance + ".",
            related_id=chapter_id,
        )

    def coordinate(
        self,
        chapter: Chapter,
        expected_sections: list[str],
        bus: MessageBus,
        chapter_type: str = "",
        expected_elements: list[str] | None = None,
    ) -> list[GapRequest]:
        """Coordinate.

        Parameters
        ----------
        chapter : Chapter
            Parameter description.
        expected_sections : list[str]
            Parameter description.
        bus : MessageBus
            Parameter description.
        chapter_type : str, optional
            Functional chapter type for this chapter.
        expected_elements : list[str] | None, optional
            Expected functional-form element ids for this chapter's sections.

        Returns
        -------
        list[GapRequest]
            Return value description.
        """
        gaps: list[GapRequest] = []
        generated_titles = [section.title for section in chapter.sections]

        for expected in expected_sections:
            if expected not in generated_titles:
                gaps.append(
                    GapRequest(
                        level="chapter",
                        component="outline",
                        request=f"Add missing section '{expected}'",
                        rationale="Outlined section was not generated in chapter output.",
                        related_id=chapter.id,
                    )
                )

        if not chapter.sections:
            gaps.append(
                GapRequest(
                    level="chapter",
                    component="outline",
                    request="Provide section structure",
                    rationale="Chapter has no generated sections and cannot cohere.",
                    related_id=chapter.id,
                )
            )

        element_map = _element_definition_map(self.functional_form)
        if expected_elements:
            generated_titles = [section.title for section in chapter.sections]
            for element_id in expected_elements:
                if not element_id:
                    continue
                element = element_map.get(element_id, {})
                if not _titles_cover_element(generated_titles, element_id=element_id, element=element):
                    gaps.append(
                        GapRequest(
                            level="chapter",
                            component="outline",
                            request=f"Add section for functional-form element '{element_id}'",
                            rationale="Chapter outline is missing an expected ontology element for this chapter role.",
                            related_id=chapter.id,
                        )
                    )

        bus.send(
            sender="chapter_coordinator",
            receiver="book_coordinator",
            message_type="chapter_status",
            content=(
                f"Chapter '{chapter.title}' includes {len(chapter.sections)} section(s)."
                + (f" Chapter role: {chapter_type}." if chapter_type else "")
            ),
            related_id=chapter.id,
        )

        return gaps


@dataclass(slots=True)
class BookCoordinationAgent:
    """Ensures chapters cohere at the manuscript/book level."""

    functional_form: dict[str, Any] | None = None
    background_prompt: str = ""

    def coordinate(self, chapters: list[Chapter], outline: list[dict], bus: MessageBus) -> list[GapRequest]:
        """Coordinate.

        Parameters
        ----------
        chapters : list[Chapter]
            Parameter description.
        outline : list[dict]
            Parameter description.
        bus : MessageBus
            Parameter description.

        Returns
        -------
        list[GapRequest]
            Return value description.
        """
        gaps: list[GapRequest] = []
        expected_titles = [chapter.get("title", "").strip() for chapter in outline]
        generated_titles = [chapter.title for chapter in chapters]

        for expected in expected_titles:
            if expected and expected not in generated_titles:
                gaps.append(
                    GapRequest(
                        level="book",
                        component="outline",
                        request=f"Add missing chapter '{expected}'",
                        rationale="Outline chapter was not produced in the manuscript.",
                    )
                )

        if not chapters:
            gaps.append(
                GapRequest(
                    level="book",
                    component="outline",
                    request="Provide chapter plan",
                    rationale="No chapters were generated; manuscript cannot cohere.",
                )
            )

        for required_type in _required_non_repeatable_chapter_types(self.functional_form):
            if not _chapter_type_signaled(required_type, expected_titles):
                gaps.append(
                    GapRequest(
                        level="book",
                        component="outline",
                        request=f"Add a chapter signaling '{required_type}' role",
                        rationale="Functional-form chapter pattern expects this non-repeatable chapter type.",
                    )
                )

        if len(chapters) >= 2:
            form_name = _string(_dict(self.functional_form).get("name"))
            bus.send(
                sender="book_coordinator",
                receiver="chapter_coordinator",
                message_type="book_flow_guidance",
                content=(
                    "Ensure chapter transitions reinforce a cumulative argument across the manuscript."
                    + (f" Align transitions to functional form '{form_name}'." if form_name else "")
                    + (f" Background guidance: {self.background_prompt}" if self.background_prompt else "")
                ),
            )

        return gaps


def _coordination_policy(functional_form: dict[str, Any] | None, level: str) -> dict[str, Any]:
    """Return a level-specific coordination policy from a functional form.

    Parameters
    ----------
    functional_form : dict[str, Any] | None
        Selected functional-form object.
    level : str
        Coordination level key (paragraph/section/chapter/book).

    Returns
    -------
    dict[str, Any]
        Level-specific policy mapping, or an empty mapping.
    """
    root = _dict(functional_form)
    ontology = _dict(root.get("coordination_ontology"))
    return _dict(ontology.get(level))


def _element_definition_map(functional_form: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    """Build an element-id lookup for functional-form ontology entries.

    Parameters
    ----------
    functional_form : dict[str, Any] | None
        Selected functional form.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping from element id to element metadata.
    """
    element_map: dict[str, dict[str, str]] = {}
    elements = _dict(functional_form).get("elements", [])
    if not isinstance(elements, list):
        return element_map
    for row in elements:
        if not isinstance(row, dict):
            continue
        element_id = _string(row.get("id"))
        if not element_id:
            continue
        element_map[element_id] = {
            "id": element_id,
            "label": _string(row.get("label")) or _humanize_identifier(element_id),
            "definition": _string(row.get("definition")),
        }
    return element_map


def _match_title_to_element_id(section_title: str, element_map: dict[str, dict[str, str]]) -> str:
    """Match a section title to the closest functional-form element id.

    Parameters
    ----------
    section_title : str
        Section heading text.
    element_map : dict[str, dict[str, str]]
        Functional element lookup.

    Returns
    -------
    str
        Matched element id, or an empty string if no match is strong enough.
    """
    title_tokens = _tokenize(section_title)
    if not title_tokens:
        return ""
    best_id = ""
    best_score = 0.0
    for element_id, element in element_map.items():
        candidates = [element_id, _string(element.get("label")), _string(element.get("definition"))]
        for candidate in candidates:
            candidate_tokens = _tokenize(candidate)
            if not candidate_tokens:
                continue
            overlap = len(title_tokens & candidate_tokens) / max(1, len(candidate_tokens))
            if overlap > best_score:
                best_score = overlap
                best_id = element_id
    return best_id if best_score >= 0.5 else ""


def _section_mentions_element(section: Section, expected_element_id: str, element: dict[str, str]) -> bool:
    """Check whether a section's generated content reflects an expected element.

    Parameters
    ----------
    section : Section
        Generated section object.
    expected_element_id : str
        Expected functional-form element id.
    element : dict[str, str]
        Element metadata mapping.

    Returns
    -------
    bool
        ``True`` when the element is reflected in section title/claims/paragraphs.
    """
    corpus = " ".join(
        [section.title]
        + [claim.text for claim in section.claims]
        + [paragraph.text for paragraph in section.paragraphs]
    )
    text_tokens = _tokenize(corpus)
    if not text_tokens:
        return False
    candidates = [
        expected_element_id,
        _string(element.get("label")),
        _string(element.get("definition")),
    ]
    for candidate in candidates:
        candidate_tokens = _tokenize(candidate)
        if not candidate_tokens:
            continue
        overlap = len(candidate_tokens & text_tokens) / max(1, len(candidate_tokens))
        if overlap >= 0.5:
            return True
    return False


def _titles_cover_element(titles: list[str], element_id: str, element: dict[str, str]) -> bool:
    """Check whether chapter section titles cover an expected ontology element.

    Parameters
    ----------
    titles : list[str]
        Section title list.
    element_id : str
        Functional-form element id.
    element : dict[str, str]
        Element metadata mapping.

    Returns
    -------
    bool
        ``True`` if any title overlaps strongly with the expected element.
    """
    title_tokens = [_tokenize(title) for title in titles]
    if not title_tokens:
        return False
    candidates = [element_id, _string(element.get("label"))]
    for candidate in candidates:
        candidate_tokens = _tokenize(candidate)
        if not candidate_tokens:
            continue
        for tokens in title_tokens:
            overlap = len(tokens & candidate_tokens) / max(1, len(candidate_tokens))
            if overlap >= 0.5:
                return True
    return False


def _required_non_repeatable_chapter_types(functional_form: dict[str, Any] | None) -> list[str]:
    """Collect non-repeatable chapter types required by a functional form.

    Parameters
    ----------
    functional_form : dict[str, Any] | None
        Selected functional form.

    Returns
    -------
    list[str]
        Required non-repeatable chapter-type ids.
    """
    root = _dict(functional_form)
    chapter_pattern = root.get("chapter_pattern", [])
    if not isinstance(chapter_pattern, list):
        chapter_pattern = root.get("chapter_or_section_pattern", root.get("section_pattern", []))
    if not isinstance(chapter_pattern, list):
        return []
    required: list[str] = []
    for row in chapter_pattern:
        if not isinstance(row, dict):
            continue
        chapter_type = _string(row.get("type"))
        if not chapter_type:
            continue
        if bool(row.get("repeatable", False)):
            continue
        required.append(chapter_type)
    return required


def _chapter_type_signaled(chapter_type: str, chapter_titles: list[str]) -> bool:
    """Check whether chapter titles signal a target chapter role.

    Parameters
    ----------
    chapter_type : str
        Chapter-type id from the functional form.
    chapter_titles : list[str]
        Chapter titles from outline/manuscript.

    Returns
    -------
    bool
        ``True`` if a title likely represents the chapter role.
    """
    keywords = {
        "introduction": ["intro", "introduction", "foundations", "background"],
        "conclusion": ["conclusion", "closing", "final", "limits"],
        "synthesis_chapter": ["synthesis", "integration", "cross-phase", "discussion"],
    }
    chapter_tokens = [_tokenize(title) for title in chapter_titles]
    target_tokens = _tokenize(chapter_type.replace("_", " "))
    candidate_keywords = _tokenize(" ".join(keywords.get(chapter_type, [])))
    for tokens in chapter_tokens:
        if target_tokens and len(tokens & target_tokens) / max(1, len(target_tokens)) >= 0.5:
            return True
        if candidate_keywords and len(tokens & candidate_keywords) > 0:
            return True
    return False


def _tokenize(value: str) -> set[str]:
    """Tokenize free text into lowercase alphanumeric terms.

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


def _humanize_identifier(value: str) -> str:
    """Convert a snake-case identifier into title-cased text.

    Parameters
    ----------
    value : str
        Identifier text.

    Returns
    -------
    str
        Human-readable label.
    """
    return " ".join(part.capitalize() for part in value.replace("-", "_").split("_") if part)


def _dict(value: object) -> dict[str, Any]:
    """Return a mapping value or an empty mapping.

    Parameters
    ----------
    value : object
        Arbitrary value.

    Returns
    -------
    dict[str, Any]
        Mapping if input is a dict; otherwise an empty dict.
    """
    if isinstance(value, dict):
        return value
    return {}


def _string(value: object) -> str:
    """Coerce a value into a stripped string.

    Parameters
    ----------
    value : object
        Arbitrary input value.

    Returns
    -------
    str
        Stripped string representation for textual inputs.
    """
    if isinstance(value, str):
        return value.strip()
    return ""


def _string_list(value: object) -> list[str]:
    """Normalize a value to a non-empty list of strings.

    Parameters
    ----------
    value : object
        Candidate list input.

    Returns
    -------
    list[str]
        Filtered list of non-empty strings.
    """
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        text = _string(item)
        if text:
            output.append(text)
    return output


def _int(value: object, default: int) -> int:
    """Convert a numeric-like value to int with fallback.

    Parameters
    ----------
    value : object
        Candidate numeric input.
    default : int
        Fallback value.

    Returns
    -------
    int
        Parsed integer or fallback.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default
