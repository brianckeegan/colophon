"""Primary multi-agent orchestration pipeline for Colophon manuscript generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents import (
    CLAIM_TEMPLATE_DEFAULT,
    EMPTY_SECTION_TEMPLATE_DEFAULT,
    FIGURE_REFERENCE_TEMPLATE_DEFAULT,
    PARAGRAPH_TEMPLATE_DEFAULT,
    CitationReviewerAgent,
    ClaimAuthorAgent,
    CoherenceReviewerAgent,
    FigureReviewerAgent,
    OutlineExpanderAgent,
    ParagraphAgent,
    SectionAgent,
)
from .coordination import (
    BookCoordinationAgent,
    ChapterCoordinationAgent,
    MessageBus,
    ParagraphCoordinationAgent,
    SectionCoordinationAgent,
)
from .genre_ontology import build_genre_ontology_context
from .graph import KnowledgeGraph
from .functional_forms import run_soft_validation, select_functional_form
from .writing_ontology import build_writing_ontology_context, run_writing_ontology_validation
from .kg_update import KGUpdateConfig, KnowledgeGraphGeneratorUpdater
from .llm import LLMClient
from .models import Chapter, GapRequest, Manuscript, RecommendationProposal, Source
from .recommendations import PaperRecommendationWorkflow, PaperSearchClient, RecommendationConfig
from .retrieval import SimpleRetriever


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for end-to-end Colophon pipeline execution.

    Parameters
    ----------
    top_k : int
        Number of retrieved sources considered per section.
    max_figures_per_section : int
        Maximum graph figure nodes attached per section.
    title : str
        Manuscript title.
    prompt_templates : dict[str, str]
        Optional prompt template overrides.
    llm_client : LLMClient | None
        Optional LLM client for claim/paragraph generation.
    llm_system_prompt : str
        Optional system prompt for LLM calls.
    enable_coordination_agents : bool
        Toggle hierarchical coordination/editing agents.
    enable_paper_recommendations : bool
        Toggle related-paper recommendation workflow.
    recommendation_config : RecommendationConfig
        Recommendation retrieval/scoring configuration.
    recommendation_client : PaperSearchClient | None
        Optional injected recommendation client.
    enable_kg_updates : bool
        Toggle bibliography-driven KG updates.
    kg_update_config : KGUpdateConfig
        KG updater configuration.
    kg_updater : KnowledgeGraphGeneratorUpdater | None
        Optional injected KG updater implementation.
    enable_outline_expander : bool
        Toggle outline expansion before drafting.
    outline_expander : OutlineExpanderAgent | None
        Optional injected outline expander.
    enable_soft_validation : bool
        Toggle functional-form soft validation.
    functional_forms : dict[str, object] | None
        Functional-form ontology payload.
    functional_form_id : str
        Selected functional-form id.
    max_soft_validation_findings : int
        Soft-validation finding cap.
    writing_ontology : dict[str, object] | None
        Companion writing ontology payload.
    max_writing_ontology_findings : int
        Writing-ontology finding cap.
    genre_ontology : dict[str, object] | None
        Genre ontology payload.
    genre_profile_id : str
        Selected genre profile id.
    narrative_tone : str
        Narrative tone hint.
    narrative_style : str
        Narrative style hint.
    narrative_audience : str
        Target audience hint.
    narrative_discipline : str
        Discipline framing hint.
    narrative_genre : str
        Genre framing hint.
    narrative_language : str
        Target language hint.
    coordination_max_revision_iterations : int
        Maximum coordination revision passes.
    """

    top_k: int = 3
    max_figures_per_section: int = 2
    title: str = "Untitled Manuscript"
    prompt_templates: dict[str, str] = field(default_factory=dict)
    llm_client: LLMClient | None = None
    llm_system_prompt: str = ""
    enable_coordination_agents: bool = True
    enable_paper_recommendations: bool = False
    recommendation_config: RecommendationConfig = field(default_factory=RecommendationConfig)
    recommendation_client: PaperSearchClient | None = None
    enable_kg_updates: bool = False
    kg_update_config: KGUpdateConfig = field(default_factory=KGUpdateConfig)
    kg_updater: KnowledgeGraphGeneratorUpdater | None = None
    enable_outline_expander: bool = False
    outline_expander: OutlineExpanderAgent | None = None
    enable_soft_validation: bool = False
    functional_forms: dict[str, object] | None = None
    functional_form_id: str = ""
    max_soft_validation_findings: int = 64
    writing_ontology: dict[str, object] | None = None
    max_writing_ontology_findings: int = 32
    genre_ontology: dict[str, object] | None = None
    genre_profile_id: str = ""
    narrative_tone: str = "neutral"
    narrative_style: str = "analytical"
    narrative_audience: str = "general"
    narrative_discipline: str = "interdisciplinary"
    narrative_genre: str = "scholarly_manuscript"
    narrative_language: str = "English"
    coordination_max_revision_iterations: int = 4


@dataclass(slots=True)
class ColophonPipeline:
    """Coordinates retrieval, drafting, coordination, validation, and diagnostics.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline runtime configuration.
    """

    config: PipelineConfig = field(default_factory=PipelineConfig)

    def run(
        self,
        bibliography: list[Source],
        outline: list[dict],
        graph: KnowledgeGraph,
    ) -> Manuscript:
        """Run.

        Parameters
        ----------
        bibliography : list[Source]
            Parameter description.
        outline : list[dict]
            Parameter description.
        graph : KnowledgeGraph
            Parameter description.

        Returns
        -------
        Manuscript
            Return value description.
        """
        effective_outline = list(outline)
        effective_prompts = dict(self.config.prompt_templates)
        outline_expansion_result = None
        selected_functional_form = select_functional_form(
            functional_forms_payload=self.config.functional_forms,
            form_id=self.config.functional_form_id,
        )
        selected_functional_form_id = _string(selected_functional_form.get("id"))
        genre_context = build_genre_ontology_context(
            genre_ontology_payload=self.config.genre_ontology,
            profile_id=self.config.genre_profile_id,
            overrides={
                "tone": _override_if_nondefault(self.config.narrative_tone, "neutral"),
                "style": _override_if_nondefault(self.config.narrative_style, "analytical"),
                "audience": _override_if_nondefault(self.config.narrative_audience, "general"),
                "discipline": _override_if_nondefault(self.config.narrative_discipline, "interdisciplinary"),
                "genre": _override_if_nondefault(self.config.narrative_genre, "scholarly_manuscript"),
                "language": _override_if_nondefault(self.config.narrative_language, "English"),
            },
        )
        narrative_metadata = _genre_metadata(genre_context)
        writing_ontology_context = build_writing_ontology_context(
            ontology_payload=self.config.writing_ontology,
            form_id=selected_functional_form_id,
        )

        if self.config.enable_outline_expander:
            expander = self.config.outline_expander or OutlineExpanderAgent(
                functional_forms_payload=self.config.functional_forms,
                functional_form_id=self.config.functional_form_id,
                tone=narrative_metadata["tone"],
                style=narrative_metadata["style"],
                audience=narrative_metadata["audience"],
                discipline=narrative_metadata["discipline"],
                language=narrative_metadata["language"],
                background_prompt=_combined_background_prompt(
                    writing_context=writing_ontology_context,
                    genre_context=genre_context,
                    role="outline_expander",
                ),
            )
            if isinstance(expander, OutlineExpanderAgent):
                if expander.functional_forms_payload is None:
                    expander.functional_forms_payload = self.config.functional_forms
                if not expander.functional_form_id:
                    expander.functional_form_id = self.config.functional_form_id
                if not expander.background_prompt:
                    expander.background_prompt = _combined_background_prompt(
                        writing_context=writing_ontology_context,
                        genre_context=genre_context,
                        role="outline_expander",
                    )
            outline_expansion_result = expander.expand({"chapters": effective_outline})
            expanded = outline_expansion_result.get("chapters", [])
            if isinstance(expanded, list) and expanded:
                effective_outline = [chapter for chapter in expanded if isinstance(chapter, dict)]
            expanded_prompts = outline_expansion_result.get("prompts", {})
            if isinstance(expanded_prompts, dict):
                for key, value in expanded_prompts.items():
                    if isinstance(key, str) and isinstance(value, str):
                        effective_prompts.setdefault(key, value)

        retriever = SimpleRetriever(bibliography)
        message_bus = MessageBus()
        if self.config.enable_coordination_agents:
            _seed_writing_ontology_messages(message_bus=message_bus, context=writing_ontology_context)
            _seed_genre_ontology_messages(message_bus=message_bus, context=genre_context)

        paragraph_coordinator = ParagraphCoordinationAgent(
            functional_form=selected_functional_form,
            background_prompt=_combined_background_prompt(
                writing_context=writing_ontology_context,
                genre_context=genre_context,
                role="paragraph_coordinator",
            ),
        )
        section_coordinator = SectionCoordinationAgent(
            functional_form=selected_functional_form,
            background_prompt=_combined_background_prompt(
                writing_context=writing_ontology_context,
                genre_context=genre_context,
                role="section_coordinator",
            ),
        )
        chapter_coordinator = ChapterCoordinationAgent(
            functional_form=selected_functional_form,
            background_prompt=_combined_background_prompt(
                writing_context=writing_ontology_context,
                genre_context=genre_context,
                role="chapter_coordinator",
            ),
        )
        book_coordinator = BookCoordinationAgent(
            functional_form=selected_functional_form,
            background_prompt=_combined_background_prompt(
                writing_context=writing_ontology_context,
                genre_context=genre_context,
                role="book_coordinator",
            ),
        )

        section_agent = SectionAgent(
            claim_agent=ClaimAuthorAgent(
                claim_template=effective_prompts.get(
                    "claim_template",
                    CLAIM_TEMPLATE_DEFAULT,
                ),
                figure_reference_template=effective_prompts.get(
                    "figure_reference_template",
                    FIGURE_REFERENCE_TEMPLATE_DEFAULT,
                ),
                llm_client=self.config.llm_client,
                llm_system_prompt=self.config.llm_system_prompt or None,
                tone=narrative_metadata["tone"],
                style=narrative_metadata["style"],
                audience=narrative_metadata["audience"],
                discipline=narrative_metadata["discipline"],
                language=narrative_metadata["language"],
                background_prompt=_combined_background_prompt(
                    writing_context=writing_ontology_context,
                    genre_context=genre_context,
                    role="claim_author_agent",
                ),
            ),
            paragraph_agent=ParagraphAgent(
                paragraph_template=effective_prompts.get(
                    "paragraph_template",
                    PARAGRAPH_TEMPLATE_DEFAULT,
                ),
                empty_section_template=effective_prompts.get(
                    "empty_section_template",
                    EMPTY_SECTION_TEMPLATE_DEFAULT,
                ),
                llm_client=self.config.llm_client,
                llm_system_prompt=self.config.llm_system_prompt or None,
                tone=narrative_metadata["tone"],
                style=narrative_metadata["style"],
                audience=narrative_metadata["audience"],
                discipline=narrative_metadata["discipline"],
                language=narrative_metadata["language"],
                background_prompt=_combined_background_prompt(
                    writing_context=writing_ontology_context,
                    genre_context=genre_context,
                    role="paragraph_agent",
                ),
            ),
        )

        chapters: list[Chapter] = []
        all_sections = []
        gap_requests: list[GapRequest] = []
        recommendation_proposals: list[RecommendationProposal] = []
        kg_update_result = None
        soft_validation_result = None
        writing_ontology_validation_result = None
        coordination_revision_result = {
            "enabled": self.config.enable_coordination_agents,
            "iterations_run": 0,
            "converged": False,
            "reason": "disabled",
            "max_iterations": max(1, self.config.coordination_max_revision_iterations),
            "history": [],
        }

        if self.config.enable_kg_updates:
            kg_updater = self.config.kg_updater or KnowledgeGraphGeneratorUpdater(config=self.config.kg_update_config)
            kg_update_result = kg_updater.run(
                bibliography=bibliography,
                graph=graph,
                message_bus=message_bus if self.config.enable_coordination_agents else None,
            )

        for chapter_idx, chapter_spec in enumerate(effective_outline, start=1):
            chapter_id = f"ch{chapter_idx}"
            chapter_title = chapter_spec.get("title", f"Chapter {chapter_idx}")
            chapter_type = _chapter_type_for_chapter_spec(
                chapter_spec=chapter_spec,
                chapter_index=chapter_idx,
                chapter_count=len(effective_outline),
                functional_form=selected_functional_form,
            )
            section_expectations = _section_expectations_for_chapter_spec(chapter_spec)
            section_titles = [row["title"] for row in section_expectations]
            expected_elements = [row["element_id"] for row in section_expectations]

            if self.config.enable_coordination_agents:
                chapter_coordinator.prepare_child_guidance(
                    chapter_title=chapter_title,
                    expected_sections=section_titles,
                    bus=message_bus,
                    chapter_id=chapter_id,
                    chapter_type=chapter_type,
                    expected_elements=[element for element in expected_elements if element],
                )

            sections = []
            for section_idx, section_expectation in enumerate(section_expectations, start=1):
                section_title = section_expectation["title"]
                expected_element_id = section_expectation["element_id"]
                if self.config.enable_coordination_agents:
                    section_coordinator.prepare_child_guidance(
                        section_title=section_title,
                        bus=message_bus,
                        chapter_id=chapter_id,
                        chapter_type=chapter_type,
                        expected_element_id=expected_element_id,
                    )
                hits = retriever.search(query=section_title, top_k=self.config.top_k)
                section = section_agent.draft(
                    chapter_id=chapter_id,
                    index=section_idx,
                    section_title=section_title,
                    hits=hits,
                    graph=graph,
                    max_figures=self.config.max_figures_per_section,
                    message_bus=message_bus if self.config.enable_coordination_agents else None,
                )

                sections.append(section)
                all_sections.append(section)

            chapter = Chapter(id=chapter_id, title=chapter_title, sections=sections)
            chapters.append(chapter)

        if self.config.enable_coordination_agents:
            chapter_expectations_by_chapter_id = {
                f"ch{index}": _chapter_expectations_for_spec(
                    chapter_spec=spec,
                    chapter_index=index,
                    chapter_count=len(effective_outline),
                    functional_form=selected_functional_form,
                )
                for index, spec in enumerate(effective_outline, start=1)
            }
            coordination_revision_result = _run_coordination_revision_loop(
                chapters=chapters,
                outline=effective_outline,
                chapter_expectations_by_chapter_id=chapter_expectations_by_chapter_id,
                paragraph_coordinator=paragraph_coordinator,
                section_coordinator=section_coordinator,
                chapter_coordinator=chapter_coordinator,
                book_coordinator=book_coordinator,
                bus=message_bus,
                gap_requests=gap_requests,
                max_iterations=max(1, self.config.coordination_max_revision_iterations),
            )

        if self.config.enable_paper_recommendations:
            recommender = PaperRecommendationWorkflow(
                config=self.config.recommendation_config,
                client=self.config.recommendation_client,
            )
            recommendation_proposals = recommender.generate_proposals(
                bibliography=bibliography,
                graph=graph,
                outline=effective_outline,
                message_bus=message_bus if self.config.enable_coordination_agents else None,
                genre_context=genre_context,
            )

            if not recommendation_proposals:
                gap_requests.append(
                    GapRequest(
                        level="book",
                        component="bibliography",
                        request="Review external literature and add missing related papers",
                        rationale=(
                            "Recommendation workflow did not return any candidate papers; "
                            "this may indicate sparse seeds or API constraints."
                        ),
                    )
                )

        citation_issues = CitationReviewerAgent().review(
            sections=all_sections,
            known_source_ids={source.id for source in bibliography},
        )
        figure_issues = FigureReviewerAgent().review(
            sections=all_sections,
            known_figure_ids=set(graph.figures.keys()),
        )
        coherence_issues = CoherenceReviewerAgent().review(sections=all_sections)
        if self.config.enable_soft_validation and self.config.functional_forms is not None:
            soft_validation_result = run_soft_validation(
                functional_forms_payload=self.config.functional_forms,
                outline=effective_outline,
                bibliography=bibliography,
                prompts=effective_prompts,
                chapters=chapters,
                agent_profile={
                    "top_k": self.config.top_k,
                    "enable_coordination_agents": self.config.enable_coordination_agents,
                    "llm_enabled": self.config.llm_client is not None,
                    "max_figures_per_section": self.config.max_figures_per_section,
                    "tone": narrative_metadata["tone"],
                    "style": narrative_metadata["style"],
                    "audience": narrative_metadata["audience"],
                    "discipline": narrative_metadata["discipline"],
                    "genre": narrative_metadata["genre"],
                    "language": narrative_metadata["language"],
                    "genre_validation": _coerce_mapping(genre_context.get("validation")),
                },
                form_id=self.config.functional_form_id,
                max_findings=max(1, self.config.max_soft_validation_findings),
            )
            for finding in soft_validation_result.findings:
                if finding.severity not in {"error", "soft_error"}:
                    continue
                gap_requests.append(
                    GapRequest(
                        level="book",
                        component=finding.component,
                        request=finding.message,
                        rationale=finding.suggestion,
                        related_id=finding.related_id,
                    )
                )

        if self.config.writing_ontology is not None:
            writing_ontology_validation_result = run_writing_ontology_validation(
                ontology_payload=self.config.writing_ontology,
                outline=effective_outline,
                bibliography=bibliography,
                prompts=effective_prompts,
                chapters=chapters,
                form_id=selected_functional_form_id,
                functional_form=selected_functional_form,
                coordination_revision=coordination_revision_result,
                max_findings=max(1, self.config.max_writing_ontology_findings),
            )
            for finding in writing_ontology_validation_result.findings:
                if finding.severity not in {"error", "soft_error"}:
                    continue
                gap_requests.append(
                    GapRequest(
                        level="book",
                        component=finding.component,
                        request=finding.message,
                        rationale=finding.suggestion,
                        related_id=finding.related_id,
                    )
                )
        referenced_figure_ids = {
            figure_id
            for section in all_sections
            for claim in section.claims
            for figure_id in claim.figure_ids
        }

        diagnostics = {
            "citation_issues": citation_issues,
            "figure_issues": figure_issues,
            "coherence_issues": coherence_issues,
            "sections_generated": len(all_sections),
            "chapters_generated": len(chapters),
            "figures_available": len(graph.figures),
            "figures_attached": sum(len(section.figures) for section in all_sections),
            "figures_referenced": len(referenced_figure_ids),
            "coordination_messages": [_message_to_dict(message) for message in message_bus.messages],
            "gap_requests": [_gap_to_dict(gap) for gap in gap_requests],
            "recommendations_enabled": self.config.enable_paper_recommendations,
            "recommendation_provider": (
                self.config.recommendation_config.provider if self.config.enable_paper_recommendations else "none"
            ),
            "recommendations_generated": len(recommendation_proposals),
            "recommendation_proposals": [_proposal_to_dict(proposal) for proposal in recommendation_proposals],
            "kg_updates_enabled": self.config.enable_kg_updates,
            "kg_update_result": kg_update_result.to_dict() if kg_update_result is not None else None,
            "outline_expander_enabled": self.config.enable_outline_expander,
            "outline_expansion_result": outline_expansion_result,
            "soft_validation_enabled": self.config.enable_soft_validation,
            "soft_validation_form_id": self.config.functional_form_id,
            "soft_validation_result": soft_validation_result.to_dict() if soft_validation_result is not None else None,
            "functional_form_context": {
                "form_id": _string(selected_functional_form.get("id")),
                "form_name": _string(selected_functional_form.get("name")),
                "catalog_loaded": self.config.functional_forms is not None,
            },
            "writing_ontology_context": writing_ontology_context,
            "writing_ontology_validation_result": (
                writing_ontology_validation_result.to_dict()
                if writing_ontology_validation_result is not None
                else None
            ),
            "genre_ontology_context": genre_context,
            "narrative_profile": {
                "tone": narrative_metadata["tone"],
                "style": narrative_metadata["style"],
                "audience": narrative_metadata["audience"],
                "discipline": narrative_metadata["discipline"],
                "genre": narrative_metadata["genre"],
                "language": narrative_metadata["language"],
            },
            "coordination_revision": coordination_revision_result,
        }

        return Manuscript(
            title=self.config.title,
            chapters=chapters,
            diagnostics=diagnostics,
            coordination_messages=message_bus.messages,
            gap_requests=gap_requests,
            recommendation_proposals=recommendation_proposals,
        )


def _message_to_dict(message) -> dict:
    """Message to dict.

    Parameters
    ----------
    message : object
        Parameter description.

    Returns
    -------
    dict
        Return value description.
    """
    return {
        "sender": message.sender,
        "receiver": message.receiver,
        "message_type": message.message_type,
        "content": message.content,
        "related_id": message.related_id,
        "priority": message.priority,
    }


def _gap_to_dict(gap: GapRequest) -> dict:
    """Gap to dict.

    Parameters
    ----------
    gap : GapRequest
        Parameter description.

    Returns
    -------
    dict
        Return value description.
    """
    return {
        "level": gap.level,
        "component": gap.component,
        "request": gap.request,
        "rationale": gap.rationale,
        "related_id": gap.related_id,
    }


def _proposal_to_dict(proposal: RecommendationProposal) -> dict:
    """Proposal to dict.

    Parameters
    ----------
    proposal : RecommendationProposal
        Parameter description.

    Returns
    -------
    dict
        Return value description.
    """
    return {
        "proposal_id": proposal.proposal_id,
        "title": proposal.title,
        "authors": proposal.authors,
        "publication": proposal.publication,
        "year": proposal.year,
        "abstract": proposal.abstract,
        "citation_count": proposal.citation_count,
        "source_url": proposal.source_url,
        "doi": proposal.doi,
        "score": proposal.score,
        "reasons": proposal.reasons,
        "based_on_source_ids": proposal.based_on_source_ids,
        "bibliography_entry": proposal.bibliography_entry,
        "knowledge_graph_update": proposal.knowledge_graph_update,
    }


def _background_prompt_for(context: dict[str, object], role: str) -> str:
    """Return merged companion-ontology background prompt for a role.

    Parameters
    ----------
    context : dict[str, object]
        Writing ontology runtime context.
    role : str
        Agent or coordinator role key.

    Returns
    -------
    str
        Background prompt string for the requested role.
    """
    if not isinstance(context, dict):
        return ""
    prompts = context.get("agent_prompts", {})
    if not isinstance(prompts, dict):
        return ""
    value = prompts.get(role, "")
    return _string(value)


def _genre_background_prompt_for(context: dict[str, object], role: str) -> str:
    """Return genre-ontology role prompt for a role.

    Parameters
    ----------
    context : dict[str, object]
        Genre ontology runtime context.
    role : str
        Agent or coordinator role key.

    Returns
    -------
    str
        Role-specific genre prompt.
    """
    if not isinstance(context, dict):
        return ""
    prompts = context.get("role_prompts", {})
    if not isinstance(prompts, dict):
        return ""
    value = _string(prompts.get(role, ""))
    if value:
        return value
    aliases = {
        "paper_recommender": "recommendation_agent",
        "paper_recommendation": "recommendation_agent",
    }
    alias = aliases.get(role, "")
    if alias:
        return _string(prompts.get(alias, ""))
    return ""


def _combined_background_prompt(
    writing_context: dict[str, object],
    genre_context: dict[str, object],
    role: str,
) -> str:
    """Combine writing-ontology and genre-ontology prompts for one role.

    Parameters
    ----------
    writing_context : dict[str, object]
        Writing ontology context.
    genre_context : dict[str, object]
        Genre ontology context.
    role : str
        Agent/coordinator role key.

    Returns
    -------
    str
        Combined prompt text.
    """
    segments = [
        _background_prompt_for(context=writing_context, role=role),
        _genre_background_prompt_for(context=genre_context, role=role),
    ]
    return " ".join(part for part in segments if part).strip()


def _seed_writing_ontology_messages(message_bus: MessageBus, context: dict[str, object]) -> None:
    """Seed baseline background guidance messages onto the coordination bus.

    Parameters
    ----------
    message_bus : MessageBus
        Shared message bus.
    context : dict[str, object]
        Writing ontology runtime context.
    """
    if not isinstance(context, dict) or not bool(context.get("enabled", False)):
        return
    receivers = [
        "claim_author_agent",
        "paragraph_agent",
        "section_coordinator",
        "chapter_coordinator",
        "book_coordinator",
    ]
    for receiver in receivers:
        prompt = _background_prompt_for(context=context, role=receiver)
        if not prompt:
            continue
        message_bus.send(
            sender="writing_ontology",
            receiver=receiver,
            message_type="background_prompt",
            content=prompt,
            priority="low",
        )


def _seed_genre_ontology_messages(message_bus: MessageBus, context: dict[str, object]) -> None:
    """Seed baseline genre prompts onto the coordination bus.

    Parameters
    ----------
    message_bus : MessageBus
        Shared message bus.
    context : dict[str, object]
        Genre ontology runtime context.
    """
    if not isinstance(context, dict) or not bool(context.get("enabled", False)):
        return
    receivers = [
        "claim_author_agent",
        "paragraph_agent",
        "outline_expander",
        "section_coordinator",
        "chapter_coordinator",
        "book_coordinator",
        "paper_recommender",
    ]
    for receiver in receivers:
        prompt = _genre_background_prompt_for(context=context, role=receiver)
        if not prompt:
            continue
        message_bus.send(
            sender="genre_ontology",
            receiver=receiver,
            message_type="genre_prompt",
            content=prompt,
            priority="low",
        )


def _genre_metadata(context: dict[str, object]) -> dict[str, str]:
    """Extract normalized narrative metadata from genre context.

    Parameters
    ----------
    context : dict[str, object]
        Genre ontology context.

    Returns
    -------
    dict[str, str]
        Normalized metadata mapping.
    """
    metadata = _coerce_mapping(context.get("metadata"))
    return {
        "tone": _string(metadata.get("tone")) or "neutral",
        "style": _string(metadata.get("style")) or "analytical",
        "audience": _string(metadata.get("audience")) or "general",
        "discipline": _string(metadata.get("discipline")) or "interdisciplinary",
        "genre": _string(metadata.get("genre")) or "scholarly_manuscript",
        "language": _string(metadata.get("language")) or "English",
    }


def _override_if_nondefault(value: str, default_value: str) -> str:
    """Return override value only when it differs from default placeholder.

    Parameters
    ----------
    value : str
        Candidate value.
    default_value : str
        Default placeholder used by config/CLI.

    Returns
    -------
    str
        Override value or empty string.
    """
    normalized = _string(value)
    if not normalized:
        return ""
    if normalized == default_value:
        return ""
    return normalized


def _run_coordination_revision_loop(
    chapters: list[Chapter],
    outline: list[dict],
    chapter_expectations_by_chapter_id: dict[str, dict[str, object]],
    paragraph_coordinator: ParagraphCoordinationAgent,
    section_coordinator: SectionCoordinationAgent,
    chapter_coordinator: ChapterCoordinationAgent,
    book_coordinator: BookCoordinationAgent,
    bus: MessageBus,
    gap_requests: list[GapRequest],
    max_iterations: int,
) -> dict:
    """Run bounded iterative coordinator passes until manuscript state converges.

    Parameters
    ----------
    chapters : list[Chapter]
        Generated chapters to revise.
    outline : list[dict]
        Effective outline used for chapter/book expectations.
    chapter_expectations_by_chapter_id : dict[str, dict[str, object]]
        Expected section titles, element ids, and chapter type for each chapter id.
    paragraph_coordinator : ParagraphCoordinationAgent
        Paragraph-level editor/coordinator.
    section_coordinator : SectionCoordinationAgent
        Section-level editor/coordinator.
    chapter_coordinator : ChapterCoordinationAgent
        Chapter-level editor/coordinator.
    book_coordinator : BookCoordinationAgent
        Book-level editor/coordinator.
    bus : MessageBus
        Shared message bus for coordination events.
    gap_requests : list[GapRequest]
        Global gap-request accumulator; updated in place.
    max_iterations : int
        Maximum number of revision passes.

    Returns
    -------
    dict
        Convergence diagnostics for iterative coordination.
    """
    history: list[dict[str, object]] = []
    converged = False
    reason = "max_iterations_reached"

    for iteration in range(1, max_iterations + 1):
        snapshot_before = _coordination_content_snapshot(chapters)
        messages_before = len(bus.messages)
        gaps_before = len(gap_requests)

        for chapter in chapters:
            expectation = chapter_expectations_by_chapter_id.get(chapter.id, {})
            expected_sections = _string_list(expectation.get("section_titles"))
            expected_elements = _string_list(expectation.get("section_element_ids"))
            chapter_type = _string(expectation.get("chapter_type"))
            for section_idx, section in enumerate(chapter.sections):
                paragraph_gaps = paragraph_coordinator.coordinate(section=section, bus=bus)
                _extend_unique_gap_requests(gap_requests, paragraph_gaps)

                expected_title = section.title
                if section_idx < len(expected_sections):
                    expected_title = expected_sections[section_idx]
                expected_element_id = ""
                if section_idx < len(expected_elements):
                    expected_element_id = expected_elements[section_idx]
                section_gaps = section_coordinator.coordinate(
                    section=section,
                    expected_title=expected_title,
                    bus=bus,
                    expected_element_id=expected_element_id,
                    chapter_type=chapter_type,
                )
                _extend_unique_gap_requests(gap_requests, section_gaps)

            chapter_gaps = chapter_coordinator.coordinate(
                chapter=chapter,
                expected_sections=expected_sections,
                bus=bus,
                chapter_type=chapter_type,
                expected_elements=expected_elements,
            )
            _extend_unique_gap_requests(gap_requests, chapter_gaps)

        book_gaps = book_coordinator.coordinate(chapters=chapters, outline=outline, bus=bus)
        _extend_unique_gap_requests(gap_requests, book_gaps)

        snapshot_after = _coordination_content_snapshot(chapters)
        messages_after = len(bus.messages)
        gaps_after = len(gap_requests)

        content_changed = snapshot_before != snapshot_after
        new_messages = messages_after - messages_before
        new_gaps = gaps_after - gaps_before

        history.append(
            {
                "iteration": iteration,
                "content_changed": content_changed,
                "new_messages": new_messages,
                "new_gap_requests": new_gaps,
                "message_count": messages_after,
                "gap_count": gaps_after,
            }
        )

        if not content_changed and new_messages == 0 and new_gaps == 0:
            converged = True
            reason = "content_and_coordination_stable"
            break

    return {
        "enabled": True,
        "iterations_run": len(history),
        "converged": converged,
        "reason": reason,
        "max_iterations": max_iterations,
        "history": history,
    }


def _coordination_content_snapshot(chapters: list[Chapter]) -> tuple:
    """Build a hashable content snapshot for convergence checks.

    Parameters
    ----------
    chapters : list[Chapter]
        Chapters to snapshot.

    Returns
    -------
    tuple
        Hashable content snapshot.
    """
    chapter_rows: list[tuple] = []
    for chapter in chapters:
        section_rows: list[tuple] = []
        for section in chapter.sections:
            paragraph_rows = tuple((paragraph.id, paragraph.text) for paragraph in section.paragraphs)
            claim_rows = tuple(
                (claim.id, claim.text, tuple(claim.evidence_ids), tuple(claim.figure_ids))
                for claim in section.claims
            )
            section_rows.append((section.id, section.title, paragraph_rows, claim_rows))
        chapter_rows.append((chapter.id, chapter.title, tuple(section_rows)))
    return tuple(chapter_rows)


def _extend_unique_gap_requests(target: list[GapRequest], incoming: list[GapRequest]) -> None:
    """Append only novel gap requests.

    Parameters
    ----------
    target : list[GapRequest]
        Existing gap-request accumulator.
    incoming : list[GapRequest]
        New gap requests to merge in.
    """
    seen = {
        (gap.level, gap.component, gap.request, gap.rationale, gap.related_id)
        for gap in target
    }
    for gap in incoming:
        key = (gap.level, gap.component, gap.request, gap.rationale, gap.related_id)
        if key in seen:
            continue
        seen.add(key)
        target.append(gap)


def _section_expectations_for_chapter_spec(chapter_spec: object) -> list[dict[str, str]]:
    """Extract ordered section title/element expectations from an outline chapter spec.

    Parameters
    ----------
    chapter_spec : object
        Outline chapter specification.

    Returns
    -------
    list[dict[str, str]]
        Section expectation rows with ``title`` and ``element_id``.
    """
    if not isinstance(chapter_spec, dict):
        return []
    section_details = chapter_spec.get("section_details", [])
    detail_lookup: dict[str, str] = {}
    if isinstance(section_details, list):
        for row in section_details:
            if not isinstance(row, dict):
                continue
            title = _string(row.get("title"))
            if not title:
                continue
            detail_lookup[title.lower()] = _string(row.get("functional_element_id"))

    raw_sections = chapter_spec.get("sections", [])
    expectations: list[dict[str, str]] = []
    seen_titles: set[str] = set()
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
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            if not element_id:
                element_id = detail_lookup.get(key, "")
            expectations.append({"title": title, "element_id": element_id})

    if expectations:
        return expectations

    if isinstance(section_details, list):
        for row in section_details:
            if not isinstance(row, dict):
                continue
            title = _string(row.get("title"))
            if not title:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            expectations.append({"title": title, "element_id": _string(row.get("functional_element_id"))})
    return expectations


def _chapter_expectations_for_spec(
    chapter_spec: object,
    chapter_index: int,
    chapter_count: int,
    functional_form: dict[str, object],
) -> dict[str, object]:
    """Resolve chapter-level expectations used by coordination revision passes.

    Parameters
    ----------
    chapter_spec : object
        Outline chapter specification.
    chapter_index : int
        1-based chapter index.
    chapter_count : int
        Total chapter count.
    functional_form : dict[str, object]
        Selected functional-form object.

    Returns
    -------
    dict[str, object]
        Mapping with chapter type, section titles, and section element ids.
    """
    chapter_type = _chapter_type_for_chapter_spec(
        chapter_spec=chapter_spec,
        chapter_index=chapter_index,
        chapter_count=chapter_count,
        functional_form=functional_form,
    )
    section_expectations = _section_expectations_for_chapter_spec(chapter_spec)
    section_titles = [row["title"] for row in section_expectations]
    section_element_ids = [row["element_id"] for row in section_expectations]

    if not section_titles:
        required_sections = _required_sections_for_chapter_type(functional_form=functional_form, chapter_type=chapter_type)
        element_lookup = _functional_form_element_lookup(functional_form)
        for section_id in required_sections:
            section_titles.append(element_lookup.get(section_id, _humanize_identifier(section_id)))
            section_element_ids.append(section_id)

    return {
        "chapter_type": chapter_type,
        "section_titles": section_titles,
        "section_element_ids": section_element_ids,
    }


def _chapter_type_for_chapter_spec(
    chapter_spec: object,
    chapter_index: int,
    chapter_count: int,
    functional_form: dict[str, object],
) -> str:
    """Infer or resolve a functional chapter type for an outline chapter spec.

    Parameters
    ----------
    chapter_spec : object
        Outline chapter specification.
    chapter_index : int
        1-based chapter index.
    chapter_count : int
        Total chapter count.
    functional_form : dict[str, object]
        Selected functional-form object.

    Returns
    -------
    str
        Chapter type label, if available.
    """
    if isinstance(chapter_spec, dict):
        explicit = _string(chapter_spec.get("functional_chapter_type"))
        if explicit:
            return explicit

    chapter_pattern = functional_form.get("chapter_pattern", [])
    if not isinstance(chapter_pattern, list):
        chapter_pattern = functional_form.get("chapter_or_section_pattern", functional_form.get("section_pattern", []))
    if not isinstance(chapter_pattern, list) or not chapter_pattern:
        return ""

    chapter_title = ""
    if isinstance(chapter_spec, dict):
        chapter_title = _string(chapter_spec.get("title")).lower()
    if chapter_title:
        title_tokens = _tokenize(chapter_title)
        for row in chapter_pattern:
            if not isinstance(row, dict):
                continue
            chapter_type = _string(row.get("type"))
            type_tokens = _tokenize(chapter_type.replace("_", " "))
            if type_tokens and len(title_tokens & type_tokens) / max(1, len(type_tokens)) >= 0.5:
                return chapter_type

    repeatable_rows = [row for row in chapter_pattern if isinstance(row, dict) and bool(row.get("repeatable", False))]
    if chapter_index == 1:
        return _string(_coerce_mapping(chapter_pattern[0]).get("type"))
    if chapter_index == chapter_count and chapter_count > 1:
        return _string(_coerce_mapping(chapter_pattern[-1]).get("type"))
    if repeatable_rows:
        return _string(_coerce_mapping(repeatable_rows[0]).get("type"))
    fallback = chapter_pattern[min(max(chapter_index - 1, 0), len(chapter_pattern) - 1)]
    return _string(_coerce_mapping(fallback).get("type"))


def _required_sections_for_chapter_type(functional_form: dict[str, object], chapter_type: str) -> list[str]:
    """Return required section-element ids for a functional chapter type.

    Parameters
    ----------
    functional_form : dict[str, object]
        Selected functional form.
    chapter_type : str
        Chapter type label.

    Returns
    -------
    list[str]
        Required section element ids for the chapter type.
    """
    chapter_pattern = functional_form.get("chapter_pattern", [])
    if not isinstance(chapter_pattern, list):
        chapter_pattern = functional_form.get("chapter_or_section_pattern", functional_form.get("section_pattern", []))
    if not isinstance(chapter_pattern, list):
        return []
    for row in chapter_pattern:
        if not isinstance(row, dict):
            continue
        if _string(row.get("type")) != chapter_type:
            continue
        required = _string_list(row.get("required_sections"))
        if required:
            return required
        return _string_list(row.get("required_elements"))
    return []


def _functional_form_element_lookup(functional_form: dict[str, object]) -> dict[str, str]:
    """Build a mapping from functional-form element id to human-readable label.

    Parameters
    ----------
    functional_form : dict[str, object]
        Selected functional form.

    Returns
    -------
    dict[str, str]
        Element id to label mapping.
    """
    elements = functional_form.get("elements", [])
    if not isinstance(elements, list):
        return {}
    mapping: dict[str, str] = {}
    for row in elements:
        if not isinstance(row, dict):
            continue
        element_id = _string(row.get("id"))
        if not element_id:
            continue
        mapping[element_id] = _string(row.get("label")) or _humanize_identifier(element_id)
    return mapping


def _coerce_mapping(value: object) -> dict[str, object]:
    """Return mapping value or empty mapping.

    Parameters
    ----------
    value : object
        Candidate mapping value.

    Returns
    -------
    dict[str, object]
        Mapping value or empty mapping.
    """
    if isinstance(value, dict):
        return value
    return {}


def _string(value: object) -> str:
    """Convert text-like values to stripped strings.

    Parameters
    ----------
    value : object
        Candidate textual value.

    Returns
    -------
    str
        Stripped string for textual inputs.
    """
    if isinstance(value, str):
        return value.strip()
    return ""


def _string_list(value: object) -> list[str]:
    """Normalize list-like values to non-empty strings.

    Parameters
    ----------
    value : object
        Candidate list value.

    Returns
    -------
    list[str]
        Non-empty string list.
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


def _humanize_identifier(value: str) -> str:
    """Convert snake-case identifiers to title-case labels.

    Parameters
    ----------
    value : str
        Identifier value.

    Returns
    -------
    str
        Human-readable label.
    """
    return " ".join(part.capitalize() for part in value.replace("-", "_").split("_") if part)
