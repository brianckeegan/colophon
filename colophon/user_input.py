"""User-input helpers for planning guidance and Claude Agent SDK integration.

This module provides:
- questionnaire templates for planning-focused guidance
- interactive answer collection with multiple-choice support
- normalization into structured planning preferences
- an Agent SDK ``can_use_tool`` handler for ``AskUserQuestion``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

MAX_USER_INPUTS_PER_STAGE = 10
CODEX_ASK_USER_QUESTION_TOOL_NAME = "AskUserQuestion"
GUIDANCE_STAGE_PLANNING = "planning"
GUIDANCE_STAGE_RECOMMENDATIONS = "recommendations"
GUIDANCE_STAGE_OUTLINE = "outline"
GUIDANCE_STAGE_COORDINATION = "coordination"
GUIDANCE_STAGES_SUPPORTED = (
    GUIDANCE_STAGE_PLANNING,
    GUIDANCE_STAGE_RECOMMENDATIONS,
    GUIDANCE_STAGE_OUTLINE,
    GUIDANCE_STAGE_COORDINATION,
)


@dataclass(slots=True)
class PlanningGuidance:
    """Structured guidance captured from user-input questions.

    Parameters
    ----------
    planning_document_focus : str
        User preference for planning-document format and depth.
    incorporate_recommendations : bool | None
        Whether recommendation proposals should be incorporated.
    expand_outline : bool | None
        Whether outline expansion should be enabled.
    additional_notes : str
        Additional free-form notes.
    answers : dict[str, str]
        Raw question-to-answer mapping captured during collection.
    """

    planning_document_focus: str = ""
    incorporate_recommendations: bool | None = None
    expand_outline: bool | None = None
    additional_notes: str = ""
    answers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RecommendationGuidance:
    """Guidance for recommendation workflow behavior.

    Parameters
    ----------
    enable_recommendations : bool | None
        Whether recommendation workflow should run.
    strategy : str
        Named recommendation strategy profile.
    top_k : int | None
        Preferred recommendation top-k target.
    per_seed_limit : int | None
        Preferred per-seed candidate limit.
    min_score : float | None
        Preferred recommendation minimum score threshold.
    focus_tags : list[str]
        Optional emphasis tags (novelty/citation/topical/etc.).
    additional_notes : str
        Free-form notes.
    answers : dict[str, str]
        Raw answers backing the parsed guidance.
    """

    enable_recommendations: bool | None = None
    strategy: str = ""
    top_k: int | None = None
    per_seed_limit: int | None = None
    min_score: float | None = None
    focus_tags: list[str] = field(default_factory=list)
    additional_notes: str = ""
    answers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class OutlineGuidance:
    """Guidance for outline-expansion behavior.

    Parameters
    ----------
    enable_outline_expansion : bool | None
        Whether outline expansion should be enabled.
    depth_profile : str
        Preferred expansion depth profile label.
    max_subsections : int | None
        Preferred maximum subsections per section.
    include_transitions : bool | None
        Preference for transition-oriented expansion prompts.
    include_counterarguments : bool | None
        Preference for counterargument prompts.
    additional_notes : str
        Free-form notes.
    answers : dict[str, str]
        Raw answers backing the parsed guidance.
    """

    enable_outline_expansion: bool | None = None
    depth_profile: str = ""
    max_subsections: int | None = None
    include_transitions: bool | None = None
    include_counterarguments: bool | None = None
    additional_notes: str = ""
    answers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CoordinationGuidance:
    """Guidance for coordination behavior and breakdown remediation.

    Parameters
    ----------
    coordination_strategy : str
        Preferred coordination strategy profile.
    preferred_components : list[str]
        Components to prioritize during remediation.
    increase_revision_iterations : bool | None
        Whether to increase coordination max iterations.
    target_revision_iterations : int | None
        Explicit revision-iteration target.
    strict_outline_alignment : bool | None
        Whether to enforce stricter outline alignment.
    breakdown_summary : str
        Human-readable breakdown summary.
    additional_notes : str
        Free-form notes.
    answers : dict[str, str]
        Raw answers backing the parsed guidance.
    """

    coordination_strategy: str = ""
    preferred_components: list[str] = field(default_factory=list)
    increase_revision_iterations: bool | None = None
    target_revision_iterations: int | None = None
    strict_outline_alignment: bool | None = None
    breakdown_summary: str = ""
    additional_notes: str = ""
    answers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class UserGuidanceBundle:
    """Aggregated multi-stage guidance bundle.

    Parameters
    ----------
    stages : list[str]
        Guidance stages requested/processed.
    planning : PlanningGuidance
        Planning-stage guidance.
    recommendations : RecommendationGuidance
        Recommendation-stage guidance.
    outline : OutlineGuidance
        Outline-stage guidance.
    coordination : CoordinationGuidance
        Coordination-stage guidance.
    answers_by_stage : dict[str, dict[str, str]]
        Raw answers keyed by stage id.
    """

    stages: list[str] = field(default_factory=list)
    planning: PlanningGuidance = field(default_factory=PlanningGuidance)
    recommendations: RecommendationGuidance = field(default_factory=RecommendationGuidance)
    outline: OutlineGuidance = field(default_factory=OutlineGuidance)
    coordination: CoordinationGuidance = field(default_factory=CoordinationGuidance)
    answers_by_stage: dict[str, dict[str, str]] = field(default_factory=dict)


def build_planning_questionnaire() -> dict[str, Any]:
    """Build a planning-focused questionnaire payload for ``AskUserQuestion``.

    Returns
    -------
    dict[str, Any]
        Questionnaire payload compatible with Agent SDK ``AskUserQuestion``
        conventions (``questions`` + ``options`` fields).
    """
    return {
        "questions": [
            {
                "header": "Plan Focus",
                "question": "What planning document style do you want?",
                "options": [
                    {
                        "label": "Detailed execution plan",
                        "description": "Break work into concrete phases, checkpoints, and outputs.",
                    },
                    {
                        "label": "High-level plan",
                        "description": "Keep planning concise and decision-oriented.",
                    },
                    {
                        "label": "Risk-first plan",
                        "description": "Prioritize assumptions, risks, and validation gates.",
                    },
                ],
            },
            {
                "header": "Recommendations",
                "question": "Should I incorporate recommendation proposals into the plan?",
                "options": [
                    {
                        "label": "Yes, include recommendations",
                        "description": "Blend recommendations into plan tasks and milestones.",
                    },
                    {
                        "label": "No, skip recommendations",
                        "description": "Keep the plan independent from recommendation proposals.",
                    },
                ],
            },
            {
                "header": "Outline",
                "question": "Should I expand the outline before drafting?",
                "options": [
                    {
                        "label": "Yes, expand outline",
                        "description": "Generate deeper section/subsection structure before drafting.",
                    },
                    {
                        "label": "No, keep current outline",
                        "description": "Use the uploaded outline as-is.",
                    },
                ],
            },
            {
                "header": "Notes",
                "question": "Any additional guidance for planning and execution?",
            },
        ]
    }


def normalize_guidance_stages(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize user-guidance stage identifiers.

    Parameters
    ----------
    raw : str | list[str] | tuple[str, ...] | None
        Raw stage descriptor(s).

    Returns
    -------
    list[str]
        Normalized stage ids in input order, deduplicated.
    """
    candidates: list[str] = []
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates = [segment.strip().lower() for segment in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        candidates = [str(item).strip().lower() for item in raw]
    else:
        candidates = [str(raw).strip().lower()]

    aliases = {
        "plan": GUIDANCE_STAGE_PLANNING,
        "recommendation": GUIDANCE_STAGE_RECOMMENDATIONS,
        "outline_expansion": GUIDANCE_STAGE_OUTLINE,
        "outline-expand": GUIDANCE_STAGE_OUTLINE,
        "coordination_breakdown": GUIDANCE_STAGE_COORDINATION,
        "coordination-breakdown": GUIDANCE_STAGE_COORDINATION,
    }

    normalized: list[str] = []
    for item in candidates:
        if not item:
            continue
        stage = aliases.get(item, item)
        if stage not in GUIDANCE_STAGES_SUPPORTED:
            continue
        if stage in normalized:
            continue
        normalized.append(stage)
    return normalized


def build_recommendation_questionnaire(
    current_enabled: bool = False,
    current_top_k: int = 8,
    current_per_seed_limit: int = 5,
    current_min_score: float = 0.2,
) -> dict[str, Any]:
    """Build recommendation-stage guidance questions.

    Parameters
    ----------
    current_enabled : bool
        Current recommendation enabled flag.
    current_top_k : int
        Current recommendation top-k.
    current_per_seed_limit : int
        Current recommendation per-seed limit.
    current_min_score : float
        Current recommendation min-score threshold.

    Returns
    -------
    dict[str, Any]
        Questionnaire payload.
    """
    status_label = "enabled" if current_enabled else "disabled"
    return {
        "questions": [
            {
                "header": "Recommendations",
                "question": f"Recommendation workflow is currently {status_label}. Enable it for this run?",
                "required": True,
                "importance": 10,
                "options": [
                    {"label": "Yes, enable recommendations"},
                    {"label": "No, keep recommendations off"},
                ],
            },
            {
                "header": "Rec Strategy",
                "question": (
                    "Choose recommendation strategy "
                    f"(current top_k={current_top_k}, per_seed={current_per_seed_limit}, min_score={current_min_score:.2f})."
                ),
                "importance": 8,
                "options": [
                    {"label": "Conservative (top_k=4, per_seed=3, min_score=0.45)"},
                    {"label": "Balanced (top_k=8, per_seed=5, min_score=0.20)"},
                    {"label": "Aggressive (top_k=12, per_seed=8, min_score=0.10)"},
                ],
            },
            {
                "header": "Rec Focus",
                "question": "Which recommendation dimensions should be prioritized?",
                "multiSelect": True,
                "importance": 6,
                "options": [
                    {"label": "Topical fit"},
                    {"label": "Citation impact"},
                    {"label": "Novelty"},
                    {"label": "Methodological diversity"},
                ],
            },
            {
                "header": "Rec Notes",
                "question": "Any additional recommendation constraints?",
                "importance": 4,
            },
        ]
    }


def build_outline_questionnaire(
    current_enabled: bool = False,
    current_max_subsections: int = 3,
) -> dict[str, Any]:
    """Build outline-expansion guidance questions.

    Parameters
    ----------
    current_enabled : bool
        Current outline-expander enabled flag.
    current_max_subsections : int
        Current max subsections per section.

    Returns
    -------
    dict[str, Any]
        Questionnaire payload.
    """
    status_label = "enabled" if current_enabled else "disabled"
    return {
        "questions": [
            {
                "header": "Outline",
                "question": f"Outline expansion is currently {status_label}. Enable expansion this run?",
                "required": True,
                "importance": 10,
                "options": [
                    {"label": "Yes, expand outline"},
                    {"label": "No, keep current outline"},
                ],
            },
            {
                "header": "Outline Depth",
                "question": f"Select outline expansion depth (current max_subsections={current_max_subsections}).",
                "importance": 8,
                "options": [
                    {"label": "Shallow (max_subsections=2)"},
                    {"label": "Balanced (max_subsections=4)"},
                    {"label": "Deep (max_subsections=6)"},
                ],
            },
            {
                "header": "Outline Signals",
                "question": "Which expansion signals should be emphasized?",
                "multiSelect": True,
                "importance": 6,
                "options": [
                    {"label": "Transitions across sections"},
                    {"label": "Counterarguments"},
                    {"label": "Evidence focus"},
                    {"label": "Deliverables/actionability"},
                ],
            },
            {
                "header": "Outline Notes",
                "question": "Any additional outline-expansion guidance?",
                "importance": 4,
            },
        ]
    }


def build_coordination_questionnaire(
    current_max_iterations: int = 4,
    gap_requests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build coordination guidance questions, optionally enriched by gap requests.

    Parameters
    ----------
    current_max_iterations : int
        Current coordination max-iteration setting.
    gap_requests : list[dict[str, Any]] | None
        Optional gap-request payload from diagnostics.

    Returns
    -------
    dict[str, Any]
        Questionnaire payload.
    """
    components = _top_gap_components(gap_requests=gap_requests)
    component_options = [{"label": component.replace("_", " ")} for component in components[:5]]
    if not component_options:
        component_options = [
            {"label": "Outline alignment"},
            {"label": "Bibliography grounding"},
            {"label": "Knowledge graph coverage"},
            {"label": "Claim coherence"},
        ]

    questions: list[dict[str, Any]] = [
        {
            "header": "Coordination",
            "question": "Choose coordination strategy for this run.",
            "required": True,
            "importance": 10,
            "options": [
                {"label": "Balanced coordination"},
                {"label": "Strict outline alignment"},
                {"label": "Evidence-first remediation"},
            ],
        },
        {
            "header": "Coord Iterations",
            "question": f"Coordination currently uses max_iterations={current_max_iterations}. Increase revision iterations?",
            "importance": 8,
            "options": [
                {"label": "Yes, increase iterations"},
                {"label": "No, keep current iterations"},
            ],
        },
        {
            "header": "Coord Priority",
            "question": "Which coordination breakdown components should be prioritized?",
            "multiSelect": True,
            "importance": 8,
            "options": component_options,
        },
        {
            "header": "Coord Notes",
            "question": "Any additional coordination breakdown instructions?",
            "importance": 5,
        },
    ]
    return {"questions": questions}


def build_coordination_breakdown_questionnaire(
    gap_requests: list[dict[str, Any]],
    coordination_messages: list[dict[str, Any]] | None = None,
    current_max_iterations: int = 4,
) -> dict[str, Any]:
    """Build breakdown-focused coordination questions using diagnostics context.

    Parameters
    ----------
    gap_requests : list[dict[str, Any]]
        Gap-request diagnostics payload.
    coordination_messages : list[dict[str, Any]] | None
        Optional coordination message payload.
    current_max_iterations : int
        Current coordination max-iteration setting.

    Returns
    -------
    dict[str, Any]
        Questionnaire payload.
    """
    _ = coordination_messages
    questionnaire = build_coordination_questionnaire(
        current_max_iterations=current_max_iterations,
        gap_requests=gap_requests,
    )
    header_note = {
        "header": "Breakdown Summary",
        "question": _coordination_breakdown_summary(gap_requests=gap_requests),
        "importance": 9,
        "required": False,
    }
    questions = questionnaire.get("questions", [])
    if isinstance(questions, list):
        questions.insert(0, header_note)
    return questionnaire


def build_stage_questionnaire(
    stage: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stage-specific questionnaire.

    Parameters
    ----------
    stage : str
        Stage id (planning/recommendations/outline/coordination).
    context : dict[str, Any] | None
        Optional context payload to tailor default values.

    Returns
    -------
    dict[str, Any]
        Questionnaire payload.
    """
    payload = context or {}
    normalized = normalize_guidance_stages([stage])
    if not normalized:
        return {"questions": []}
    selected = normalized[0]

    if selected == GUIDANCE_STAGE_PLANNING:
        return build_planning_questionnaire()
    if selected == GUIDANCE_STAGE_RECOMMENDATIONS:
        return build_recommendation_questionnaire(
            current_enabled=bool(payload.get("enabled", False)),
            current_top_k=_safe_int(payload.get("top_k"), 8),
            current_per_seed_limit=_safe_int(payload.get("per_seed_limit"), 5),
            current_min_score=_safe_float(payload.get("min_score"), 0.2),
        )
    if selected == GUIDANCE_STAGE_OUTLINE:
        return build_outline_questionnaire(
            current_enabled=bool(payload.get("enabled", False)),
            current_max_subsections=_safe_int(payload.get("max_subsections"), 3),
        )
    if selected == GUIDANCE_STAGE_COORDINATION:
        gaps = payload.get("gap_requests", [])
        if isinstance(gaps, list) and gaps:
            return build_coordination_breakdown_questionnaire(
                gap_requests=[gap for gap in gaps if isinstance(gap, dict)],
                coordination_messages=payload.get("coordination_messages", []),
                current_max_iterations=_safe_int(payload.get("max_iterations"), 4),
            )
        return build_coordination_questionnaire(
            current_max_iterations=_safe_int(payload.get("max_iterations"), 4),
            gap_requests=[],
        )
    return {"questions": []}


def collect_stage_guidance(
    stage: str,
    context: dict[str, Any] | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    max_inputs: int = MAX_USER_INPUTS_PER_STAGE,
) -> dict[str, str]:
    """Collect guidance answers for one stage.

    Parameters
    ----------
    stage : str
        Stage id.
    context : dict[str, Any] | None
        Optional stage context.
    input_fn : Callable[[str], str]
        Input callback.
    output_fn : Callable[[str], None]
        Output callback.
    max_inputs : int
        Maximum stage inputs.

    Returns
    -------
    dict[str, str]
        Answer mapping for the stage.
    """
    questionnaire = build_stage_questionnaire(stage=stage, context=context)
    return collect_user_answers(
        input_data=questionnaire,
        input_fn=input_fn,
        output_fn=output_fn,
        max_inputs=max_inputs,
    )


def collect_user_guidance_bundle(
    stages: list[str],
    context_by_stage: dict[str, dict[str, Any]] | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    max_inputs: int = MAX_USER_INPUTS_PER_STAGE,
) -> UserGuidanceBundle:
    """Collect and infer multi-stage user guidance.

    Parameters
    ----------
    stages : list[str]
        Stage list to collect.
    context_by_stage : dict[str, dict[str, Any]] | None
        Optional mapping from stage id to context.
    input_fn : Callable[[str], str]
        Input callback.
    output_fn : Callable[[str], None]
        Output callback.
    max_inputs : int
        Maximum stage inputs.

    Returns
    -------
    UserGuidanceBundle
        Collected bundle.
    """
    normalized = normalize_guidance_stages(stages)
    contexts = context_by_stage or {}
    bundle = UserGuidanceBundle(stages=list(normalized))

    for stage in normalized:
        stage_context = contexts.get(stage, {})
        answers = collect_stage_guidance(
            stage=stage,
            context=stage_context,
            input_fn=input_fn,
            output_fn=output_fn,
            max_inputs=max_inputs,
        )
        bundle.answers_by_stage[stage] = answers
        if stage == GUIDANCE_STAGE_PLANNING:
            bundle.planning = infer_planning_guidance(answers)
        elif stage == GUIDANCE_STAGE_RECOMMENDATIONS:
            bundle.recommendations = infer_recommendation_guidance(answers)
        elif stage == GUIDANCE_STAGE_OUTLINE:
            bundle.outline = infer_outline_guidance(answers)
        elif stage == GUIDANCE_STAGE_COORDINATION:
            bundle.coordination = infer_coordination_guidance(answers)

    return bundle


def collect_user_answers(
    input_data: dict[str, Any],
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    max_inputs: int = MAX_USER_INPUTS_PER_STAGE,
) -> dict[str, str]:
    """Collect answers from a user for an ``AskUserQuestion`` payload.

    Parameters
    ----------
    input_data : dict[str, Any]
        Payload containing either ``questions`` (recommended) or a single
        ``question`` with optional choices.
    input_fn : Callable[[str], str]
        Function used to read user input.
    output_fn : Callable[[str], None]
        Function used to emit prompts.
    max_inputs : int
        Maximum number of user inputs to request for this task/stage.
        If more questions are supplied, only the top-scored subset is asked.

    Returns
    -------
    dict[str, str]
        Mapping from question text to normalized answer text.
    """
    bounded_input, total_questions, selected_count, selected_questions = _bounded_input_data(
        input_data=input_data,
        max_inputs=max_inputs,
    )
    _ = bounded_input
    if selected_count < total_questions:
        output_fn(
            f"Collected top {selected_count} prioritized prompts out of {total_questions} available for this stage."
        )
    answers: dict[str, str] = {}

    for question in selected_questions:
        question_text = str(question.get("question", "")).strip() or "Question"
        header = str(question.get("header", "")).strip()
        options = _question_options(question)
        multi_select = bool(question.get("multiSelect", False))

        if header:
            output_fn(f"\n{header}")
        output_fn(question_text)

        if options:
            for index, option in enumerate(options, start=1):
                output_fn(f"{index}. {option}")
            output_fn(f"{len(options) + 1}. Other (type your own response)")
            try:
                response = input_fn("> ").strip()
            except EOFError:
                response = ""
            answer = _parse_choice_response(response=response, options=options, multi_select=multi_select)
            if answer == "__OTHER__":
                try:
                    answer = input_fn("Enter response: ").strip()
                except EOFError:
                    answer = ""
        else:
            try:
                answer = input_fn("> ").strip()
            except EOFError:
                answer = ""

        answers[question_text] = answer

    return answers


def infer_planning_guidance(answers: dict[str, str]) -> PlanningGuidance:
    """Infer structured planning preferences from raw user answers.

    Parameters
    ----------
    answers : dict[str, str]
        Raw answer mapping.

    Returns
    -------
    PlanningGuidance
        Parsed planning guidance.
    """
    guidance = PlanningGuidance(answers=dict(answers))
    notes: list[str] = []

    for question, answer in answers.items():
        q = question.lower()
        a = answer.strip()
        if not a:
            continue
        if "plan" in q and ("style" in q or "focus" in q or "document" in q):
            guidance.planning_document_focus = a
            continue
        if "recommendation" in q:
            guidance.incorporate_recommendations = _parse_boolean_preference(a)
            continue
        if "outline" in q and ("expand" in q or "expansion" in q):
            guidance.expand_outline = _parse_boolean_preference(a)
            continue
        notes.append(a)

    guidance.additional_notes = " ".join(notes).strip()
    return guidance


def infer_recommendation_guidance(answers: dict[str, str]) -> RecommendationGuidance:
    """Infer recommendation-stage guidance from answers.

    Parameters
    ----------
    answers : dict[str, str]
        Raw answer mapping.

    Returns
    -------
    RecommendationGuidance
        Parsed recommendation guidance.
    """
    guidance = RecommendationGuidance(answers=dict(answers))
    notes: list[str] = []

    for question, answer in answers.items():
        q = question.lower()
        a = answer.strip()
        lowered = a.lower()
        if not a:
            continue
        if "enable" in q and "recommendation" in q:
            guidance.enable_recommendations = _parse_boolean_preference(a)
            continue
        if "strategy" in q:
            guidance.strategy = a
            if "conservative" in lowered:
                guidance.top_k = 4
                guidance.per_seed_limit = 3
                guidance.min_score = 0.45
            elif "balanced" in lowered:
                guidance.top_k = 8
                guidance.per_seed_limit = 5
                guidance.min_score = 0.20
            elif "aggressive" in lowered:
                guidance.top_k = 12
                guidance.per_seed_limit = 8
                guidance.min_score = 0.10
            continue
        if "dimension" in q or "prioritized" in q:
            guidance.focus_tags = [segment.strip() for segment in a.split(",") if segment.strip()]
            continue
        notes.append(a)

    guidance.additional_notes = " ".join(notes).strip()
    return guidance


def infer_outline_guidance(answers: dict[str, str]) -> OutlineGuidance:
    """Infer outline-expansion guidance from answers.

    Parameters
    ----------
    answers : dict[str, str]
        Raw answer mapping.

    Returns
    -------
    OutlineGuidance
        Parsed outline guidance.
    """
    guidance = OutlineGuidance(answers=dict(answers))
    notes: list[str] = []

    for question, answer in answers.items():
        q = question.lower()
        a = answer.strip()
        lowered = a.lower()
        if not a:
            continue
        if "enable expansion" in q or ("outline expansion" in q and "enable" in q):
            guidance.enable_outline_expansion = _parse_boolean_preference(a)
            continue
        if "expansion depth" in q:
            guidance.depth_profile = a
            if "shallow" in lowered:
                guidance.max_subsections = 2
            elif "balanced" in lowered:
                guidance.max_subsections = 4
            elif "deep" in lowered:
                guidance.max_subsections = 6
            continue
        if "signals" in q or "emphasized" in q:
            tags = [segment.strip().lower() for segment in a.split(",") if segment.strip()]
            guidance.include_transitions = any("transition" in tag for tag in tags) if tags else None
            guidance.include_counterarguments = any("counter" in tag for tag in tags) if tags else None
            continue
        notes.append(a)

    guidance.additional_notes = " ".join(notes).strip()
    return guidance


def infer_coordination_guidance(answers: dict[str, str]) -> CoordinationGuidance:
    """Infer coordination-stage guidance from answers.

    Parameters
    ----------
    answers : dict[str, str]
        Raw answer mapping.

    Returns
    -------
    CoordinationGuidance
        Parsed coordination guidance.
    """
    guidance = CoordinationGuidance(answers=dict(answers))
    notes: list[str] = []

    for question, answer in answers.items():
        q = question.lower()
        a = answer.strip()
        lowered = a.lower()
        if not a:
            continue
        if "coordination strategy" in q:
            guidance.coordination_strategy = a
            if "strict outline" in lowered:
                guidance.strict_outline_alignment = True
            continue
        if "increase revision iterations" in q:
            guidance.increase_revision_iterations = _parse_boolean_preference(a)
            if guidance.increase_revision_iterations:
                guidance.target_revision_iterations = 6
            continue
        if "breakdown components" in q:
            guidance.preferred_components = [segment.strip().lower().replace(" ", "_") for segment in a.split(",") if segment.strip()]
            continue
        if "breakdown summary" in q:
            guidance.breakdown_summary = a
            continue
        notes.append(a)

    guidance.additional_notes = " ".join(notes).strip()
    return guidance


def apply_recommendation_guidance(
    guidance: RecommendationGuidance,
    enable_recommendations: bool,
    top_k: int,
    per_seed_limit: int,
    min_score: float,
) -> tuple[bool, int, int, float]:
    """Apply recommendation guidance to runtime config values.

    Parameters
    ----------
    guidance : RecommendationGuidance
        Recommendation guidance.
    enable_recommendations : bool
        Existing recommendation toggle.
    top_k : int
        Existing top-k value.
    per_seed_limit : int
        Existing per-seed value.
    min_score : float
        Existing min-score value.

    Returns
    -------
    tuple[bool, int, int, float]
        Updated ``(enabled, top_k, per_seed_limit, min_score)``.
    """
    enabled = enable_recommendations
    resolved_top_k = max(0, top_k)
    resolved_per_seed = max(1, per_seed_limit)
    resolved_min_score = max(0.0, min(1.0, min_score))

    if guidance.enable_recommendations is not None:
        enabled = guidance.enable_recommendations
    if guidance.top_k is not None:
        resolved_top_k = max(0, guidance.top_k)
    if guidance.per_seed_limit is not None:
        resolved_per_seed = max(1, guidance.per_seed_limit)
    if guidance.min_score is not None:
        resolved_min_score = max(0.0, min(1.0, guidance.min_score))

    return enabled, resolved_top_k, resolved_per_seed, resolved_min_score


def apply_outline_guidance(
    guidance: OutlineGuidance,
    enable_outline_expander: bool,
    max_subsections: int,
) -> tuple[bool, int]:
    """Apply outline guidance to runtime config values.

    Parameters
    ----------
    guidance : OutlineGuidance
        Outline guidance.
    enable_outline_expander : bool
        Existing expander toggle.
    max_subsections : int
        Existing max subsections.

    Returns
    -------
    tuple[bool, int]
        Updated ``(enable_outline_expander, max_subsections)``.
    """
    enabled = enable_outline_expander
    resolved_max_subsections = max(1, max_subsections)

    if guidance.enable_outline_expansion is not None:
        enabled = guidance.enable_outline_expansion
    if guidance.max_subsections is not None:
        resolved_max_subsections = max(1, guidance.max_subsections)

    return enabled, resolved_max_subsections


def apply_coordination_guidance(
    guidance: CoordinationGuidance,
    coordination_max_iterations: int,
) -> int:
    """Apply coordination guidance to revision-iteration setting.

    Parameters
    ----------
    guidance : CoordinationGuidance
        Coordination guidance.
    coordination_max_iterations : int
        Existing revision-iteration cap.

    Returns
    -------
    int
        Updated revision-iteration cap.
    """
    iterations = max(1, coordination_max_iterations)
    if guidance.target_revision_iterations is not None:
        iterations = max(1, guidance.target_revision_iterations)
    elif guidance.increase_revision_iterations is True:
        iterations = max(iterations, iterations + 2)
    return iterations


def apply_guidance_to_pipeline_flags(
    guidance: PlanningGuidance,
    enable_paper_recommendations: bool,
    enable_outline_expander: bool,
) -> tuple[bool, bool]:
    """Apply guidance-derived overrides to pipeline feature flags.

    Parameters
    ----------
    guidance : PlanningGuidance
        Structured guidance preferences.
    enable_paper_recommendations : bool
        Existing recommendation flag.
    enable_outline_expander : bool
        Existing outline-expansion flag.

    Returns
    -------
    tuple[bool, bool]
        ``(recommendations_enabled, outline_expander_enabled)``
    """
    recommendations = enable_paper_recommendations
    outline_expander = enable_outline_expander

    if guidance.incorporate_recommendations is not None:
        recommendations = guidance.incorporate_recommendations
    if guidance.expand_outline is not None:
        outline_expander = guidance.expand_outline

    return recommendations, outline_expander


def build_planning_guidance_prompt(task_description: str) -> str:
    """Build an Agent SDK prompt that requests user guidance before planning.

    Parameters
    ----------
    task_description : str
        Task context supplied by the caller.

    Returns
    -------
    str
        Prompt text instructing the agent to use ``AskUserQuestion``.
    """
    cleaned = task_description.strip()
    return (
        "Prepare a planning document for the following task.\n\n"
        f"Task: {cleaned}\n\n"
        "Before drafting the plan, call AskUserQuestion to confirm:\n"
        "1) preferred planning-document style,\n"
        "2) whether to incorporate recommendation proposals,\n"
        "3) whether to expand the outline before drafting,\n"
        "4) any additional constraints.\n"
        "Do not request more than 10 user inputs at this step; "
        "if more candidates exist, score by importance and ask only the top 10.\n\n"
        "After collecting answers, produce a concise implementation plan."
    )


async def request_planning_guidance_via_agent_sdk(
    task_description: str,
    max_turns: int = 12,
    tools: list[str] | None = None,
) -> tuple[str, PlanningGuidance | None]:
    """Run a planning session through Claude Agent SDK with ``AskUserQuestion``.

    Parameters
    ----------
    task_description : str
        Planning task context.
    max_turns : int
        Maximum turns allowed in the agent session.
    tools : list[str] | None
        Optional tool allow-list. Defaults to read/search + ``AskUserQuestion``.

    Returns
    -------
    tuple[str, PlanningGuidance | None]
        Final assistant text and inferred planning guidance (if captured).
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except Exception as exc:
        raise RuntimeError(
            "Claude Agent SDK is required. Install with: pip install -e .[agent_sdk]"
        ) from exc

    default_tools = ["Read", "Glob", "Grep", "AskUserQuestion"]
    handler = AgentSDKUserInputHandler()
    prompt = build_planning_guidance_prompt(task_description=task_description)

    option_kwargs: dict[str, Any] = {}
    try:
        from claude_agent_sdk.types import HookMatcher

        async def _dummy_hook(input_data: dict[str, Any], tool_response: dict[str, Any], context: Any) -> dict[str, Any]:
            _ = input_data, tool_response, context
            return {}

        option_kwargs["hooks"] = {"PreToolUse": [HookMatcher(matcher="", hooks=[_dummy_hook])]}
    except Exception:
        option_kwargs = {}

    options = ClaudeAgentOptions(
        tools=list(tools or default_tools),
        can_use_tool=handler.can_use_tool,
        max_turns=max(1, max_turns),
        **option_kwargs,
    )

    text_chunks: list[str] = []
    async for message in query(prompt=prompt, options=options):
        text = _extract_message_text(message)
        if text:
            text_chunks.append(text)

    return "\n".join(text_chunks).strip(), handler.last_guidance


def build_codex_ask_user_question_tool(
    tool_name: str = CODEX_ASK_USER_QUESTION_TOOL_NAME,
) -> dict[str, Any]:
    """Build an OpenAI/Codex function-tool schema for requesting user input.

    Parameters
    ----------
    tool_name : str
        Function tool name exposed to the model.

    Returns
    -------
    dict[str, Any]
        Function tool schema compatible with OpenAI Responses API.
    """
    return {
        "type": "function",
        "name": tool_name,
        "description": (
            "Ask the user for focused guidance. "
            "Limit to at most 10 questions per stage; if more, pass all candidates and the host will prioritize."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "header": {"type": "string"},
                            "question": {"type": "string"},
                            "required": {"type": "boolean"},
                            "importance": {"type": "number"},
                            "priority": {"type": "number"},
                            "multiSelect": {"type": "boolean"},
                            "options": {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {
                                            "type": "object",
                                            "properties": {
                                                "label": {"type": "string"},
                                                "description": {"type": "string"},
                                            },
                                        },
                                    ]
                                },
                            },
                        },
                        "required": ["question"],
                    },
                },
            },
        },
    }


class OpenAICodexUserInputHandler:
    """Handle OpenAI/Codex function-tool calls for user-question collection.

    Parameters
    ----------
    input_fn : Callable[[str], str]
        Interactive input reader.
    output_fn : Callable[[str], None]
        Interactive output writer.
    max_inputs_per_stage : int
        Hard cap of user inputs requested in a single stage/task.
    tool_names : tuple[str, ...]
        Accepted tool names treated as AskUserQuestion equivalents.
    """

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        max_inputs_per_stage: int = MAX_USER_INPUTS_PER_STAGE,
        tool_names: tuple[str, ...] = (CODEX_ASK_USER_QUESTION_TOOL_NAME, "ask_user_question"),
    ) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.max_inputs_per_stage = max(1, max_inputs_per_stage)
        self.tool_names = set(tool_names)
        self.last_answers: dict[str, str] = {}
        self.last_guidance: PlanningGuidance | None = None
        self._call_lock = Lock()

    def handle_function_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        """Handle a function call payload and return tool output content.

        Parameters
        ----------
        tool_name : str
            Called function tool name.
        arguments : dict[str, Any] | str | None
            Function arguments payload.

        Returns
        -------
        dict[str, Any]
            Tool output payload containing answers and stage metadata.
        """
        if tool_name not in self.tool_names:
            return {
                "status": "ignored",
                "message": f"Unhandled tool call: {tool_name}",
            }

        if not self._call_lock.acquire(blocking=False):
            return {
                "status": "busy",
                "message": "AskUserQuestion call already in progress for this handler; retry once it completes.",
            }
        try:
            payload = _coerce_function_arguments(arguments)
            bounded_input, total_questions, selected_count, _ = _bounded_input_data(
                input_data=payload,
                max_inputs=self.max_inputs_per_stage,
            )
            answers = collect_user_answers(
                input_data=bounded_input,
                input_fn=self.input_fn,
                output_fn=self.output_fn,
                max_inputs=self.max_inputs_per_stage,
            )
            guidance = infer_planning_guidance(answers)
            self.last_answers = dict(answers)
            self.last_guidance = guidance

            return {
                "status": "ok",
                "answers": answers,
                "meta": {
                    "tool_name": tool_name,
                    "max_inputs_per_stage": self.max_inputs_per_stage,
                    "questions_requested": total_questions,
                    "questions_asked": selected_count,
                },
                "guidance": {
                    "planning_document_focus": guidance.planning_document_focus,
                    "incorporate_recommendations": guidance.incorporate_recommendations,
                    "expand_outline": guidance.expand_outline,
                    "additional_notes": guidance.additional_notes,
                },
            }
        finally:
            self._call_lock.release()


async def request_planning_guidance_via_openai_codex(
    task_description: str,
    model: str = "gpt-5-codex",
    max_turns: int = 12,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, PlanningGuidance | None]:
    """Run a planning session through OpenAI/Codex with AskUserQuestion function tools.

    Parameters
    ----------
    task_description : str
        Planning task context.
    model : str
        Codex/OpenAI model name.
    max_turns : int
        Maximum responses loop iterations.
    tools : list[dict[str, Any]] | None
        Optional tool definitions. Defaults to ``AskUserQuestion`` tool schema.

    Returns
    -------
    tuple[str, PlanningGuidance | None]
        Final assistant text and inferred guidance captured from user input.
    """
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("OpenAI SDK is required. Install with: pip install openai") from exc

    client = OpenAI()
    handler = OpenAICodexUserInputHandler()
    tool_defs = list(tools or [build_codex_ask_user_question_tool()])
    prompt = build_planning_guidance_prompt(task_description=task_description)

    response = client.responses.create(model=model, input=prompt, tools=tool_defs)
    text_chunks: list[str] = []

    for _ in range(max(1, max_turns)):
        text = _extract_codex_output_text(response)
        if text:
            text_chunks.append(text)

        calls = _extract_codex_function_calls(response)
        if not calls:
            break

        function_outputs: list[dict[str, Any]] = []
        for call in calls:
            output_payload = handler.handle_function_call(
                tool_name=call["name"],
                arguments=call.get("arguments"),
            )
            function_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(output_payload),
                }
            )

        next_kwargs: dict[str, Any] = {
            "model": model,
            "input": function_outputs,
            "tools": tool_defs,
        }
        response_id = _safe_attr(response, "id")
        if isinstance(response_id, str) and response_id:
            next_kwargs["previous_response_id"] = response_id
        response = client.responses.create(**next_kwargs)

    return "\n".join(chunk for chunk in text_chunks if chunk.strip()).strip(), handler.last_guidance


class AgentSDKUserInputHandler:
    """Handle Agent SDK ``AskUserQuestion`` tool calls with local user input.

    Parameters
    ----------
    input_fn : Callable[[str], str]
        Interactive input reader.
    output_fn : Callable[[str], None]
        Interactive output writer.
    """

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        max_inputs_per_stage: int = MAX_USER_INPUTS_PER_STAGE,
    ) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.max_inputs_per_stage = max(1, max_inputs_per_stage)
        self.last_answers: dict[str, str] = {}
        self.last_guidance: PlanningGuidance | None = None
        self._call_lock = Lock()

    async def can_use_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        """Approve or deny tool use and inject user answers for ``AskUserQuestion``.

        Parameters
        ----------
        tool_name : str
            Name of tool being requested.
        input_data : dict[str, Any]
            Tool input payload.
        context : Any | None
            Optional Agent SDK context object.

        Returns
        -------
        Any
            Agent SDK permission decision object (or dict fallback).
        """
        _ = context
        if tool_name != "AskUserQuestion":
            return _permission_allow(input_data=input_data)

        if not isinstance(input_data, dict):
            return _permission_deny(message="AskUserQuestion payload must be a JSON object.")

        if not self._call_lock.acquire(blocking=False):
            return _permission_deny(
                message="AskUserQuestion call already in progress for this handler; retry once it completes."
            )

        try:
            bounded_input, _, _, _ = _bounded_input_data(
                input_data=input_data,
                max_inputs=self.max_inputs_per_stage,
            )
            answers = collect_user_answers(
                input_data=bounded_input,
                input_fn=self.input_fn,
                output_fn=self.output_fn,
                max_inputs=self.max_inputs_per_stage,
            )
            updated_input = dict(bounded_input)
            updated_input["answers"] = answers

            self.last_answers = dict(answers)
            self.last_guidance = infer_planning_guidance(answers)
            return _permission_allow(input_data=updated_input)
        finally:
            self._call_lock.release()


def _normalize_questions(input_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize single-question and multi-question payload shapes."""
    raw_questions = input_data.get("questions")
    if isinstance(raw_questions, list):
        normalized = [question for question in raw_questions if isinstance(question, dict)]
        if normalized:
            return normalized

    question_text = input_data.get("question")
    if isinstance(question_text, str) and question_text.strip():
        options = input_data.get("choices", input_data.get("options", []))
        payload: dict[str, Any] = {"question": question_text}
        if isinstance(options, list):
            payload["options"] = options
        return [payload]

    return []


def _bounded_input_data(
    input_data: dict[str, Any],
    max_inputs: int,
) -> tuple[dict[str, Any], int, int, list[dict[str, Any]]]:
    """Cap an AskUserQuestion payload to a manageable number of inputs."""
    questions = _normalize_questions(input_data)
    selected_questions = _select_stage_questions(questions=questions, max_inputs=max_inputs)
    bounded_input = dict(input_data)

    if "questions" in input_data and isinstance(input_data.get("questions"), list):
        bounded_input["questions"] = selected_questions

    return bounded_input, len(questions), len(selected_questions), selected_questions


def _select_stage_questions(questions: list[dict[str, Any]], max_inputs: int) -> list[dict[str, Any]]:
    """Select manageable top-priority questions for a single stage/task.

    Parameters
    ----------
    questions : list[dict[str, Any]]
        Candidate questions requested by the agent.
    max_inputs : int
        Maximum number of user inputs allowed in this stage.

    Returns
    -------
    list[dict[str, Any]]
        Selected questions capped by ``max_inputs``.
    """
    if max_inputs <= 0:
        return []
    if len(questions) <= max_inputs:
        return questions

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, question in enumerate(questions):
        score = _question_importance_score(question)
        scored.append((score, index, question))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:max_inputs]
    selected.sort(key=lambda item: item[1])
    return [item[2] for item in selected]


def _question_importance_score(question: dict[str, Any]) -> float:
    """Score a question for prioritization when question count exceeds stage limits."""
    score = 0.0

    explicit = question.get("importance", question.get("priority", None))
    if isinstance(explicit, (int, float)):
        score += float(explicit) * 100.0

    if bool(question.get("required", False)):
        score += 50.0

    text = str(question.get("question", "")).strip().lower()
    header = str(question.get("header", "")).strip().lower()
    combined = f"{header} {text}".strip()

    keyword_weights = (
        ("must", 18.0),
        ("required", 18.0),
        ("critical", 16.0),
        ("blocking", 16.0),
        ("goal", 12.0),
        ("scope", 12.0),
        ("constraint", 12.0),
        ("deadline", 10.0),
        ("risk", 10.0),
        ("recommendation", 8.0),
        ("outline", 8.0),
        ("plan", 8.0),
    )
    for token, weight in keyword_weights:
        if token in combined:
            score += weight

    # Questions with options are generally faster to answer and good for early triage.
    if _question_options(question):
        score += 3.0

    return score


def _question_options(question: dict[str, Any]) -> list[str]:
    """Extract option labels from a single question payload."""
    raw = question.get("options", question.get("choices", []))
    if not isinstance(raw, list):
        return []

    labels: list[str] = []
    for option in raw:
        if isinstance(option, str):
            label = option.strip()
            if label:
                labels.append(label)
            continue
        if isinstance(option, dict):
            label = str(option.get("label", "")).strip()
            if label:
                labels.append(label)
    return labels


def _parse_choice_response(response: str, options: list[str], multi_select: bool) -> str:
    """Parse a numeric/text response against option labels."""
    if not response:
        return ""

    if multi_select:
        parts = [part.strip() for part in response.split(",") if part.strip()]
        selected: list[str] = []
        for part in parts:
            selected_value = _resolve_choice_value(part=part, options=options)
            if selected_value == "__OTHER__":
                return "__OTHER__"
            if selected_value:
                selected.append(selected_value)
        return ", ".join(selected) if selected else response

    selected_value = _resolve_choice_value(part=response.strip(), options=options)
    if selected_value:
        return selected_value
    return response


def _resolve_choice_value(part: str, options: list[str]) -> str:
    """Resolve an individual choice token into an option label."""
    if not part:
        return ""

    if part.isdigit():
        index = int(part)
        if 1 <= index <= len(options):
            return options[index - 1]
        if index == len(options) + 1:
            return "__OTHER__"
        return ""

    lowered = part.lower()
    if lowered in {"other", "custom"}:
        return "__OTHER__"
    for option in options:
        if option.lower() == lowered:
            return option
    return ""


def _parse_boolean_preference(answer: str) -> bool | None:
    """Parse yes/no style preferences from free-form answer text."""
    lowered = answer.strip().lower()
    if not lowered:
        return None

    positive_tokens = ("yes", "include", "incorporate", "enable", "expand", "true")
    negative_tokens = ("no", "skip", "exclude", "disable", "keep current", "false")

    if any(token in lowered for token in positive_tokens):
        return True
    if any(token in lowered for token in negative_tokens):
        return False
    return None


def _top_gap_components(gap_requests: list[dict[str, Any]] | None, max_items: int = 6) -> list[str]:
    """Return top components by frequency from gap requests."""
    if not isinstance(gap_requests, list):
        return []
    counts: dict[str, int] = {}
    for gap in gap_requests:
        if not isinstance(gap, dict):
            continue
        component = str(gap.get("component", "")).strip().lower().replace(" ", "_")
        if not component:
            continue
        counts[component] = counts.get(component, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [component for component, _ in ranked[:max(1, max_items)]]


def _coordination_breakdown_summary(gap_requests: list[dict[str, Any]]) -> str:
    """Build a compact human-readable summary of coordination breakdown signals."""
    components = _top_gap_components(gap_requests=gap_requests, max_items=4)
    if not components:
        return "No explicit coordination gaps were detected. Provide any proactive coordination preferences."
    rendered = ", ".join(component.replace("_", " ") for component in components)
    return (
        "Detected coordination pressure in: "
        + rendered
        + ". Which breakdowns should be addressed first in the next run?"
    )


def _safe_int(value: Any, default: int) -> int:
    """Coerce an int with fallback."""
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    """Coerce a float with fallback."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _permission_allow(input_data: dict[str, Any]) -> Any:
    """Create a permission-allow result with graceful fallback when SDK types are unavailable."""
    try:
        from claude_agent_sdk.types import PermissionResultAllow

        return PermissionResultAllow(updated_input=input_data)
    except Exception:
        return {"behavior": "allow", "updated_input": input_data}


def _permission_deny(message: str) -> Any:
    """Create a permission-deny result with graceful fallback when SDK types are unavailable."""
    try:
        from claude_agent_sdk.types import PermissionResultDeny

        return PermissionResultDeny(message=message)
    except Exception:
        return {"behavior": "deny", "message": message}


def _extract_message_text(message: Any) -> str:
    """Extract plain-text content from Agent SDK stream messages."""
    payload = message
    if hasattr(message, "model_dump"):
        try:
            payload = message.model_dump()
        except Exception:
            payload = message

    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        content = payload.get("content")
        return _coerce_content_text(content)

    content = getattr(payload, "content", None)
    return _coerce_content_text(content)


def _coerce_content_text(content: Any) -> str:
    """Coerce content payloads into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(part for part in parts if part.strip()).strip()
    return ""


def _coerce_function_arguments(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    """Coerce function-call arguments into a mapping."""
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"question": stripped}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _safe_attr(payload: Any, name: str) -> Any:
    """Read an attribute/key safely across dict and object payload types."""
    if isinstance(payload, dict):
        return payload.get(name)
    return getattr(payload, name, None)


def _extract_codex_output_text(response: Any) -> str:
    """Extract plain output text from an OpenAI Responses API payload."""
    text = _safe_attr(response, "output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = _safe_attr(response, "output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            item_type = _safe_attr(item, "type")
            if item_type not in {"message", "output_text"}:
                continue
            content = _safe_attr(item, "content")
            if isinstance(content, list):
                for block in content:
                    block_text = _safe_attr(block, "text")
                    if isinstance(block_text, str) and block_text.strip():
                        parts.append(block_text.strip())
        return "\n".join(parts).strip()

    return ""


def _extract_codex_function_calls(response: Any) -> list[dict[str, Any]]:
    """Extract function-call items from an OpenAI Responses API payload."""
    output = _safe_attr(response, "output")
    if not isinstance(output, list):
        return []

    calls: list[dict[str, Any]] = []
    for item in output:
        item_type = _safe_attr(item, "type")
        if item_type != "function_call":
            continue
        name = _safe_attr(item, "name")
        call_id = _safe_attr(item, "call_id")
        arguments = _safe_attr(item, "arguments")
        if isinstance(name, str) and isinstance(call_id, str):
            calls.append({"name": name, "call_id": call_id, "arguments": arguments})
    return calls
