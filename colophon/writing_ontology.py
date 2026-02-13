"""Companion writing ontology context and validation routines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Chapter, Source


@dataclass(slots=True)
class WritingOntologyValidationFinding:
    """Structured validation finding from a companion writing ontology.

    Parameters
    ----------
    code : str
        Stable rule identifier.
    severity : str
        Severity label (for example ``warning`` or ``info``).
    category : str
        Validation category such as ``structure`` or ``rhetoric``.
    component : str
        Target component, such as ``outline``, ``bibliography``, or ``prompts``.
    message : str
        Human-readable issue summary.
    suggestion : str
        Suggested remediation action.
    related_id : str, optional
        Optional chapter/section identifier associated with the finding.
    """

    code: str
    severity: str
    category: str
    component: str
    message: str
    suggestion: str
    related_id: str = ""

    def to_dict(self) -> dict[str, str]:
        """Convert finding into JSON-serializable mapping.

        Returns
        -------
        dict[str, str]
            Serialized finding payload.
        """
        return {
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "component": self.component,
            "message": self.message,
            "suggestion": self.suggestion,
            "related_id": self.related_id,
        }


@dataclass(slots=True)
class WritingOntologyValidationResult:
    """Aggregate companion-ontology validation result.

    Parameters
    ----------
    enabled : bool
        Whether a companion ontology payload was available.
    ontology_id : str
        Companion ontology identifier.
    ontology_name : str
        Companion ontology display name.
    ontology_version : str
        Companion ontology version string.
    form_id : str
        Functional form id used for compatibility checks.
    compatible : bool
        Whether the selected functional form is declared compatible.
    assumptions : list[str]
        Assumption ids declared in the ontology.
    rule_ids : list[str]
        Validation rule ids evaluated.
    findings : list[WritingOntologyValidationFinding]
        Collected findings.
    """

    enabled: bool
    ontology_id: str = ""
    ontology_name: str = ""
    ontology_version: str = ""
    form_id: str = ""
    compatible: bool = True
    assumptions: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    findings: list[WritingOntologyValidationFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result into JSON-serializable mapping.

        Returns
        -------
        dict[str, Any]
            Serialized result including finding counts and payloads.
        """
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return {
            "enabled": self.enabled,
            "ontology_id": self.ontology_id,
            "ontology_name": self.ontology_name,
            "ontology_version": self.ontology_version,
            "form_id": self.form_id,
            "compatible": self.compatible,
            "assumptions": self.assumptions,
            "rule_ids": self.rule_ids,
            "finding_counts": counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def build_writing_ontology_context(
    ontology_payload: dict[str, Any] | None,
    form_id: str = "",
) -> dict[str, Any]:
    """Build runtime prompt context from companion ontology payload.

    Parameters
    ----------
    ontology_payload : dict[str, Any] | None
        Companion ontology payload.
    form_id : str, optional
        Selected functional-form id for compatibility checks.

    Returns
    -------
    dict[str, Any]
        Runtime context including compatibility flags and merged agent prompts.
    """
    context = {
        "enabled": False,
        "ontology_id": "",
        "ontology_name": "",
        "ontology_version": "",
        "form_id": form_id,
        "compatible": True,
        "global_prompts": [],
        "agent_prompts": {},
        "assumptions": [],
        "validation_rule_ids": [],
    }
    if not isinstance(ontology_payload, dict):
        return context

    context["enabled"] = True
    context["ontology_id"] = _string(ontology_payload.get("id"))
    context["ontology_name"] = _string(ontology_payload.get("name"))
    context["ontology_version"] = _string(ontology_payload.get("version"))

    compatibility = _mapping(ontology_payload.get("compatibility"))
    compatible_forms = _string_list(compatibility.get("compatible_form_ids"))
    if form_id and compatible_forms:
        context["compatible"] = form_id.lower() in {value.lower() for value in compatible_forms}

    prompts = _mapping(ontology_payload.get("background_prompts"))
    global_prompts = _string_list(prompts.get("global"))
    context["global_prompts"] = global_prompts
    raw_agents = _mapping(prompts.get("agents"))
    form_overrides = _mapping(prompts.get("form_overrides"))
    form_specific = _string_list(form_overrides.get(form_id))

    merged: dict[str, str] = {}
    for role in _known_agent_roles():
        role_prompts = _string_list(raw_agents.get(role))
        pieces = [value for value in [*global_prompts, *form_specific, *role_prompts] if value]
        merged[role] = " ".join(pieces).strip()
    context["agent_prompts"] = merged

    assumptions = ontology_payload.get("assumptions", [])
    if isinstance(assumptions, list):
        context["assumptions"] = [
            {"id": _string(row.get("id")), "statement": _string(row.get("statement"))}
            for row in assumptions
            if isinstance(row, dict) and _string(row.get("id"))
        ]

    rules = _mapping(ontology_payload.get("validations")).get("rules", [])
    if isinstance(rules, list):
        context["validation_rule_ids"] = [
            _string(rule.get("id")) for rule in rules if isinstance(rule, dict) and _string(rule.get("id"))
        ]
    return context


def run_writing_ontology_validation(
    ontology_payload: dict[str, Any] | None,
    outline: list[dict],
    bibliography: list[Source],
    prompts: dict[str, str],
    chapters: list[Chapter],
    form_id: str = "",
    functional_form: dict[str, Any] | None = None,
    coordination_revision: dict[str, Any] | None = None,
    max_findings: int = 32,
) -> WritingOntologyValidationResult:
    """Run validation checks declared in companion writing ontology.

    Parameters
    ----------
    ontology_payload : dict[str, Any] | None
        Companion ontology payload.
    outline : list[dict]
        Effective outline used by the pipeline.
    bibliography : list[Source]
        Bibliography used for retrieval.
    prompts : dict[str, str]
        Prompt template overrides.
    chapters : list[Chapter]
        Generated chapter payload.
    form_id : str, optional
        Selected functional-form id.
    functional_form : dict[str, Any] | None, optional
        Selected functional-form object.
    coordination_revision : dict[str, Any] | None, optional
        Revision-loop diagnostics for process checks.
    max_findings : int, optional
        Maximum finding count.

    Returns
    -------
    WritingOntologyValidationResult
        Validation result payload.
    """
    result = WritingOntologyValidationResult(enabled=isinstance(ontology_payload, dict), form_id=form_id)
    if not isinstance(ontology_payload, dict):
        return result

    result.ontology_id = _string(ontology_payload.get("id"))
    result.ontology_name = _string(ontology_payload.get("name"))
    result.ontology_version = _string(ontology_payload.get("version"))

    compatibility = _mapping(ontology_payload.get("compatibility"))
    compatible_forms = {value.lower() for value in _string_list(compatibility.get("compatible_form_ids"))}
    if form_id and compatible_forms:
        result.compatible = form_id.lower() in compatible_forms

    assumptions = ontology_payload.get("assumptions", [])
    if isinstance(assumptions, list):
        result.assumptions = [
            _string(row.get("id")) for row in assumptions if isinstance(row, dict) and _string(row.get("id"))
        ]

    rules = _mapping(ontology_payload.get("validations")).get("rules", [])
    if not isinstance(rules, list):
        return result
    result.rule_ids = [_string(rule.get("id")) for rule in rules if isinstance(rule, dict) and _string(rule.get("id"))]

    chapter_titles, section_titles = _outline_titles(outline)
    outline_tokens = " ".join([*chapter_titles, *section_titles]).lower()

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = _string(rule.get("id")).lower()
        if not rule_id:
            continue

        if rule_id == "question_problem_presence":
            keywords = _lower_set(_string_list(rule.get("keywords"))) or {"question", "problem", "puzzle", "dilemma"}
            fallback_element_ids = _string_list(rule.get("fallback_element_ids"))
            has_keyword = any(keyword in outline_tokens for keyword in keywords)
            has_element = any(
                element_id.lower() in outline_tokens or element_id.lower().replace("_", " ") in outline_tokens
                for element_id in fallback_element_ids
            )
            if not has_keyword and not has_element:
                _append_finding(
                    result,
                    rule=rule,
                    message=f"Outline does not clearly surface a question/problem framing ({', '.join(sorted(keywords))}).",
                    suggestion="Add an early section that states a non-obvious, answerable problem and stakes.",
                    max_findings=max_findings,
                )
            continue

        if rule_id == "method_visibility":
            keywords = _lower_set(_string_list(rule.get("keywords"))) or {"method", "methods", "approach", "data"}
            if not any(keyword in outline_tokens for keyword in keywords):
                _append_finding(
                    result,
                    rule=rule,
                    message="Outline does not visibly signal method or evidence-handling sections.",
                    suggestion="Add a method/evidence section clarifying how claims are warranted.",
                    max_findings=max_findings,
                )
            continue

        if rule_id == "implication_visibility":
            keywords = _lower_set(_string_list(rule.get("keywords"))) or {"implication", "conclusion", "stakes", "limits"}
            if not any(keyword in outline_tokens for keyword in keywords):
                _append_finding(
                    result,
                    rule=rule,
                    message="Outline does not visibly surface implications, limits, or conclusion framing.",
                    suggestion="Add implications/conclusion sections to state what follows and where claims stop.",
                    max_findings=max_findings,
                )
            continue

        if rule_id == "bibliography_grounding_density":
            min_text_chars = _int(rule.get("min_text_chars"), 80)
            min_ratio = _float(rule.get("min_grounded_ratio"), 0.6)
            grounded = 0
            for source in bibliography:
                if len(source.text.strip()) >= min_text_chars:
                    grounded += 1
                    continue
                abstract = _string(source.metadata.get("abstract")) if isinstance(source.metadata, dict) else ""
                if len(abstract.strip()) >= min_text_chars:
                    grounded += 1
            ratio = grounded / len(bibliography) if bibliography else 0.0
            if ratio < min_ratio:
                _append_finding(
                    result,
                    rule=rule,
                    message=(
                        f"Only {grounded}/{len(bibliography) if bibliography else 0} bibliography entries "
                        f"meet grounding length >= {min_text_chars} chars."
                    ),
                    suggestion="Add abstracts/full text in bibliography so retrieval and claim grounding have enough evidence.",
                    max_findings=max_findings,
                )
            continue

        if rule_id == "assertion_evidence_analysis_ratio":
            min_ratio = _float(rule.get("min_ratio"), 0.5)
            analysis_keywords = _lower_set(_string_list(rule.get("analysis_keywords"))) or {
                "because",
                "therefore",
                "suggests",
                "indicates",
                "implies",
                "demonstrates",
                "shows",
            }
            paragraph_rows = [paragraph.text for chapter in chapters for section in chapter.sections for paragraph in section.paragraphs]
            if not paragraph_rows:
                continue
            balanced = 0
            for paragraph_text in paragraph_rows:
                normalized = paragraph_text.lower()
                has_citation = "[" in paragraph_text and "]" in paragraph_text
                has_analysis = any(keyword in normalized for keyword in analysis_keywords)
                if has_citation and has_analysis:
                    balanced += 1
            ratio = balanced / len(paragraph_rows)
            if ratio < min_ratio:
                _append_finding(
                    result,
                    rule=rule,
                    message=(
                        f"Only {balanced}/{len(paragraph_rows)} paragraphs show both citation markers and analysis signals "
                        f"(ratio {ratio:.2f} < {min_ratio:.2f})."
                    ),
                    suggestion="Revise paragraphs to follow an assertion-evidence-analysis pattern with explicit warrants.",
                    max_findings=max_findings,
                )
            continue

        if rule_id == "prompt_argument_scaffolding":
            required_prompt_keys = _string_list(rule.get("required_prompt_keys")) or [
                "claim_template",
                "paragraph_template",
            ]
            missing = [key for key in required_prompt_keys if key not in prompts]
            claim_placeholders = _string_list(rule.get("claim_template_placeholders")) or [
                "{lead_entity}",
                "{source_title}",
            ]
            claim_template = _string(prompts.get("claim_template"))
            missing_placeholders = [token for token in claim_placeholders if token not in claim_template] if claim_template else []
            if missing or missing_placeholders:
                message = "Prompt overrides omit argumentative scaffolding expected by the ontology."
                details: list[str] = []
                if missing:
                    details.append("missing prompt keys: " + ", ".join(missing))
                if missing_placeholders:
                    details.append("missing claim placeholders: " + ", ".join(missing_placeholders))
                if details:
                    message += " " + "; ".join(details) + "."
                _append_finding(
                    result,
                    rule=rule,
                    message=message,
                    suggestion=(
                        "Provide claim/paragraph templates preserving evidence linkage placeholders "
                        "and explicit argument progression cues."
                    ),
                    max_findings=max_findings,
                )
            continue

        if rule_id == "iterative_revision_process":
            if not isinstance(coordination_revision, dict):
                continue
            enabled = bool(coordination_revision.get("enabled", False))
            min_iterations = _int(rule.get("min_iterations_if_enabled"), 2)
            iterations = _int(coordination_revision.get("iterations_run"), 0)
            if enabled and iterations < min_iterations:
                _append_finding(
                    result,
                    rule=rule,
                    message=(
                        f"Coordination revise loop ran {iterations} iteration(s); ontology expects >= {min_iterations} "
                        "when iterative editing is enabled."
                    ),
                    suggestion="Increase coordination iteration budget or inspect why loop stabilized prematurely.",
                    max_findings=max_findings,
                )
            continue

    return result


def _known_agent_roles() -> list[str]:
    """Return supported agent-role keys for background prompts.

    Returns
    -------
    list[str]
        Ordered list of known role names.
    """
    return [
        "claim_author_agent",
        "paragraph_agent",
        "outline_expander",
        "paragraph_coordinator",
        "section_coordinator",
        "chapter_coordinator",
        "book_coordinator",
    ]


def _outline_titles(outline: list[dict]) -> tuple[list[str], list[str]]:
    """Collect chapter and section titles from outline payload.

    Parameters
    ----------
    outline : list[dict]
        Outline chapter rows.

    Returns
    -------
    tuple[list[str], list[str]]
        Chapter titles and section titles.
    """
    chapter_titles: list[str] = []
    section_titles: list[str] = []
    for chapter in outline:
        if not isinstance(chapter, dict):
            continue
        chapter_title = _string(chapter.get("title"))
        if chapter_title:
            chapter_titles.append(chapter_title)
        sections = chapter.get("sections", [])
        if not isinstance(sections, list):
            continue
        for row in sections:
            if isinstance(row, str):
                text = _string(row)
            elif isinstance(row, dict):
                text = _string(row.get("title"))
            else:
                text = ""
            if text:
                section_titles.append(text)
    return chapter_titles, section_titles


def _append_finding(
    result: WritingOntologyValidationResult,
    rule: dict[str, Any],
    message: str,
    suggestion: str,
    max_findings: int,
) -> None:
    """Append finding while respecting max-finding bound.

    Parameters
    ----------
    result : WritingOntologyValidationResult
        Result accumulator.
    rule : dict[str, Any]
        Rule definition payload.
    message : str
        Human-readable finding message.
    suggestion : str
        Suggested remediation.
    max_findings : int
        Maximum retained findings.
    """
    if len(result.findings) >= max(1, max_findings):
        return
    result.findings.append(
        WritingOntologyValidationFinding(
            code=_string(rule.get("id")) or "writing_ontology_rule",
            severity=_string(rule.get("severity")) or "warning",
            category=_string(rule.get("category")) or "rhetoric",
            component=_string(rule.get("component")) or "outline",
            message=message,
            suggestion=suggestion,
        )
    )


def _mapping(value: object) -> dict[str, Any]:
    """Coerce unknown value into mapping.

    Parameters
    ----------
    value : object
        Input object.

    Returns
    -------
    dict[str, Any]
        Mapping if input is a dictionary, otherwise empty.
    """
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str:
    """Normalize value into stripped string.

    Parameters
    ----------
    value : object
        Input value.

    Returns
    -------
    str
        Normalized string.
    """
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: object) -> list[str]:
    """Normalize value into list of non-empty strings.

    Parameters
    ----------
    value : object
        Candidate list-like value.

    Returns
    -------
    list[str]
        List of normalized strings.
    """
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _int(value: object, default: int) -> int:
    """Convert value to integer with fallback.

    Parameters
    ----------
    value : object
        Input value.
    default : int
        Fallback integer.

    Returns
    -------
    int
        Parsed integer or fallback.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _float(value: object, default: float) -> float:
    """Convert value to float with fallback.

    Parameters
    ----------
    value : object
        Input value.
    default : float
        Fallback value.

    Returns
    -------
    float
        Parsed float or fallback.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _lower_set(values: list[str]) -> set[str]:
    """Lowercase and deduplicate string list.

    Parameters
    ----------
    values : list[str]
        Input values.

    Returns
    -------
    set[str]
        Lowercased unique values.
    """
    return {value.lower() for value in values if value}
