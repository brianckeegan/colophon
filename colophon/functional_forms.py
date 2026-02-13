"""Functional-form selection and soft-validation logic for narrative structure checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Chapter, Source


@dataclass(slots=True)
class SoftValidationFinding:
    """Structured soft-validation finding.

    Parameters
    ----------
    code : str
        Stable identifier for the check result.
    severity : str
        Severity label such as ``error``, ``soft_error``, ``warning``, or ``info``.
    category : str
        High-level validation category (for example ``structure`` or ``rhetoric``).
    component : str
        Target component such as ``outline``, ``claims``, ``bibliography``, ``prompts``, or ``agents``.
    message : str
        Human-readable summary of the issue.
    suggestion : str
        Suggested corrective action.
    related_id : str, optional
        Optional chapter/section/claim identifier associated with the finding.
    """

    code: str
    severity: str
    category: str
    component: str
    message: str
    suggestion: str
    related_id: str = ""

    def to_dict(self) -> dict[str, str]:
        """Convert the finding to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, str]
            Mapping representation of the finding.
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
class SoftValidationResult:
    """Aggregate output from functional-form soft validation.

    Parameters
    ----------
    enabled : bool
        Whether soft validation was enabled for the run.
    form_id : str
        Selected functional-form identifier.
    form_name : str
        Selected functional-form display name.
    profile_name : str
        Soft-validation profile used for thresholds and keyword heuristics.
    rule_checks_declared : list[str]
        Required check identifiers declared in the selected functional form.
    required_outline_elements : list[str]
        Required section identifiers derived from the selected functional form.
    matched_outline_elements : list[str]
        Required section identifiers observed in the outline.
    missing_outline_elements : list[str]
        Required section identifiers not observed in the outline.
    findings : list[SoftValidationFinding]
        Collected findings.
    """

    enabled: bool
    form_id: str = ""
    form_name: str = ""
    profile_name: str = ""
    rule_checks_declared: list[str] = field(default_factory=list)
    required_outline_elements: list[str] = field(default_factory=list)
    matched_outline_elements: list[str] = field(default_factory=list)
    missing_outline_elements: list[str] = field(default_factory=list)
    findings: list[SoftValidationFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the validation result to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing summary counters and serialized findings.
        """
        severity_counts = {"error": 0, "soft_error": 0, "warning": 0, "info": 0}
        for finding in self.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

        return {
            "enabled": self.enabled,
            "form_id": self.form_id,
            "form_name": self.form_name,
            "profile_name": self.profile_name,
            "rule_checks_declared": self.rule_checks_declared,
            "required_outline_elements": self.required_outline_elements,
            "matched_outline_elements": self.matched_outline_elements,
            "missing_outline_elements": self.missing_outline_elements,
            "finding_counts": severity_counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def run_soft_validation(
    functional_forms_payload: dict[str, Any],
    outline: list[dict],
    bibliography: list[Source],
    prompts: dict[str, str],
    chapters: list[Chapter],
    agent_profile: dict[str, Any],
    form_id: str = "",
    max_findings: int = 64,
) -> SoftValidationResult:
    """Run soft validation against functional-form structure and rhetoric heuristics.

    Parameters
    ----------
    functional_forms_payload : dict[str, Any]
        Parsed functional forms payload.
    outline : list[dict]
        Outline chapters and sections used for generation.
    bibliography : list[Source]
        Bibliography records used for retrieval and grounding.
    prompts : dict[str, str]
        Prompt template overrides used by drafting agents.
    chapters : list[Chapter]
        Generated chapters containing sections, claims, and paragraphs.
    agent_profile : dict[str, Any]
        Runtime agent configuration values relevant to soft validation.
    form_id : str, optional
        Requested functional form id. If omitted, the catalog default is used.
    max_findings : int, optional
        Maximum number of findings retained in the result.

    Returns
    -------
    SoftValidationResult
        Result containing selected form metadata and collected findings.
    """
    result = SoftValidationResult(enabled=True)
    if not isinstance(functional_forms_payload, dict):
        _append_finding(
            result,
            SoftValidationFinding(
                code="ff_invalid_payload",
                severity="warning",
                category="structure",
                component="functional_forms",
                message="Functional forms payload is not a JSON object.",
                suggestion="Provide a JSON object with a top-level functional_forms list.",
            ),
            max_findings=max_findings,
        )
        return result

    forms = functional_forms_payload.get("functional_forms", [])
    if not isinstance(forms, list) or not forms:
        _append_finding(
            result,
            SoftValidationFinding(
                code="ff_missing_forms",
                severity="warning",
                category="structure",
                component="functional_forms",
                message="No functional forms were found in the configuration.",
                suggestion="Populate functional_forms with at least one genre form.",
            ),
            max_findings=max_findings,
        )
        return result

    selected_form = select_functional_form(functional_forms_payload=functional_forms_payload, form_id=form_id)
    result.form_id = _string(selected_form.get("id"))
    result.form_name = _string(selected_form.get("name"))

    profile_name, profile = _resolve_soft_profile(functional_forms_payload)
    result.profile_name = profile_name

    validation_block = selected_form.get("validation", {})
    if isinstance(validation_block, dict):
        required_checks = validation_block.get("required_checks", [])
        if isinstance(required_checks, list):
            result.rule_checks_declared = [_string(value) for value in required_checks if _string(value)]

    chapter_pattern = selected_form.get("chapter_pattern", [])
    required_sections = _required_sections_from_pattern(chapter_pattern)
    result.required_outline_elements = required_sections

    chapter_titles, section_titles = _outline_titles(outline)
    matched, missing = _match_required_sections(required_sections, section_titles, chapter_titles, profile)
    result.matched_outline_elements = matched
    result.missing_outline_elements = missing

    _validate_outline(
        result=result,
        selected_form=selected_form,
        chapter_titles=chapter_titles,
        section_titles=section_titles,
        matched_sections=matched,
        missing_sections=missing,
        profile=profile,
        max_findings=max_findings,
    )
    _validate_bibliography(result=result, bibliography=bibliography, profile=profile, max_findings=max_findings)
    _validate_prompts(result=result, selected_form=selected_form, prompts=prompts, profile=profile, max_findings=max_findings)
    _validate_agents(result=result, agent_profile=agent_profile, profile=profile, max_findings=max_findings)
    _validate_claims(result=result, selected_form=selected_form, chapters=chapters, profile=profile, max_findings=max_findings)
    _validate_declared_rule_coverage(result=result, profile=profile, max_findings=max_findings)

    return result


def select_functional_form(
    functional_forms_payload: dict[str, Any] | None,
    form_id: str = "",
) -> dict[str, Any]:
    """Select a functional-form ontology from a catalog payload.

    Parameters
    ----------
    functional_forms_payload : dict[str, Any] | None
        Parsed functional forms payload. May be ``None``.
    form_id : str, optional
        Explicit requested form id. If omitted, the catalog default is used.

    Returns
    -------
    dict[str, Any]
        Selected functional-form object, or an empty mapping when unavailable.
    """
    if not isinstance(functional_forms_payload, dict):
        return {}
    forms = functional_forms_payload.get("functional_forms", [])
    if not isinstance(forms, list) or not forms:
        return {}
    selected = _select_form(forms=forms, payload=functional_forms_payload, requested_id=form_id)
    return _normalize_selected_form(selected_form=selected, payload=functional_forms_payload)


def _select_form(forms: list[dict[str, Any]], payload: dict[str, Any], requested_id: str) -> dict[str, Any]:
    """Select a functional form from a catalog.

    Parameters
    ----------
    forms : list[dict[str, Any]]
        Available form objects.
    payload : dict[str, Any]
        Top-level payload, used for optional ``default_form_id``.
    requested_id : str
        Explicit requested form id.

    Returns
    -------
    dict[str, Any]
        Selected form object.
    """
    preferred_id = requested_id.strip() or _string(payload.get("default_form_id"))
    if preferred_id:
        for form in forms:
            if _string(form.get("id")) == preferred_id:
                return form
    return forms[0]


def _normalize_selected_form(selected_form: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize heterogeneous form schemas into runtime-compatible shape.

    Parameters
    ----------
    selected_form : dict[str, Any]
        Raw selected form object.
    payload : dict[str, Any]
        Top-level forms catalog payload.

    Returns
    -------
    dict[str, Any]
        Normalized form object containing runtime keys expected by pipeline agents.
    """
    normalized = dict(selected_form)
    if not isinstance(normalized.get("chapter_pattern"), list):
        pattern_rows = _pattern_rows_from_form(normalized)
        if pattern_rows:
            normalized["chapter_pattern"] = pattern_rows

    if not isinstance(normalized.get("coordination_ontology"), dict):
        normalized["coordination_ontology"] = _synthesized_coordination_ontology(
            form=normalized,
            payload=payload,
        )

    if not isinstance(normalized.get("outline_expansion"), dict):
        normalized["outline_expansion"] = _synthesized_outline_expansion(
            form=normalized,
            payload=payload,
        )

    validation = _dict(normalized.get("validation"))
    if not validation.get("required_checks"):
        rules = validation.get("rules", [])
        if isinstance(rules, list):
            required = [_string(row.get("id")) for row in rules if isinstance(row, dict) and _string(row.get("id"))]
            if required:
                validation["required_checks"] = required
    normalized["validation"] = validation

    if not isinstance(normalized.get("agent_roles"), dict):
        roles = payload.get("agent_roles", {})
        if isinstance(roles, dict):
            normalized["agent_roles"] = roles

    return normalized


def _pattern_rows_from_form(form: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized pattern rows from alternative form keys.

    Parameters
    ----------
    form : dict[str, Any]
        Form mapping that may use different pattern key names.

    Returns
    -------
    list[dict[str, Any]]
        Pattern rows with ``required_sections`` field populated.
    """
    raw_rows = form.get("chapter_pattern")
    if not isinstance(raw_rows, list):
        for key in ("chapter_or_section_pattern", "section_pattern"):
            candidate = form.get(key)
            if isinstance(candidate, list):
                raw_rows = candidate
                break
    if not isinstance(raw_rows, list):
        return []

    output: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        normalized_row = dict(row)
        required_sections = normalized_row.get("required_sections")
        if not isinstance(required_sections, list):
            required_sections = normalized_row.get("required_elements")
        if isinstance(required_sections, list):
            normalized_row["required_sections"] = [
                _string(value) for value in required_sections if _string(value)
            ]
        else:
            normalized_row["required_sections"] = []
        output.append(normalized_row)
    return output


def _synthesized_coordination_ontology(form: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Build default coordination ontology when one is not declared.

    Parameters
    ----------
    form : dict[str, Any]
        Selected functional form.
    payload : dict[str, Any]
        Forms catalog payload.

    Returns
    -------
    dict[str, Any]
        Synthetic coordination ontology for paragraph/section/chapter/book levels.
    """
    role_targets = [entry for entry in _core_role_names(form) if entry]
    base_targets = role_targets or ["technical rigor", "traceable evidence", "coherent argument flow"]
    roles = payload.get("agent_roles", {})
    validator_prompt = ""
    if isinstance(roles, dict):
        validator = roles.get("validator_agent", {})
        if isinstance(validator, dict):
            validator_prompt = _string(validator.get("prompt"))
    concise_suffix = " This paragraph strengthens technical traceability and interpretive clarity."
    if validator_prompt:
        concise_suffix = " This paragraph aligns with validator expectations for technical rigor and traceability."
    return {
        "paragraph": {
            "quality_targets": ["readability", "evidence linkage", *base_targets[:2]],
            "require_citation_markers": True,
            "min_words": 14,
            "concise_suffix": concise_suffix,
        },
        "section": {
            "quality_targets": ["section coherence", "element coverage", *base_targets[:2]],
        },
        "chapter": {
            "quality_targets": ["chapter role alignment", "ordering coherence", *base_targets[:2]],
        },
        "book": {
            "quality_targets": ["cross-chapter continuity", "global argument consistency", *base_targets[:2]],
        },
    }


def _synthesized_outline_expansion(form: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Build default outline-expansion templates when not declared.

    Parameters
    ----------
    form : dict[str, Any]
        Selected functional form.
    payload : dict[str, Any]
        Forms catalog payload.

    Returns
    -------
    dict[str, Any]
        Synthetic outline-expansion template block.
    """
    pattern_rows = _pattern_rows_from_form(form)
    chapter_goal_templates: dict[str, str] = {}
    for row in pattern_rows:
        chapter_type = _string(row.get("type"))
        if not chapter_type:
            continue
        chapter_goal_templates[chapter_type] = (
            "Advance {chapter_title} as a '{chapter_type}' unit in {form_name}, "
            "explicitly connecting technical requirements, evidence, and implications."
        )

    section_objective_templates: dict[str, str] = {}
    subsection_templates: dict[str, list[str]] = {}
    elements = form.get("elements", [])
    if isinstance(elements, list):
        for row in elements:
            if not isinstance(row, dict):
                continue
            element_id = _string(row.get("id"))
            if not element_id:
                continue
            definition = _string(row.get("definition"))
            section_objective_templates[element_id] = (
                f"Develop '{{section_title}}' in '{{chapter_title}}' by addressing {definition or element_id.replace('_', ' ')}."
            )
            subsection_templates[element_id] = _default_subsections_for_element_id(element_id)

    roles = payload.get("agent_roles", {})
    section_planning_hint = (
        "Use section objectives to draft claims with explicit metrics, assumptions, and cited evidence."
    )
    if isinstance(roles, dict):
        evaluation_agent = roles.get("evaluation_agent", {})
        if isinstance(evaluation_agent, dict):
            prompt = _string(evaluation_agent.get("prompt"))
            if prompt:
                section_planning_hint = (
                    "Use section objectives to draft claims with explicit metrics, baselines, and ablations."
                )

    return {
        "chapter_goal_templates": chapter_goal_templates,
        "section_objective_templates": section_objective_templates,
        "subsection_templates": subsection_templates,
        "prompt_hints": {
            "section_planning_template": section_planning_hint,
            "synthesis_template": (
                "Synthesize strongest claims into a transition that preserves technical scope and evidentiary limits."
            ),
        },
    }


def _core_role_names(form: dict[str, Any]) -> list[str]:
    """Extract normalized role names from core ontology rows.

    Parameters
    ----------
    form : dict[str, Any]
        Selected form.

    Returns
    -------
    list[str]
        Role names from ``core_ontology``.
    """
    roles = form.get("core_ontology", [])
    if not isinstance(roles, list):
        return []
    output: list[str] = []
    for row in roles:
        if not isinstance(row, dict):
            continue
        role = _string(row.get("role"))
        if role:
            output.append(role.replace("_", " "))
    return output


def _default_subsections_for_element_id(element_id: str) -> list[str]:
    """Return default subsection labels for a technical ontology element.

    Parameters
    ----------
    element_id : str
        Element identifier.

    Returns
    -------
    list[str]
        Suggested subsection titles.
    """
    lowered = element_id.lower()
    if "assumption" in lowered:
        return ["Assumption Set", "Justification", "Failure Cases"]
    if "artifact" in lowered or "manifest" in lowered:
        return ["Artifact Inventory", "Version and Hashes", "Build Steps"]
    if "baseline" in lowered or "benchmark" in lowered:
        return ["Benchmark Setup", "Comparator Systems", "Result Summary"]
    if "proof" in lowered or "theorem" in lowered:
        return ["Statement", "Dependencies", "Proof Sketch"]
    if "taxonomy" in lowered or "category" in lowered:
        return ["Criteria", "Representative Exemplars", "Boundary Cases"]
    return ["Scope and Definition", "Evidence and Analysis", "Implications and Limits"]


def _resolve_soft_profile(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve soft-validation profile settings.

    Parameters
    ----------
    payload : dict[str, Any]
        Functional forms payload.

    Returns
    -------
    tuple[str, dict[str, Any]]
        Selected profile name and normalized profile mapping.
    """
    defaults = _default_soft_profile()
    profiles = payload.get("soft_validation_profiles", {})
    if not isinstance(profiles, dict):
        return "default", defaults

    profile_name = _string(payload.get("default_soft_validation_profile")) or "common"
    candidate = profiles.get(profile_name, profiles.get("common", {}))
    if not isinstance(candidate, dict):
        return "default", defaults

    merged = _deep_merge(defaults, candidate)
    return profile_name, merged


def _validate_outline(
    result: SoftValidationResult,
    selected_form: dict[str, Any],
    chapter_titles: list[str],
    section_titles: list[str],
    matched_sections: list[str],
    missing_sections: list[str],
    profile: dict[str, Any],
    max_findings: int,
) -> None:
    """Validate outline coverage against selected functional-form structure.

    Parameters
    ----------
    result : SoftValidationResult
        Result accumulator.
    selected_form : dict[str, Any]
        Selected functional form.
    chapter_titles : list[str]
        Chapter titles from outline.
    section_titles : list[str]
        Section titles from outline.
    matched_sections : list[str]
        Required section ids matched in outline.
    missing_sections : list[str]
        Required section ids missing in outline.
    profile : dict[str, Any]
        Soft profile configuration.
    max_findings : int
        Maximum number of findings retained.
    """
    outline_profile = _dict(profile.get("outline"))
    total_required = len(matched_sections) + len(missing_sections)
    if total_required > 0 and missing_sections:
        ratio_missing = len(missing_sections) / total_required
        ratio_soft_error = _float(outline_profile.get("missing_required_ratio_soft_error"), 0.4)
        severity = "soft_error" if ratio_missing >= ratio_soft_error else "warning"
        _append_finding(
            result,
            SoftValidationFinding(
                code="outline_missing_required_sections",
                severity=severity,
                category="structure",
                component="outline",
                message=(
                    f"Outline is missing {len(missing_sections)} of {total_required} required sections "
                    f"for form '{result.form_id}'."
                ),
                suggestion=(
                    "Add or rename sections to cover required elements: "
                    + ", ".join(missing_sections[:8])
                    + ("..." if len(missing_sections) > 8 else "")
                ),
            ),
            max_findings=max_findings,
        )

    lower_titles = " ".join(title.lower() for title in chapter_titles)
    intro_keywords = _string_list(outline_profile.get("introduction_keywords"))
    conclusion_keywords = _string_list(outline_profile.get("conclusion_keywords"))
    if chapter_titles and intro_keywords and not any(keyword in lower_titles for keyword in intro_keywords):
        _append_finding(
            result,
            SoftValidationFinding(
                code="outline_missing_introduction_signal",
                severity="warning",
                category="genre",
                component="outline",
                message="Outline does not signal an explicit introduction chapter.",
                suggestion="Add an introduction chapter that frames problem, scope, and thesis.",
            ),
            max_findings=max_findings,
        )
    if chapter_titles and conclusion_keywords and not any(keyword in lower_titles for keyword in conclusion_keywords):
        _append_finding(
            result,
            SoftValidationFinding(
                code="outline_missing_conclusion_signal",
                severity="warning",
                category="genre",
                component="outline",
                message="Outline does not signal an explicit synthesis or conclusion chapter.",
                suggestion="Add a synthesis/conclusion chapter to restate claims, limits, and implications.",
            ),
            max_findings=max_findings,
        )

    ontology = selected_form.get("core_ontology", [])
    if isinstance(ontology, list):
        combined_titles = chapter_titles + section_titles
        missing_roles: list[str] = []
        threshold = _float(outline_profile.get("ontology_match_threshold"), 0.45)
        for row in ontology:
            if not isinstance(row, dict):
                continue
            role = _string(row.get("role"))
            label = _string(row.get("label"))
            phrase = role.replace("_", " ") if role else label
            if phrase and not _matches_phrase(phrase, combined_titles, threshold=threshold):
                missing_roles.append(role or label)
        if missing_roles:
            _append_finding(
                result,
                SoftValidationFinding(
                    code="outline_missing_ontology_roles",
                    severity="info",
                    category="coverage",
                    component="outline",
                    message="Outline does not clearly map all functional-form ontology roles.",
                    suggestion="Consider adding sections that cover roles: " + ", ".join(missing_roles[:8]),
                ),
                max_findings=max_findings,
            )


def _validate_bibliography(
    result: SoftValidationResult,
    bibliography: list[Source],
    profile: dict[str, Any],
    max_findings: int,
) -> None:
    """Validate bibliography completeness and grounding readiness.

    Parameters
    ----------
    result : SoftValidationResult
        Result accumulator.
    bibliography : list[Source]
        Bibliography records.
    profile : dict[str, Any]
        Soft profile configuration.
    max_findings : int
        Maximum number of findings retained.
    """
    bib_profile = _dict(profile.get("bibliography"))
    if not bibliography:
        _append_finding(
            result,
            SoftValidationFinding(
                code="bibliography_empty",
                severity="soft_error",
                category="evidence",
                component="bibliography",
                message="Bibliography is empty; evidence-grounded drafting will be weak.",
                suggestion="Add seed papers with titles, authors, publication metadata, and abstracts/text.",
            ),
            max_findings=max_findings,
        )
        return

    min_text_chars = _int(bib_profile.get("min_text_chars"), 60)
    max_missing_ratio = _float(bib_profile.get("max_missing_ratio"), 0.35)

    missing_text = sum(1 for src in bibliography if len(src.text.strip()) < min_text_chars)
    missing_authors = sum(1 for src in bibliography if not src.authors)
    missing_publication = sum(1 for src in bibliography if not _string(src.metadata.get("publication")))

    for code, missing_count, field_name in (
        ("bibliography_missing_abstracts", missing_text, "abstract/text"),
        ("bibliography_missing_authors", missing_authors, "authors"),
        ("bibliography_missing_publications", missing_publication, "publication"),
    ):
        ratio = missing_count / len(bibliography)
        if ratio >= max_missing_ratio:
            _append_finding(
                result,
                SoftValidationFinding(
                    code=code,
                    severity="warning",
                    category="evidence",
                    component="bibliography",
                    message=(
                        f"{missing_count} of {len(bibliography)} sources are missing {field_name} information."
                    ),
                    suggestion="Backfill missing bibliography fields to improve retrieval and claim grounding.",
                ),
                max_findings=max_findings,
            )


def _validate_prompts(
    result: SoftValidationResult,
    selected_form: dict[str, Any],
    prompts: dict[str, str],
    profile: dict[str, Any],
    max_findings: int,
) -> None:
    """Validate prompt-template coverage for rhetorical scaffolding.

    Parameters
    ----------
    result : SoftValidationResult
        Result accumulator.
    selected_form : dict[str, Any]
        Selected functional form.
    prompts : dict[str, str]
        Prompt templates.
    profile : dict[str, Any]
        Soft profile configuration.
    max_findings : int
        Maximum number of findings retained.
    """
    prompt_profile = _dict(profile.get("prompts"))
    required_keys = _string_list(prompt_profile.get("required_prompt_keys"))
    missing = [key for key in required_keys if key not in prompts]
    if missing:
        _append_finding(
            result,
            SoftValidationFinding(
                code="prompts_missing_required_templates",
                severity="warning",
                category="rhetoric",
                component="prompts",
                message="Prompt template bundle is missing recommended templates.",
                suggestion="Add prompt keys: " + ", ".join(missing),
            ),
            max_findings=max_findings,
        )

    claim_template = prompts.get("claim_template", "")
    claim_placeholders = _string_list(prompt_profile.get("claim_required_placeholders"))
    if claim_template and claim_placeholders and not any(token in claim_template for token in claim_placeholders):
        _append_finding(
            result,
            SoftValidationFinding(
                code="prompts_claim_template_low_grounding",
                severity="info",
                category="evidence",
                component="prompts",
                message="Claim template does not reference common grounding placeholders.",
                suggestion="Include placeholders such as {source_title}, {lead_entity}, or section context tokens.",
            ),
            max_findings=max_findings,
        )

    paragraph_template = prompts.get("paragraph_template", "")
    if paragraph_template and "{citations}" not in paragraph_template:
        _append_finding(
            result,
            SoftValidationFinding(
                code="prompts_paragraph_missing_citations_slot",
                severity="warning",
                category="citation",
                component="prompts",
                message="Paragraph template does not expose a citation placeholder.",
                suggestion="Include {citations} in paragraph_template to keep evidence traceable.",
            ),
            max_findings=max_findings,
        )

    elements = selected_form.get("elements", [])
    element_ids = {_string(row.get("id")) for row in elements if isinstance(row, dict)}
    if "rival_accounts" in element_ids:
        has_counterarg_prompt = any(
            key in prompts for key in ("counterargument_template", "rival_accounts_template", "alternative_explanations_template")
        )
        if not has_counterarg_prompt:
            _append_finding(
                result,
                SoftValidationFinding(
                    code="prompts_missing_rival_accounts_prompt",
                    severity="info",
                    category="rhetoric",
                    component="prompts",
                    message="Selected form includes rival-account analysis but prompts do not scaffold it explicitly.",
                    suggestion="Add a counterargument/rival-account prompt template.",
                ),
                max_findings=max_findings,
            )


def _validate_agents(
    result: SoftValidationResult,
    agent_profile: dict[str, Any],
    profile: dict[str, Any],
    max_findings: int,
) -> None:
    """Validate agent runtime configuration against baseline genre-support expectations.

    Parameters
    ----------
    result : SoftValidationResult
        Result accumulator.
    agent_profile : dict[str, Any]
        Runtime agent configuration map.
    profile : dict[str, Any]
        Soft profile configuration.
    max_findings : int
        Maximum number of findings retained.
    """
    agent_rules = _dict(profile.get("agents"))
    min_top_k = _int(agent_rules.get("min_top_k"), 1)

    top_k = _int(agent_profile.get("top_k"), 0)
    if top_k < min_top_k:
        _append_finding(
            result,
            SoftValidationFinding(
                code="agents_low_retrieval_k",
                severity="warning",
                category="evidence",
                component="agents",
                message=f"Retriever top-k is {top_k}, below recommended minimum {min_top_k}.",
                suggestion="Increase --top-k to improve evidence coverage per section.",
            ),
            max_findings=max_findings,
        )

    if not bool(agent_profile.get("enable_coordination_agents", False)):
        _append_finding(
            result,
            SoftValidationFinding(
                code="agents_coordination_disabled",
                severity="warning",
                category="coherence",
                component="agents",
                message="Coordination/editing agents are disabled.",
                suggestion="Enable coordination agents to improve paragraph/section/chapter coherence checks.",
            ),
            max_findings=max_findings,
        )

    if not bool(agent_profile.get("llm_enabled", False)):
        _append_finding(
            result,
            SoftValidationFinding(
                code="agents_llm_hooks_disabled",
                severity="info",
                category="rhetoric",
                component="agents",
                message="LLM hook is not configured; deterministic templates will drive rhetoric.",
                suggestion="Configure an LLM provider if you want adaptive rhetorical shaping.",
            ),
            max_findings=max_findings,
        )

    genre_validation = _dict(agent_profile.get("genre_validation"))
    genre = _string(agent_profile.get("genre")).lower()
    language = _string(agent_profile.get("language"))
    min_top_k_for_profile = _int(genre_validation.get("min_top_k"), 0)
    if min_top_k_for_profile > 0 and top_k < min_top_k_for_profile:
        _append_finding(
            result,
            SoftValidationFinding(
                code="agents_genre_profile_low_retrieval_k",
                severity="warning",
                category="evidence",
                component="agents",
                message=(
                    f"Genre profile expects top-k >= {min_top_k_for_profile}, but runtime uses {top_k}."
                ),
                suggestion="Increase --top-k or choose a genre profile with lower evidence-density expectations.",
            ),
            max_findings=max_findings,
        )

    technical_tokens = {"technical", "imrad", "theory", "spec", "survey", "systems"}
    if genre and any(token in genre for token in technical_tokens) and top_k < 2:
        _append_finding(
            result,
            SoftValidationFinding(
                code="agents_technical_genre_low_retrieval_k",
                severity="warning",
                category="evidence",
                component="agents",
                message="Technical writing profiles are likely under-grounded with top-k below 2.",
                suggestion="Set --top-k to at least 2 for stronger technical evidence coverage.",
            ),
            max_findings=max_findings,
        )

    require_llm_for_non_english = bool(genre_validation.get("require_llm_for_non_english", False))
    if language and language.lower() != "english" and require_llm_for_non_english and not bool(agent_profile.get("llm_enabled", False)):
        _append_finding(
            result,
            SoftValidationFinding(
                code="agents_non_english_without_llm",
                severity="warning",
                category="rhetoric",
                component="agents",
                message=(
                    f"Genre profile requests LLM-backed generation for non-English output, but language is '{language}' with LLM disabled."
                ),
                suggestion="Enable an LLM provider to better satisfy non-English genre/style expectations.",
            ),
            max_findings=max_findings,
        )


def _validate_claims(
    result: SoftValidationResult,
    selected_form: dict[str, Any],
    chapters: list[Chapter],
    profile: dict[str, Any],
    max_findings: int,
) -> None:
    """Validate generated claims and prose against evidence and rhetoric heuristics.

    Parameters
    ----------
    result : SoftValidationResult
        Result accumulator.
    selected_form : dict[str, Any]
        Selected functional form.
    chapters : list[Chapter]
        Generated manuscript chapters.
    profile : dict[str, Any]
        Soft profile configuration.
    max_findings : int
        Maximum number of findings retained.
    """
    claim_profile = _dict(profile.get("claims"))
    rhetorical_profile = _dict(profile.get("rhetoric"))

    claims = [claim for chapter in chapters for section in chapter.sections for claim in section.claims]
    paragraphs = [paragraph for chapter in chapters for section in chapter.sections for paragraph in section.paragraphs]

    if not claims:
        _append_finding(
            result,
            SoftValidationFinding(
                code="claims_none_generated",
                severity="warning",
                category="coverage",
                component="claims",
                message="No claims were generated; soft validation cannot assess argument quality.",
                suggestion="Review retrieval coverage and prompt templates to produce evidence-grounded claims.",
            ),
            max_findings=max_findings,
        )
        return

    min_ratio = _float(claim_profile.get("min_evidence_link_ratio"), 0.7)
    linked_ratio = sum(1 for claim in claims if claim.evidence_ids) / len(claims)
    if linked_ratio < min_ratio:
        _append_finding(
            result,
            SoftValidationFinding(
                code="claims_low_evidence_linkage",
                severity="soft_error",
                category="evidence",
                component="claims",
                message=(
                    f"Only {linked_ratio:.2f} of claims include evidence ids, below target {min_ratio:.2f}."
                ),
                suggestion="Increase retrieval coverage and require explicit evidence linking per claim.",
            ),
            max_findings=max_findings,
        )

    teleology_terms = _string_list(rhetorical_profile.get("teleology_terms"))
    if teleology_terms:
        teleology_hits = 0
        claim_texts = [claim.text for claim in claims] + [paragraph.text for paragraph in paragraphs]
        for text in claim_texts:
            lowered = text.lower()
            if any(term in lowered for term in teleology_terms):
                teleology_hits += 1
        ratio = teleology_hits / max(1, len(claim_texts))
        max_ratio = _float(rhetorical_profile.get("max_teleology_ratio_without_warrant"), 0.25)
        if ratio > max_ratio and linked_ratio < min_ratio:
            _append_finding(
                result,
                SoftValidationFinding(
                    code="claims_teleology_risk",
                    severity="warning",
                    category="rhetoric",
                    component="claims",
                    message="Language indicates teleological framing without strong evidence linkage.",
                    suggestion="Add explicit mechanisms, contingencies, and citations around inevitability language.",
                ),
                max_findings=max_findings,
            )

    element_ids = {
        _string(row.get("id"))
        for row in selected_form.get("elements", [])
        if isinstance(row, dict)
    }
    if "transition_claim" in element_ids:
        transition_keywords = _string_list(claim_profile.get("transition_keywords"))
        if transition_keywords and not any(
            any(keyword in claim.text.lower() for keyword in transition_keywords) for claim in claims
        ):
            _append_finding(
                result,
                SoftValidationFinding(
                    code="claims_missing_transition_language",
                    severity="info",
                    category="genre",
                    component="claims",
                    message="Selected form expects transition claims but generated claims lack transition cues.",
                    suggestion="Encourage transition-oriented claims linking one phase or section to the next.",
                ),
                max_findings=max_findings,
            )


def _validate_declared_rule_coverage(
    result: SoftValidationResult,
    profile: dict[str, Any],
    max_findings: int,
) -> None:
    """Report declared rule coverage relative to implemented soft heuristics.

    Parameters
    ----------
    result : SoftValidationResult
        Result accumulator.
    profile : dict[str, Any]
        Soft profile configuration.
    max_findings : int
        Maximum number of findings retained.
    """
    declared = result.rule_checks_declared
    if not declared:
        return

    implemented_prefixes = _string_list(_dict(profile.get("rule_coverage")).get("implemented_prefixes"))
    if not implemented_prefixes:
        return

    unsupported = [rule_id for rule_id in declared if not any(rule_id.startswith(prefix) for prefix in implemented_prefixes)]
    if unsupported:
        _append_finding(
            result,
            SoftValidationFinding(
                code="functional_form_partial_rule_coverage",
                severity="info",
                category="scope",
                component="functional_forms",
                message=(
                    f"{len(unsupported)} declared rule ids are outside currently implemented heuristic prefixes."
                ),
                suggestion=(
                    "Treat these checks as advisory placeholders or extend validator support for: "
                    + ", ".join(unsupported[:6])
                    + ("..." if len(unsupported) > 6 else "")
                ),
            ),
            max_findings=max_findings,
        )


def _required_sections_from_pattern(chapter_pattern: object) -> list[str]:
    """Extract required section identifiers from a chapter pattern.

    Parameters
    ----------
    chapter_pattern : object
        Pattern object from a functional form.

    Returns
    -------
    list[str]
        Unique required section identifiers.
    """
    if not isinstance(chapter_pattern, list):
        return []

    sections: list[str] = []
    seen: set[str] = set()
    for row in chapter_pattern:
        if not isinstance(row, dict):
            continue
        required = row.get("required_sections", [])
        if not isinstance(required, list):
            required = row.get("required_elements", [])
        if not isinstance(required, list):
            continue
        for value in required:
            section_id = _string(value)
            if section_id and section_id not in seen:
                seen.add(section_id)
                sections.append(section_id)
    return sections


def _outline_titles(outline: list[dict]) -> tuple[list[str], list[str]]:
    """Extract chapter and section titles from an outline object.

    Parameters
    ----------
    outline : list[dict]
        Outline chapter list.

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
        if isinstance(sections, list):
            for item in sections:
                section_title = _string(item)
                if section_title:
                    section_titles.append(section_title)

        section_details = chapter.get("section_details", [])
        if isinstance(section_details, list):
            for row in section_details:
                if not isinstance(row, dict):
                    continue
                detail_title = _string(row.get("title"))
                if detail_title:
                    section_titles.append(detail_title)
    return chapter_titles, section_titles


def _match_required_sections(
    required_sections: list[str],
    section_titles: list[str],
    chapter_titles: list[str],
    profile: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Match required section ids against outline chapter/section titles.

    Parameters
    ----------
    required_sections : list[str]
        Required section identifiers.
    section_titles : list[str]
        Outline section titles.
    chapter_titles : list[str]
        Outline chapter titles.
    profile : dict[str, Any]
        Soft profile configuration.

    Returns
    -------
    tuple[list[str], list[str]]
        Matched and missing required section identifiers.
    """
    threshold = _float(_dict(profile.get("outline")).get("section_match_threshold"), 0.5)
    titles = section_titles + chapter_titles

    matched: list[str] = []
    missing: list[str] = []
    for section_id in required_sections:
        phrase = section_id.replace("_", " ")
        if _matches_phrase(phrase, titles, threshold=threshold):
            matched.append(section_id)
        else:
            missing.append(section_id)
    return matched, missing


def _matches_phrase(phrase: str, titles: list[str], threshold: float) -> bool:
    """Check whether a phrase is covered by a set of outline titles.

    Parameters
    ----------
    phrase : str
        Phrase to match.
    titles : list[str]
        Candidate titles.
    threshold : float
        Minimum token-overlap ratio for fuzzy matches.

    Returns
    -------
    bool
        ``True`` if the phrase is considered covered.
    """
    phrase_tokens = _tokenize(phrase)
    if not phrase_tokens:
        return False

    phrase_normalized = " ".join(phrase_tokens)
    for title in titles:
        title_tokens = _tokenize(title)
        if not title_tokens:
            continue
        title_normalized = " ".join(title_tokens)
        if phrase_normalized in title_normalized:
            return True
        overlap = len(phrase_tokens & title_tokens) / max(1, len(phrase_tokens))
        if overlap >= threshold:
            return True
        if _jaccard(phrase_tokens, title_tokens) >= threshold:
            return True
    return False


def _append_finding(result: SoftValidationResult, finding: SoftValidationFinding, max_findings: int) -> None:
    """Append a finding if the configured limit has not been reached.

    Parameters
    ----------
    result : SoftValidationResult
        Result accumulator.
    finding : SoftValidationFinding
        Finding to append.
    max_findings : int
        Maximum number of findings retained.
    """
    if len(result.findings) >= max(1, max_findings):
        return
    result.findings.append(finding)


def _default_soft_profile() -> dict[str, Any]:
    """Return default soft-validation thresholds and keyword lexicons.

    Returns
    -------
    dict[str, Any]
        Default profile mapping.
    """
    return {
        "outline": {
            "section_match_threshold": 0.5,
            "ontology_match_threshold": 0.45,
            "missing_required_ratio_soft_error": 0.4,
            "introduction_keywords": [
                "intro",
                "introduction",
                "problem",
                "background",
                "foundations",
                "scope",
                "requirements",
                "preliminaries",
            ],
            "conclusion_keywords": [
                "conclusion",
                "synthesis",
                "implications",
                "discussion",
                "limits",
                "open problems",
                "versioning",
            ],
        },
        "bibliography": {
            "min_text_chars": 60,
            "max_missing_ratio": 0.35,
        },
        "prompts": {
            "required_prompt_keys": [
                "claim_template",
                "paragraph_template",
                "empty_section_template",
            ],
            "claim_required_placeholders": [
                "{source_title}",
                "{lead_entity}",
                "{section_title",
            ],
        },
        "agents": {
            "min_top_k": 1,
        },
        "claims": {
            "min_evidence_link_ratio": 0.7,
            "transition_keywords": ["transition", "turning", "shift", "phase", "pivot", "therefore"],
        },
        "rhetoric": {
            "teleology_terms": ["inevitable", "bound to", "destined", "naturally", "unavoidably"],
            "max_teleology_ratio_without_warrant": 0.25,
        },
        "rule_coverage": {
            "implemented_prefixes": [
                "st_",
                "sc_",
                "cs_",
                "le_",
                "id_",
                "imrad_",
                "sys_",
                "theory_",
                "spec_",
                "survey_",
            ],
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Parameters
    ----------
    base : dict[str, Any]
        Base dictionary.
    override : dict[str, Any]
        Override dictionary.

    Returns
    -------
    dict[str, Any]
        Merged dictionary.
    """
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge(_dict(output[key]), value)
        else:
            output[key] = value
    return output


def _jaccard(left: set[str], right: set[str]) -> float:
    """Compute Jaccard similarity between two token sets.

    Parameters
    ----------
    left : set[str]
        Left token set.
    right : set[str]
        Right token set.

    Returns
    -------
    float
        Jaccard index in ``[0, 1]``.
    """
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _tokenize(text: str) -> set[str]:
    """Tokenize free text into lowercase alphanumeric tokens.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    set[str]
        Unique tokens.
    """
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {part for part in cleaned.split() if part}


def _string(value: object) -> str:
    """Convert a value to a stripped string.

    Parameters
    ----------
    value : object
        Value to normalize.

    Returns
    -------
    str
        Normalized string or empty string.
    """
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _string_list(value: object) -> list[str]:
    """Normalize a list-like value into a list of non-empty strings.

    Parameters
    ----------
    value : object
        Candidate sequence.

    Returns
    -------
    list[str]
        Normalized non-empty string list.
    """
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        text = _string(item)
        if text:
            output.append(text)
    return output


def _dict(value: object) -> dict[str, Any]:
    """Coerce a mapping-like object into a dictionary.

    Parameters
    ----------
    value : object
        Candidate mapping.

    Returns
    -------
    dict[str, Any]
        Dictionary value or empty dictionary.
    """
    if isinstance(value, dict):
        return value
    return {}


def _int(value: object, default: int) -> int:
    """Convert a value to an integer with fallback.

    Parameters
    ----------
    value : object
        Candidate value.
    default : int
        Fallback integer.

    Returns
    -------
    int
        Parsed integer or fallback.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float) -> float:
    """Convert a value to a float with fallback.

    Parameters
    ----------
    value : object
        Candidate value.
    default : float
        Fallback float.

    Returns
    -------
    float
        Parsed float or fallback.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
