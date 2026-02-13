"""Normalization helpers for consistent ontology typing and schema metadata."""

from __future__ import annotations

from typing import Any


def normalize_functional_forms_catalog(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a functional-forms ontology payload for runtime use.

    Parameters
    ----------
    payload : dict[str, Any] | None
        Raw functional-forms payload.

    Returns
    -------
    dict[str, Any]
        Normalized payload with consistent metadata and form list structure.
    """
    normalized = _normalize_metadata(
        payload=payload,
        ontology_type="functional_forms_catalog",
        fallback_id="colophon_functional_forms_catalog",
        fallback_name="Colophon Functional Forms Catalog",
        fallback_version="1.0",
    )
    forms = normalized.get("functional_forms")
    if not isinstance(forms, list):
        normalized["functional_forms"] = []
        return normalized

    normalized_forms: list[dict[str, Any]] = []
    for index, row in enumerate(forms, start=1):
        if not isinstance(row, dict):
            continue
        form = dict(row)
        form_id = _string(form.get("id")) or f"form_{index}"
        form["id"] = form_id
        normalized_forms.append(form)
    normalized["functional_forms"] = normalized_forms
    return normalized


def normalize_genre_ontology(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a genre ontology payload for runtime use.

    Parameters
    ----------
    payload : dict[str, Any] | None
        Raw genre ontology payload.

    Returns
    -------
    dict[str, Any]
        Normalized payload with consistent metadata and profile structure.
    """
    normalized = _normalize_metadata(
        payload=payload,
        ontology_type="genre_profile_ontology",
        fallback_id="colophon_genre_ontology",
        fallback_name="Colophon Genre Ontology",
        fallback_version="1.0",
    )
    profiles = normalized.get("profiles")
    if not isinstance(profiles, list):
        profiles = []

    normalized_profiles: list[dict[str, Any]] = []
    for index, row in enumerate(profiles, start=1):
        if not isinstance(row, dict):
            continue
        profile = dict(row)
        profile_id = _string(profile.get("id")) or f"profile_{index}"
        profile["id"] = profile_id
        profile["name"] = _string(profile.get("name")) or _humanize(profile_id)
        profile["audience"] = _string(profile.get("audience")) or "general"
        profile["discipline"] = _string(profile.get("discipline")) or "interdisciplinary"
        profile["style"] = _string(profile.get("style")) or "analytical"
        profile["genre"] = _string(profile.get("genre")) or "scholarly_manuscript"
        profile["language"] = _string(profile.get("language")) or "English"
        profile["tone"] = _string(profile.get("tone")) or "neutral"
        if not isinstance(profile.get("agent_prompts"), dict):
            profile["agent_prompts"] = {}
        if not isinstance(profile.get("recommendation"), dict):
            profile["recommendation"] = {}
        if not isinstance(profile.get("validation"), dict):
            profile["validation"] = {}
        normalized_profiles.append(profile)

    normalized["profiles"] = normalized_profiles
    default_profile_id = _string(normalized.get("default_profile_id"))
    if not default_profile_id and normalized_profiles:
        default_profile_id = _string(normalized_profiles[0].get("id"))
    normalized["default_profile_id"] = default_profile_id
    return normalized


def normalize_writing_companion_ontology(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a companion writing ontology payload for runtime use.

    Parameters
    ----------
    payload : dict[str, Any] | None
        Raw writing companion ontology payload.

    Returns
    -------
    dict[str, Any]
        Normalized payload with consistent metadata and validation containers.
    """
    normalized = _normalize_metadata(
        payload=payload,
        ontology_type="writing_companion_ontology",
        fallback_id="colophon_writing_companion_ontology",
        fallback_name="Colophon Writing Companion Ontology",
        fallback_version="1.0",
    )
    compatibility = normalized.get("compatibility")
    if not isinstance(compatibility, dict):
        compatibility = {}
    compatibility.setdefault("mode", "alongside_functional_forms")
    compatibility.setdefault("compatible_form_ids", [])
    normalized["compatibility"] = compatibility

    assumptions = normalized.get("assumptions")
    if not isinstance(assumptions, list):
        assumptions = []
    normalized_assumptions: list[dict[str, Any]] = []
    for row in assumptions:
        if not isinstance(row, dict):
            continue
        assumption_id = _string(row.get("id"))
        if not assumption_id:
            continue
        normalized_assumptions.append(
            {
                "id": assumption_id,
                "statement": _string(row.get("statement")),
            }
        )
    normalized["assumptions"] = normalized_assumptions

    validations = normalized.get("validations")
    if not isinstance(validations, dict):
        validations = {}
    rules = validations.get("rules")
    if not isinstance(rules, list):
        rules = []
    normalized_rules: list[dict[str, Any]] = []
    for index, row in enumerate(rules, start=1):
        if not isinstance(row, dict):
            continue
        rule = dict(row)
        rule_id = _string(rule.get("id")) or f"rule_{index}"
        rule["id"] = rule_id
        rule["severity"] = _string(rule.get("severity")) or "info"
        rule["category"] = _string(rule.get("category")) or "structure"
        rule["component"] = _string(rule.get("component")) or "outline"
        normalized_rules.append(rule)
    validations["rules"] = normalized_rules
    normalized["validations"] = validations
    return normalized


def _normalize_metadata(
    payload: dict[str, Any] | None,
    ontology_type: str,
    fallback_id: str,
    fallback_name: str,
    fallback_version: str,
) -> dict[str, Any]:
    """Normalize shared top-level ontology metadata.

    Parameters
    ----------
    payload : dict[str, Any] | None
        Raw ontology payload.
    ontology_type : str
        Expected ontology type label.
    fallback_id : str
        Fallback ontology identifier.
    fallback_name : str
        Fallback ontology name.
    fallback_version : str
        Fallback ontology/schema version.

    Returns
    -------
    dict[str, Any]
        Mapping with normalized metadata fields.
    """
    normalized = dict(payload) if isinstance(payload, dict) else {}
    normalized["id"] = _string(normalized.get("id")) or fallback_id
    normalized["name"] = _string(normalized.get("name")) or fallback_name
    normalized["ontology_type"] = _string(normalized.get("ontology_type")) or ontology_type
    version = _string(normalized.get("version"))
    schema_version = _string(normalized.get("schema_version"))
    if not version and schema_version:
        version = schema_version
    if not schema_version and version:
        schema_version = version
    if not version:
        version = fallback_version
    if not schema_version:
        schema_version = version
    normalized["version"] = version
    normalized["schema_version"] = schema_version
    return normalized


def _string(value: object) -> str:
    """Convert a value to stripped string.

    Parameters
    ----------
    value : object
        Input value.

    Returns
    -------
    str
        Trimmed string value, or empty string.
    """
    if value is None:
        return ""
    return str(value).strip()


def _humanize(identifier: str) -> str:
    """Convert an identifier into a title-like label.

    Parameters
    ----------
    identifier : str
        Identifier text.

    Returns
    -------
    str
        Humanized label.
    """
    return " ".join(piece.capitalize() for piece in identifier.replace("-", "_").split("_") if piece) or identifier
