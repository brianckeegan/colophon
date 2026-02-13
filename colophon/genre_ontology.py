"""Genre ontology context resolution for narrative profile customization."""

from __future__ import annotations

from typing import Any


_DEFAULT_GENRE_ONTOLOGY: dict[str, Any] = {
    "version": "1.0",
    "default_profile_id": "general_academic",
    "profiles": [
        {
            "id": "general_academic",
            "name": "General Academic",
            "audience": "general",
            "discipline": "interdisciplinary",
            "style": "analytical",
            "genre": "scholarly_manuscript",
            "language": "English",
            "tone": "neutral",
            "agent_prompts": {
                "claim_author_agent": "Prioritize evidence-grounded claims with clear scope conditions.",
                "paragraph_agent": "Preserve assertion-evidence-analysis paragraph logic and citation transparency.",
                "outline_expander": "Balance problem framing, method visibility, evidence, and implications across sections.",
                "section_coordinator": "Maintain argumentative continuity and explicit evidence mapping within section flow.",
            },
            "recommendation": {
                "query_terms": ["evidence", "method", "findings", "implications"],
                "keyword_weight": 0.08,
            },
            "validation": {
                "min_top_k": 1,
            },
        },
        {
            "id": "technical_research",
            "name": "Technical Research",
            "audience": "technical experts",
            "discipline": "computer science",
            "style": "technical",
            "genre": "technical_report",
            "language": "English",
            "tone": "precise",
            "agent_prompts": {
                "claim_author_agent": "State measurable conditions, baselines, and assumptions for each claim when available.",
                "paragraph_agent": "Use precise metric language and avoid qualitative overstatement without quantitative support.",
                "outline_expander": "Ensure methods, artifact details, benchmark design, and limitations are explicitly represented.",
                "paragraph_coordinator": "Flag unquantified comparative claims and missing evidentiary anchors.",
                "section_coordinator": "Maintain traceability from requirements and methods to results and limitations.",
                "chapter_coordinator": "Check chapter structure for replicability-critical components.",
            },
            "recommendation": {
                "query_terms": ["benchmark", "baseline", "ablation", "artifact", "reproducibility"],
                "keyword_weight": 0.12,
            },
            "validation": {
                "min_top_k": 2,
                "require_llm_for_non_english": True,
            },
        },
        {
            "id": "policy_brief",
            "name": "Policy Brief",
            "audience": "policy analysts",
            "discipline": "public policy",
            "style": "persuasive",
            "genre": "policy_brief",
            "language": "English",
            "tone": "formal",
            "agent_prompts": {
                "claim_author_agent": "Surface decision relevance and practical constraints with explicit evidence warrants.",
                "paragraph_agent": "Emphasize implications, tradeoffs, and actionable recommendations.",
                "outline_expander": "Include stakeholder impacts, implementation risks, and policy implications.",
                "book_coordinator": "Ensure chapter flow supports clear policy conclusions and scope limits.",
            },
            "recommendation": {
                "query_terms": ["governance", "implementation", "evaluation", "impact"],
                "keyword_weight": 0.1,
            },
            "validation": {
                "min_top_k": 1,
            },
        },
    ],
}


def build_genre_ontology_context(
    genre_ontology_payload: dict[str, Any] | None,
    profile_id: str = "",
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a runtime genre/style context for downstream Colophon components.

    Parameters
    ----------
    genre_ontology_payload : dict[str, Any] | None
        Optional ontology payload containing one or more profiles.
    profile_id : str, optional
        Requested profile id; falls back to payload default and then built-in default.
    overrides : dict[str, str] | None, optional
        Optional field overrides for ``audience``, ``discipline``, ``style``, ``genre``,
        ``language``, and ``tone``.

    Returns
    -------
    dict[str, Any]
        Runtime context with selected metadata, role prompts, recommendation hints, and
        validation hints.
    """
    payload = _merge_payload_with_defaults(genre_ontology_payload)
    profiles = _profile_map(payload.get("profiles"))
    requested = _string(profile_id)
    default_profile_id = _string(payload.get("default_profile_id")) or "general_academic"
    selected_id = requested or default_profile_id
    selected = profiles.get(selected_id)
    if selected is None:
        selected_id = "general_academic"
        selected = profiles.get(selected_id, {})

    selected = dict(selected) if isinstance(selected, dict) else {}
    applied_overrides = overrides or {}

    tone = _override_or_default(applied_overrides.get("tone"), selected.get("tone"), "neutral")
    style = _override_or_default(applied_overrides.get("style"), selected.get("style"), "analytical")
    audience = _override_or_default(applied_overrides.get("audience"), selected.get("audience"), "general")
    discipline = _override_or_default(applied_overrides.get("discipline"), selected.get("discipline"), "interdisciplinary")
    genre = _override_or_default(applied_overrides.get("genre"), selected.get("genre"), "scholarly_manuscript")
    language = _override_or_default(applied_overrides.get("language"), selected.get("language"), "English")

    role_prompts = _build_role_prompts(
        selected=_mapping(selected.get("agent_prompts")),
        metadata={
            "tone": tone,
            "style": style,
            "audience": audience,
            "discipline": discipline,
            "genre": genre,
            "language": language,
        },
    )

    recommendation = _mapping(selected.get("recommendation"))
    validation = _mapping(selected.get("validation"))

    return {
        "enabled": True,
        "profile_id": selected_id,
        "profile_name": _string(selected.get("name")) or _humanize_identifier(selected_id),
        "metadata": {
            "tone": tone,
            "style": style,
            "audience": audience,
            "discipline": discipline,
            "genre": genre,
            "language": language,
        },
        "role_prompts": role_prompts,
        "recommendation": {
            "query_terms": _string_list(recommendation.get("query_terms")),
            "keyword_weight": _float(recommendation.get("keyword_weight"), 0.08),
        },
        "validation": {
            "min_top_k": max(1, _int(validation.get("min_top_k"), 1)),
            "require_llm_for_non_english": bool(validation.get("require_llm_for_non_english", False)),
        },
    }


def _merge_payload_with_defaults(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Merge optional payload with built-in default genre ontology.

    Parameters
    ----------
    payload : dict[str, Any] | None
        Optional user-supplied payload.

    Returns
    -------
    dict[str, Any]
        Merged ontology payload.
    """
    if not isinstance(payload, dict):
        return dict(_DEFAULT_GENRE_ONTOLOGY)

    merged = dict(_DEFAULT_GENRE_ONTOLOGY)
    merged["default_profile_id"] = _string(payload.get("default_profile_id")) or merged.get("default_profile_id")

    default_profiles = _profile_map(_DEFAULT_GENRE_ONTOLOGY.get("profiles"))
    user_profiles = _profile_map(payload.get("profiles"))
    combined: dict[str, dict[str, Any]] = {}
    for profile_id, profile in default_profiles.items():
        combined[profile_id] = dict(profile)
    for profile_id, profile in user_profiles.items():
        base = combined.get(profile_id, {})
        combined[profile_id] = _deep_merge(_mapping(base), profile)

    merged["profiles"] = list(combined.values())
    return merged


def _build_role_prompts(selected: dict[str, Any], metadata: dict[str, str]) -> dict[str, str]:
    """Construct merged role prompts with metadata framing.

    Parameters
    ----------
    selected : dict[str, Any]
        Profile-level role prompt mapping.
    metadata : dict[str, str]
        Selected metadata fields.

    Returns
    -------
    dict[str, str]
        Role prompt mapping.
    """
    metadata_prefix = (
        "Profile guidance: "
        f"Audience={metadata['audience']}; Discipline={metadata['discipline']}; "
        f"Style={metadata['style']}; Genre={metadata['genre']}; "
        f"Language={metadata['language']}; Tone={metadata['tone']}."
    )
    roles = [
        "claim_author_agent",
        "paragraph_agent",
        "outline_expander",
        "paragraph_coordinator",
        "section_coordinator",
        "chapter_coordinator",
        "book_coordinator",
        "retrieval_agent",
        "recommendation_agent",
        "validator_agent",
    ]
    prompts: dict[str, str] = {}
    for role in roles:
        role_prompt = _string(selected.get(role))
        prompts[role] = (metadata_prefix + (" " + role_prompt if role_prompt else "")).strip()
    return prompts


def _profile_map(value: object) -> dict[str, dict[str, Any]]:
    """Normalize profile list/dict into profile-id keyed mapping.

    Parameters
    ----------
    value : object
        Profile container.

    Returns
    -------
    dict[str, dict[str, Any]]
        Profile mapping keyed by ``id``.
    """
    result: dict[str, dict[str, Any]] = {}
    if isinstance(value, dict):
        for key, row in value.items():
            if not isinstance(row, dict):
                continue
            profile_id = _string(row.get("id")) or _string(key)
            if not profile_id:
                continue
            mapped = dict(row)
            mapped["id"] = profile_id
            result[profile_id] = mapped
        return result
    if isinstance(value, list):
        for row in value:
            if not isinstance(row, dict):
                continue
            profile_id = _string(row.get("id"))
            if not profile_id:
                continue
            result[profile_id] = dict(row)
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Parameters
    ----------
    base : dict[str, Any]
        Base mapping.
    override : dict[str, Any]
        Override mapping.

    Returns
    -------
    dict[str, Any]
        Merged mapping.
    """
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge(_mapping(output.get(key)), value)
        else:
            output[key] = value
    return output


def _override_or_default(override_value: object, profile_value: object, fallback: str) -> str:
    """Resolve field value from override, profile default, and fallback.

    Parameters
    ----------
    override_value : object
        Explicit override value.
    profile_value : object
        Profile field value.
    fallback : str
        Final fallback.

    Returns
    -------
    str
        Resolved value.
    """
    override = _string(override_value)
    if override:
        return override
    profile = _string(profile_value)
    if profile:
        return profile
    return fallback


def _mapping(value: object) -> dict[str, Any]:
    """Normalize arbitrary value into a dictionary.

    Parameters
    ----------
    value : object
        Input value.

    Returns
    -------
    dict[str, Any]
        Mapping value or empty mapping.
    """
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str:
    """Normalize arbitrary value into stripped string.

    Parameters
    ----------
    value : object
        Input value.

    Returns
    -------
    str
        Stripped string or empty value.
    """
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: object) -> list[str]:
    """Normalize arbitrary value into list of strings.

    Parameters
    ----------
    value : object
        Candidate list-like value.

    Returns
    -------
    list[str]
        Filtered list of strings.
    """
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _int(value: object, default: int) -> int:
    """Convert arbitrary value into integer with fallback.

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
    """Convert arbitrary value into float with fallback.

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


def _humanize_identifier(value: str) -> str:
    """Humanize snake-case identifiers.

    Parameters
    ----------
    value : str
        Identifier string.

    Returns
    -------
    str
        Humanized title.
    """
    text = value.replace("_", " ").strip()
    if not text:
        return ""
    return " ".join(token.capitalize() for token in text.split())
