"""AgentSkills-compatible discovery, validation, matching, and activation utilities."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_MAX_COMPATIBILITY_LENGTH = 500
_ALLOWED_FRONTMATTER_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}


@dataclass(slots=True)
class AgentSkill:
    """AgentSkills metadata for a discovered skill directory.

    Parameters
    ----------
    name : str
        Frontmatter ``name`` value.
    description : str
        Frontmatter ``description`` value.
    skill_dir : str
        Absolute path to the skill directory.
    skill_md_path : str
        Absolute path to ``SKILL.md`` (or ``skill.md``).
    license : str
        Optional frontmatter ``license`` value.
    compatibility : str
        Optional frontmatter ``compatibility`` value.
    allowed_tools : str
        Optional frontmatter ``allowed-tools`` value.
    metadata : dict[str, str]
        Optional frontmatter ``metadata`` mapping.
    """

    name: str
    description: str
    skill_dir: str
    skill_md_path: str
    license: str = ""
    compatibility: str = ""
    allowed_tools: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AgentSkillValidationIssue:
    """Validation/parsing issue for a discovered skill directory.

    Parameters
    ----------
    skill_dir : str
        Absolute path to the skill directory.
    skill_md_path : str
        Absolute path to ``SKILL.md`` (or ``skill.md``), if present.
    errors : list[str]
        Validation/parsing errors.
    """

    skill_dir: str
    skill_md_path: str
    errors: list[str]


@dataclass(slots=True)
class AgentSkillActivation:
    """Activated skill bundle used for an individual task.

    Parameters
    ----------
    skill : AgentSkill
        Matched skill metadata.
    instructions : str
        Activated skill instructions (markdown body from ``SKILL.md``).
    matched_tokens : list[str]
        Task tokens that overlapped with this skill's metadata tokens.
    score : float
        Match score used for ranking.
    """

    skill: AgentSkill
    instructions: str
    matched_tokens: list[str]
    score: float


@dataclass(slots=True)
class AgentSkillsRuntime:
    """Runtime container for AgentSkills metadata and activation.

    Parameters
    ----------
    skills : list[AgentSkill]
        Valid discovered skills.
    invalid_skills : list[AgentSkillValidationIssue]
        Invalid/malformed skill directories.
    available_skills_xml : str
        XML metadata block for prompt injection.
    max_instruction_chars : int
        Maximum instruction body characters loaded per activated skill.
    """

    skills: list[AgentSkill]
    invalid_skills: list[AgentSkillValidationIssue]
    available_skills_xml: str
    max_instruction_chars: int = 8000
    _content_cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_directories(cls, skill_dirs: list[str], max_instruction_chars: int = 8000) -> AgentSkillsRuntime:
        """Discover and validate AgentSkills directories.

        Parameters
        ----------
        skill_dirs : list[str]
            Root directories to scan for skill folders.
        max_instruction_chars : int
            Maximum characters retained from each activated skill body.

        Returns
        -------
        AgentSkillsRuntime
            Runtime metadata and validation results.
        """
        discovered_skills: list[AgentSkill] = []
        invalid: list[AgentSkillValidationIssue] = []

        seen_dirs: set[str] = set()
        for root_dir in skill_dirs:
            normalized_root = str(root_dir).strip()
            if not normalized_root:
                continue
            root_path = Path(normalized_root).expanduser().resolve()
            if not root_path.exists():
                invalid.append(
                    AgentSkillValidationIssue(
                        skill_dir=str(root_path),
                        skill_md_path="",
                        errors=[f"Path does not exist: {root_path}"],
                    )
                )
                continue
            if not root_path.is_dir():
                invalid.append(
                    AgentSkillValidationIssue(
                        skill_dir=str(root_path),
                        skill_md_path="",
                        errors=[f"Not a directory: {root_path}"],
                    )
                )
                continue

            for skill_dir in _discover_skill_directories(root_path):
                skill_dir_key = str(skill_dir)
                if skill_dir_key in seen_dirs:
                    continue
                seen_dirs.add(skill_dir_key)
                skill_md = _find_skill_md(skill_dir)
                if skill_md is None:
                    continue
                parsed = _parse_skill_file(skill_dir=skill_dir, skill_md=skill_md)
                if isinstance(parsed, AgentSkillValidationIssue):
                    invalid.append(parsed)
                    continue
                discovered_skills.append(parsed)

        discovered_skills.sort(key=lambda skill: skill.name)
        invalid.sort(key=lambda issue: (issue.skill_dir, issue.skill_md_path))
        return cls(
            skills=discovered_skills,
            invalid_skills=invalid,
            available_skills_xml=build_available_skills_xml(discovered_skills),
            max_instruction_chars=max(1, int(max_instruction_chars)),
        )

    def match_and_activate(
        self,
        task: str,
        max_matches: int = 3,
        min_token_overlap: int = 1,
    ) -> list[AgentSkillActivation]:
        """Match skills to a task and load activated instructions.

        Parameters
        ----------
        task : str
            Task/query text used for skill matching.
        max_matches : int
            Maximum activated skills to return.
        min_token_overlap : int
            Minimum token overlap required for activation.

        Returns
        -------
        list[AgentSkillActivation]
            Activated skill instructions ordered by descending score.
        """
        if not self.skills or max_matches <= 0:
            return []
        task_tokens = _tokenize(task)
        if not task_tokens:
            return []

        scored: list[tuple[float, AgentSkill, list[str]]] = []
        minimum_overlap = max(1, int(min_token_overlap))
        for skill in self.skills:
            skill_tokens = _tokenize(f"{skill.name} {skill.description}")
            overlap = sorted(task_tokens & skill_tokens)
            if len(overlap) < minimum_overlap:
                continue
            denominator = max(1, len(skill_tokens))
            overlap_score = len(overlap) / denominator
            # Strong boost when task explicitly names the skill.
            normalized_task = task.lower()
            explicit_name = skill.name in normalized_task or skill.name.replace("-", " ") in normalized_task
            score = overlap_score + (1.0 if explicit_name else 0.0)
            scored.append((score, skill, overlap))

        scored.sort(key=lambda row: (-row[0], row[1].name))
        activations: list[AgentSkillActivation] = []
        for score, skill, overlap in scored[: max(0, int(max_matches))]:
            instructions = self._skill_instructions(skill)
            if not instructions:
                continue
            activations.append(
                AgentSkillActivation(
                    skill=skill,
                    instructions=instructions,
                    matched_tokens=overlap,
                    score=score,
                )
            )
        return activations

    def _skill_instructions(self, skill: AgentSkill) -> str:
        """Load and cache the body portion of a skill's markdown file.

        Parameters
        ----------
        skill : AgentSkill
            Skill metadata pointing to a markdown file.

        Returns
        -------
        str
            Truncated markdown body.
        """
        cached = self._content_cache.get(skill.skill_md_path)
        if cached is not None:
            return cached
        path = Path(skill.skill_md_path)
        if not path.exists():
            self._content_cache[skill.skill_md_path] = ""
            return ""
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            self._content_cache[skill.skill_md_path] = ""
            return ""

        try:
            _, body = _parse_frontmatter(content)
        except ValueError:
            body = content.strip()
        truncated = body.strip()[: self.max_instruction_chars]
        self._content_cache[skill.skill_md_path] = truncated
        return truncated


def build_available_skills_xml(skills: list[AgentSkill]) -> str:
    """Build AgentSkills-compatible ``<available_skills>`` XML metadata.

    Parameters
    ----------
    skills : list[AgentSkill]
        Valid discovered skills.

    Returns
    -------
    str
        XML metadata block for prompt injection.
    """
    if not skills:
        return "<available_skills>\n</available_skills>"

    lines = ["<available_skills>"]
    for skill in skills:
        lines.append("<skill>")
        lines.append("<name>")
        lines.append(html.escape(skill.name))
        lines.append("</name>")
        lines.append("<description>")
        lines.append(html.escape(skill.description))
        lines.append("</description>")
        lines.append("<location>")
        lines.append(html.escape(skill.skill_md_path))
        lines.append("</location>")
        lines.append("</skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _discover_skill_directories(root: Path) -> list[Path]:
    """Return unique directories containing ``SKILL.md``/``skill.md`` under a root.

    Parameters
    ----------
    root : Path
        Root directory to scan.

    Returns
    -------
    list[Path]
        Discovered skill directories.
    """
    discovered: set[Path] = set()
    for candidate in root.rglob("SKILL.md"):
        if candidate.is_file():
            discovered.add(candidate.parent.resolve())
    for candidate in root.rglob("skill.md"):
        if candidate.is_file():
            discovered.add(candidate.parent.resolve())
    # Include root itself when it is directly a skill directory.
    if _find_skill_md(root) is not None:
        discovered.add(root.resolve())
    return sorted(discovered)


def _find_skill_md(skill_dir: Path) -> Path | None:
    """Locate ``SKILL.md`` (preferred) or ``skill.md`` in one directory.

    Parameters
    ----------
    skill_dir : Path
        Candidate skill directory.

    Returns
    -------
    Path | None
        Path to skill markdown file when present.
    """
    for filename in ("SKILL.md", "skill.md"):
        candidate = (skill_dir / filename).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _parse_skill_file(skill_dir: Path, skill_md: Path) -> AgentSkill | AgentSkillValidationIssue:
    """Parse and validate one skill file.

    Parameters
    ----------
    skill_dir : Path
        Skill directory.
    skill_md : Path
        ``SKILL.md`` (or lowercase variant).

    Returns
    -------
    AgentSkill | AgentSkillValidationIssue
        Parsed skill metadata or validation issues.
    """
    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError as exc:
        return AgentSkillValidationIssue(
            skill_dir=str(skill_dir),
            skill_md_path=str(skill_md),
            errors=[f"Could not read skill file: {exc}"],
        )

    try:
        frontmatter, _ = _parse_frontmatter(content)
    except ValueError as exc:
        return AgentSkillValidationIssue(
            skill_dir=str(skill_dir),
            skill_md_path=str(skill_md),
            errors=[str(exc)],
        )

    errors = _validate_frontmatter(frontmatter=frontmatter, skill_dir=skill_dir)
    if errors:
        return AgentSkillValidationIssue(
            skill_dir=str(skill_dir),
            skill_md_path=str(skill_md),
            errors=errors,
        )

    metadata_value = frontmatter.get("metadata", {})
    metadata: dict[str, str] = {}
    if isinstance(metadata_value, dict):
        metadata = {str(key): str(value) for key, value in metadata_value.items()}

    return AgentSkill(
        name=_as_string(frontmatter.get("name")),
        description=_as_string(frontmatter.get("description")),
        skill_dir=str(skill_dir),
        skill_md_path=str(skill_md),
        license=_as_string(frontmatter.get("license")),
        compatibility=_as_string(frontmatter.get("compatibility")),
        allowed_tools=_as_string(frontmatter.get("allowed-tools")),
        metadata=metadata,
    )


def _parse_frontmatter(content: str) -> tuple[dict[str, object], str]:
    """Parse YAML-style frontmatter from skill markdown content.

    Parameters
    ----------
    content : str
        Raw markdown content.

    Returns
    -------
    tuple[dict[str, object], str]
        Parsed frontmatter mapping and markdown body.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must begin with YAML frontmatter delimiter '---'.")

    end_index = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break
    if end_index <= 0:
        raise ValueError("SKILL.md frontmatter must be closed by a second '---' delimiter.")

    frontmatter_lines = lines[1:end_index]
    body = "\n".join(lines[end_index + 1 :]).strip()
    frontmatter = _parse_frontmatter_mapping(frontmatter_lines)
    return frontmatter, body


def _parse_frontmatter_mapping(lines: list[str]) -> dict[str, object]:
    """Parse a restricted YAML mapping used by AgentSkills frontmatter.

    Parameters
    ----------
    lines : list[str]
        Frontmatter lines between ``---`` delimiters.

    Returns
    -------
    dict[str, object]
        Parsed frontmatter mapping.
    """
    result: dict[str, object] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if line.startswith(" ") or line.startswith("\t"):
            raise ValueError(f"Invalid frontmatter indentation at line: {line!r}")
        match = re.match(r"^([A-Za-z0-9_-]+):(.*)$", line)
        if match is None:
            raise ValueError(f"Invalid frontmatter entry: {line!r}")
        key = match.group(1)
        remainder = match.group(2).strip()

        if remainder:
            result[key] = _coerce_scalar(remainder)
            index += 1
            continue

        nested, next_index = _parse_nested_mapping(lines=lines, start=index + 1)
        if nested:
            result[key] = nested
        else:
            result[key] = ""
        index = next_index
    return result


def _parse_nested_mapping(lines: list[str], start: int) -> tuple[dict[str, str], int]:
    """Parse one nested mapping block indented by two spaces.

    Parameters
    ----------
    lines : list[str]
        Frontmatter lines.
    start : int
        Start index immediately after the parent ``key:`` line.

    Returns
    -------
    tuple[dict[str, str], int]
        Parsed nested mapping and index of next top-level line.
    """
    nested: dict[str, str] = {}
    index = start
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if not line.startswith("  "):
            break
        nested_line = line[2:]
        if nested_line.startswith(" ") or nested_line.startswith("\t"):
            raise ValueError(f"Only one nested mapping level is supported: {line!r}")
        match = re.match(r"^([A-Za-z0-9_-]+):(.*)$", nested_line)
        if match is None:
            raise ValueError(f"Invalid nested frontmatter entry: {line!r}")
        nested_key = match.group(1)
        nested_value = match.group(2).strip()
        nested[nested_key] = _coerce_scalar(nested_value)
        index += 1
    return nested, index


def _coerce_scalar(value: str) -> str:
    """Normalize scalar frontmatter values.

    Parameters
    ----------
    value : str
        Raw scalar value.

    Returns
    -------
    str
        Trimmed scalar with optional quote wrappers removed.
    """
    trimmed = value.strip()
    if len(trimmed) >= 2:
        if (trimmed[0] == '"' and trimmed[-1] == '"') or (trimmed[0] == "'" and trimmed[-1] == "'"):
            return trimmed[1:-1]
    return trimmed


def _validate_frontmatter(frontmatter: dict[str, object], skill_dir: Path) -> list[str]:
    """Validate AgentSkills frontmatter against core specification constraints.

    Parameters
    ----------
    frontmatter : dict[str, object]
        Parsed frontmatter mapping.
    skill_dir : Path
        Skill directory path for name-directory matching.

    Returns
    -------
    list[str]
        Validation errors, if any.
    """
    errors: list[str] = []
    unknown = sorted(set(frontmatter.keys()) - _ALLOWED_FRONTMATTER_FIELDS)
    if unknown:
        errors.append(
            "Unexpected frontmatter fields: "
            + ", ".join(unknown)
            + f". Allowed fields: {sorted(_ALLOWED_FRONTMATTER_FIELDS)}."
        )

    if "name" not in frontmatter:
        errors.append("Missing required field in frontmatter: name")
    else:
        errors.extend(_validate_name(_as_string(frontmatter.get("name")), skill_dir))

    if "description" not in frontmatter:
        errors.append("Missing required field in frontmatter: description")
    else:
        errors.extend(_validate_description(_as_string(frontmatter.get("description"))))

    if "compatibility" in frontmatter:
        errors.extend(_validate_compatibility(frontmatter.get("compatibility")))

    if "metadata" in frontmatter and not isinstance(frontmatter.get("metadata"), dict):
        errors.append("Field 'metadata' must be a mapping when provided.")

    if "allowed-tools" in frontmatter and not isinstance(frontmatter.get("allowed-tools"), str):
        errors.append("Field 'allowed-tools' must be a string when provided.")

    if "license" in frontmatter and not isinstance(frontmatter.get("license"), str):
        errors.append("Field 'license' must be a string when provided.")

    return errors


def _validate_name(name: str, skill_dir: Path) -> list[str]:
    """Validate skill name format and directory-name conventions.

    Parameters
    ----------
    name : str
        Skill name value.
    skill_dir : Path
        Skill directory path.

    Returns
    -------
    list[str]
        Validation errors for ``name``.
    """
    errors: list[str] = []
    normalized = unicodedata.normalize("NFKC", name.strip())
    if not normalized:
        errors.append("Field 'name' must be a non-empty string.")
        return errors
    if len(normalized) > _MAX_NAME_LENGTH:
        errors.append(f"Field 'name' exceeds {_MAX_NAME_LENGTH} characters ({len(normalized)}).")
    if normalized != normalized.lower():
        errors.append("Field 'name' must be lowercase.")
    if normalized.startswith("-") or normalized.endswith("-"):
        errors.append("Field 'name' must not start or end with '-'.")
    if "--" in normalized:
        errors.append("Field 'name' must not contain consecutive hyphens.")
    if not all(char.isalnum() or char == "-" for char in normalized):
        errors.append("Field 'name' may only contain letters, numbers, and hyphens.")
    if unicodedata.normalize("NFKC", skill_dir.name) != normalized:
        errors.append(f"Skill directory '{skill_dir.name}' must match frontmatter name '{normalized}'.")
    return errors


def _validate_description(description: str) -> list[str]:
    """Validate description length and non-empty constraints.

    Parameters
    ----------
    description : str
        Description value.

    Returns
    -------
    list[str]
        Validation errors for ``description``.
    """
    errors: list[str] = []
    normalized = description.strip()
    if not normalized:
        errors.append("Field 'description' must be a non-empty string.")
        return errors
    if len(normalized) > _MAX_DESCRIPTION_LENGTH:
        errors.append(f"Field 'description' exceeds {_MAX_DESCRIPTION_LENGTH} characters ({len(normalized)}).")
    return errors


def _validate_compatibility(value: object) -> list[str]:
    """Validate optional compatibility field constraints.

    Parameters
    ----------
    value : object
        Compatibility value from frontmatter.

    Returns
    -------
    list[str]
        Validation errors for ``compatibility``.
    """
    if not isinstance(value, str):
        return ["Field 'compatibility' must be a string when provided."]
    if len(value) > _MAX_COMPATIBILITY_LENGTH:
        return [f"Field 'compatibility' exceeds {_MAX_COMPATIBILITY_LENGTH} characters ({len(value)})."]
    return []


def _as_string(value: object) -> str:
    """Return a safe string value.

    Parameters
    ----------
    value : object
        Value to coerce.

    Returns
    -------
    str
        Coerced string (empty for non-strings).
    """
    if isinstance(value, str):
        return value.strip()
    return ""


def _tokenize(value: str) -> set[str]:
    """Tokenize text for skill matching.

    Parameters
    ----------
    value : str
        Input text.

    Returns
    -------
    set[str]
        Lowercased token set.
    """
    return {token for token in re.findall(r"[a-z0-9]{3,}", value.lower())}
