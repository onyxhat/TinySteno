"""Persona loading and discovery for TinySteno."""
from __future__ import annotations

import keyword
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

BUILTIN_ORDER = ["default", "rca", "irm", "sprint", "kickoff", "executive-summary"]

_RESERVED_NAMES = frozenset({"title", "date", "duration", "transcript", "detected_language"})
_VALID_TYPES = frozenset({"string", "list"})
_REQUIRED_FILES = ("persona.yaml", "system_prompt.md", "template.md")

_BUILTIN_DIR = Path(__file__).parent
_USER_DIR = Path.home() / ".tinysteno" / "personas"


@dataclass
class Persona:
    slug: str
    name: str
    description: str
    schema: dict  # field name -> {"type": str, "description": str}; insertion-ordered
    system_prompt: str
    template: str
    template_path: Path  # absolute path to template.md, for error reporting


class PersonaNotFoundError(Exception):
    """No persona directory exists for the requested slug."""


class PersonaInvalidError(Exception):
    """A persona directory exists but is malformed."""


def _validate_dir(path: Path, slug: str) -> dict:
    """Validate persona directory files and schema. Returns parsed persona.yaml data.

    Raises PersonaInvalidError with a descriptive message on any problem.
    """
    for filename in _REQUIRED_FILES:
        if not (path / filename).exists():
            raise PersonaInvalidError(
                f"Persona '{slug}': missing required file {path / filename}"
            )

    try:
        data = yaml.safe_load((path / "persona.yaml").read_text())
    except Exception as e:
        raise PersonaInvalidError(
            f"Persona '{slug}': failed to parse persona.yaml: {e}"
        )

    if not isinstance(data, dict):
        raise PersonaInvalidError(
            f"Persona '{slug}': persona.yaml must be a YAML mapping"
        )

    for key in ("name", "description", "schema"):
        if key not in data:
            raise PersonaInvalidError(
                f"Persona '{slug}': persona.yaml missing required key '{key}'"
            )

    schema = data.get("schema") or {}
    for field_name, field_def in schema.items():
        field_name_str = str(field_name)
        if not field_name_str.isidentifier():
            raise PersonaInvalidError(
                f"Persona '{slug}': field name '{field_name_str}' is not a valid Python identifier"
            )
        if keyword.iskeyword(field_name_str):
            raise PersonaInvalidError(
                f"Persona '{slug}': field name '{field_name_str}' is a Python keyword"
            )
        if field_name_str in _RESERVED_NAMES:
            raise PersonaInvalidError(
                f"Persona '{slug}': field name '{field_name_str}' collides with reserved "
                f"metadata variable name"
            )
        if not isinstance(field_def, dict) or field_def.get("type") not in _VALID_TYPES:
            raise PersonaInvalidError(
                f"Persona '{slug}': field '{field_name_str}' has invalid type "
                f"(must be 'string' or 'list', got {field_def!r})"
            )

    return data


def _load_from_dir(slug: str, path: Path) -> Persona:
    """Load a Persona from a directory. Raises PersonaInvalidError if malformed."""
    data = _validate_dir(path, slug)
    return Persona(
        slug=slug,
        name=data["name"],
        description=data["description"],
        schema=data.get("schema") or {},
        system_prompt=(path / "system_prompt.md").read_text(),
        template=(path / "template.md").read_text(),
        template_path=(path / "template.md").resolve(),
    )


def load_persona(slug: str) -> Persona:
    """Load persona by slug.

    User persona in ~/.tinysteno/personas/<slug>/ takes priority over built-ins.
    If user directory exists for the slug, it is used exclusively (no fallback to built-in).

    Raises PersonaNotFoundError if no directory for the slug exists.
    Raises PersonaInvalidError if the directory exists but is malformed.
    """
    user_path = _USER_DIR / slug
    builtin_path = _BUILTIN_DIR / slug

    if user_path.exists() and user_path.is_dir():
        return _load_from_dir(slug, user_path)

    if builtin_path.exists() and builtin_path.is_dir():
        return _load_from_dir(slug, builtin_path)

    available = list_personas()
    raise PersonaNotFoundError(
        f"Unknown persona '{slug}'. Available: {', '.join(available)}"
    )


def list_personas() -> list[str]:
    """Return available persona slugs in canonical order.

    Order: built-ins in BUILTIN_ORDER, then user-only personas sorted
    case-insensitively. A user persona that overrides a built-in slug stays at
    the built-in's position. Malformed user persona directories are skipped
    (with a warning) and do not suppress built-ins.
    """
    # Start with built-in order; user valid overrides replace in-place at load time
    result = list(BUILTIN_ORDER)
    user_only: list[str] = []

    if not _USER_DIR.exists():
        return result

    for entry in sorted(_USER_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        slug = entry.name
        try:
            _validate_dir(entry, slug)
        except PersonaInvalidError as e:
            logger.warning(str(e))
            # If it shadows a built-in, the built-in remains in the list
            continue

        if slug not in BUILTIN_ORDER:
            user_only.append(slug)
        # Valid built-in override: slug already in result, load_persona will pick user's version

    result.extend(user_only)
    return result
