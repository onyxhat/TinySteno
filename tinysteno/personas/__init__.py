"""Persona loading and discovery for TinySteno."""
from __future__ import annotations

import keyword
import logging
import shutil
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
    """Encapsulates a meeting note persona: schema, system prompt, and Jinja2 template."""

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
        ) from e

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


def seed_builtin_personas(interactive: bool = False, force: bool = False) -> dict[str, list[str]]:
    """Copy built-in persona dirs to _USER_DIR.

    Returns {"copied": [...], "skipped": [...]} for callers to display or discard.
    """
    if force and interactive:
        raise ValueError("force and interactive are mutually exclusive")
    _USER_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    skipped: list[str] = []

    for slug in BUILTIN_ORDER:
        src = _BUILTIN_DIR / slug
        dst = _USER_DIR / slug

        if not src.exists():
            logger.warning("Built-in persona source missing: %s", src)
            continue

        if dst.exists():
            if force:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                copied.append(slug)
            elif interactive:
                answer = input(
                    f"  Built-in persona '{slug}' already exists. Overwrite? [y/N] "
                ).strip().lower()
                if answer in ("y", "yes"):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    copied.append(slug)
                else:
                    skipped.append(slug)
            else:
                skipped.append(slug)
        else:
            shutil.copytree(src, dst)
            copied.append(slug)

    return {"copied": copied, "skipped": skipped}


def load_persona(slug: str) -> Persona:
    """Load persona by slug from _USER_DIR.

    Raises PersonaNotFoundError if no directory for the slug exists in _USER_DIR.
    Raises PersonaInvalidError if the directory exists but is malformed.
    """
    user_path = _USER_DIR / slug

    if user_path.exists() and user_path.is_dir():
        return _load_from_dir(slug, user_path)

    available = list_personas()
    raise PersonaNotFoundError(
        f"Unknown persona '{slug}'. Available: {', '.join(available)}"
    )


def list_personas() -> list[str]:
    """Return available persona slugs in canonical order.

    Order: built-ins in BUILTIN_ORDER (only those present and valid in _USER_DIR),
    then user-only personas sorted case-insensitively.
    Malformed persona directories are skipped with a warning.
    """
    if not _USER_DIR.exists():
        return []

    valid_slugs: set[str] = set()
    user_only: list[str] = []

    for entry in sorted(_USER_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        slug = entry.name
        try:
            _validate_dir(entry, slug)
        except PersonaInvalidError as e:
            logger.warning(str(e))
            continue
        valid_slugs.add(slug)
        if slug not in BUILTIN_ORDER:
            user_only.append(slug)

    result = [s for s in BUILTIN_ORDER if s in valid_slugs]
    result.extend(user_only)
    return result
