"""Tests for tinysteno.personas module."""
import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from tinysteno.personas import (
    Persona,
    PersonaNotFoundError,
    PersonaInvalidError,
    load_persona,
    list_personas,
    BUILTIN_ORDER,
    seed_builtin_personas,
    _BUILTIN_DIR,
)

# pylint: disable=missing-function-docstring,protected-access

# --- list_personas ---

def test_list_personas_returns_all_builtins(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        slugs = list_personas()
    assert slugs[:len(BUILTIN_ORDER)] == BUILTIN_ORDER


def test_list_personas_builtin_order_is_fixed():
    assert BUILTIN_ORDER == [
        "default", "rca", "irm", "sprint", "kickoff", "executive-summary"
    ]


def test_list_personas_no_duplicates(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        slugs = list_personas()
    assert len(slugs) == len(set(slugs))


def test_list_personas_skips_malformed_user_persona(tmp_path):
    user_dir = tmp_path / "personas" / "my-bad-persona"
    user_dir.mkdir(parents=True)
    # missing all required files — malformed

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        slugs = list_personas()

    assert "my-bad-persona" not in slugs


def test_list_personas_malformed_builtin_slug_is_excluded(tmp_path):
    # Malformed user dir for a built-in slug: no fallback to package, slug excluded
    user_dir = tmp_path / "personas" / "default"
    user_dir.mkdir(parents=True)
    # malformed: no files

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        slugs = list_personas()

    assert "default" not in slugs


def test_list_personas_valid_user_only_appended_alphabetically(tmp_path):
    for slug in ["zebra", "alpha", "Middle"]:
        d = tmp_path / "personas" / slug
        d.mkdir(parents=True)
        _write_valid_persona_dir(d, slug)

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        slugs = list_personas()

    user_slugs = [s for s in slugs if s not in BUILTIN_ORDER]
    assert user_slugs == sorted(["zebra", "alpha", "Middle"], key=str.lower)


# --- load_persona ---

def test_load_persona_default_returns_persona(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        p = load_persona("default")
    assert isinstance(p, Persona)
    assert p.slug == "default"
    assert p.name
    assert p.description
    assert "overview" in p.schema
    assert p.system_prompt
    assert p.template


def test_load_persona_unknown_slug_raises(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        with pytest.raises(PersonaNotFoundError, match="unknown-slug"):
            load_persona("unknown-slug")


def test_load_persona_unknown_lists_available_in_error(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        with pytest.raises(PersonaNotFoundError) as exc_info:
            load_persona("no-such-persona")
    assert "default" in str(exc_info.value)


def test_load_persona_missing_file_raises_invalid(tmp_path):
    user_dir = tmp_path / "personas" / "incomplete"
    user_dir.mkdir(parents=True)
    (user_dir / "persona.yaml").write_text(
        "name: Inc\ndescription: d\nschema: {}\n"
    )
    # missing system_prompt.md and template.md

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        with pytest.raises(PersonaInvalidError, match="system_prompt.md"):
            load_persona("incomplete")


def test_load_persona_invalid_field_name_raises(tmp_path):
    user_dir = tmp_path / "personas" / "badfield"
    user_dir.mkdir(parents=True)
    (user_dir / "persona.yaml").write_text(
        "name: B\ndescription: d\n"
        "schema:\n  123bad: {type: string, description: x}\n"
    )
    (user_dir / "system_prompt.md").write_text("x")
    (user_dir / "template.md").write_text("x")

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        with pytest.raises(PersonaInvalidError, match="123bad"):
            load_persona("badfield")


def test_load_persona_keyword_field_name_raises(tmp_path):
    user_dir = tmp_path / "personas" / "kwpersona"
    user_dir.mkdir(parents=True)
    (user_dir / "persona.yaml").write_text(
        "name: K\ndescription: d\n"
        "schema:\n  for: {type: string, description: x}\n"
    )
    (user_dir / "system_prompt.md").write_text("x")
    (user_dir / "template.md").write_text("x")

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        with pytest.raises(PersonaInvalidError, match="for"):
            load_persona("kwpersona")


def test_load_persona_reserved_field_name_raises(tmp_path):
    user_dir = tmp_path / "personas" / "reserved"
    user_dir.mkdir(parents=True)
    (user_dir / "persona.yaml").write_text(
        "name: R\ndescription: d\n"
        "schema:\n  transcript: {type: string, description: x}\n"
    )
    (user_dir / "system_prompt.md").write_text("x")
    (user_dir / "template.md").write_text("x")

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        with pytest.raises(PersonaInvalidError, match="transcript"):
            load_persona("reserved")


def test_load_persona_invalid_field_type_raises(tmp_path):
    user_dir = tmp_path / "personas" / "badtype"
    user_dir.mkdir(parents=True)
    (user_dir / "persona.yaml").write_text(
        "name: BT\ndescription: d\n"
        "schema:\n  myfield: {type: dict, description: x}\n"
    )
    (user_dir / "system_prompt.md").write_text("x")
    (user_dir / "template.md").write_text("x")

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        with pytest.raises(PersonaInvalidError, match="myfield"):
            load_persona("badtype")


def test_load_persona_user_overrides_builtin(tmp_path):
    user_dir = tmp_path / "personas" / "default"
    user_dir.mkdir(parents=True)
    _write_valid_persona_dir(user_dir, "default", name="My Custom Default")

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        p = load_persona("default")

    assert p.name == "My Custom Default"


def test_load_persona_malformed_user_shadows_builtin_raises(tmp_path):
    """Malformed user dir for a built-in slug: load raises, no built-in fallback."""
    user_dir = tmp_path / "personas" / "default"
    user_dir.mkdir(parents=True)
    # malformed: no files

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        with pytest.raises(PersonaInvalidError):
            load_persona("default")


# --- Persona dataclass ---

def test_persona_schema_preserves_insertion_order(tmp_path):
    user_dir = tmp_path / "personas" / "ordered"
    user_dir.mkdir(parents=True)
    (user_dir / "persona.yaml").write_text(
        "name: O\ndescription: d\n"
        "schema:\n"
        "  zzz: {type: string, description: last}\n"
        "  aaa: {type: list, description: first}\n"
    )
    (user_dir / "system_prompt.md").write_text("x")
    (user_dir / "template.md").write_text("x")

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        p = load_persona("ordered")

    assert list(p.schema.keys()) == ["zzz", "aaa"]


# --- seed_builtin_personas ---


def test_seed_copies_all_builtins_to_empty_dir(tmp_path):
    personas_dir = tmp_path / "personas"
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas()
    assert set(result["copied"]) == set(BUILTIN_ORDER)
    assert not result["skipped"]
    for slug in BUILTIN_ORDER:
        assert (personas_dir / slug / "persona.yaml").exists()


def test_seed_skips_existing_when_not_forced(tmp_path):
    personas_dir = tmp_path / "personas"
    shutil.copytree(_BUILTIN_DIR / "default", personas_dir / "default")
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas()
    assert "default" in result["skipped"]
    assert "default" not in result["copied"]


def test_seed_overwrites_when_forced(tmp_path):
    personas_dir = tmp_path / "personas"
    shutil.copytree(_BUILTIN_DIR / "default", personas_dir / "default")
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas(force=True)
    assert "default" in result["copied"]
    assert "default" not in result["skipped"]


def test_seed_interactive_overwrites_on_yes(tmp_path, monkeypatch):
    personas_dir = tmp_path / "personas"
    shutil.copytree(_BUILTIN_DIR / "default", personas_dir / "default")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas(interactive=True)
    assert "default" in result["copied"]
    assert "default" not in result["skipped"]


def test_seed_interactive_skips_on_no(tmp_path, monkeypatch):
    personas_dir = tmp_path / "personas"
    shutil.copytree(_BUILTIN_DIR / "default", personas_dir / "default")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas(interactive=True)
    assert "default" in result["skipped"]
    assert "default" not in result["copied"]


def test_seed_warns_and_skips_missing_source(tmp_path, caplog):
    fake_builtin = tmp_path / "fake_builtin"
    fake_builtin.mkdir()
    for slug in BUILTIN_ORDER[1:]:  # all except "default"
        shutil.copytree(_BUILTIN_DIR / slug, fake_builtin / slug)
    personas_dir = tmp_path / "personas"
    with patch("tinysteno.personas._BUILTIN_DIR", fake_builtin):
        with patch("tinysteno.personas._USER_DIR", personas_dir):
            with caplog.at_level(logging.WARNING, logger="tinysteno.personas"):
                result = seed_builtin_personas()
    assert "default" not in result["copied"]
    assert "default" not in result["skipped"]
    assert any("default" in r.message for r in caplog.records)


# --- helpers ---

def _write_valid_persona_dir(d: Path, slug: str, name: str = None):
    (d / "persona.yaml").write_text(
        f"name: {name or slug.title()}\n"
        "description: Test description.\n"
        "schema:\n"
        "  summary: {type: string, description: A summary}\n"
    )
    (d / "system_prompt.md").write_text("You are a test assistant.")
    (d / "template.md").write_text("# {{ title }}\n{{ summary }}")
