"""Tests for scripts/bump_version.py."""

# pylint: disable=missing-function-docstring,redefined-outer-name,protected-access

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bump_version", Path(__file__).resolve().parent.parent / "scripts" / "bump_version.py"
)
bump_version = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bump_version)


SAMPLE_PYPROJECT = '''\
[project]
name = "tinysteno"
version = "0.2.0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.4.3",
]
'''

SAMPLE_INIT = '''\
"""TinySteno - Minimal meeting recorder with Obsidian export."""

__version__ = "0.2.0"
'''


# --- pure helpers ---

@pytest.mark.parametrize(
    ("current", "expected"),
    [("0.2.0", "0.2.1"), ("1.9.9", "1.9.10"), ("0.0.0", "0.0.1"), ("10.20.30", "10.20.31")],
)
def test_bump_patch(current, expected):
    assert bump_version.bump_patch(current) == expected


@pytest.mark.parametrize("bad", ["1.2", "1.2.3.4", "v1.2.3", "1.2.x", ""])
def test_bump_patch_rejects_non_semver(bad):
    with pytest.raises(ValueError):
        bump_version.bump_patch(bad)


def test_validate_version_roundtrips_good_input():
    assert bump_version.validate_version("1.0.0") == "1.0.0"


@pytest.mark.parametrize("bad", ["1.0", "1.0.0-rc1", "1.0.0a", " 1.0.0"])
def test_validate_version_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        bump_version.validate_version(bad)


def test_parse_version_reads_project_table():
    assert bump_version.parse_version(SAMPLE_PYPROJECT) == "0.2.0"


def test_replace_pyproject_version_touches_only_project_version():
    out = bump_version.replace_pyproject_version(SAMPLE_PYPROJECT, "0.2.1")
    assert 'version = "0.2.1"' in out
    assert 'version = "0.2.0"' not in out
    # dependency pins and requires-python must be untouched
    assert 'requires-python = ">=3.12"' in out
    assert '"numpy>=2.4.3"' in out


def test_replace_pyproject_version_raises_when_absent():
    with pytest.raises(ValueError):
        bump_version.replace_pyproject_version("[project]\nname = \"x\"\n", "0.2.1")


def test_replace_init_version():
    out = bump_version.replace_init_version(SAMPLE_INIT, "0.3.0")
    assert '__version__ = "0.3.0"' in out
    assert out.startswith('"""TinySteno')


def test_replace_init_version_raises_when_absent():
    with pytest.raises(ValueError):
        bump_version.replace_init_version("# no version here\n", "0.3.0")


# --- changelog rendering ---

def test_render_section_with_subjects():
    section = bump_version.render_section("0.2.1", "2026-09-09", ["feat: a", "fix: b"])
    assert section == "## [0.2.1] - 2026-09-09\n- feat: a\n- fix: b\n"


def test_render_section_without_subjects():
    section = bump_version.render_section("0.2.1", "2026-09-09", [])
    assert section == "## [0.2.1] - 2026-09-09\n- No changes recorded.\n"


def test_insert_section_into_empty_changelog():
    out = bump_version.insert_section("", "## [0.2.1] - 2026-09-09\n- feat: a\n")
    assert out.startswith("# Changelog\n")
    assert out.rstrip().endswith("- feat: a")


def test_insert_section_header_only_appends():
    existing = bump_version.CHANGELOG_HEADER
    out = bump_version.insert_section(existing, "## [0.2.1] - 2026-09-09\n- feat: a\n")
    assert out == existing.rstrip() + "\n\n## [0.2.1] - 2026-09-09\n- feat: a\n"


def test_insert_section_prepends_above_newest_entry():
    existing = bump_version.CHANGELOG_HEADER + "\n## [0.2.0] - 2026-09-01\n- old\n"
    out = bump_version.insert_section(existing, "## [0.2.1] - 2026-09-09\n- new\n")
    assert out.index("## [0.2.1]") < out.index("## [0.2.0]")
    assert "- old" in out and "- new" in out


# --- main() integration ---

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    init = tmp_path / "__init__.py"
    changelog = tmp_path / "CHANGELOG.md"
    pyproject.write_text(SAMPLE_PYPROJECT)
    init.write_text(SAMPLE_INIT)
    changelog.write_text(bump_version.CHANGELOG_HEADER + "\n## [0.2.0] - 2026-09-01\n- baseline\n")
    monkeypatch.setattr(bump_version, "PYPROJECT", pyproject)
    monkeypatch.setattr(bump_version, "INIT", init)
    monkeypatch.setattr(bump_version, "CHANGELOG", changelog)
    monkeypatch.setattr(bump_version, "head_subject", lambda: "feat: something")
    monkeypatch.setattr(bump_version, "commit_subjects_since_last_tag", lambda: ["feat: something"])
    monkeypatch.setattr(bump_version, "_relock", lambda: None)
    return tmp_path


def test_main_patch_bump_updates_all_three_files(sandbox, capsys):
    rc = bump_version.main([])
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "0.2.1"
    assert 'version = "0.2.1"' in (sandbox / "pyproject.toml").read_text()
    assert '__version__ = "0.2.1"' in (sandbox / "__init__.py").read_text()
    changelog = (sandbox / "CHANGELOG.md").read_text()
    assert changelog.index("## [0.2.1]") < changelog.index("## [0.2.0]")
    assert "- feat: something" in changelog


def test_main_set_flag_overrides_patch(sandbox, capsys):
    bump_version.main(["--set", "1.0.0"])
    assert capsys.readouterr().out.strip().splitlines()[-1] == "1.0.0"
    assert 'version = "1.0.0"' in (sandbox / "pyproject.toml").read_text()


def test_main_is_idempotent_when_head_is_a_bump(sandbox, capsys, monkeypatch):
    monkeypatch.setattr(
        bump_version, "head_subject", lambda: "chore: bump version to 0.2.1 [skip ci]"
    )
    rc = bump_version.main([])
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "0.2.0"
    assert (sandbox / "pyproject.toml").read_text() == SAMPLE_PYPROJECT


def test_relock_is_best_effort(monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise OSError("no uv")

    monkeypatch.setattr(bump_version.subprocess, "run", boom)
    bump_version._relock()  # must not raise
    assert "could not refresh uv.lock" in capsys.readouterr().err


def test_main_no_op_when_set_equals_current(sandbox, capsys):
    rc = bump_version.main(["--set", "0.2.0"])
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "0.2.0"
    assert (sandbox / "CHANGELOG.md").read_text().count("## [0.2.0]") == 1
