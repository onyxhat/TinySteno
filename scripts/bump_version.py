"""Bump the project version and refresh the changelog.

Single source of truth for version changes, shared by the ``bump-version`` skill
and the ``version-bump`` GitHub Actions workflow. Rewrites the version string in
``pyproject.toml`` and ``tinysteno/__init__.py`` together, then prepends a dated
section to ``CHANGELOG.md`` built from the commit subjects since the last tag.

Diagnostics go to stderr; the resulting version is the only line on stdout, so
callers can read it with ``python scripts/bump_version.py | tail -1``.

Usage::

    python scripts/bump_version.py            # patch bump (0.2.0 -> 0.2.1)
    python scripts/bump_version.py --set 1.0.0
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "tinysteno" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"

BUMP_SUBJECT_PREFIX = "chore: bump version"

CHANGELOG_HEADER = """# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
"""


def parse_version(pyproject_text: str) -> str:
    """Return the ``[project]`` version declared in ``pyproject.toml`` text."""
    return tomllib.loads(pyproject_text)["project"]["version"]


def validate_version(version: str) -> str:
    """Return ``version`` unchanged if it is a bare ``MAJOR.MINOR.PATCH`` string."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    return version


def bump_patch(version: str) -> str:
    """Return ``version`` with its patch component incremented by one."""
    major, minor, patch = (int(part) for part in validate_version(version).split("."))
    return f"{major}.{minor}.{patch + 1}"


def replace_pyproject_version(text: str, new: str) -> str:
    """Return ``text`` with the first line-anchored ``version = "..."`` set to ``new``."""
    updated, count = re.subn(
        r'(?m)^version = "[^"]+"', f'version = "{new}"', text, count=1
    )
    if count != 1:
        raise ValueError("could not find a 'version = \"...\"' line in pyproject.toml")
    return updated


def replace_init_version(text: str, new: str) -> str:
    """Return ``text`` with the ``__version__ = "..."`` assignment set to ``new``."""
    updated, count = re.subn(
        r'(?m)^__version__ = "[^"]+"', f'__version__ = "{new}"', text, count=1
    )
    if count != 1:
        raise ValueError("could not find a '__version__ = \"...\"' line in __init__.py")
    return updated


def render_section(version: str, date: str, subjects: list[str]) -> str:
    """Render one ``## [version] - date`` changelog block from commit subjects."""
    lines = [f"## [{version}] - {date}"]
    lines += [f"- {subject}" for subject in subjects] or ["- No changes recorded."]
    return "\n".join(lines) + "\n"


def insert_section(existing: str, section: str) -> str:
    """Return the changelog with ``section`` inserted above the newest entry."""
    if not existing.strip():
        return f"{CHANGELOG_HEADER}\n{section}"
    marker = existing.find("\n## [")
    if marker == -1:
        return f"{existing.rstrip()}\n\n{section}"
    head, rest = existing[:marker].rstrip(), existing[marker + 1:]
    return f"{head}\n\n{section}\n{rest}"


def _git(*args: str) -> str:
    """Run ``git`` in the repo root and return its stripped stdout."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=ROOT
    )
    return result.stdout.strip()


def head_subject() -> str:
    """Return the subject line of HEAD, or ``""`` outside a git checkout."""
    try:
        return _git("log", "-1", "--pretty=%s")
    except subprocess.CalledProcessError:
        return ""


def commit_subjects_since_last_tag() -> list[str]:
    """Return commit subjects since the last tag, dropping prior bump commits."""
    try:
        commit_range = f"{_git('describe', '--tags', '--abbrev=0')}..HEAD"
    except subprocess.CalledProcessError:
        commit_range = "HEAD"
    try:
        out = _git("log", commit_range, "--no-merges", "--pretty=%s")
    except subprocess.CalledProcessError:
        return []
    return [
        line
        for line in out.splitlines()
        if line and not line.startswith(BUMP_SUBJECT_PREFIX)
    ]


def main(argv: list[str] | None = None) -> int:
    """Bump the version across both files and the changelog; print the new version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="set_version",
        metavar="X.Y.Z",
        help="set an explicit version instead of a patch bump",
    )
    args = parser.parse_args(argv)

    pyproject_text = PYPROJECT.read_text()
    current = parse_version(pyproject_text)

    subject = head_subject()
    if subject.startswith(BUMP_SUBJECT_PREFIX):
        print(f"HEAD is already a version bump ({subject!r}); nothing to do.", file=sys.stderr)
        print(current)
        return 0

    new = validate_version(args.set_version) if args.set_version else bump_patch(current)
    if new == current:
        print(f"version is already {current}; nothing to do.", file=sys.stderr)
        print(current)
        return 0

    PYPROJECT.write_text(replace_pyproject_version(pyproject_text, new))
    INIT.write_text(replace_init_version(INIT.read_text(), new))

    section = render_section(new, dt.date.today().isoformat(), commit_subjects_since_last_tag())
    existing = CHANGELOG.read_text() if CHANGELOG.exists() else ""
    CHANGELOG.write_text(insert_section(existing, section))

    print(f"bumped {current} -> {new}", file=sys.stderr)
    print(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
