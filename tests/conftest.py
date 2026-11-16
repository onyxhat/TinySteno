"""Shared pytest fixtures for TinySteno tests."""
import os
import shutil
from pathlib import Path

import pytest

import tinysteno.personas
from tinysteno.personas import _BUILTIN_DIR, BUILTIN_ORDER

# Computed before any patching, so the guard test can prove nothing resolves under it.
REAL_HOME = os.path.expanduser("~")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Point HOME, Path.home and the personas dir at a per-test temp directory.

    Child processes inherit the redirected HOME environment variable.
    """
    fake_home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(tinysteno.personas, "_USER_DIR", fake_home / ".tinysteno" / "personas")
    return fake_home


@pytest.fixture(name="real_home")
def real_home_fixture():
    """The real home directory, captured at conftest import time."""
    return Path(REAL_HOME)


@pytest.fixture
def seeded_user_dir(tmp_path):
    """A tmp personas dir pre-seeded with all built-in personas."""
    personas_dir = tmp_path / "personas"
    for slug in BUILTIN_ORDER:
        shutil.copytree(_BUILTIN_DIR / slug, personas_dir / slug)
    return personas_dir
