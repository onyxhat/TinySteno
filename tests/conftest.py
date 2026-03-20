"""Shared pytest fixtures for TinySteno tests."""
import shutil

import pytest

from tinysteno.personas import _BUILTIN_DIR, BUILTIN_ORDER


@pytest.fixture
def seeded_user_dir(tmp_path):
    """A tmp personas dir pre-seeded with all built-in personas."""
    personas_dir = tmp_path / "personas"
    for slug in BUILTIN_ORDER:
        shutil.copytree(_BUILTIN_DIR / slug, personas_dir / slug)
    return personas_dir
