"""Shared pytest fixtures for TinySteno tests."""
import pytest


@pytest.fixture
def tmp_persona_dir(tmp_path):
    """Create a temporary user personas directory with a valid test persona."""
    persona_dir = tmp_path / "personas" / "test-persona"
    persona_dir.mkdir(parents=True)

    (persona_dir / "persona.yaml").write_text(
        "name: Test Persona\n"
        "description: A persona used in tests.\n"
        "schema:\n"
        "  summary:\n"
        "    type: string\n"
        "    description: A short summary\n"
        "  points:\n"
        "    type: list\n"
        "    description: Key points as strings\n"
    )
    (persona_dir / "system_prompt.md").write_text(
        "You are a test assistant."
    )
    (persona_dir / "template.md").write_text(
        "# {{ title }}\n\n{{ summary }}\n\n{% for p in points %}- {{ p }}\n{% endfor %}"
    )
    return tmp_path / "personas"
