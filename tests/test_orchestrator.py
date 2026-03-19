"""Tests for tinysteno.orchestrator (Orchestrator)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from tinysteno.personas import Persona
from tinysteno.orchestrator import Orchestrator

# pylint: disable=missing-function-docstring,protected-access

def _make_persona(schema: dict) -> Persona:
    return Persona(
        slug="test",
        name="Test",
        description="desc",
        schema=schema,
        system_prompt="You are a test assistant.",
        template="{{ title }}",
        template_path=Path("/fake/template.md"),
    )


def _make_orchestrator() -> Orchestrator:
    with patch("tinysteno.orchestrator.OpenAI"):
        s = Orchestrator(api_key="test", base_url="http://localhost", model="test-model")
    return s


# --- _build_user_message ---

def test_build_user_message_includes_field_names():
    persona = _make_persona({
        "overview": {"type": "string", "description": "A summary"},
        "points":   {"type": "list",   "description": "Key points"},
    })
    s = _make_orchestrator()
    msg = s._build_user_message("Hello world", persona)
    assert '"overview"' in msg
    assert '"points"' in msg


def test_build_user_message_string_field_shown_as_string():
    persona = _make_persona({"title_field": {"type": "string", "description": "x"}})
    s = _make_orchestrator()
    msg = s._build_user_message("t", persona)
    assert '"title_field": "string"' in msg


def test_build_user_message_list_field_shown_as_array():
    persona = _make_persona({"items": {"type": "list", "description": "x"}})
    s = _make_orchestrator()
    msg = s._build_user_message("t", persona)
    assert '"items": [' in msg


def test_build_user_message_truncates_transcript():
    persona = _make_persona({"s": {"type": "string", "description": "x"}})
    s = _make_orchestrator()
    long_transcript = "x" * 20000
    msg = s._build_user_message(long_transcript, persona)
    # The transcript in the message should be at most 15000 chars
    transcript_section = msg.split("Transcript:\n", 1)[1]
    assert len(transcript_section) <= 15000


def test_build_user_message_includes_field_descriptions():
    persona = _make_persona({
        "key": {"type": "string", "description": "The primary key value"},
    })
    s = _make_orchestrator()
    msg = s._build_user_message("transcript", persona)
    assert "The primary key value" in msg


# --- summarize ---

def test_summarize_returns_dict_with_schema_fields():
    persona = _make_persona({
        "overview": {"type": "string", "description": "summary"},
        "items":    {"type": "list",   "description": "items"},
    })
    s = _make_orchestrator()
    s._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "overview": "Great meeting",
            "items": ["point 1", "point 2"],
        })))]
    )
    result = s.summarize("transcript", persona)
    assert result == {"overview": "Great meeting", "items": ["point 1", "point 2"]}


def test_summarize_defaults_missing_string_field():
    persona = _make_persona({"title_field": {"type": "string", "description": "x"}})
    s = _make_orchestrator()
    s._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({})))]
    )
    result = s.summarize("t", persona)
    assert result["title_field"] == ""


def test_summarize_defaults_missing_list_field():
    persona = _make_persona({"items": {"type": "list", "description": "x"}})
    s = _make_orchestrator()
    s._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({})))]
    )
    result = s.summarize("t", persona)
    assert result["items"] == []


def test_summarize_ignores_extra_llm_fields():
    persona = _make_persona({"overview": {"type": "string", "description": "x"}})
    s = _make_orchestrator()
    s._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "overview": "hi", "unexpected_field": "ignored"
        })))]
    )
    result = s.summarize("t", persona)
    assert "unexpected_field" not in result
    assert result["overview"] == "hi"


def test_summarize_returns_defaults_on_llm_exception():
    persona = _make_persona({
        "overview": {"type": "string", "description": "x"},
        "items": {"type": "list", "description": "x"},
    })
    s = _make_orchestrator()
    s._client.chat.completions.create.side_effect = Exception("timeout")
    result = s.summarize("t", persona)
    assert result == {"overview": "", "items": []}


def test_summarize_uses_persona_system_prompt():
    persona = _make_persona({"s": {"type": "string", "description": "x"}})
    persona = Persona(
        slug=persona.slug, name=persona.name, description=persona.description,
        schema=persona.schema,
        system_prompt="CUSTOM SYSTEM PROMPT",
        template=persona.template, template_path=persona.template_path,
    )
    s = _make_orchestrator()
    s._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({"s": ""})))]
    )
    s.summarize("t", persona)
    call_kwargs = s._client.chat.completions.create.call_args
    messages = call_kwargs[1]["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"] == "CUSTOM SYSTEM PROMPT"


# --- generate_title ---

def test_generate_title_returns_string_on_success():
    s = _make_orchestrator()
    s._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="My Meeting Title"))]
    )
    result = s.generate_title("Some overview text")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_title_returns_none_on_exception():
    s = _make_orchestrator()
    s._client.chat.completions.create.side_effect = Exception("API down")
    result = s.generate_title("overview")
    assert result is None
