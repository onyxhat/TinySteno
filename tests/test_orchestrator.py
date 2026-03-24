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


def test_build_user_message_includes_full_transcript():
    persona = _make_persona({"s": {"type": "string", "description": "x"}})
    s = _make_orchestrator()
    long_transcript = "x" * 20000
    msg = s._build_user_message(long_transcript, persona)
    transcript_section = msg.split("Transcript:\n", 1)[1]
    assert len(transcript_section) == 20000


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


# --- generate_tags ---

def test_generate_tags_returns_list_on_success():
    s = _make_orchestrator()
    s._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="budget, planning, q4"))]
    )
    result = s.generate_tags("We discussed Q4 budget planning")
    assert isinstance(result, list)
    assert result == ["budget", "planning", "q4"]


def test_generate_tags_returns_empty_list_on_exception():
    s = _make_orchestrator()
    s._client.chat.completions.create.side_effect = Exception("API down")
    result = s.generate_tags("some content")
    assert not result


def test_generate_tags_returns_empty_list_on_empty_response():
    s = _make_orchestrator()
    s._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=""))]
    )
    result = s.generate_tags("content")
    assert not result


# --- _clean_tags ---

def test_clean_tags_lowercases():
    s = _make_orchestrator()
    assert s._clean_tags("Budget, PLANNING") == ["budget", "planning"]


def test_clean_tags_removes_special_characters():
    s = _make_orchestrator()
    assert s._clean_tags("q4-review, cost!") == ["q4review", "cost"]


def test_clean_tags_converts_spaces_to_underscores():
    s = _make_orchestrator()
    assert s._clean_tags("budget planning") == ["budget_planning"]


def test_clean_tags_collapses_repeated_underscores():
    s = _make_orchestrator()
    assert s._clean_tags("q4__review") == ["q4_review"]


def test_clean_tags_strips_leading_trailing_underscores():
    s = _make_orchestrator()
    assert s._clean_tags("_budget_") == ["budget"]


def test_clean_tags_skips_empty_parts():
    s = _make_orchestrator()
    assert s._clean_tags("budget,,planning") == ["budget", "planning"]


# --- _chunk_transcript ---

def test_chunk_transcript_short_returns_single_chunk():
    s = _make_orchestrator()
    transcript = "hello world"
    assert s._chunk_transcript(transcript) == [transcript]


def test_chunk_transcript_exactly_chunk_size_returns_single_chunk():
    from tinysteno.orchestrator import _CHUNK_SIZE_CHARS
    s = _make_orchestrator()
    transcript = "x" * _CHUNK_SIZE_CHARS
    assert s._chunk_transcript(transcript) == [transcript]


def test_chunk_transcript_long_creates_multiple_chunks():
    from tinysteno.orchestrator import _CHUNK_SIZE_CHARS
    s = _make_orchestrator()
    transcript = "x" * (_CHUNK_SIZE_CHARS * 2 + 1)
    chunks = s._chunk_transcript(transcript)
    assert len(chunks) > 1


def test_chunk_transcript_all_content_covered():
    """Every character of the transcript appears in at least one chunk."""
    from tinysteno.orchestrator import _CHUNK_SIZE_CHARS, _CHUNK_OVERLAP_CHARS
    s = _make_orchestrator()
    # Build a transcript with unique characters at known positions
    transcript = "".join(str(i % 10) for i in range(_CHUNK_SIZE_CHARS * 3))
    chunks = s._chunk_transcript(transcript)
    # Reconstruct: non-overlapping portions should cover the whole transcript
    reconstructed = chunks[0]
    for chunk in chunks[1:]:
        reconstructed += chunk[_CHUNK_OVERLAP_CHARS:]
    assert reconstructed == transcript


def test_chunk_transcript_chunks_have_overlap():
    from tinysteno.orchestrator import _CHUNK_SIZE_CHARS, _CHUNK_OVERLAP_CHARS
    s = _make_orchestrator()
    transcript = "x" * (_CHUNK_SIZE_CHARS + _CHUNK_OVERLAP_CHARS + 100)
    chunks = s._chunk_transcript(transcript)
    assert len(chunks) == 2
    # The tail of chunk 0 should match the head of chunk 1
    assert chunks[0][-_CHUNK_OVERLAP_CHARS:] == chunks[1][:_CHUNK_OVERLAP_CHARS]


# --- _build_merge_message ---

def test_build_merge_message_includes_dedup_instruction():
    persona = _make_persona({"overview": {"type": "string", "description": "summary"}})
    s = _make_orchestrator()
    partials = [{"overview": "Part one"}, {"overview": "Part two"}]
    msg = s._build_merge_message(partials, persona)
    assert "duplicate" in msg.lower()
    assert "Part one" in msg
    assert "Part two" in msg


def test_build_merge_message_includes_schema_fields():
    persona = _make_persona({
        "overview": {"type": "string", "description": "summary"},
        "points": {"type": "list", "description": "key points"},
    })
    s = _make_orchestrator()
    msg = s._build_merge_message([{}], persona)
    assert '"overview"' in msg
    assert '"points"' in msg


# --- summarize with chunking ---

def _llm_response(content: dict) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(content)))])


def test_summarize_single_chunk_makes_one_llm_call():
    persona = _make_persona({"overview": {"type": "string", "description": "x"}})
    s = _make_orchestrator()
    s._client.chat.completions.create.return_value = _llm_response({"overview": "hi"})
    s.summarize("short transcript", persona)
    assert s._client.chat.completions.create.call_count == 1


def test_summarize_long_transcript_makes_chunk_plus_merge_calls():
    from tinysteno.orchestrator import _CHUNK_SIZE_CHARS
    persona = _make_persona({"overview": {"type": "string", "description": "x"}})
    s = _make_orchestrator()
    # Two chunks → 2 extract calls + 1 merge call = 3 total
    long_transcript = "a" * (_CHUNK_SIZE_CHARS + 1000)
    s._client.chat.completions.create.return_value = _llm_response({"overview": "merged"})
    result = s.summarize(long_transcript, persona)
    assert s._client.chat.completions.create.call_count == 3
    assert result["overview"] == "merged"


def test_summarize_merge_call_receives_partial_results():
    from tinysteno.orchestrator import _CHUNK_SIZE_CHARS
    persona = _make_persona({"overview": {"type": "string", "description": "x"}})
    s = _make_orchestrator()
    long_transcript = "b" * (_CHUNK_SIZE_CHARS + 1000)

    chunk_response = _llm_response({"overview": "chunk summary"})
    merge_response = _llm_response({"overview": "final merged"})
    s._client.chat.completions.create.side_effect = [
        chunk_response, chunk_response, merge_response
    ]

    result = s.summarize(long_transcript, persona)
    # Third call is the merge; its user message should contain partial results
    merge_call_messages = s._client.chat.completions.create.call_args_list[2][1]["messages"]
    user_msg = next(m["content"] for m in merge_call_messages if m["role"] == "user")
    assert "chunk summary" in user_msg
    assert result["overview"] == "final merged"


def test_summarize_all_chunks_fail_returns_defaults():
    from tinysteno.orchestrator import _CHUNK_SIZE_CHARS
    persona = _make_persona({
        "overview": {"type": "string", "description": "x"},
        "items": {"type": "list", "description": "x"},
    })
    s = _make_orchestrator()
    long_transcript = "c" * (_CHUNK_SIZE_CHARS + 1000)
    s._client.chat.completions.create.side_effect = Exception("API down")
    result = s.summarize(long_transcript, persona)
    assert result == {"overview": "", "items": []}
