"""Tests for tinysteno.settings: the single description of every editable setting."""
import re

import pytest

from tinysteno.config import DEFAULT_CONFIG, validate_config
from tinysteno.settings import SECTIONS, SETTINGS, collect, from_text, setting, to_text


def by_key(key):
    return setting(key)


def test_every_default_key_is_editable():
    assert set(DEFAULT_CONFIG) <= {s.key for s in SETTINGS}


def test_keys_are_unique_and_sections_known():
    keys = [s.key for s in SETTINGS]
    assert len(keys) == len(set(keys))
    assert {s.section for s in SETTINGS} == set(SECTIONS)


@pytest.mark.parametrize("key,text,expected", [
    ("session_timeout_minutes", "90", 90), ("session_timeout_minutes", "7.5", 7.5),
    ("session_timeout_minutes", "0", 0), ("channels", "2", 2),
    ("llm_context_tokens", "8192", 8192), ("max_speakers", "", None),
    ("recordings_path", "  ", None), ("whisper_api_timeout", "600", 600),
    ("session_timeout_minutes", "75.0", 75), ("whisper_vad", "Off", False),
    ("whisper_model", "distil-large-v3", "distil-large-v3"),
])
def test_from_text(key, text, expected):
    assert from_text(by_key(key), text) == expected


@pytest.mark.parametrize("key,text", [
    ("session_timeout_minutes", "-1"), ("session_timeout_minutes", "soon"),
    ("session_timeout_minutes", "inf"), ("llm_context_tokens", "0"),
    ("llm_context_tokens", "8k"), ("max_speakers", "0"), ("sample_rate", "44.1"),
    ("obsidian_vault", ""), ("api_key", " "), ("whisper_api_timeout", "0"),
    ("channels", "3"), ("transcription_backend", "cloud"), ("auto_tags", "maybe"),
])
def test_from_text_rejects_and_names_the_field(key, text):
    setting_ = by_key(key)
    with pytest.raises(ValueError, match=re.escape(setting_.label)):
        from_text(setting_, text)


def test_round_trip_for_every_default():
    for s in SETTINGS:
        default = DEFAULT_CONFIG.get(s.key)
        if s.kind == "bool" or default is None:
            continue
        if s.optional and default == "":     # a blank optional reads back as None (key removed)
            assert from_text(s, to_text(s, default)) is None
        else:
            assert from_text(s, to_text(s, default)) == default


def test_whole_numbers_stay_ints():
    assert isinstance(from_text(by_key("session_timeout_minutes"), "75"), int)


def test_bool_round_trip():
    s = by_key("auto_title")
    assert to_text(s, True) == "true" and to_text(s, False) == "false"
    assert from_text(s, to_text(s, False)) is False


def test_collected_defaults_pass_validation():
    validate_config(collect({s.key: DEFAULT_CONFIG[s.key] for s in SETTINGS
                             if s.key in DEFAULT_CONFIG}))


def test_collect_mixes_text_and_typed_values_and_blank_optional_is_none():
    got = collect({"session_timeout_minutes": "30", "auto_title": False, "max_speakers": " "})
    assert got == {"session_timeout_minutes": 30, "auto_title": False, "max_speakers": None}


def test_collect_rejects_unknown_key_and_first_invalid_value():
    with pytest.raises(ValueError):
        collect({"nonsense": "1"})
    with pytest.raises(ValueError, match=re.escape(by_key("sample_rate").label)):
        collect({"sample_rate": "x", "channels": "9"})


def test_secret_errors_never_echo_the_text():
    with pytest.raises(ValueError) as info:
        from_text(by_key("api_key"), "   ")
    assert by_key("api_key").label in str(info.value)
