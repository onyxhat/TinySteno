"""Tests for ApiTranscriber."""
# pylint: disable=redefined-outer-name,unused-argument

from unittest.mock import MagicMock, patch

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tinysteno.api_transcriber import ApiTranscriber


@pytest.fixture
def mock_openai():
    """Patch OpenAI constructor and return (mock_class, mock_client)."""
    with patch("tinysteno.api_transcriber.OpenAI") as mock_class:
        client = MagicMock()
        mock_class.return_value = client
        yield mock_class, client


def _response(text="", segments=None, language="en"):
    """Build a response explicitly (MagicMock would claim every attribute exists)."""
    return SimpleNamespace(text=text, language=language, duration=1.0, segments=segments)


def _seg(start, text):
    return SimpleNamespace(start=start, end=start + 1.0, text=text)


def _write(tmp_path, stereo, silent=False, seconds=2):
    """Write a wav whose channels carry a tone (or silence)."""
    n = 16000 * seconds
    tone = np.full(n, 0.0 if silent else 0.5, dtype=np.float32)
    data = np.stack([tone, tone], axis=1) if stereo else tone
    path = tmp_path / "audio.wav"
    sf.write(str(path), data, 16000)
    return str(path)


def _run(  # pylint: disable=too-many-arguments
    client, tmp_path, left, right=None, *, diarize=True, on_progress=None
):
    """Transcribe a stereo (or mono if right is None) file; uploads return left, right."""
    client.audio.transcriptions.create.side_effect = [left] if right is None else [left, right]
    t = ApiTranscriber(api_key="k")
    return t.transcribe(
        _write(tmp_path, stereo=right is not None), diarize=diarize, on_progress=on_progress
    )


def test_init(mock_openai):
    """ApiTranscriber should store model and create an OpenAI client."""
    mock_class, _ = mock_openai

    t = ApiTranscriber(
        api_key="test-key", base_url="https://test.example.com/v1", model="whisper-1"
    )

    assert t.model == "whisper-1"
    mock_class.assert_called_once_with(
        api_key="test-key", base_url="https://test.example.com/v1", timeout=180.0
    )


def test_transcribe_file_not_found(mock_openai):
    """ApiTranscriber.transcribe should raise FileNotFoundError for missing files."""

    t = ApiTranscriber(api_key="test-key")

    with pytest.raises(FileNotFoundError):
        t.transcribe("/nonexistent/file.wav")


def test_transcribe_calls_api(mock_openai, tmp_path):
    """ApiTranscriber.transcribe should call OpenAI audio.transcriptions.create."""
    _, client = mock_openai

    # Create a short audio file
    audio = np.full(16000, 0.5, dtype=np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, 16000)

    # Mock the API response
    mock_response = MagicMock()
    mock_response.text = "hello world"
    mock_response.language = "en"
    mock_response.duration = 1.0
    client.audio.transcriptions.create.return_value = mock_response

    t = ApiTranscriber(api_key="test-key")
    result = t.transcribe(str(wav_path))

    assert result["text"] == "hello world"
    assert result["detected_language"] == "en"
    assert result["duration_seconds"] == 1.0
    client.audio.transcriptions.create.assert_called_once()


def test_transcribe_mono_no_diarize(mock_openai, tmp_path):
    """Mono file should be uploaded as-is, no diarization."""
    _, client = mock_openai

    audio = np.full(16000, 0.5, dtype=np.float32)
    wav_path = tmp_path / "mono.wav"
    sf.write(str(wav_path), audio, 16000)

    mock_response = MagicMock()
    mock_response.text = "transcript"
    mock_response.language = "en"
    mock_response.duration = 1.0
    client.audio.transcriptions.create.return_value = mock_response

    t = ApiTranscriber(api_key="test-key")
    result = t.transcribe(str(wav_path), diarize=False)

    assert result["text"] == "transcript"
    # Single API call
    assert client.audio.transcriptions.create.call_count == 1


def test_transcribe_stereo_diarize(mock_openai, tmp_path):
    """Stereo file with diarize=True should make two API calls (one per channel)."""
    _, client = mock_openai

    # Stereo audio: 2 seconds at 16kHz
    stereo = np.zeros((32000, 2), dtype=np.float32)
    stereo[:8000, 0] = 0.5  # Left channel has signal (You)
    stereo[8000:, 1] = 0.5  # Right channel has signal (Others)
    wav_path = tmp_path / "stereo.wav"
    sf.write(str(wav_path), stereo, 16000)

    # Return different text for left vs right calls
    mock_response_left = _response("left speaker", [_seg(0.0, "left speaker")])
    mock_response_right = _response("right speaker", [_seg(1.0, "right speaker")])

    client.audio.transcriptions.create.side_effect = [
        mock_response_left,
        mock_response_right,
    ]

    t = ApiTranscriber(api_key="test-key")
    result = t.transcribe(str(wav_path), diarize=True)

    assert result["diarised_text"] == "[You] left speaker\n[Others] right speaker"
    assert client.audio.transcriptions.create.call_count == 2


def test_progress_callback_called(mock_openai, tmp_path):
    """on_progress callback should be called at 0.0 and 1.0."""
    _, client = mock_openai

    audio = np.full(16000, 0.5, dtype=np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, 16000)

    mock_response = MagicMock()
    mock_response.text = "test"
    mock_response.language = "en"
    mock_response.duration = 1.0
    client.audio.transcriptions.create.return_value = mock_response

    progress_values = []

    t = ApiTranscriber(api_key="test-key")
    t.transcribe(str(wav_path), on_progress=progress_values.append)

    assert len(progress_values) == 2
    assert progress_values[0] == 0.0
    assert progress_values[1] == 1.0


def test_language_fallback(mock_openai, tmp_path):
    """When API response lacks language, default to 'en'."""
    _, client = mock_openai

    audio = np.full(16000, 0.5, dtype=np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, 16000)

    class FakeResponse:
        """Minimal response-like object without language attribute."""

        text = "hello"
        duration = 1.0

    client.audio.transcriptions.create.return_value = FakeResponse()

    t = ApiTranscriber(api_key="test-key")
    result = t.transcribe(str(wav_path))
    assert result["detected_language"] == "en"


def test_stereo_diarize_interleaves_segments_from_both_uploads(mock_openai, tmp_path):
    """Segments from both uploads are merged by start time."""
    _, client = mock_openai
    result = _run(
        client, tmp_path, _response("second", [_seg(5.0, "second")]),
        _response("first", [_seg(1.0, "first")]),
    )
    assert result["diarised_text"] == "[Others] first\n[You] second"


def test_stereo_without_diarize_uploads_both_channels(mock_openai, tmp_path):
    """System audio is transcribed even when diarization is off."""
    _, client = mock_openai
    result = _run(
        client, tmp_path, _response("me", [_seg(0.0, "me")]),
        _response("them", [_seg(1.0, "them")]), diarize=False,
    )
    assert result["text"] == "me them" and result["diarised_text"] is None
    assert client.audio.transcriptions.create.call_count == 2


def test_server_without_segments_falls_back_to_one_block_per_channel(mock_openai, tmp_path):
    """Text without segments (None, missing or empty) becomes one segment per channel."""
    _, client = mock_openai
    result = _run(
        client, tmp_path, _response("all of me", None), _response("all of them", []),
    )
    assert result["diarised_text"] == "[You] all of me\n[Others] all of them"
    assert result["segments"][0].end == 2.0


def test_response_without_segments_attribute_falls_back(mock_openai, tmp_path):
    """A response object lacking a segments attribute still yields its text."""
    _, client = mock_openai
    bare = SimpleNamespace(text="plain", language="de")
    result = _run(client, tmp_path, bare)
    assert result["text"] == "plain" and result["detected_language"] == "de"


def test_silent_stereo_recording_is_no_speech(mock_openai, tmp_path):
    """Both channels silent: nothing uploaded, no speech reported."""
    _, client = mock_openai
    t = ApiTranscriber(api_key="k")
    result = t.transcribe(_write(tmp_path, stereo=True, silent=True), diarize=True)
    assert result["text"] == "" and result["diarised_text"] is None
    client.audio.transcriptions.create.assert_not_called()


def test_silent_channel_is_not_uploaded(mock_openai, tmp_path):
    """Only the channel with signal is uploaded."""
    _, client = mock_openai
    stereo = np.zeros((32000, 2), dtype=np.float32)
    stereo[:, 1] = 0.5
    path = tmp_path / "half.wav"
    sf.write(str(path), stereo, 16000)
    client.audio.transcriptions.create.side_effect = [_response("them", [_seg(0.0, "them")])]
    result = ApiTranscriber(api_key="k").transcribe(str(path), diarize=True)
    assert client.audio.transcriptions.create.call_count == 1
    assert result["diarised_text"] == "[Others] them"


def test_segments_given_as_dicts_are_accepted(mock_openai, tmp_path):
    """Dict segments work, empty text is skipped and text is stripped."""
    _, client = mock_openai
    segments = [
        {"start": 0.0, "end": 1.0, "text": "  hi  "},
        {"start": 1.0, "end": 2.0, "text": "   "},
    ]
    result = _run(client, tmp_path, _response("hi", segments))
    assert [(s.start, s.text) for s in result["segments"]] == [(0.0, "hi")]


def test_language_prefers_microphone_then_other(mock_openai, tmp_path):
    """The mic channel's language wins; otherwise the other channel's."""
    _, client = mock_openai
    result = _run(
        client, tmp_path, _response("a", [_seg(0.0, "a")], "fr"),
        _response("b", [_seg(1.0, "b")], "de"),
    )
    assert result["detected_language"] == "fr"
    result = _run(
        client, tmp_path, _response("", None, "fr"), _response("b", [_seg(1.0, "b")], "de")
    )
    assert result["detected_language"] == "de"


def test_stereo_progress_is_0_half_1(mock_openai, tmp_path):
    """Two uploads report 0.0, 0.5, 1.0."""
    _, client = mock_openai
    seen = []
    _run(
        client, tmp_path, _response("a", [_seg(0.0, "a")]), _response("b", [_seg(1.0, "b")]),
        on_progress=seen.append,
    )
    assert seen == [0.0, 0.5, 1.0]


def test_request_has_only_model_file_and_format(mock_openai, tmp_path):
    """The upload request carries no non-standard fields."""
    _, client = mock_openai
    _run(client, tmp_path, _response("a", [_seg(0.0, "a")]))
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert set(kwargs) == {"model", "file", "response_format"}
    assert kwargs["response_format"] == "verbose_json"


def test_api_request_carries_prompt_only_when_set(mock_openai, tmp_path):
    """A vocabulary prompt is sent as the standard prompt field; none, no field."""
    _, client = mock_openai
    client.audio.transcriptions.create.return_value = _response("a", [_seg(0.0, "a")])
    ApiTranscriber(api_key="k", prompt="Kubernetes, Priya").transcribe(
        _write(tmp_path, stereo=False)
    )
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["prompt"] == "Kubernetes, Priya"

    client.audio.transcriptions.create.reset_mock()
    ApiTranscriber(api_key="k", prompt="").transcribe(_write(tmp_path, stereo=False))
    assert "prompt" not in client.audio.transcriptions.create.call_args.kwargs


def test_malformed_start_falls_back_to_one_segment(mock_openai, tmp_path):
    """A non-numeric start makes the channel untimed; its text is still kept."""
    _, client = mock_openai
    bad = [{"start": "00:01.5", "end": 2.0, "text": "hello"}]
    result = _run(client, tmp_path, _response("hello there", bad))
    assert result["text"] == "hello there"
    assert [(s.start, s.end) for s in result["segments"]] == [(0.0, 2.0)]


def test_malformed_start_with_empty_text_joins_segment_texts(mock_openai, tmp_path):
    """With no top-level text, the usable segment texts are joined."""
    _, client = mock_openai
    bad = [
        {"start": "n/a", "end": 1.0, "text": "one"},
        {"start": 1.0, "end": 2.0, "text": "two"},
    ]
    result = _run(client, tmp_path, _response("", bad))
    assert result["text"] == "one two"


def test_numeric_string_times_and_missing_end_are_parsed(mock_openai, tmp_path):
    """Numeric strings parse; an unreadable end uses start; end is never below start."""
    _, client = mock_openai
    segs = [
        {"start": "1.5", "end": None, "text": "a"},
        {"start": 3.0, "end": 2.0, "text": "b"},
        {"start": 4.0, "end": "later", "text": "c"},
    ]
    result = _run(client, tmp_path, _response("a b c", segs))
    got = [(s.start, s.end) for s in result["segments"]]
    assert got == [(1.5, 1.5), (3.0, 3.0), (4.0, 4.0)]


def test_one_channel_malformed_other_still_interleaved(mock_openai, tmp_path):
    """Malformed times on one channel do not affect the other channel's segments."""
    _, client = mock_openai
    bad = _response("mine", [{"start": "x", "end": 1.0, "text": "mine"}])
    good = _response("theirs", [_seg(1.0, "theirs"), _seg(9.0, "later")])
    result = _run(client, tmp_path, bad, good)
    assert result["diarised_text"] == "[You] mine\n[Others] theirs\n[Others] later"
