"""Tests for ApiTranscriber."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_openai():
    """Patch OpenAI constructor and return (mock_class, mock_client)."""
    with patch("tinysteno.api_transcriber.OpenAI") as mock_class:
        client = MagicMock()
        mock_class.return_value = client
        yield mock_class, client


def test_init(mock_openai):
    """ApiTranscriber should store model and create an OpenAI client."""
    mock_class, _ = mock_openai
    from tinysteno.api_transcriber import ApiTranscriber

    t = ApiTranscriber(api_key="test-key", base_url="https://test.example.com/v1", model="whisper-1")

    assert t.model == "whisper-1"
    mock_class.assert_called_once_with(
        api_key="test-key", base_url="https://test.example.com/v1", timeout=180.0
    )


def test_transcribe_file_not_found(mock_openai):
    """ApiTranscriber.transcribe should raise FileNotFoundError for missing files."""
    from tinysteno.api_transcriber import ApiTranscriber

    t = ApiTranscriber(api_key="test-key")

    with pytest.raises(FileNotFoundError):
        t.transcribe("/nonexistent/file.wav")


def test_transcribe_calls_api(mock_openai, tmp_path):
    """ApiTranscriber.transcribe should call OpenAI audio.transcriptions.create."""
    _, client = mock_openai
    import soundfile as sf
    from tinysteno.api_transcriber import ApiTranscriber

    # Create a short audio file
    audio = np.zeros(16000, dtype=np.float32)
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
    import soundfile as sf
    from tinysteno.api_transcriber import ApiTranscriber

    audio = np.zeros(16000, dtype=np.float32)
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
    import soundfile as sf
    from tinysteno.api_transcriber import ApiTranscriber

    # Stereo audio: 2 seconds at 16kHz
    stereo = np.zeros((32000, 2), dtype=np.float32)
    stereo[:8000, 0] = 0.5  # Left channel has signal (You)
    stereo[8000:, 1] = 0.5  # Right channel has signal (Others)
    wav_path = tmp_path / "stereo.wav"
    sf.write(str(wav_path), stereo, 16000)

    # Return different text for left vs right calls
    mock_response_left = MagicMock()
    mock_response_left.text = "left speaker"
    mock_response_left.language = "en"
    mock_response_left.duration = 2.0

    mock_response_right = MagicMock()
    mock_response_right.text = "right speaker"
    mock_response_right.language = "en"
    mock_response_right.duration = 2.0

    client.audio.transcriptions.create.side_effect = [
        mock_response_left,
        mock_response_right,
    ]

    t = ApiTranscriber(api_key="test-key")
    result = t.transcribe(str(wav_path), diarize=True)

    assert "[You]" in result["diarised_text"]
    assert "[Others]" in result["diarised_text"]
    assert "left speaker" in result["diarised_text"]
    assert "right speaker" in result["diarised_text"]
    assert client.audio.transcriptions.create.call_count == 2


def test_progress_callback_called(mock_openai, tmp_path):
    """on_progress callback should be called at 0.0 and 1.0."""
    _, client = mock_openai
    import soundfile as sf
    from tinysteno.api_transcriber import ApiTranscriber

    audio = np.zeros(16000, dtype=np.float32)
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
    import soundfile as sf
    from tinysteno.api_transcriber import ApiTranscriber

    audio = np.zeros(16000, dtype=np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, 16000)

    class FakeResponse:
        text = "hello"
        duration = 1.0

    client.audio.transcriptions.create.return_value = FakeResponse()

    t = ApiTranscriber(api_key="test-key")
    result = t.transcribe(str(wav_path))
    assert result["detected_language"] == "en"