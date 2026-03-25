"""Whisper transcription module for TinySteno."""

from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample as scipy_resample


class WhisperTranscriber:
    """Transcribe audio files using whisper."""

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str, diarize: bool = False) -> dict:
        """Transcribe an audio file and return results.

        Returns:
            dict with keys: text, diarised_text, duration_seconds, detected_language
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Read original file before converting — conversion drops to mono,
        # so we must check for stereo here to support diarization.
        data, sr = sf.read(str(path))
        is_stereo = data.ndim == 2 and data.shape[1] >= 2

        audio_16k = self._convert_to_16khz_array(data, sr)
        text, language = self._run_whisper(audio_16k)
        duration = len(audio_16k) / 16000.0

        diarised_text = None
        if diarize and is_stereo:
            diarised_text = self._diarize(data, sr)

        return {
            "text": text.strip(),
            "diarised_text": diarised_text,
            "duration_seconds": duration,
            "detected_language": language,
        }

    def _run_whisper(self, audio: np.ndarray) -> tuple[str, str]:
        """Run Whisper on a 16kHz mono float32 numpy array."""
        segments, info = self._model.transcribe(audio, beam_size=5)
        text = "".join(segment.text for segment in segments)
        return text, info.language

    def _run_whisper_segments(self, audio: np.ndarray) -> list[tuple[float, str]]:
        """Run Whisper and return (start_seconds, text) tuples."""
        segments, _ = self._model.transcribe(audio, beam_size=5)
        return [(seg.start, seg.text.strip()) for seg in segments if seg.text.strip()]

    def _diarize(self, data: np.ndarray, sr: int) -> Optional[str]:
        """Split stereo channels and transcribe each independently."""
        if data.ndim != 2:
            return None

        left_16k = self._convert_to_16khz_array(data[:, 0].copy(), sr)
        right_16k = self._convert_to_16khz_array(data[:, 1].copy(), sr)

        left_segs = self._run_whisper_segments(left_16k)
        right_segs = self._run_whisper_segments(right_16k)

        tagged = (
            [("You", start, text) for start, text in left_segs] +
            [("Others", start, text) for start, text in right_segs]
        )
        tagged.sort(key=lambda x: x[1])
        return "\n".join(f"[{speaker}] {text}" for speaker, _, text in tagged)

    def _convert_to_16khz_array(
        self,
        data: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Convert audio data to 16kHz mono float32 numpy array.

        Uses scipy FFT-based resampling instead of np.interp to avoid
        large index array allocations and improve accuracy.
        """
        # Collapse to mono
        if data.ndim > 1:
            data = data[:, 0]

        data = data.astype(np.float32)

        if sr != 16000:
            num_samples = int(len(data) * 16000 / sr)
            data = scipy_resample(data, num_samples).astype(np.float32)

        return data
