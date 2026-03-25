"""Whisper transcription module for TinySteno."""

import wave
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

        audio_16khz = self._convert_to_16khz(path, data, sr)
        try:
            text, language = self._run_whisper(audio_16khz)
            duration = self._calculate_duration(audio_16khz)
        finally:
            audio_16khz.unlink(missing_ok=True)

        diarised_text = None
        if diarize and is_stereo:
            diarised_text = self._diarize(path, data, sr)

        return {
            "text": text.strip(),
            "diarised_text": diarised_text,
            "duration_seconds": duration,
            "detected_language": language,
        }

    def _run_whisper(self, audio_path: Path) -> tuple[str, str]:
        """Run Whisper on a 16kHz mono WAV file."""
        segments, info = self._model.transcribe(str(audio_path), beam_size=5)
        text = "".join(segment.text for segment in segments)
        return text, info.language

    def _run_whisper_segments(
        self, audio_path: Path
    ) -> list[tuple[float, str]]:
        """Run Whisper and return (start_seconds, text) tuples."""
        segments, _ = self._model.transcribe(str(audio_path), beam_size=5)
        return [(seg.start, seg.text.strip()) for seg in segments if seg.text.strip()]

    def _convert_to_16khz(
        self,
        audio_path: Path,
        data: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> Path:
        """Convert audio to 16kHz mono WAV, returning the temp file path."""
        if data is None or sr is None:
            data, sr = sf.read(str(audio_path))

        if data.ndim > 1:
            data = data[:, 0]

        if sr != 16000:
            num_samples = int(len(data) * 16000 / sr)
            x = np.linspace(0, len(data) - 1, num_samples)
            data = np.interp(x, np.arange(len(data)), data)

        temp_path = audio_path.with_suffix(".16khz.wav")
        sf.write(str(temp_path), data, 16000)
        return temp_path

    def _diarize(self, audio_path: Path, data: np.ndarray, sr: int) -> Optional[str]:
        """Split stereo channels and label speakers."""
        if data.ndim != 2:
            return None

        # Resample both channels together so they share identical timestamps.
        if sr != 16000:
            num_samples = int(data.shape[0] * 16000 / sr)
            x = np.linspace(0, data.shape[0] - 1, num_samples)
            data = np.column_stack([
                np.interp(x, np.arange(data.shape[0]), data[:, 0]),
                np.interp(x, np.arange(data.shape[0]), data[:, 1]),
            ])

        left_path = audio_path.with_stem(audio_path.stem + "_left")
        right_path = audio_path.with_stem(audio_path.stem + "_right")
        sf.write(str(left_path), data[:, 0], 16000)
        sf.write(str(right_path), data[:, 1], 16000)

        try:
            left_segs = self._run_whisper_segments(left_path)
            right_segs = self._run_whisper_segments(right_path)
        finally:
            left_path.unlink(missing_ok=True)
            right_path.unlink(missing_ok=True)

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

    def _calculate_duration(self, audio_path: Path) -> float:
        """Calculate audio duration in seconds."""
        with wave.open(str(audio_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / rate
