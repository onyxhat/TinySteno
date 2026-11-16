"""Audio recording module for TinySteno."""

import os
import platform
import tempfile
import threading
import wave
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from tinysteno.levels import save_levels

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore  # pylint: disable=invalid-name


# A channel whose peak is below this is converted at unit gain, not amplified into noise.
NORMALISE_FLOOR = 0.003


def _to_int16(audio: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Convert float audio [n, channels] to int16, scaling each column to its own peak.

    A column whose peak is below ``NORMALISE_FLOOR`` (including all zeros) is converted at
    unit gain so a near-silent channel is not boosted into audible noise. Also returns each
    column's gain relative to full scale (1.0 for a column that was not scaled).
    """
    audio = np.asarray(audio, dtype=np.float32)
    out = np.empty(audio.shape, dtype=np.int16)
    gains = []
    for i in range(audio.shape[1]):
        column = audio[:, i]
        peak = float(np.max(np.abs(column))) if column.size else 0.0
        gains.append(1.0 / peak if peak >= NORMALISE_FLOOR else 1.0)
        out[:, i] = np.clip(column * 32767.0 * gains[-1], -32767, 32767).astype(np.int16)
    return out, gains


def _rms_level(frame) -> float:
    """RMS over all samples of a frame, clamped to 0.0..1.0 (0.0 for an empty frame)."""
    samples = np.asarray(frame, dtype=np.float32)
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return min(rms, 1.0) if np.isfinite(rms) else 0.0


class RecorderState(Enum):
    """Where a recorder is in its life. Every public method checks and moves this, nothing else.

    idle ──start()──▶ recording ──stop()──▶ idle          (WAV written, or nothing captured)
                          │    ──discard()─▶ idle          (audio thrown away)
                          └────stop() fails─▶ save_failed  (raw audio kept, see ``raw_paths``)
    save_failed ──discard()──▶ idle                        (the only way the kept audio goes)

    A start() that fails part-way undoes itself and leaves the recorder idle.
    """

    IDLE = "idle"
    RECORDING = "recording"
    SAVE_FAILED = "save_failed"


class AudioRecorder:
    """Record mic + system audio loopback to WAV files.

    On Windows and Linux the system audio loopback is captured via PortAudio
    (WASAPI loopback on Windows, PulseAudio/PipeWire monitor on Linux).
    On macOS, system audio is captured via ScreenCaptureKit
    (requires ``pyobjc-framework-ScreenCaptureKit`` and Screen Recording permission).

    When a loopback source is found, the output WAV is stereo:
      left  channel = microphone
      right channel = system audio
    This makes the recording compatible with the diarization feature
    (left = "You", right = "Others").
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        recordings_dir: Optional[Path] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recordings_dir = recordings_dir or Path.cwd() / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: list = []  # kept for is-empty check; never appended to
        self._mic_raw_path: Optional[Path] = None
        self._loopback_raw_path: Optional[Path] = None
        self._mic_fh = None
        self._loopback_fh = None
        self._mic_channels_written = 1
        self._loopback_channels_written = 1
        self._state = RecorderState.IDLE
        self._audio_interface = None
        self._loopback_interface = None
        self._macos_loopback = None
        self._active_channels = channels
        self._has_loopback = False
        self.output_path: Optional[Path] = None
        self._fh_lock = threading.Lock()
        self._mic_level = 0.0
        self._system_level = 0.0
        # Raw audio files left on disk because saving the WAV failed (empty otherwise).
        self.raw_paths: tuple[Path, ...] = ()
        # Why macOS system-audio capture could not start, when it was attempted and failed.
        self.loopback_error: Optional[str] = None

    # ── live levels (written by audio threads, read by the UI thread) ─────────

    @property
    def mic_level(self) -> float:
        """RMS of the most recent microphone frame, 0.0..1.0 (0.0 when not recording)."""
        return self._mic_level

    @property
    def system_level(self) -> float:
        """RMS of the most recent system-audio frame, 0.0..1.0 (0.0 when none)."""
        return self._system_level

    @property
    def has_loopback(self) -> bool:
        """Whether a system-audio source is being recorded."""
        return self._has_loopback

    # ── audio callbacks ───────────────────────────────────────────────────────

    def _audio_callback(self, indata, _frames, _time, status):
        if status:
            print(f"Audio callback status: {status}")
        self._write_mic_frame(indata)

    def _write_mic_frame(self, indata: np.ndarray) -> None:
        level = _rms_level(indata)
        with self._fh_lock:
            if self._mic_fh is not None:
                self._mic_level = level
                self._mic_fh.write(indata.astype(np.float32).tobytes())
                self._mic_channels_written = indata.shape[1] if indata.ndim > 1 else 1

    def _loopback_callback(self, indata, _frames, _time, status):
        if status:
            print(f"Loopback callback status: {status}")
        level = _rms_level(indata)
        with self._fh_lock:
            if self._loopback_fh is not None:
                self._system_level = level
                self._loopback_fh.write(indata.astype(np.float32).tobytes())
                self._loopback_channels_written = indata.shape[1] if indata.ndim > 1 else 1

    def _write_loopback_frame(self, indata) -> None:
        level = _rms_level(indata)
        with self._fh_lock:
            if self._loopback_fh is not None:
                self._system_level = level
                self._loopback_fh.write(np.array(indata, dtype=np.float32).tobytes())

    # ── device discovery ──────────────────────────────────────────────────────

    def _get_default_input_device(self) -> Optional[tuple[int, int]]:
        """Return (device_index, max_input_channels) for the default input device."""
        try:
            device = sd.query_devices(device=None, kind="input")
            max_ch = device.get("max_input_channels", 0)
            return (device["index"], max_ch) if max_ch > 0 else None
        except Exception:
            return None

    def _find_loopback_device(self) -> Optional[tuple[int, int]]:
        """Return (device_index, max_input_channels) for the system loopback device.

        Only relevant on Windows and Linux; returns None on macOS.
        """
        system = platform.system()
        if system == "Windows":
            return self._find_wasapi_loopback()
        if system == "Linux":
            return self._find_pulse_monitor()
        return None

    def _find_wasapi_loopback(self) -> Optional[tuple[int, int]]:
        """Find a WASAPI loopback device on Windows."""
        hostapis = sd.query_hostapis()
        wasapi_idx = next(
            (i for i, h in enumerate(hostapis) if "WASAPI" in h.get("name", "")),
            None,
        )
        if wasapi_idx is None:
            return None

        devices = sd.query_devices()

        # Prefer a device explicitly named "loopback"
        for i, dev in enumerate(devices):
            if (
                dev.get("hostapi") == wasapi_idx
                and dev.get("max_input_channels", 0) > 0
                and "loopback" in dev.get("name", "").lower()
            ):
                return (i, dev["max_input_channels"])

        # Fall back: use the default output device via WASAPI (PortAudio exposes it)
        try:
            default_out = sd.query_devices(kind="output")
            out_idx = default_out.get("index")
            if out_idx is not None:
                dev = sd.query_devices(out_idx)
                if dev.get("hostapi") == wasapi_idx:
                    return (out_idx, dev.get("max_output_channels", 2))
        except Exception:
            pass

        return None

    def _find_pulse_monitor(self) -> Optional[tuple[int, int]]:
        """Find a PulseAudio/PipeWire monitor source on Linux."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if (
                "monitor" in dev.get("name", "").lower()
                and dev.get("max_input_channels", 0) > 0
            ):
                return (i, dev["max_input_channels"])
        return None

    # ── start / stop ──────────────────────────────────────────────────────────

    def start(self, name: Optional[str] = None) -> str:
        """Start recording and return the output WAV path.

        If anything fails, whatever was opened is closed again and the recorder is left idle.
        """
        if self._state is RecorderState.RECORDING:
            raise RuntimeError("Recorder is already recording")
        if self._state is RecorderState.SAVE_FAILED:
            raise RuntimeError(
                "The previous recording failed to save; discard() it before starting another"
            )
        self.output_path = self._generate_output_path(name)
        self._buffer = []
        self._has_loopback = False
        self.raw_paths = ()
        self.loopback_error = None
        try:
            self._open_streams()
        except Exception as e:
            self._abandon_start()
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Failed to start audio recording: {e}") from e
        self._state = RecorderState.RECORDING
        return str(self.output_path)

    def _abandon_start(self) -> None:
        """Undo a start() that failed part-way: no stream left capturing, no raw file left."""
        try:
            self._stop_streams()
        except Exception:
            pass  # best effort: the start error is the one worth reporting
        finally:
            self._audio_interface = self._loopback_interface = self._macos_loopback = None
            self._cleanup_temp_files()
            self._reset_levels()

    def _open_streams(self) -> None:
        """Open the raw files, the microphone stream and, where available, system audio."""
        # ── microphone ────────────────────────────────────────────────────────
        result = self._get_default_input_device()
        if result is None:
            raise RuntimeError("No audio input devices found")

        device, max_channels = result
        self._active_channels = min(self.channels, max_channels)
        if self._active_channels != self.channels:
            print(
                f"Warning: requested {self.channels} channel(s) but device supports "
                f"{max_channels}; recording with {self._active_channels}."
            )

        # open temp raw files for streaming audio (after device check passes)
        fd, mic_path = tempfile.mkstemp(suffix=".f32")
        os.close(fd)
        self._mic_raw_path = Path(mic_path)
        self._mic_fh = open(mic_path, "wb")  # pylint: disable=consider-using-with

        fd, lb_path = tempfile.mkstemp(suffix=".f32")
        os.close(fd)
        self._loopback_raw_path = Path(lb_path)
        self._loopback_fh = open(lb_path, "wb")  # pylint: disable=consider-using-with

        try:
            self._audio_interface = sd.InputStream(
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                channels=self._active_channels,
                device=device,
            )
            self._audio_interface.start()
        except sd.PortAudioError as e:
            raise RuntimeError(f"Failed to start audio recording: {e}") from e

        # ── system audio loopback ─────────────────────────────────────────────
        system = platform.system()
        if system == "Darwin":
            self._has_loopback = self._start_macos_loopback()
        else:
            loopback = self._find_loopback_device()
            if loopback is not None:
                lb_device, lb_channels = loopback
                try:
                    self._loopback_interface = sd.InputStream(
                        callback=self._loopback_callback,
                        samplerate=self.sample_rate,
                        channels=min(self._active_channels, lb_channels),
                        device=lb_device,
                    )
                    self._loopback_interface.start()
                    self._has_loopback = True
                except sd.PortAudioError:
                    self._loopback_interface = None

    def _start_macos_loopback(self) -> bool:
        """Start macOS system audio capture via ScreenCaptureKit."""
        try:
            from tinysteno._macos_loopback import MacOSLoopback

            self._macos_loopback = MacOSLoopback(
                sample_rate=self.sample_rate,
                callback=self._write_loopback_frame,
            )
            self._macos_loopback.start()
            return True
        except ImportError:
            # pyobjc-framework-ScreenCaptureKit not installed
            return False
        except Exception as e:
            print(f"Warning: macOS system audio capture unavailable: {e}")
            self.loopback_error = str(e) or type(e).__name__
            return False

    def _stop_streams(self) -> Optional[int]:
        """Stop all active audio streams; return macOS loopback sample rate if applicable."""
        if self._audio_interface:
            self._audio_interface.stop()
            self._audio_interface.close()
            self._audio_interface = None

        if self._loopback_interface:
            self._loopback_interface.stop()
            self._loopback_interface.close()
            self._loopback_interface = None

        loopback_sr = None
        if self._macos_loopback:
            loopback_sr = self._macos_loopback.detected_sample_rate
            self._macos_loopback.stop()
            self._macos_loopback = None
        return loopback_sr

    def stop(self) -> bool:
        """Stop recording and save the WAV file; False when no audio was captured.

        The raw audio files are deleted only once the WAV has been written. If saving fails
        they are kept, listed in ``raw_paths``, and the RuntimeError names them.
        """
        if self._state is RecorderState.SAVE_FAILED:
            raise RuntimeError(f"Recording already failed to save. {self._kept_audio()}")
        if self._state is not RecorderState.RECORDING:
            return False
        try:
            loopback_sr = self._stop_streams()
            self._close_raw_files()
            raw = np.frombuffer(self._mic_raw_path.read_bytes(), dtype=np.float32)
            loopback_raw_bytes = b""
            if self._loopback_raw_path and self._loopback_raw_path.exists():
                loopback_raw_bytes = self._loopback_raw_path.read_bytes()
            # Nothing at all was captured. (System audio alone is still a recording.)
            if raw.size == 0 and not (self._has_loopback and loopback_raw_bytes):
                self._cleanup_temp_files()
                self._state = RecorderState.IDLE
                return False

            ch = self._mic_channels_written
            mic_data = raw.reshape(-1, ch) if ch > 1 else raw.reshape(-1, 1)

            audio_data, out_channels = self._build_audio_data(
                mic_data, loopback_raw_bytes, loopback_sr
            )

            audio_data, gains = _to_int16(audio_data)

            with wave.open(str(self.output_path), "w") as wav_file:
                wav_file.setnchannels(out_channels)  # pylint: disable=no-member
                wav_file.setsampwidth(2)  # pylint: disable=no-member
                wav_file.setframerate(self.sample_rate)  # pylint: disable=no-member
                wav_file.writeframes(audio_data.tobytes())  # pylint: disable=no-member
        except Exception as e:
            self._close_raw_files()
            self.raw_paths = tuple(
                p for p in (self._mic_raw_path, self._loopback_raw_path)
                if p is not None and p.exists()
            )
            self._state = RecorderState.SAVE_FAILED
            raise RuntimeError(f"Failed to save recording: {e}. {self._kept_audio()}") from e
        finally:
            self._reset_levels()

        save_levels(str(self.output_path), gains)  # never raises; the WAV is already saved
        self._cleanup_temp_files()
        self._state = RecorderState.IDLE
        return True

    def _kept_audio(self) -> str:
        if self.raw_paths:
            return f"Raw audio kept at {' and '.join(str(p) for p in self.raw_paths)}"
        return "No raw audio was kept"

    def discard(self) -> None:
        """Throw the audio away: no WAV is written and the raw files are deleted.

        This covers a recording in progress and one whose save failed (its kept raw audio).
        """
        if self._state is RecorderState.IDLE:
            return
        try:
            self._stop_streams()
        finally:
            self._state = RecorderState.IDLE
            self._cleanup_temp_files()
            self._reset_levels()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _reset_levels(self) -> None:
        self._mic_level = 0.0
        self._system_level = 0.0

    def _close_raw_files(self) -> None:
        """Close the raw audio file handles (the audio threads stop writing to them)."""
        with self._fh_lock:
            if self._mic_fh:
                self._mic_fh.close()
                self._mic_fh = None
            if self._loopback_fh:
                self._loopback_fh.close()
                self._loopback_fh = None

    def _cleanup_temp_files(self) -> None:
        """Close and delete temp raw audio files if they exist."""
        self._close_raw_files()
        self.raw_paths = ()
        if self._mic_raw_path and self._mic_raw_path.exists():
            self._mic_raw_path.unlink(missing_ok=True)
            self._mic_raw_path = None
        if self._loopback_raw_path and self._loopback_raw_path.exists():
            self._loopback_raw_path.unlink(missing_ok=True)
            self._loopback_raw_path = None

    def _build_audio_data(
        self,
        mic_data: np.ndarray,
        loopback_raw_bytes: bytes,
        loopback_sr: Optional[int],
    ) -> tuple[np.ndarray, int]:
        """Mix mic and loopback raw bytes into final audio array."""
        if not (self._has_loopback and loopback_raw_bytes):
            # No system audio: a 2-channel file would be read as mic + system, so write mono.
            if mic_data.ndim > 1 and mic_data.shape[1] > 1:
                return mic_data.mean(axis=1, keepdims=True), 1
            return mic_data, 1

        from scipy.signal import resample as scipy_resample  # pylint: disable=import-outside-toplevel

        lb_raw = np.frombuffer(loopback_raw_bytes, dtype=np.float32)
        lbch = self._loopback_channels_written
        lb_data = lb_raw.reshape(-1, lbch) if lbch > 1 else lb_raw.reshape(-1, 1)
        if loopback_sr and abs(loopback_sr - self.sample_rate) > 1:
            n_out = int(round(len(lb_data) * self.sample_rate / loopback_sr))
            lb_data = scipy_resample(lb_data.squeeze(), n_out).astype(np.float32).reshape(-1, 1)
        return self._mix_stereo(mic_data, lb_data), 2

    def _mix_stereo(self, mic: np.ndarray, loopback: np.ndarray) -> np.ndarray:
        """Combine mic and loopback into a stereo array [n, 2] (L=mic, R=loopback)."""
        # Reduce to mono
        mic_mono = mic[:, 0] if mic.ndim > 1 else mic
        lb_mono = loopback.mean(axis=1) if loopback.ndim > 1 else loopback

        # Pad the shorter stream with silence. Cutting to the shorter one would throw away
        # the rest of the meeting whenever one capture ends early (or never delivers).
        n = max(len(mic_mono), len(lb_mono))
        mic_mono = np.pad(mic_mono, (0, n - len(mic_mono)))
        lb_mono = np.pad(lb_mono, (0, n - len(lb_mono)))
        return np.column_stack([mic_mono, lb_mono])

    def _generate_output_path(self, name: Optional[str] = None) -> Path:
        timestamp = datetime.now()
        base_name = name.replace(" ", "-") if name else "Meeting"
        filename = f"{base_name}-{timestamp.strftime('%Y%m%d-%H%M%S')}.wav"
        return self.recordings_dir / filename
