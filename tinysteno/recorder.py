"""Audio recording module for TinySteno."""

import os
import platform
import tempfile
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore


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
        self._is_recording = False
        self._audio_interface = None
        self._loopback_interface = None
        self._macos_loopback = None
        self._active_channels = channels
        self._has_loopback = False
        self.output_path: Optional[Path] = None
        self._fh_lock = threading.Lock()

    # ── audio callbacks ───────────────────────────────────────────────────────

    def _audio_callback(self, indata, _frames, _time, status):
        if status:
            print(f"Audio callback status: {status}")
        self._write_mic_frame(indata)

    def _write_mic_frame(self, indata: np.ndarray) -> None:
        with self._fh_lock:
            if self._mic_fh is not None:
                self._mic_fh.write(indata.astype(np.float32).tobytes())
                self._mic_channels_written = indata.shape[1] if indata.ndim > 1 else 1

    def _loopback_callback(self, indata, _frames, _time, status):
        if status:
            print(f"Loopback callback status: {status}")
        with self._fh_lock:
            if self._loopback_fh is not None:
                self._loopback_fh.write(indata.astype(np.float32).tobytes())
                self._loopback_channels_written = indata.shape[1] if indata.ndim > 1 else 1

    def _write_loopback_frame(self, indata) -> None:
        with self._fh_lock:
            if self._loopback_fh is not None:
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
        """Start recording and return the output WAV path."""
        self.output_path = self._generate_output_path(name)
        self._buffer = []
        self._is_recording = True
        self._has_loopback = False

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
        self._mic_fh = open(mic_path, "wb")

        fd, lb_path = tempfile.mkstemp(suffix=".f32")
        os.close(fd)
        self._loopback_raw_path = Path(lb_path)
        self._loopback_fh = open(lb_path, "wb")

        try:
            self._audio_interface = sd.InputStream(
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                channels=self._active_channels,
                device=device,
            )
            self._audio_interface.start()
        except sd.PortAudioError as e:
            self._cleanup_temp_files()
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

        return str(self.output_path)

    def _start_macos_loopback(self) -> bool:
        """Start macOS system audio capture via ScreenCaptureKit."""
        try:
            from tinysteno._macos_loopback import MacOSLoopback

            self._macos_loopback = MacOSLoopback(
                sample_rate=self.sample_rate,
                callback=lambda indata: self._write_loopback_frame(indata),
            )
            self._macos_loopback.start()
            return True
        except ImportError:
            # pyobjc-framework-ScreenCaptureKit not installed
            return False
        except Exception as e:
            print(f"Warning: macOS system audio capture unavailable: {e}")
            return False

    def stop(self) -> bool:
        """Stop recording and save WAV file."""
        if not self._is_recording:
            return False

        try:
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

            # close file handles
            with self._fh_lock:
                if self._mic_fh:
                    self._mic_fh.close()
                    self._mic_fh = None
                if self._loopback_fh:
                    self._loopback_fh.close()
                    self._loopback_fh = None

            try:
                raw = np.frombuffer(self._mic_raw_path.read_bytes(), dtype=np.float32)
                loopback_raw_bytes = b""
                if self._loopback_raw_path and self._loopback_raw_path.exists():
                    loopback_raw_bytes = self._loopback_raw_path.read_bytes()
            except Exception:
                return False
            finally:
                if self._mic_raw_path and self._mic_raw_path.exists():
                    self._mic_raw_path.unlink(missing_ok=True)
                    self._mic_raw_path = None
                if self._loopback_raw_path and self._loopback_raw_path.exists():
                    self._loopback_raw_path.unlink(missing_ok=True)
                    self._loopback_raw_path = None

            if raw.size == 0:
                return False

            ch = self._mic_channels_written
            mic_data = raw.reshape(-1, ch) if ch > 1 else raw.reshape(-1, 1)

            if self._has_loopback and loopback_raw_bytes:
                lb_raw = np.frombuffer(loopback_raw_bytes, dtype=np.float32)
                lbch = self._loopback_channels_written
                lb_data = lb_raw.reshape(-1, lbch) if lbch > 1 else lb_raw.reshape(-1, 1)
                if loopback_sr and abs(loopback_sr - self.sample_rate) > 1:
                    from scipy.signal import resample as scipy_resample
                    n_out = int(round(len(lb_data) * self.sample_rate / loopback_sr))
                    lb_data = scipy_resample(lb_data.squeeze(), n_out).astype(np.float32).reshape(-1, 1)
                audio_data = self._mix_stereo(mic_data, lb_data)
                out_channels = 2
            else:
                audio_data = mic_data
                out_channels = ch

            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = np.int16(audio_data / max_val * 32767)
            else:
                audio_data = np.zeros_like(audio_data, dtype=np.int16)

            with wave.open(str(self.output_path), "w") as wav_file:
                wav_file.setnchannels(out_channels)  # pylint: disable=no-member
                wav_file.setsampwidth(2)  # pylint: disable=no-member
                wav_file.setframerate(self.sample_rate)  # pylint: disable=no-member
                wav_file.writeframes(audio_data.tobytes())  # pylint: disable=no-member

            self._is_recording = False
            return True
        except Exception as e:
            self._is_recording = False
            raise RuntimeError(f"Failed to save recording: {e}") from e

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cleanup_temp_files(self) -> None:
        """Close and delete temp raw audio files if they exist."""
        if self._mic_fh:
            self._mic_fh.close()
            self._mic_fh = None
        if self._loopback_fh:
            self._loopback_fh.close()
            self._loopback_fh = None
        if self._mic_raw_path and self._mic_raw_path.exists():
            self._mic_raw_path.unlink(missing_ok=True)
            self._mic_raw_path = None
        if self._loopback_raw_path and self._loopback_raw_path.exists():
            self._loopback_raw_path.unlink(missing_ok=True)
            self._loopback_raw_path = None

    def _mix_stereo(self, mic: np.ndarray, loopback: np.ndarray) -> np.ndarray:
        """Combine mic and loopback into a stereo array [n, 2] (L=mic, R=loopback)."""
        # Reduce to mono
        mic_mono = mic[:, 0] if mic.ndim > 1 else mic
        lb_mono = loopback.mean(axis=1) if loopback.ndim > 1 else loopback

        # Align lengths
        n = min(len(mic_mono), len(lb_mono))
        return np.column_stack([mic_mono[:n], lb_mono[:n]])

    def _generate_output_path(self, name: Optional[str] = None) -> Path:
        timestamp = datetime.now()
        base_name = name.replace(" ", "-") if name else "Meeting"
        filename = f"{base_name}-{timestamp.strftime('%Y%m%d-%H%M%S')}.wav"
        return self.recordings_dir / filename
