# Performance Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 identified performance, memory, and UX issues in TinySteno's audio recording and transcription pipeline.

**Architecture:** Changes are confined to four existing files — `recorder.py`, `transcriber.py`, `orchestrator.py`, `main.py`, and `_macos_loopback.py`. No new modules needed. Tasks are ordered so later tasks build on earlier ones (e.g. Task 2 depends on Task 3's resampling fix).

**Tech Stack:** Python 3.12, numpy, scipy (new dep), faster-whisper, soundfile, wave, concurrent.futures

---

## File Map

| File | Issues Addressed | What Changes |
|------|-----------------|--------------|
| `tinysteno/recorder.py` | 1, 2 | Buffer replaced with streaming to temp raw file |
| `tinysteno/_macos_loopback.py` | 8 | Spin loop interval 0.05 → 0.1 |
| `tinysteno/transcriber.py` | 3, 4, 5, 9, 10 | Numpy-direct Whisper, scipy resample, model cache, dead branch removal |
| `tinysteno/orchestrator.py` | 6 | Progress callback threading |
| `tinysteno/main.py` | 6, 7 | Parallel title/tags, progress display |
| `pyproject.toml` | 5 | Add scipy dependency |
| `tests/test_recorder.py` | — | New test file |
| `tests/test_transcriber.py` | — | New test file |

---

## Task 1: Stream Recording to Disk (Issues 1 & 2)

**Problem:** Mic and loopback audio are accumulated in `list[np.ndarray]` buffers that grow unbounded. On stop, `np.concatenate` over the full buffer causes a 2–3× memory spike.

**Fix:** Write float32 frames to temporary raw binary files during recording. On stop, read them back in one pass, mix, then write the final WAV. Memory usage becomes O(chunk) instead of O(recording length).

**Files:**
- Modify: `tinysteno/recorder.py`
- Create: `tests/test_recorder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_recorder.py
"""Tests for AudioRecorder streaming behavior."""
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from tinysteno.recorder import AudioRecorder


def _make_recorder(tmp_path) -> AudioRecorder:
    return AudioRecorder(sample_rate=44100, channels=1, recordings_dir=tmp_path)


def test_buffer_not_used_during_recording(tmp_path):
    """After start(), _buffer should remain empty (data goes to file)."""
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        mock_sd.query_devices.return_value = {"index": 0, "max_input_channels": 1}
        mock_sd.InputStream.return_value.__enter__ = lambda s: s
        mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)
        mock_sd.InputStream.return_value.start = MagicMock()
        rec.start()
    assert rec._buffer == [], "Buffer should be empty; data streams to file"


def test_stop_produces_valid_wav(tmp_path):
    """stop() should write a readable WAV file with correct properties."""
    rec = _make_recorder(tmp_path)
    # Simulate 1 second of audio at 44100 Hz mono
    fake_chunk = np.zeros((1024, 1), dtype=np.float32)

    with patch("tinysteno.recorder.sd") as mock_sd:
        mock_sd.query_devices.return_value = {"index": 0, "max_input_channels": 1}
        mock_sd.InputStream.return_value.start = MagicMock()
        mock_sd.InputStream.return_value.stop = MagicMock()
        mock_sd.InputStream.return_value.close = MagicMock()
        rec.start()
        # Manually write some frames via the streaming writer
        for _ in range(10):
            rec._write_mic_frame(fake_chunk)
        rec.stop()

    assert rec.output_path is not None
    assert rec.output_path.exists()
    with wave.open(str(rec.output_path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 44100
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_recorder.py -v
```
Expected: `AttributeError: 'AudioRecorder' object has no attribute '_write_mic_frame'`

- [ ] **Step 3: Implement streaming write in `recorder.py`**

Replace the `__init__` buffer fields and the callback/stop methods:

```python
# In __init__, replace:
#   self._buffer: list[np.ndarray] = []
#   self._loopback_buffer: list[np.ndarray] = []
# With:
import tempfile, os

self._mic_raw_path: Optional[Path] = None
self._loopback_raw_path: Optional[Path] = None
self._mic_fh = None        # file handle for writing raw float32 mic frames
self._loopback_fh = None   # file handle for writing raw float32 loopback frames
self._mic_channels_written = 1
self._loopback_channels_written = 1
# Keep _buffer = [] for the is-empty check; never appended to
self._buffer: list = []
```

```python
# Replace _audio_callback:
def _audio_callback(self, indata, _frames, _time, status):
    if status:
        print(f"Audio callback status: {status}")
    self._write_mic_frame(indata)

def _write_mic_frame(self, indata: np.ndarray) -> None:
    if self._mic_fh is not None:
        self._mic_fh.write(indata.astype(np.float32).tobytes())
        self._mic_channels_written = indata.shape[1] if indata.ndim > 1 else 1

# Replace _loopback_callback:
def _loopback_callback(self, indata, _frames, _time, status):
    if status:
        print(f"Loopback callback status: {status}")
    if self._loopback_fh is not None:
        self._loopback_fh.write(indata.astype(np.float32).tobytes())
        self._loopback_channels_written = indata.shape[1] if indata.ndim > 1 else 1
```

```python
# In start(), after self._buffer = []:
fd, mic_path = tempfile.mkstemp(suffix=".f32")
os.close(fd)
self._mic_raw_path = Path(mic_path)
self._mic_fh = open(mic_path, "wb")

fd, lb_path = tempfile.mkstemp(suffix=".f32")
os.close(fd)
self._loopback_raw_path = Path(lb_path)
self._loopback_fh = open(lb_path, "wb")
```

```python
# Replace the buffer-reading section in stop() (lines 232-244):
if self._mic_fh:
    self._mic_fh.close()
    self._mic_fh = None
if self._loopback_fh:
    self._loopback_fh.close()
    self._loopback_fh = None

try:
    raw = np.frombuffer(self._mic_raw_path.read_bytes(), dtype=np.float32)
except Exception:
    return False
finally:
    if self._mic_raw_path and self._mic_raw_path.exists():
        self._mic_raw_path.unlink(missing_ok=True)

if raw.size == 0:
    return False

ch = self._mic_channels_written
mic_data = raw.reshape(-1, ch) if ch > 1 else raw.reshape(-1, 1)

loopback_raw_bytes = b""
if self._loopback_raw_path and self._loopback_raw_path.exists():
    loopback_raw_bytes = self._loopback_raw_path.read_bytes()
    self._loopback_raw_path.unlink(missing_ok=True)

if self._has_loopback and loopback_raw_bytes:
    lb_raw = np.frombuffer(loopback_raw_bytes, dtype=np.float32)
    lbch = self._loopback_channels_written
    lb_data = lb_raw.reshape(-1, lbch) if lbch > 1 else lb_raw.reshape(-1, 1)
    if loopback_sr and abs(loopback_sr - self.sample_rate) > 1:
        n_out = int(round(len(lb_data) * self.sample_rate / loopback_sr))
        x = np.linspace(0, len(lb_data) - 1, n_out)
        lb_data = np.interp(x, np.arange(len(lb_data)), lb_data.squeeze()).reshape(-1, 1)
    audio_data = self._mix_stereo(mic_data, lb_data)
    out_channels = 2
else:
    audio_data = mic_data
    out_channels = ch
```

Also remove `self._loopback_buffer` references from `_start_macos_loopback` — the macOS loopback callback is already `self._loopback_buffer.append`; change that to:
```python
callback=lambda indata: self._loopback_fh and self._loopback_fh.write(
    np.array(indata, dtype=np.float32).tobytes()
),
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_recorder.py -v
```
Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add tinysteno/recorder.py tests/test_recorder.py
git commit -m "perf: stream recording to disk instead of buffering in memory"
```

---

## Task 2: Reduce macOS Spin Loop CPU Usage (Issue 8)

**Problem:** The NSRunLoop keep-alive loop in `_macos_loopback.py` spins every 50ms (20 Hz) while recording, burning CPU for no benefit. ScreenCaptureKit delivers audio on its own schedule; the run loop just needs to be alive.

**Files:**
- Modify: `tinysteno/_macos_loopback.py:274-275, 335-336`

- [ ] **Step 1: Change both spin intervals from 0.05 to 0.1**

In `_run_loop()`, there are two `runUntilDate_` calls. Both use `0.05`:

```python
# Line ~275 (startup wait loop):
# BEFORE:
run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
# AFTER:
run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))

# Line ~336 (keep-alive loop):
# BEFORE:
run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
# AFTER:
run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
```

- [ ] **Step 2: Run full test suite** (macOS loopback has no unit tests; manual smoke test if on macOS)

```bash
uv run pytest -v
```
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tinysteno/_macos_loopback.py
git commit -m "perf: halve macOS ScreenCaptureKit spin loop frequency (50ms -> 100ms)"
```

---

## Task 3: Add scipy and Replace np.interp Resampling (Issue 5)

**Problem:** `np.interp(x, np.arange(len(data)), data)` allocates a full integer index array (`np.arange`) for every resample call. For a 10-minute 44.1 kHz recording this is ~20MB of throwaway allocation. `scipy.signal.resample` uses FFT-based resampling: more accurate, no redundant allocation.

**Files:**
- Modify: `pyproject.toml`
- Modify: `tinysteno/transcriber.py:78-80, 92-98`
- Modify: `tinysteno/recorder.py:241-243` (loopback resample on stop)
- Create: `tests/test_transcriber.py` (initial skeleton; more tests added in Task 4)

- [ ] **Step 1: Add scipy to pyproject.toml**

```toml
# In [project] dependencies, add:
"scipy>=1.13",
```

- [ ] **Step 2: Install**

```bash
uv sync
```

- [ ] **Step 3: Write failing test for resampling**

```python
# tests/test_transcriber.py
"""Tests for WhisperTranscriber."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_transcriber(model_size="tiny"):
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_wm.return_value = MagicMock()
        from tinysteno.transcriber import WhisperTranscriber
        t = WhisperTranscriber(model_size=model_size)
    return t


def test_resample_uses_scipy_not_np_interp():
    """_convert_to_16khz should resample without calling np.interp."""
    import numpy as np
    from unittest.mock import patch, MagicMock
    import tinysteno.transcriber as mod

    t = _make_transcriber()
    data_44k = np.random.rand(44100).astype(np.float32)  # 1 second at 44.1kHz

    with patch("tinysteno.transcriber.np.interp") as mock_interp:
        result = t._convert_to_16khz_array(data_44k, sr=44100)
        mock_interp.assert_not_called()

    assert result.shape[0] == 16000
    assert result.dtype == np.float32


def test_resample_passthrough_at_16khz():
    """No resampling performed when input is already 16kHz."""
    t = _make_transcriber()
    data = np.ones(16000, dtype=np.float32)
    result = t._convert_to_16khz_array(data, sr=16000)
    np.testing.assert_array_equal(result, data)
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
uv run pytest tests/test_transcriber.py::test_resample_uses_scipy_not_np_interp -v
```
Expected: `AttributeError: 'WhisperTranscriber' object has no attribute '_convert_to_16khz_array'`

- [ ] **Step 5: Implement `_convert_to_16khz_array` in `transcriber.py`**

Add import at top of file:
```python
from scipy.signal import resample as scipy_resample
```

Add the new method (to replace the file-path version in Task 4):
```python
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
```

Also update the loopback resample in `recorder.py` stop() method:
```python
# Replace the np.interp block in stop():
# BEFORE:
n_out = int(round(len(lb_data) * self.sample_rate / loopback_sr))
x = np.linspace(0, len(lb_data) - 1, n_out)
lb_data = np.interp(x, np.arange(len(lb_data)), lb_data.squeeze()).reshape(-1, 1)
# AFTER:
from scipy.signal import resample as scipy_resample
n_out = int(round(len(lb_data) * self.sample_rate / loopback_sr))
lb_data = scipy_resample(lb_data.squeeze(), n_out).astype(np.float32).reshape(-1, 1)
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_transcriber.py -v
```
Expected: PASS

- [ ] **Step 7: Run full test suite**

```bash
uv run pytest -v
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock tinysteno/transcriber.py tinysteno/recorder.py tests/test_transcriber.py
git commit -m "perf: replace np.interp resampling with scipy FFT-based resample"
```

---

## Task 4: Pass Numpy Arrays Directly to Whisper (Issues 3, 4, 9)

**Problem:** Every transcription writes a `.16khz.wav` temp file to disk then passes the path to Whisper. Diarization writes two more (left/right channels), totalling 3 unnecessary disk write+read cycles per transcription. `faster_whisper` accepts `np.ndarray` directly.

**Files:**
- Modify: `tinysteno/transcriber.py`
- Modify: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_transcriber.py`:

```python
def test_transcribe_writes_no_temp_files(tmp_path):
    """transcribe() should not write any .wav temp files."""
    import soundfile as sf
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from tinysteno.transcriber import WhisperTranscriber

    # Write a short test WAV
    audio = np.zeros(16000, dtype=np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, 44100)

    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), MagicMock(language="en"))
        mock_wm.return_value = mock_model
        from tinysteno.transcriber import WhisperTranscriber
        t = WhisperTranscriber()
        t.transcribe(str(wav_path))

    # No temp files should remain
    assert not list(tmp_path.glob("*.16khz.wav")), "Temp 16khz file should not exist"
    assert not list(tmp_path.glob("*_left.wav")), "Left channel temp file should not exist"
    assert not list(tmp_path.glob("*_right.wav")), "Right channel temp file should not exist"


def test_run_whisper_accepts_array():
    """_run_whisper should accept numpy array, not require a file path."""
    from unittest.mock import MagicMock, patch
    from tinysteno.transcriber import WhisperTranscriber
    import numpy as np

    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_model = MagicMock()
        seg = MagicMock()
        seg.text = " hello"
        mock_model.transcribe.return_value = (iter([seg]), MagicMock(language="en"))
        mock_wm.return_value = mock_model
        t = WhisperTranscriber()

    audio = np.zeros(16000, dtype=np.float32)
    text, lang = t._run_whisper(audio)
    assert text == " hello"
    assert lang == "en"
    # Verify transcribe was called with ndarray, not a string
    call_args = mock_model.transcribe.call_args
    assert isinstance(call_args[0][0], np.ndarray)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_transcriber.py::test_run_whisper_accepts_array -v
```
Expected: FAIL — `_run_whisper` currently passes a string path

- [ ] **Step 3: Refactor `transcriber.py` to use numpy arrays throughout**

Change `_run_whisper` signature and body:
```python
def _run_whisper(self, audio: np.ndarray) -> tuple[str, str]:
    """Run Whisper on a 16kHz mono float32 numpy array."""
    segments, info = self._model.transcribe(audio, beam_size=5)
    text = "".join(segment.text for segment in segments)
    return text, info.language
```

Change `_run_whisper_segments`:
```python
def _run_whisper_segments(self, audio: np.ndarray) -> list[tuple[float, str]]:
    """Run Whisper and return (start_seconds, text) tuples."""
    segments, _ = self._model.transcribe(audio, beam_size=5)
    return [(seg.start, seg.text.strip()) for seg in segments if seg.text.strip()]
```

Remove `_convert_to_16khz` (file-based) entirely. The `_convert_to_16khz_array` from Task 3 replaces it.

Update `transcribe()`:
```python
def transcribe(self, audio_path: str, diarize: bool = False) -> dict:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

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
```

Remove `_calculate_duration` (duration is now `len(audio_16k) / 16000.0`).

Update `_diarize` to remove temp file writes:
```python
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
```

Update call site in `transcribe()` — `_diarize(path, data, sr)` → `_diarize(data, sr)`.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_transcriber.py -v
```
Expected: All PASS

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```

- [ ] **Step 6: Commit**

```bash
git add tinysteno/transcriber.py tests/test_transcriber.py
git commit -m "perf: pass numpy arrays directly to Whisper, remove all temp file writes"
```

---

## Task 5: Cache WhisperModel (Issue 10)

**Problem:** `WhisperTranscriber.__init__` creates a new `WhisperModel` every time it's instantiated. Model loading is 100–500MB and takes several seconds. If batch processing is added, this would reload per file.

**Fix:** Module-level dict cache keyed by `(model_size, device, compute_type)`.

**Files:**
- Modify: `tinysteno/transcriber.py`
- Modify: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_transcriber.py`:

```python
def test_whisper_model_cached_across_instances():
    """Two WhisperTranscriber instances with the same model_size share one WhisperModel."""
    from unittest.mock import patch, MagicMock, call
    import tinysteno.transcriber as mod

    # Clear cache before test
    mod._MODEL_CACHE.clear()

    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_wm.return_value = MagicMock()
        t1 = mod.WhisperTranscriber(model_size="tiny")
        t2 = mod.WhisperTranscriber(model_size="tiny")

    # WhisperModel() should have been constructed exactly once
    assert mock_wm.call_count == 1
    assert t1._model is t2._model
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_transcriber.py::test_whisper_model_cached_across_instances -v
```
Expected: FAIL — `AttributeError: module 'tinysteno.transcriber' has no attribute '_MODEL_CACHE'`

- [ ] **Step 3: Add module-level cache to `transcriber.py`**

After the imports, add:
```python
# Module-level cache: (model_size, device, compute_type) -> WhisperModel
_MODEL_CACHE: dict[tuple[str, str, str], "WhisperModel"] = {}
```

Update `__init__`:
```python
def __init__(self, model_size: str = "small"):
    self.model_size = model_size
    cache_key = (model_size, "cpu", "int8")
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = WhisperModel(model_size, device="cpu", compute_type="int8")
    self._model = _MODEL_CACHE[cache_key]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_transcriber.py -v
```
Expected: All PASS

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```

- [ ] **Step 6: Commit**

```bash
git add tinysteno/transcriber.py tests/test_transcriber.py
git commit -m "perf: cache WhisperModel at module level to avoid reload across instances"
```

---

## Task 6: Add Progress Feedback (Issue 6)

**Problem:** Transcription and diarization run silently for minutes. Users see "Transcribing..." then nothing until completion.

**Fix:** faster-whisper's `transcribe` returns a generator. Consume it segment-by-segment, calling an optional `on_progress(ratio: float)` callback. In `main.py`, display a Rich live progress bar.

**Files:**
- Modify: `tinysteno/transcriber.py`
- Modify: `tinysteno/main.py`
- Modify: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_transcriber.py`:

```python
def test_transcribe_calls_progress_callback(tmp_path):
    """on_progress callback should be called with values between 0.0 and 1.0."""
    import soundfile as sf
    from unittest.mock import patch, MagicMock
    import tinysteno.transcriber as mod

    audio = np.zeros(32000, dtype=np.float32)  # 2 seconds at 16kHz
    wav_path = tmp_path / "t.wav"
    sf.write(str(wav_path), audio, 16000)

    # Build two fake segments at t=0.5 and t=1.5 (out of 2s total)
    seg1 = MagicMock()
    seg1.text = " hello"
    seg1.start = 0.5

    seg2 = MagicMock()
    seg2.text = " world"
    seg2.start = 1.5

    progress_values = []

    mod._MODEL_CACHE.clear()
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg1, seg2]), MagicMock(language="en"))
        mock_wm.return_value = mock_model
        t = mod.WhisperTranscriber()

    t.transcribe(str(wav_path), on_progress=lambda r: progress_values.append(r))

    assert len(progress_values) >= 1
    assert all(0.0 <= v <= 1.0 for v in progress_values)
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_transcriber.py::test_transcribe_calls_progress_callback -v
```
Expected: FAIL — `transcribe()` doesn't accept `on_progress`

- [ ] **Step 3: Add `on_progress` to `transcriber.py`**

Update `transcribe` signature:
```python
def transcribe(
    self,
    audio_path: str,
    diarize: bool = False,
    on_progress: Optional[Callable[[float], None]] = None,
) -> dict:
```

Add `Callable` to the imports:
```python
from typing import Optional, Callable
```

Update `_run_whisper` to accept and forward the callback:
```python
def _run_whisper(
    self,
    audio: np.ndarray,
    on_progress: Optional[Callable[[float], None]] = None,
) -> tuple[str, str]:
    """Run Whisper on a 16kHz mono float32 numpy array."""
    duration = len(audio) / 16000.0
    segments, info = self._model.transcribe(audio, beam_size=5)
    parts = []
    for seg in segments:
        parts.append(seg.text)
        if on_progress is not None and duration > 0:
            on_progress(min(seg.start / duration, 1.0))
    if on_progress is not None:
        on_progress(1.0)
    return "".join(parts), info.language
```

Update the call site in `transcribe()`:
```python
text, language = self._run_whisper(audio_16k, on_progress=on_progress)
```

- [ ] **Step 4: Update `main.py` to display progress**

```python
# In _process_audio(), replace:
#   print("Transcribing...")
#   result = transcriber.transcribe(wav_path, diarize=config.get("diarization", False))
# With:
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

print("Transcribing...")
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.percentage:>3.0f}%"),
    transient=True,
) as progress:
    task = progress.add_task("Transcribing...", total=100)
    result = transcriber.transcribe(
        wav_path,
        diarize=config.get("diarization", False),
        on_progress=lambda r: progress.update(task, completed=int(r * 100)),
    )
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_transcriber.py -v
```
Expected: All PASS

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -v
```

- [ ] **Step 7: Commit**

```bash
git add tinysteno/transcriber.py tinysteno/main.py tests/test_transcriber.py
git commit -m "feat: add progress callback to transcription with Rich progress bar in CLI"
```

---

## Task 7: Parallelize Title and Tag Generation (Issue 7)

**Problem:** In `main.py`, `generate_title()` and `generate_tags()` are sequential HTTP calls to the LLM. Both use the same input (`first_string_value`) and are independent — they add 2–4 seconds in series.

**Files:**
- Modify: `tinysteno/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write failing test**

Check what test_main.py looks like first, then add:

```python
# Add to tests/test_main.py
def test_title_and_tags_generated_in_parallel():
    """generate_title and generate_tags should be called concurrently."""
    import threading
    from unittest.mock import MagicMock, patch
    from tinysteno.main import _process_audio
    from tinysteno.personas import Persona
    from pathlib import Path
    from datetime import datetime

    call_times = {}

    def fake_title(text):
        call_times["title_start"] = datetime.now()
        import time; time.sleep(0.05)
        call_times["title_end"] = datetime.now()
        return "Test Title"

    def fake_tags(text):
        call_times["tags_start"] = datetime.now()
        import time; time.sleep(0.05)
        call_times["tags_end"] = datetime.now()
        return ["test"]

    # ... setup mocks for transcriber, orchestrator, exporter ...
    # Assert that title_start and tags_start are within 20ms of each other
    # (i.e., they started concurrently, not sequentially)
    assert abs(
        (call_times["tags_start"] - call_times["title_start"]).total_seconds()
    ) < 0.02, "Title and tag generation should start at the same time"
```

> Note: The full mock setup for `_process_audio` is complex. Check `tests/test_main.py` for existing patterns and match them.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_main.py -k "parallel" -v
```
Expected: FAIL — calls are sequential

- [ ] **Step 3: Implement parallel generation in `main.py`**

```python
# In _process_audio(), replace the sequential title/tags block (lines ~149-161):
from concurrent.futures import ThreadPoolExecutor, as_completed

title_future = None
tags_future = None

with ThreadPoolExecutor(max_workers=2) as executor:
    if config.get("auto_title") and orchestrator and first_string_value:
        print("Generating title and tags...")
        title_future = executor.submit(orchestrator.generate_title, first_string_value)
        tags_future = executor.submit(orchestrator.generate_tags, first_string_value)

generated = title_future.result() if title_future else None
generated_tags = tags_future.result() if tags_future else []

# Resolve title
title = name
if not title:
    if generated:
        title = generated
    else:
        title = Path(wav_path).stem
```

Remove the separate `auto_tags` block — tags are now resolved above. Remove the old `print("Generating tags...")` and `print("Generating title...")` prints (replaced by single "Generating title and tags...").

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_main.py -v
```
Expected: All PASS

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```

- [ ] **Step 6: Commit**

```bash
git add tinysteno/main.py tests/test_main.py
git commit -m "perf: parallelize title and tag generation with ThreadPoolExecutor"
```

---

## Summary Table

| Task | Issues | Files Changed | Risk |
|------|--------|---------------|------|
| 1: Stream to disk | 1, 2 | `recorder.py` | Medium — audio callback path changes |
| 2: Spin loop | 8 | `_macos_loopback.py` | Low — cosmetic timing change |
| 3: scipy resample | 5 | `transcriber.py`, `recorder.py`, `pyproject.toml` | Low — new dep, same output |
| 4: Numpy to Whisper | 3, 4, 9 | `transcriber.py` | Medium — removes temp file paths |
| 5: Model cache | 10 | `transcriber.py` | Low — transparent singleton |
| 6: Progress | 6 | `transcriber.py`, `main.py` | Low — additive only |
| 7: Parallel LLM | 7 | `main.py` | Low — independent futures |

Execute in order — Task 4 depends on the `_convert_to_16khz_array` method introduced in Task 3.
