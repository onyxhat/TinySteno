"""Thin adapter around the optional ``sherpa-onnx`` speaker diarizer.

It fetches the two model files on first use, runs the offline diarizer on one channel of
16 kHz audio and returns plain :class:`~tinysteno.speaker_split.Turn` objects. Whether those
turns are trustworthy is decided elsewhere (``speaker_split.apply_policy``).

Guarantees: ``diarize`` and ``ensure_models`` raise only :class:`DiarizerUnavailable` (with the
original exception as ``__cause__``) for any ordinary ``Exception``; ``KeyboardInterrupt`` and
``SystemExit`` still propagate. Downloads are HTTPS-only, capped at ``MAX_DOWNLOAD_BYTES`` and
checked against pinned SHA-256 digests before use, and a failed download leaves no file behind.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from tinysteno.speaker_split import Turn, cluster_threshold

logger = logging.getLogger(__name__)

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)

# SHA-256 of the files exactly as downloaded from the URLs above (the archive, not its contents).
SEGMENTATION_SHA256 = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
EMBEDDING_SHA256 = "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024

_CHUNK_BYTES = 64 * 1024
_SAMPLE_RATE = 16000
_DOWNLOAD_TIMEOUT_SECONDS = 60
_ALLOWED_URL_PREFIX = "https://github.com/"
_SEGMENTATION_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
_SEGMENTATION_FILE = "model.onnx"
_DOWNLOAD_FAILED = "Could not download the speaker-separation models"
_MISSING_EXTRA = 'Speaker separation needs the optional extra: pip install "tinysteno[diarize]"'


class DiarizerUnavailable(RuntimeError):
    """Speaker separation cannot run; the message is meant to be shown to the user."""


def default_model_dir() -> Path:
    """Where the models live, resolved from the current home directory at call time."""
    return Path.home() / ".tinysteno" / "models" / "diarize"


def _reason(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _download(url: str, dest: Path) -> None:
    """Fetch ``url`` (GitHub over HTTPS only) into ``dest``, at most ``MAX_DOWNLOAD_BYTES``."""
    if not url.startswith(_ALLOWED_URL_PREFIX):
        raise ValueError(f"refusing to download from {url}")
    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        if not str(response.url).startswith("https://"):
            raise ValueError(f"refusing redirect to {response.url}")
        expected = response.headers.get("Content-Length")
        written = 0
        with open(dest, "wb") as out:
            while chunk := response.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    raise OSError("download is larger than expected")
                out.write(chunk)
    if expected is not None and written != int(expected):
        raise OSError("download was cut short")


def _verify(path: Path, expected_sha256: str, name: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise DiarizerUnavailable(f"{_DOWNLOAD_FAILED}: checksum mismatch for {name}")


def _find_segmentation_model(extracted: Path) -> Path | None:
    expected = extracted / _SEGMENTATION_DIR / _SEGMENTATION_FILE
    if expected.exists() or expected.is_symlink():
        return expected
    return next((p for p in sorted(extracted.rglob(_SEGMENTATION_FILE))), None)


def _fetch_segmentation(fetch: Callable[[str, Path], None], work: Path, target: Path) -> None:
    archive = work / "segmentation.tar.bz2"
    fetch(SEGMENTATION_URL, archive)
    _verify(archive, SEGMENTATION_SHA256, SEGMENTATION_URL.rsplit("/", 1)[-1])
    extracted = work / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as tar:
        if sum(member.size for member in tar.getmembers()) > MAX_DOWNLOAD_BYTES:
            raise DiarizerUnavailable(f"{_DOWNLOAD_FAILED}: archive is larger than expected")
        tar.extractall(extracted, filter="data")
    found = _find_segmentation_model(extracted)
    if found is None or found.is_symlink() or not found.is_file():
        raise DiarizerUnavailable(f"{_DOWNLOAD_FAILED}: unexpected archive layout")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(found, target)


def _fetch_embedding(fetch: Callable[[str, Path], None], work: Path, target: Path) -> None:
    partial = work / "embedding.onnx"
    fetch(EMBEDDING_URL, partial)
    _verify(partial, EMBEDDING_SHA256, EMBEDDING_URL.rsplit("/", 1)[-1])
    os.replace(partial, target)


def ensure_models(
    model_dir: Path | None = None,
    fetch: Callable[[str, Path], None] | None = None,
) -> tuple[Path, Path]:
    """Return the segmentation and embedding model paths, downloading whichever is missing.

    Downloads happen in a scratch directory inside ``model_dir`` and move into place only when
    complete, so a failure never leaves a partial file behind.
    """
    work = None
    try:
        model_dir = default_model_dir() if model_dir is None else model_dir
        fetch = _download if fetch is None else fetch
        segmentation = model_dir / _SEGMENTATION_DIR / _SEGMENTATION_FILE
        embedding = model_dir / EMBEDDING_URL.rsplit("/", 1)[-1]
        steps = [(segmentation, _fetch_segmentation), (embedding, _fetch_embedding)]
        missing = [(target, step) for target, step in steps if not target.is_file()]
        if not missing:
            return segmentation, embedding
        model_dir.mkdir(parents=True, exist_ok=True)
        for stale in model_dir.glob(".download-*"):
            shutil.rmtree(stale, ignore_errors=True)
        work = Path(tempfile.mkdtemp(prefix=".download-", dir=model_dir))
        for target, step in missing:
            step(fetch, work, target)
    except DiarizerUnavailable:
        raise
    except Exception as exc:
        raise DiarizerUnavailable(f"{_DOWNLOAD_FAILED}: {_reason(exc)}") from exc
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
    return segmentation, embedding


def _progress_callback(on_progress: Callable[[float], None] | None):
    """Library callback that forwards a 0-1 fraction and never aborts processing."""
    reporter = on_progress

    def callback(processed: int, total: int) -> int:
        nonlocal reporter
        if reporter is not None and total > 0:
            try:
                reporter(min(1.0, max(0.0, processed / total)))
            except Exception:
                logger.debug("progress callback failed; no further progress reported",
                             exc_info=True)
                reporter = None
        return 0

    return callback


def diarize(
    audio: np.ndarray,
    *,
    on_progress: Callable[[float], None] | None = None,
    num_threads: int = 2,
) -> list[Turn]:
    """Split 16 kHz mono float32 ``audio`` into speaker turns, sorted by start time."""
    try:
        return _diarize(audio, on_progress, num_threads)
    except DiarizerUnavailable:
        raise
    except Exception as exc:
        raise DiarizerUnavailable(f"Speaker separation could not run: {_reason(exc)}") from exc


def _diarize(
    audio: np.ndarray, on_progress: Callable[[float], None] | None, num_threads: int,
) -> list[Turn]:
    if len(audio) == 0:
        return []
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise DiarizerUnavailable(_MISSING_EXTRA) from exc
    segmentation, embedding = ensure_models()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation)),
            num_threads=num_threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding), num_threads=num_threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=-1, threshold=cluster_threshold(len(audio) / _SAMPLE_RATE)),
        min_duration_on=0.2,
        min_duration_off=0.5,
    )
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    if diarizer.sample_rate != _SAMPLE_RATE:
        raise DiarizerUnavailable(
            f"Speaker separation could not run: the model expects "
            f"{diarizer.sample_rate} Hz audio, not {_SAMPLE_RATE}")
    samples = np.ascontiguousarray(audio, dtype=np.float32)
    result = diarizer.process(samples, callback=_progress_callback(on_progress))
    turns = [Turn(float(s.start), float(s.end), int(s.speaker))
             for s in result.sort_by_start_time()]
    return sorted(turns, key=lambda t: (t.start, t.end))
