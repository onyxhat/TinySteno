"""Tests for the sherpa-onnx diarizer adapter. Nothing is downloaded and no real model runs."""
import hashlib
import io
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tinysteno import sherpa_diarizer as sherpa_diarizer_module
from tinysteno.sherpa_diarizer import (
    EMBEDDING_URL, SEGMENTATION_URL, DiarizerUnavailable, default_model_dir, diarize,
    ensure_models,
)
from tinysteno.speaker_split import Turn

download = sherpa_diarizer_module._download  # pylint: disable=protected-access
MODEL_PATH = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"


def _archive_bytes(members: dict[str, bytes], extra=None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra is not None:
            extra(tar)
    return buffer.getvalue()


def _write_archive(dest: Path, members: dict[str, bytes], extra=None) -> None:
    data = _archive_bytes(members, extra)
    dest.write_bytes(data)
    sherpa_diarizer_module.SEGMENTATION_SHA256 = hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def _pins(monkeypatch):
    """Pin the digests of the fake files; `_write_archive` re-pins for custom archives."""
    monkeypatch.setattr(sherpa_diarizer_module, "EMBEDDING_SHA256",
                        hashlib.sha256(b"emb").hexdigest())
    monkeypatch.setattr(sherpa_diarizer_module, "SEGMENTATION_SHA256",
                        hashlib.sha256(_archive_bytes({MODEL_PATH: b"seg"})).hexdigest())


def _fake_fetch(fetched, members=None):
    def fetch(url, dest):
        fetched.append(url)
        if url == SEGMENTATION_URL:
            _write_archive(dest, members if members is not None else {MODEL_PATH: b"seg"})
        else:
            dest.write_bytes(b"emb")
    return fetch


def _fake_sherpa(segments=(), sample_rate=16000, process_error=None):
    fake = MagicMock()
    diarizer = fake.OfflineSpeakerDiarization.return_value
    diarizer.sample_rate = sample_rate
    result = diarizer.process.return_value
    result.sort_by_start_time.return_value = [
        SimpleNamespace(start=s, end=e, speaker=k) for s, e, k in segments]
    if process_error is not None:
        diarizer.process.side_effect = process_error
    return fake


def _run(fake, audio, tmp_path, **kwargs):
    with patch.dict(sys.modules, {"sherpa_onnx": fake}), \
         patch("tinysteno.sherpa_diarizer.ensure_models",
               return_value=(tmp_path / "s", tmp_path / "e")):
        return diarize(audio, **kwargs)


def _silence(seconds):
    return np.zeros(16000 * seconds, dtype=np.float32)


def test_missing_extra_is_reported_with_the_install_command():
    with patch.dict(sys.modules, {"sherpa_onnx": None}):
        with pytest.raises(DiarizerUnavailable, match=r'pip install "tinysteno\[diarize\]"'):
            diarize(_silence(40))


def test_models_are_downloaded_once_then_reused(tmp_path):
    fetched = []
    paths = ensure_models(tmp_path, fetch=_fake_fetch(fetched))
    assert all(p.exists() for p in paths) and len(fetched) == 2
    assert paths[0].name == "model.onnx"
    ensure_models(tmp_path, fetch=_fake_fetch(fetched))
    assert len(fetched) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [paths[0].parent.name, paths[1].name])


def test_only_the_missing_model_is_fetched(tmp_path):
    fetched = []
    seg, _ = ensure_models(tmp_path, fetch=_fake_fetch(fetched))
    _, emb = ensure_models(tmp_path, fetch=_fake_fetch([]))
    emb.unlink()
    again = []
    ensure_models(tmp_path, fetch=_fake_fetch(again))
    assert again == [EMBEDDING_URL] and seg.exists()


def test_default_model_dir_follows_the_current_home(tmp_path):
    with patch.object(Path, "home", return_value=tmp_path):
        assert default_model_dir() == tmp_path / ".tinysteno" / "models" / "diarize"


def test_failed_download_leaves_nothing_behind(tmp_path):
    def offline(_url, _dest):
        raise OSError("network unreachable")
    with pytest.raises(DiarizerUnavailable, match="Could not download"):
        ensure_models(tmp_path, fetch=offline)
    assert not list(tmp_path.rglob("*"))


def test_interrupted_download_removes_its_partial_file(tmp_path):
    def partial(_url, dest):
        dest.write_bytes(b"half")
        raise TimeoutError("timed out")
    with pytest.raises(DiarizerUnavailable, match="Could not download.*timed out"):
        ensure_models(tmp_path, fetch=partial)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_failure_after_the_first_model_keeps_only_the_complete_one(tmp_path):
    def second_fails(url, dest):
        if url == SEGMENTATION_URL:
            _write_archive(dest, {MODEL_PATH: b"seg"})
        else:
            dest.write_bytes(b"half")
            raise OSError("reset")
    with pytest.raises(DiarizerUnavailable):
        ensure_models(tmp_path, fetch=second_fails)
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["model.onnx"]


def test_corrupt_archive_is_reported_and_cleaned_up(tmp_path):
    def corrupt(_url, dest):
        dest.write_bytes(b"not a tarball")
    with pytest.raises(DiarizerUnavailable, match="Could not download"):
        ensure_models(tmp_path, fetch=corrupt)
    assert not list(tmp_path.rglob("*"))


def test_unexpected_archive_layout_is_reported_and_cleaned_up(tmp_path):
    fetch = _fake_fetch([], members={"other/readme.txt": b"hi"})
    with pytest.raises(DiarizerUnavailable, match="unexpected archive layout"):
        ensure_models(tmp_path, fetch=fetch)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_model_found_under_a_different_directory_name(tmp_path):
    fetch = _fake_fetch([], members={"renamed/sub/model.onnx": b"seg"})
    seg, _ = ensure_models(tmp_path, fetch=fetch)
    assert seg.read_bytes() == b"seg"


def test_archive_members_outside_the_destination_are_refused(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    def evil(_url, dest):
        _write_archive(dest, {MODEL_PATH: b"seg", "../escaped.txt": b"x"})
    with pytest.raises(DiarizerUnavailable, match="Could not download"):
        ensure_models(model_dir, fetch=evil)
    assert not (tmp_path / "escaped.txt").exists()
    assert not [p for p in model_dir.rglob("*") if p.is_file()]


def test_diarize_maps_library_output_to_turns_and_sets_the_ramped_threshold(tmp_path):
    fake = _fake_sherpa(segments=[(12.0, 20.0, 1), (0.0, 11.5, 0)])
    turns = _run(fake, _silence(1800), tmp_path, num_threads=3)
    assert turns == [Turn(0.0, 11.5, 0), Turn(12.0, 20.0, 1)]
    assert fake.FastClusteringConfig.call_args.kwargs == {
        "num_clusters": -1, "threshold": pytest.approx(0.6333, abs=1e-3)}
    assert fake.OfflineSpeakerDiarizationConfig.call_args.kwargs["min_duration_on"] == 0.2
    assert fake.OfflineSpeakerDiarizationConfig.call_args.kwargs["min_duration_off"] == 0.5
    assert fake.OfflineSpeakerSegmentationModelConfig.call_args.kwargs["num_threads"] == 3
    assert fake.SpeakerEmbeddingExtractorConfig.call_args.kwargs["num_threads"] == 3


def test_empty_audio_returns_nothing_without_loading_anything():
    with patch.dict(sys.modules, {"sherpa_onnx": None}):
        assert diarize(np.zeros(0, dtype=np.float32)) == []


def test_unsupported_sample_rate_is_reported(tmp_path):
    with pytest.raises(DiarizerUnavailable, match="16000"):
        _run(_fake_sherpa(sample_rate=8000), _silence(40), tmp_path)


def test_library_runtime_error_becomes_diarizer_unavailable(tmp_path):
    fake = _fake_sherpa(process_error=RuntimeError("bad model"))
    with pytest.raises(DiarizerUnavailable, match="Speaker separation could not run: bad model"):
        _run(fake, _silence(40), tmp_path)


def test_construction_error_becomes_diarizer_unavailable(tmp_path):
    fake = _fake_sherpa()
    fake.OfflineSpeakerDiarization.side_effect = RuntimeError("missing file")
    with pytest.raises(DiarizerUnavailable, match="could not run: missing file"):
        _run(fake, _silence(40), tmp_path)


def test_progress_is_passed_on_as_a_fraction_and_never_aborts(tmp_path):
    fake = _fake_sherpa()
    seen = []

    def process(_samples, callback=None):
        assert callback(1, 4) == 0
        assert callback(9, 4) == 0
        assert callback(0, 0) == 0
        return fake.OfflineSpeakerDiarization.return_value.process.return_value
    fake.OfflineSpeakerDiarization.return_value.process.side_effect = process
    _run(fake, _silence(40), tmp_path, on_progress=seen.append)
    assert seen == [0.25, 1.0]


def test_failing_progress_callback_is_dropped_not_propagated(tmp_path):
    fake = _fake_sherpa()
    calls = []

    def bad(fraction):
        calls.append(fraction)
        raise ValueError("ui gone")

    def process(_samples, callback=None):
        assert callback(1, 4) == 0
        assert callback(2, 4) == 0
        return fake.OfflineSpeakerDiarization.return_value.process.return_value
    fake.OfflineSpeakerDiarization.return_value.process.side_effect = process
    assert _run(fake, _silence(40), tmp_path, on_progress=bad) == []
    assert calls == [0.25]


def test_download_only_accepts_github_over_https(tmp_path):
    with pytest.raises(ValueError):
        download("file:///etc/passwd", tmp_path / "x")
    with pytest.raises(ValueError):
        download("http://github.com/x", tmp_path / "x")


def _fake_response(body: bytes, length: str):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.side_effect = [body, b""]
    response.headers = {"Content-Length": length}
    response.url = "https://objects.githubusercontent.com/model"
    return response


def test_download_writes_the_body_and_rejects_a_truncated_one(tmp_path):
    dest = tmp_path / "x"
    with patch("urllib.request.urlopen", return_value=_fake_response(b"abcd", "4")) as opened:
        download(SEGMENTATION_URL, dest)
    assert dest.read_bytes() == b"abcd" and opened.call_args.kwargs["timeout"] == 60
    with patch("urllib.request.urlopen", return_value=_fake_response(b"abcd", "9")):
        with pytest.raises(OSError, match="cut short"):
            download(SEGMENTATION_URL, dest)


@pytest.mark.parametrize("break_it", ["process", "config", "numpy"])
def test_any_library_exception_becomes_diarizer_unavailable(tmp_path, break_it):
    fake = _fake_sherpa()
    if break_it == "process":
        fake.OfflineSpeakerDiarization.return_value.process.side_effect = ValueError("bad shape")
    elif break_it == "config":
        fake.FastClusteringConfig.side_effect = TypeError("bad arg")
    else:
        fake.OfflineSpeakerDiarization.return_value.process.return_value \
            .sort_by_start_time.return_value = [SimpleNamespace(start="x", end=1, speaker=0)]
    with pytest.raises(DiarizerUnavailable, match="could not run") as caught:
        _run(fake, _silence(40), tmp_path)
    assert caught.value.__cause__ is not None


def test_a_non_import_error_while_importing_is_contained():
    with patch("builtins.__import__", side_effect=OSError("dlopen failed")):
        with pytest.raises(DiarizerUnavailable, match="could not run: dlopen failed"):
            diarize(_silence(40))


def test_exception_without_a_message_names_its_class(tmp_path):
    fake = _fake_sherpa(process_error=KeyError())
    with pytest.raises(DiarizerUnavailable, match="could not run: KeyError"):
        _run(fake, _silence(40), tmp_path)


def test_keyboard_interrupt_is_not_swallowed(tmp_path):
    fake = _fake_sherpa(process_error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _run(fake, _silence(40), tmp_path)


def test_unresolvable_home_is_reported():
    with patch.object(Path, "home", side_effect=RuntimeError("no home")):
        with pytest.raises(DiarizerUnavailable, match="Could not download.*no home"):
            ensure_models(fetch=_fake_fetch([]))


def test_unreadable_model_dir_is_reported(tmp_path):
    with patch.object(Path, "is_file", side_effect=PermissionError("denied")):
        with pytest.raises(DiarizerUnavailable, match="denied"):
            ensure_models(tmp_path, fetch=_fake_fetch([]))


def test_checksum_mismatch_on_the_embedding_leaves_nothing(tmp_path):
    def wrong(url, dest):
        dest.write_bytes(b"emb-tampered" if url == EMBEDDING_URL else b"")
        if url == SEGMENTATION_URL:
            _write_archive(dest, {MODEL_PATH: b"seg"})
    with pytest.raises(DiarizerUnavailable, match="checksum mismatch for 3dspeaker"):
        ensure_models(tmp_path, fetch=wrong)
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["model.onnx"]


def test_checksum_mismatch_on_the_archive_leaves_nothing(tmp_path):
    def wrong(_url, dest):
        _write_archive(dest, {MODEL_PATH: b"seg"})
        sherpa_diarizer_module.SEGMENTATION_SHA256 = "0" * 64
    with pytest.raises(DiarizerUnavailable, match="checksum mismatch for sherpa-onnx-pyannote"):
        ensure_models(tmp_path, fetch=wrong)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_symlinked_model_in_the_archive_is_refused(tmp_path):
    def link(tar):
        info = tarfile.TarInfo("seg/model.onnx")
        info.type = tarfile.SYMTYPE
        info.linkname = "readme.txt"
        tar.addfile(info)

    def fetch(url, dest):
        if url == SEGMENTATION_URL:
            _write_archive(dest, {"seg/readme.txt": b"x"}, extra=link)
        else:
            dest.write_bytes(b"emb")
    with pytest.raises(DiarizerUnavailable, match="unexpected archive layout"):
        ensure_models(tmp_path, fetch=fetch)
    assert not [p for p in tmp_path.rglob("*") if p.is_file() or p.is_symlink()]


def test_oversized_archive_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sherpa_diarizer_module, "MAX_DOWNLOAD_BYTES", 2)
    with pytest.raises(DiarizerUnavailable, match="larger than expected"):
        ensure_models(tmp_path, fetch=_fake_fetch([]))
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_stale_scratch_directories_are_removed(tmp_path):
    stale = tmp_path / ".download-old"
    stale.mkdir()
    (stale / "half.bin").write_bytes(b"x")
    ensure_models(tmp_path, fetch=_fake_fetch([]))
    assert not stale.exists()


def test_download_stops_past_the_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(sherpa_diarizer_module, "MAX_DOWNLOAD_BYTES", 10)
    response = _fake_response(b"x" * 20, "20")
    response.read.side_effect = [b"x" * 8, b"x" * 8, b""]
    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(OSError, match="larger than expected"):
            download(SEGMENTATION_URL, tmp_path / "x")


def test_download_refuses_a_redirect_off_https(tmp_path):
    response = _fake_response(b"abcd", "4")
    response.url = "http://example.com/model"
    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(ValueError, match="redirect"):
            download(SEGMENTATION_URL, tmp_path / "x")
