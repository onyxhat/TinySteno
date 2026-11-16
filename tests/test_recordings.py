"""Tests for listing and deleting recordings."""
import os

import pytest

from tinysteno.recordings import delete_recording, list_recordings


def touch(path, mtime=None):
    path.write_bytes(b"x" * 10)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_list_is_newest_first_with_transcript_flag(tmp_path):
    touch(tmp_path / "old.wav", 1_000_000)
    touch(tmp_path / "new.WAV", 2_000_000)
    touch(tmp_path / "new.WAV.transcript.json")
    touch(tmp_path / "notes.md")
    (tmp_path / "dir.wav").mkdir()
    infos = list_recordings(tmp_path)
    assert [i.path.name for i in infos] == ["new.WAV", "old.wav"]
    assert [i.has_transcript for i in infos] == [True, False]
    assert infos[0].size_bytes == 10
    assert infos[0].modified.tzinfo is None


def test_list_of_a_missing_directory_is_empty(tmp_path):
    assert not list_recordings(tmp_path / "missing")


def test_list_includes_symlinked_wavs_and_skips_broken_ones(tmp_path):
    outside = touch(tmp_path / "elsewhere.dat")
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "link.wav").symlink_to(outside)
    (rec / "broken.wav").symlink_to(tmp_path / "gone")
    assert [i.path.name for i in list_recordings(rec)] == ["link.wav"]


def test_delete_removes_wav_and_its_sidecars_only(tmp_path):
    for name in ("a.wav", "a.wav.transcript.json", "a.wav.levels.json", "b.wav", "a.md"):
        touch(tmp_path / name)
    removed = delete_recording(tmp_path, tmp_path / "a.wav")
    assert [p.name for p in removed] == ["a.wav", "a.wav.transcript.json", "a.wav.levels.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.md", "b.wav"]


def test_delete_with_no_sidecars_returns_just_the_wav(tmp_path):
    touch(tmp_path / "a.wav")
    assert delete_recording(tmp_path, tmp_path / "a.wav") == [tmp_path / "a.wav"]


def test_delete_refuses_outside_the_directory(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    other = touch(tmp_path / "other.wav")
    with pytest.raises(ValueError):
        delete_recording(rec, other)
    nested = rec / "sub"
    nested.mkdir()
    deep = touch(nested / "deep.wav")
    with pytest.raises(ValueError):
        delete_recording(rec, deep)
    with pytest.raises(ValueError):
        delete_recording(rec, rec / ".." / "other.wav")
    assert other.exists() and deep.exists()


def test_delete_refuses_a_symlink_and_a_non_wav(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    target = touch(tmp_path / "target.wav")
    link = rec / "link.wav"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        delete_recording(rec, link)
    notes = touch(rec / "a.md")
    with pytest.raises(ValueError):
        delete_recording(rec, notes)
    assert target.exists() and link.is_symlink() and notes.exists()


def test_delete_does_not_follow_a_symlinked_sidecar(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    precious = touch(tmp_path / "precious.json")
    touch(rec / "a.wav")
    (rec / "a.wav.transcript.json").symlink_to(precious)
    removed = delete_recording(rec, rec / "a.wav")
    assert not (rec / "a.wav.transcript.json").is_symlink()
    assert rec / "a.wav.transcript.json" in removed
    assert precious.exists()


def test_delete_leaves_a_directory_sidecar_alone(tmp_path):
    touch(tmp_path / "a.wav")
    (tmp_path / "a.wav.levels.json").mkdir()
    assert delete_recording(tmp_path, tmp_path / "a.wav") == [tmp_path / "a.wav"]
    assert (tmp_path / "a.wav.levels.json").is_dir()


def test_delete_of_a_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        delete_recording(tmp_path, tmp_path / "gone.wav")
    (tmp_path / "d.wav").mkdir()
    with pytest.raises(FileNotFoundError):
        delete_recording(tmp_path, tmp_path / "d.wav")


def test_sidecar_failure_propagates_after_the_wav_is_gone(tmp_path, monkeypatch):
    touch(tmp_path / "a.wav")
    touch(tmp_path / "a.wav.transcript.json")
    real_unlink = type(tmp_path).unlink

    def flaky(self, *args, **kwargs):
        if self.name.endswith(".transcript.json"):
            raise PermissionError("nope")
        return real_unlink(self, *args, **kwargs)
    monkeypatch.setattr(type(tmp_path), "unlink", flaky)
    with pytest.raises(PermissionError):
        delete_recording(tmp_path, tmp_path / "a.wav")
    assert not (tmp_path / "a.wav").exists()
