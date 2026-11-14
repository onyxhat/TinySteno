"""Keep a started recorder reachable, so no way out of the app can leave it unsaved."""

import threading

from tinysteno.session import saved_message

_SAVED = "saved"
_EMPTY = "empty"
_FAILED = "failed"
_DISCARDED = "discarded"


def save_failed_message(err: Exception, raw_paths) -> str:
    """What to tell the user when the WAV could not be written: why, and where the audio is."""
    cause = err.__cause__ or err
    reason = str(cause) or type(cause).__name__
    kept = (
        f"Raw audio kept at {' and '.join(str(p) for p in raw_paths)}"
        if raw_paths else "No raw audio was kept"
    )
    return f"Could not save the recording: {reason}. {kept}"


class RecorderGuard:
    """The recorder of the session in progress.

    ``stop()`` and ``discard()`` reach the recorder at most once between them, whichever
    thread gets there first; a later call returns the first outcome (or re-raises its error).
    """

    def __init__(self, recorder, wav_path: str) -> None:
        self.recorder = recorder
        self.wav_path = wav_path
        self._lock = threading.Lock()
        self._outcome: str | None = None
        self._error: Exception | None = None

    def stop(self) -> bool:
        """Stop and save; True when a WAV was written. Raises what ``recorder.stop()`` raised."""
        with self._lock:
            if self._outcome is None:
                try:
                    self._outcome = _SAVED if self.recorder.stop() else _EMPTY
                except Exception as err:
                    self._outcome, self._error = _FAILED, err
            if self._error is not None:
                raise self._error
            return self._outcome == _SAVED

    def discard(self) -> None:
        """Throw the audio away (only if nothing has stopped the recorder yet)."""
        with self._lock:
            if self._outcome is None:
                self._outcome = _DISCARDED
                self.recorder.discard()

    def final_message(self) -> tuple[str | None, bool]:
        """Stop the recorder if nothing has; return (what to tell the user, whether it failed)."""
        try:
            saved = self.stop()
        except Exception as err:
            raw_paths = tuple(getattr(self.recorder, "raw_paths", ()) or ())
            return save_failed_message(err, raw_paths), True
        return (saved_message(self.wav_path) if saved else None), False
