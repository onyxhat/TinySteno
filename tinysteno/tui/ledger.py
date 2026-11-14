"""What became of every recording this session touched."""

from tinysteno.session import saved_message


class SessionLedger:
    """The one place the app's parting message comes from.

    Screens come and go; what the user must be told on the way out does not depend on which
    of them happens to be showing. Each recording is entered here as its state changes:

        saved(wav)          a WAV exists and has no note yet (just recorded, or picked on Home)
        processed(wav)      a note was written from it: nothing left to say
        save_failed(text)   the WAV could not be written; ``text`` says where the raw audio is
    """

    def __init__(self) -> None:
        self._unprocessed: dict[str, None] = {}  # insertion-ordered set of WAV paths
        self._failures: list[str] = []

    def saved(self, wav_path: str) -> None:
        """A WAV is on disk and still needs processing."""
        self._unprocessed[wav_path] = None

    def processed(self, wav_path: str) -> None:
        """A note was exported from this WAV."""
        self._unprocessed.pop(wav_path, None)

    def save_failed(self, message: str) -> None:
        """A recording could not be saved; ``message`` is what to tell the user."""
        if message not in self._failures:
            self._failures.append(message)

    def summary(self) -> tuple[str | None, bool]:
        """(What to print on exit, whether any recording failed to save)."""
        lines = self._failures + [saved_message(wav) for wav in self._unprocessed]
        return ("\n".join(lines) or None), bool(self._failures)
