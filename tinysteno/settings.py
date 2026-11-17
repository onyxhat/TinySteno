"""The editable settings, described once: what the Settings screen shows and validates.

This module is UI-free: no Textual and no file access.
"""
import math
from dataclasses import dataclass

SECTIONS: tuple[str, ...] = (
    "Notes", "Summary (LLM)", "Transcription", "Speakers", "Recording", "Appearance",
)


@dataclass(frozen=True)
class Setting:
    """One editable config key.

    kind is one of "text", "path", "secret", "int", "number", "bool", "choice".
    A blank `optional` setting is stored as None, meaning "remove the key from the file".
    `essential` settings are the ones the first-run guide asks for.
    """
    key: str
    label: str
    kind: str
    section: str
    help: str = ""
    choices: tuple[str, ...] = ()
    optional: bool = False
    essential: bool = False


def _bools(section: str, *items: tuple[str, str, str]) -> tuple[Setting, ...]:
    return tuple(Setting(key, label, "bool", section, help_text) for key, label, help_text in items)


SETTINGS: tuple[Setting, ...] = (
    Setting("obsidian_vault", "Obsidian vault", "path", "Notes",
            "Folder of the Obsidian vault that receives your notes.", essential=True),
    Setting("output_folder", "Notes folder", "text", "Notes",
            "Folder inside the vault where meeting notes are saved.", essential=True),
    Setting("recordings_path", "Recordings folder", "path", "Notes",
            "Where audio is kept; blank uses <vault>/<folder>/audio.", optional=True),
    Setting("persona", "Persona", "choice", "Notes",
            "The note style used by default.", essential=True),
    *_bools(
        "Notes",
        ("auto_title", "Generate a title", "Ask the model to write a title for each note."),
        ("auto_tags", "Generate tags", "Ask the model to suggest tags for each note."),
        ("transcript_timestamps", "Transcript timestamps",
         "Put a time stamp on each transcript line."),
    ),
    Setting("base_url", "LLM base URL", "text", "Summary (LLM)",
            "Address of the OpenAI-compatible service that writes the summary.", essential=True),
    Setting("model", "LLM model", "text", "Summary (LLM)",
            "Name of the model used for the summary.", essential=True),
    Setting("api_key", "LLM API key", "secret", "Summary (LLM)",
            "Key for the summary service; local services accept any text.", essential=True),
    Setting("llm_context_tokens", "LLM context tokens", "int", "Summary (LLM)",
            "How many tokens of transcript to send to the model at once."),
    Setting("transcription_backend", "Transcription backend", "choice", "Transcription",
            "Transcribe on this computer (local) or through a Whisper API (api).",
            choices=("local", "api"), essential=True),
    Setting("whisper_model", "Whisper model", "choice", "Transcription",
            "Local model: tiny, base, small, medium or large-v3, or any other model name.",
            choices=("tiny", "base", "small", "medium", "large-v3")),
    Setting("whisper_device", "Whisper device", "choice", "Transcription",
            "Hardware for local transcription.", choices=("auto", "cpu", "cuda")),
    Setting("whisper_compute_type", "Whisper compute type", "text", "Transcription",
            "Numeric precision for local transcription; auto picks one."),
    Setting("whisper_vad", "Skip silence", "bool", "Transcription",
            "Skip silent stretches when transcribing locally."),
    Setting("whisper_prompt", "Whisper prompt", "text", "Transcription",
            "Words or names to help the transcriber spell things right.", optional=True),
    Setting("whisper_base_url", "Whisper API base URL", "text", "Transcription",
            "Address of the Whisper API; blank uses the summary service address.",
            optional=True),
    Setting("whisper_api_key", "Whisper API key", "secret", "Transcription",
            "Key for the Whisper API; blank uses the summary service key.", optional=True),
    Setting("whisper_api_model", "Whisper API model", "text", "Transcription",
            "Model name sent to the Whisper API."),
    Setting("whisper_api_timeout", "Whisper API timeout", "number", "Transcription",
            "Seconds to wait for the Whisper API; must be more than 0."),
    *_bools(
        "Speakers",
        ("diarization", "Speaker labels", "Label who is speaking in the transcript."),
        ("hallucination_filter", "Hallucination filter",
         "Remove lines the transcriber invents during silence."),
        ("bleed_filter", "Bleed filter",
         "Remove your own voice that leaked into the system audio."),
        ("speaker_split", "Separate remote speakers",
         "Tell apart the people on the remote side of a call."),
        ("whisper_api_diarize", "API speaker labels",
         "Ask the Whisper API to label speakers."),
    ),
    Setting("max_speakers", "Maximum speakers", "int", "Speakers",
            "Most speakers to look for; blank lets the program decide.", optional=True),
    Setting("session_timeout_minutes", "Recording time limit (minutes)", "number", "Recording",
            "Stop a recording after this many minutes; 0 disables."),
    Setting("sample_rate", "Sample rate", "int", "Recording",
            "Audio samples per second; must be a whole number above 0."),
    Setting("channels", "Channels", "choice", "Recording",
            "1 for mono or 2 for stereo.", choices=("1", "2")),
    Setting("tui_theme", "Theme", "text", "Appearance",
            "Name of the terminal interface theme.", optional=True),
)

_BY_KEY = {s.key: s for s in SETTINGS}
# Choice settings whose listed choices are suggestions: an existing value outside them is kept.
_OPEN_CHOICES = frozenset({"whisper_model"})
_BOOL_WORDS = {
    "true": True, "yes": True, "on": True, "1": True,
    "false": False, "no": False, "off": False, "0": False,
}


def accepts_any(item: Setting) -> bool:
    """True for a choice setting whose choices are only suggestions: any value is accepted."""
    return item.key in _OPEN_CHOICES


def setting(key: str) -> Setting:
    """Return the Setting for `key`; raise ValueError when no such setting exists."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown setting: {key}") from None


def to_text(item: Setting, value: object) -> str:
    """Render a stored value as the text shown in its field."""
    if value is None:
        return ""
    if item.kind == "bool" or isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fail(item: Setting, reason: str) -> ValueError:
    return ValueError(f"{item.label}: {reason}")


def _parse_number(item: Setting, text: str) -> int | float:
    """Parse a finite number; a whole value comes back as int."""
    try:
        number = float(text) if "_" not in text else math.nan
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        raise _fail(item, "must be a number")
    return int(number) if number.is_integer() else number


def _check_range(item: Setting, number: int | float) -> None:
    key = item.key
    if key == "session_timeout_minutes" and number < 0:
        raise _fail(item, "must be 0 or more (0 disables)")
    if key in ("whisper_api_timeout", "llm_context_tokens", "sample_rate") and number <= 0:
        raise _fail(item, "must be more than 0")
    if key == "max_speakers" and number < 1:
        raise _fail(item, "must be 1 or more")


def from_text(item: Setting, text: str) -> object:
    """Parse the text of a field; raise ValueError starting with the label when invalid.

    Surrounding whitespace is ignored. A blank optional setting gives None. Whole numbers
    come back as int. Paths are kept as typed (load_config expands them).
    """
    text = text.strip()
    if not text:
        if item.optional:
            return None
        raise _fail(item, "is required")
    kind = item.kind
    if kind == "bool":
        if text.lower() not in _BOOL_WORDS:
            raise _fail(item, "must be yes or no")
        return _BOOL_WORDS[text.lower()]
    if kind in ("int", "number"):
        number = _parse_number(item, text)
        if kind == "int" and not isinstance(number, int):
            raise _fail(item, "must be a whole number")
        _check_range(item, number)
        return number
    if kind == "choice":
        if item.choices and not accepts_any(item) and text not in item.choices:
            raise _fail(item, f"must be one of {', '.join(item.choices)}")
        return int(text) if item.key == "channels" else text
    return text
