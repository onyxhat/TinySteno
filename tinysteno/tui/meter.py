"""A fixed-scale audio level meter: dBFS history, two rows tall, newest sample at the right.

Unlike Textual's Sparkline, which rescales to its data's own range, the scale here is fixed
at -60..0 dBFS, so a silent input draws only a faint baseline and real sound draws bars.
"""

import math
from collections import deque

from rich.style import Style
from rich.text import Text
from textual.color import Color
from textual.widget import Widget

FLOOR_DBFS = -60.0
ROWS = 2
STEPS = 8 * ROWS  # vertical resolution: eight block heights per row
BLOCKS = " ▁▂▃▄▅▆▇█"
BASELINE = "▁"
HISTORY = 120  # samples kept: 12 seconds at the 0.1 s refresh
# Fraction of full scale where the colour leaves $success, and where it reaches $warning
# (then $error at 0 dBFS): the usual green to -18 dBFS, amber to -6 dBFS, red above.
QUIET_UNTIL = 0.7  # -18 dBFS
WARNING_AT = 0.9  # -6 dBFS


def level_fraction(rms: float) -> float:
    """Map an RMS sample (1.0 = full scale) onto 0..1 of the -60..0 dBFS scale, clamped."""
    if math.isnan(rms) or rms <= 0:
        return 0.0
    dbfs = 20 * math.log10(rms)
    return min(1.0, max(0.0, (dbfs - FLOOR_DBFS) / -FLOOR_DBFS))


def level_steps(rms: float) -> int:
    """Bar height for one sample, 0 (empty) to STEPS (full)."""
    return round(level_fraction(rms) * STEPS)


def column_steps(samples, width: int) -> list[int]:
    """Bar heights for a meter ``width`` cells wide: the most recent samples, newest at the
    right edge; columns without a sample yet are 0."""
    recent = list(samples)[-width:] if width > 0 else []
    return [0] * (width - len(recent)) + [level_steps(s) for s in recent]


def meter_rows(steps: list[int]) -> tuple[str, str]:
    """The (top, bottom) rows of block characters for a list of bar heights."""
    top = "".join(BLOCKS[max(0, s - 8)] for s in steps)
    bottom = "".join(BLOCKS[min(s, 8)] if s else BASELINE for s in steps)
    return top, bottom


class LevelMeter(Widget):
    """Rolling level history for one audio stream."""

    COMPONENT_CLASSES = {
        "level-meter--quiet",
        "level-meter--warning",
        "level-meter--loud",
        "level-meter--baseline",
    }

    DEFAULT_CSS = """
    LevelMeter {
        height: 2;
    }
    LevelMeter > .level-meter--quiet {
        color: $success;
    }
    LevelMeter > .level-meter--warning {
        color: $warning;
    }
    LevelMeter > .level-meter--loud {
        color: $error;
    }
    LevelMeter > .level-meter--baseline {
        color: $foreground 25%;
    }
    """

    def __init__(self, *, id: str | None = None,  # pylint: disable=redefined-builtin
                 classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self.data: deque[float] = deque(maxlen=HISTORY)

    def push(self, sample: float) -> None:
        """Add the newest sample and redraw."""
        self.data.append(float(sample))
        self.refresh()

    def _colour(self, name: str) -> Color:
        _, base = self.background_colors
        return base + self.get_component_styles(f"level-meter--{name}").color

    def _bar_colour(self, steps: int, quiet: Color, warning: Color, loud: Color) -> Color:
        fraction = steps / STEPS
        if fraction <= QUIET_UNTIL:
            return quiet
        if fraction <= WARNING_AT:
            return quiet.blend(warning, (fraction - QUIET_UNTIL) / (WARNING_AT - QUIET_UNTIL))
        return warning.blend(loud, (fraction - WARNING_AT) / (1 - WARNING_AT))

    def render(self) -> Text:
        steps = column_steps(self.data, self.size.width)
        top, bottom = meter_rows(steps)
        quiet, warning, loud = (self._colour(n) for n in ("quiet", "warning", "loud"))
        baseline = Style(color=self._colour("baseline").rich_color)
        top_text, bottom_text = Text(), Text()
        for column, height in enumerate(steps):
            if height:
                style = Style(color=self._bar_colour(height, quiet, warning, loud).rich_color)
                top_text.append(top[column], style=style)
                bottom_text.append(bottom[column], style=style)
            else:
                top_text.append(top[column])
                bottom_text.append(bottom[column], style=baseline)
        return Text("\n").join([top_text, bottom_text])
