"""The fixed-scale level meter: dBFS mapping, rolling history and rendering."""
import pytest
from textual.app import App, ComposeResult

from tinysteno.tui.meter import (
    BASELINE, HISTORY, STEPS, LevelMeter, column_steps, level_fraction, level_steps, meter_rows,
)


@pytest.mark.parametrize("rms", [0.0, 0.0005, 0.001, -0.5, float("nan")])
def test_at_or_below_minus_60_dbfs_is_empty(rms):
    assert level_fraction(rms) == 0.0
    assert level_steps(rms) == 0


def test_minus_30_dbfs_is_half_scale():
    assert level_fraction(0.0316) == pytest.approx(0.5, abs=1e-3)
    assert level_steps(0.0316) == STEPS // 2


@pytest.mark.parametrize("rms", [1.0, 1.5, 40.0, float("inf")])
def test_full_scale_and_above_clamp_to_full(rms):
    assert level_fraction(rms) == 1.0
    assert level_steps(rms) == STEPS


def test_scale_is_monotonic():
    samples = [0.001, 0.002, 0.01, 0.0316, 0.1, 0.3, 1.0]
    fractions = [level_fraction(s) for s in samples]
    assert fractions == sorted(fractions)


def test_columns_show_the_newest_samples_at_the_right_edge():
    assert column_steps([1.0, 0.0316], 4) == [0, 0, STEPS, STEPS // 2]
    assert column_steps([0.0316] * 3 + [1.0], 2) == [STEPS // 2, STEPS]
    assert column_steps([1.0], 0) == []


def test_rows_split_sixteen_steps_over_two_rows():
    top, bottom = meter_rows([0, 4, 8, 12, 16])
    assert bottom == BASELINE + "▄███"
    assert top == "   ▄█"


def test_noise_floor_history_renders_no_bars():
    # Samples at or below the -60 dBFS floor (0.001 RMS), as from an idle or muted input.
    steps = column_steps([0.001, 0.0008, 0.0005, 0.0] * 30, 60)
    assert steps == [0] * 60
    top, bottom = meter_rows(steps)
    assert top.strip() == "" and set(bottom) == {BASELINE}


class _MeterApp(App):
    def compose(self) -> ComposeResult:
        yield LevelMeter(id="meter")


async def test_widget_keeps_120_samples_and_renders_two_rows():
    app = _MeterApp()
    async with app.run_test(size=(30, 4)) as pilot:
        meter = app.query_one("#meter", LevelMeter)
        for _ in range(HISTORY + 30):
            meter.push(0.001)
        meter.push(1.0)
        await pilot.pause()
        assert len(meter.data) == HISTORY and meter.data[-1] == 1.0
        rendered = meter.render()
        top, bottom = rendered.plain.split("\n")
        assert len(top) == len(bottom) == meter.size.width
        assert top[-1] == "█" and bottom[-1] == "█"
        assert set(top[:-1]) == {" "} and set(bottom[:-1]) == {BASELINE}
        assert meter.size.height == 2
