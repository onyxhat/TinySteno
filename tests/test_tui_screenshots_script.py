"""scripts/tui_screenshots.py keeps HOME inside its temporary directory while it runs."""
import importlib.util
import os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "tui_screenshots", Path(__file__).resolve().parent.parent / "scripts" / "tui_screenshots.py"
)
tui_screenshots = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tui_screenshots)


def test_home_points_at_the_temporary_directory_and_is_restored(isolated_home):
    before = os.environ["HOME"]
    assert before == str(isolated_home)
    with tui_screenshots._demo_app():  # pylint: disable=protected-access
        inside = os.environ["HOME"]
        assert Path(inside).resolve() == Path.cwd().resolve()
        assert Path(inside).resolve() != isolated_home.resolve()
    assert os.environ["HOME"] == before
    assert not Path(inside).exists()
