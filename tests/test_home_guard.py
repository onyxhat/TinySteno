"""Guard: no test may resolve the real home directory or its ~/.tinysteno."""
import os
import subprocess
import sys
from pathlib import Path

import tinysteno.personas
from tinysteno.config import DEFAULT_CONFIG, config_path, load_config, recordings_dir


def _under(path: Path, root: Path) -> bool:
    return Path(os.path.realpath(path)).is_relative_to(os.path.realpath(root))


def test_home_and_config_resolve_under_the_pytest_temp_root(tmp_path_factory, real_home):
    temp_root = tmp_path_factory.getbasetemp()
    assert _under(Path.home(), temp_root) and _under(config_path(), temp_root)
    assert _under(Path(os.environ["HOME"]), temp_root)
    assert not _under(Path.home(), real_home) and not _under(config_path(), real_home)


def test_personas_dir_is_under_the_pytest_temp_root(tmp_path_factory, real_home):
    user_dir = tinysteno.personas._USER_DIR  # pylint: disable=protected-access
    assert _under(user_dir, tmp_path_factory.getbasetemp())
    assert not _under(user_dir, real_home)


def test_no_default_config_value_points_at_the_real_home(real_home):
    for key, value in DEFAULT_CONFIG.items():
        if isinstance(value, str):
            assert not value.startswith(str(real_home)), key


def test_recordings_dirs_resolve_under_the_temp_root(tmp_path_factory):
    temp_root = tmp_path_factory.getbasetemp()
    assert _under(recordings_dir(DEFAULT_CONFIG), temp_root)
    assert _under(recordings_dir(load_config()), temp_root)


def test_child_processes_see_the_redirected_home(tmp_path_factory):
    out = subprocess.run(
        [sys.executable, "-c", "from pathlib import Path; print(Path.home())"],
        check=True, capture_output=True, text=True).stdout.strip()
    assert _under(Path(out), tmp_path_factory.getbasetemp())
