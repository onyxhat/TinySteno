"""Guard: no test may resolve the real home directory or its ~/.tinysteno."""
import os
from pathlib import Path

import tinysteno.personas
from tinysteno.config import config_path


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
