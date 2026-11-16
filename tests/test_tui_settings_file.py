"""Settings screen against the real config file, in the per-test home that conftest sets up."""
import yaml

from tinysteno.config import config_path, read_stored, update_config
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.home import HomeScreen
from tests.test_tui_settings import CONFIG, _error, _field, _open, _save
from tests.tui_fakes import make_deps


def _write_config(text: str):
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def test_real_malformed_file_is_reported_without_its_content_and_left_alone(tmp_path):
    path = _write_config("model: a\napi_key sk-live-123\n: [\n")
    before = path.read_bytes()
    deps = make_deps(tmp_path, load_settings=read_stored, save_settings=update_config)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        text = str(screen.query_one("#load-error").content)
        assert text.startswith("Config file cannot be read: Config file is not valid YAML")
        assert "sk-live-123" not in text
        await _save(pilot)
    assert path.read_bytes() == before


async def test_real_read_only_file_is_reported_and_left_alone(tmp_path):
    path = _write_config("model: a\n")
    path.chmod(0o444)
    before = path.read_bytes()
    deps = make_deps(tmp_path, load_settings=read_stored, save_settings=update_config)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        _field(screen, "model").value = "b"
        await _save(pilot)
        assert app.screen is screen
        assert _error(screen).startswith("Could not save settings: ")
    assert path.read_bytes() == before


async def test_real_save_keeps_keys_the_form_does_not_change(tmp_path):
    path = _write_config("model: a\nwhisper_device: mps\nmy_own_key: 7\n")
    deps = make_deps(tmp_path, load_settings=read_stored, save_settings=update_config)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        _field(screen, "model").value = "b"
        await _save(pilot)
        assert isinstance(app.screen, HomeScreen), _error(screen)
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "model": "b", "whisper_device": "mps", "my_own_key": 7,
    }


async def test_real_invalid_stored_value_blocks_saving_until_corrected(tmp_path):
    path = _write_config("model: a\nsample_rate: fast\n")
    deps = make_deps(tmp_path, load_settings=read_stored, save_settings=update_config)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        _field(screen, "model").value = "b"
        await _save(pilot)
        assert app.screen is screen and "sample_rate" in _error(screen)
        _field(screen, "sample_rate").value = "48000"
        await _save(pilot)
        assert isinstance(app.screen, HomeScreen)
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"model": "b", "sample_rate": 48000}
