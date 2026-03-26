"""Tests for updated ObsidianExporter."""
from pathlib import Path

import pytest

from tinysteno.obsidian import ObsidianExporter
from tinysteno.personas import Persona

# pylint: disable=missing-function-docstring

def _make_persona(template: str, schema: dict | None = None, tags: list | None = None) -> Persona:
    return Persona(
        slug="test",
        name="Test",
        description="desc",
        schema=schema or {"summary": {"type": "string", "description": "x"}},
        system_prompt="x",
        template=template,
        template_path=Path("/fake/test/template.md"),
        tags=tags or [],
    )


def _make_exporter(tmp_path: Path) -> ObsidianExporter:
    return ObsidianExporter(
        vault_path=str(tmp_path),
        output_folder="notes",
    )


def _make_metadata(title: str = "Test Note", date: str = "2026-03-18 10:00") -> dict:
    return {
        "title": title,
        "date": date,
        "duration": "00:30:00",
        "transcript": "Hello world transcript.",
        "detected_language": "en",
    }


# --- export ---

def test_export_creates_file(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("# {{ title }}\n{{ summary }}")
    metadata = _make_metadata()
    data = {"summary": "Great meeting."}

    path = exporter.export(data, persona, metadata)
    assert Path(path).exists()


def test_export_filename_uses_title_and_date(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("# {{ title }}")
    metadata = _make_metadata(title="My Meeting", date="2026-03-18 14:30")
    data = {}

    path = exporter.export(data, persona, metadata)
    assert "My Meeting (2026-03-18)" in Path(path).name


def test_export_renders_metadata_variables(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("date={{ date }} lang={{ detected_language }}")
    metadata = _make_metadata()
    metadata["detected_language"] = "fr"

    path = exporter.export({}, persona, metadata)
    content = Path(path).read_text(encoding="utf-8")
    assert "date=2026-03-18 10:00" in content
    assert "lang=fr" in content


def test_export_renders_schema_fields(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona(
        "{{ summary }}\n{% for p in points %}- {{ p }}\n{% endfor %}",
        schema={
            "summary": {"type": "string", "description": "x"},
            "points": {"type": "list", "description": "x"},
        }
    )
    metadata = _make_metadata()
    data = {"summary": "Discussed budgets.", "points": ["Point A", "Point B"]}

    path = exporter.export(data, persona, metadata)
    content = Path(path).read_text(encoding="utf-8")
    assert "Discussed budgets." in content
    assert "- Point A" in content
    assert "- Point B" in content


def test_export_transcript_absent_means_not_included(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("# {{ title }}\n{{ summary }}")  # no {{ transcript }}
    metadata = _make_metadata()
    metadata["transcript"] = "SECRET TRANSCRIPT"
    data = {"summary": "hello"}

    path = exporter.export(data, persona, metadata)
    content = Path(path).read_text(encoding="utf-8")
    # transcript variable is available but not referenced in the template
    assert "SECRET TRANSCRIPT" not in content


def test_export_transcript_included_when_referenced(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("{{ transcript }}")
    metadata = _make_metadata()
    metadata["transcript"] = "THE TRANSCRIPT"
    data = {}

    path = exporter.export(data, persona, metadata)
    content = Path(path).read_text(encoding="utf-8")
    assert "THE TRANSCRIPT" in content


def test_export_raises_runtime_error_on_bad_template(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("{% for x in %}broken{% endfor %}")  # invalid Jinja2
    metadata = _make_metadata()

    with pytest.raises(RuntimeError, match="template.md"):
        exporter.export({}, persona, metadata)


def test_export_data_and_metadata_merged_in_context(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("{{ title }} {{ summary }}")
    metadata = _make_metadata(title="The Title")
    data = {"summary": "The Summary"}

    path = exporter.export(data, persona, metadata)
    content = Path(path).read_text(encoding="utf-8")
    assert "The Title" in content
    assert "The Summary" in content


def test_export_file_in_correct_output_folder(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("x")
    path = exporter.export({}, persona, _make_metadata())
    assert "notes" in str(path)


def test_export_tags_combines_static_and_generated(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("{{ tags | join(',') }}", tags=["meeting"])
    metadata = {**_make_metadata(), "generated_tags": ["ai", "weekly"]}

    path = exporter.export({}, persona, metadata)
    content = Path(path).read_text(encoding="utf-8")
    assert "meeting" in content
    assert "ai" in content
    assert "weekly" in content


def test_export_tags_deduplicates(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("{{ tags | join(',') }}", tags=["meeting", "ai"])
    metadata = {**_make_metadata(), "generated_tags": ["ai", "weekly"]}

    path = exporter.export({}, persona, metadata)
    content = Path(path).read_text(encoding="utf-8")
    assert content.count("ai") == 1


def test_export_tags_static_order_preserved(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("{{ tags | join(',') }}", tags=["incident", "rca"])
    metadata = {**_make_metadata(), "generated_tags": ["postmortem"]}

    path = exporter.export({}, persona, metadata)
    content = Path(path).read_text(encoding="utf-8")
    assert content.startswith("incident,rca,postmortem")


def test_export_tags_no_generated_tags(tmp_path):
    exporter = _make_exporter(tmp_path)
    persona = _make_persona("{{ tags | join(',') }}", tags=["sprint"])
    metadata = _make_metadata()  # no generated_tags key

    path = exporter.export({}, persona, metadata)
    content = Path(path).read_text(encoding="utf-8")
    assert content == "sprint"
