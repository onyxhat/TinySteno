"""Obsidian export module for TinySteno."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinysteno.personas import Persona


class ObsidianExporter:
    """Export meeting data to Obsidian vault as Markdown."""

    def __init__(
        self,
        vault_path: str,
        output_folder: str = "meetings",
    ):
        self.vault_path = Path(vault_path)
        self.output_folder = output_folder
        self.vault_path.mkdir(parents=True, exist_ok=True)
        if not os.access(self.vault_path, os.W_OK):
            raise PermissionError(f"Obsidian vault is not writable: {self.vault_path}")

    # pylint: disable=too-many-locals
    def export(self, data: dict, persona: "Persona", metadata: dict) -> str:
        """Export a note using the persona's Jinja2 template.

        Args:
            data: Dict of schema field values from orchestrator.
            persona: The Persona whose template to render.
            metadata: Dict with keys: title, date, duration, transcript, detected_language.

        Returns:
            Absolute path to the written file.

        Raises:
            RuntimeError: If the Jinja2 template fails to render.
        """
        from jinja2 import Environment, BaseLoader, TemplateError, StrictUndefined

        meetings_dir = self.vault_path / self.output_folder
        meetings_dir.mkdir(parents=True, exist_ok=True)

        # Context: metadata first, then data fields (data overwrites on collision)
        context = {**metadata, **data}

        # Build deduplicated tag list: persona static tags first, then generated tags
        static = list(persona.tags)
        generated = metadata.get("generated_tags") or []
        context["tags"] = list(dict.fromkeys(static + generated))

        try:
            env = Environment(
                loader=BaseLoader(), keep_trailing_newline=True, undefined=StrictUndefined
            )
            tmpl = env.from_string(persona.template)
            content = tmpl.render(**context)
        except TemplateError as e:
            raise RuntimeError(
                f"Template render error in {persona.template_path}: {e}"
            ) from e

        date_prefix = metadata["date"][:10]
        filename = self._sanitize_filename(f"{metadata['title']} ({date_prefix})")
        filepath = meetings_dir / f"{filename}.md"
        filepath.write_text(content)
        return str(filepath)

    def _sanitize_filename(self, name: str) -> str:
        import re

        name = re.sub(r'[:\\/\*\?"\|<>]', "-", name)
        name = re.sub(r"-+", "-", name)
        return name.strip("-")
