"""Obsidian export module for TinySteno."""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import yaml

if TYPE_CHECKING:
    from tinysteno.personas import Persona


class ObsidianExporter:
    """Export meeting data to Obsidian vault as Markdown."""

    def __init__(
        self,
        vault_path: str,
        output_folder: str = "meetings",
        tags: Optional[List[str]] = None,
    ):
        self.vault_path = Path(vault_path)
        self.output_folder = output_folder
        self.tags = tags or ["meeting"]
        self.vault_path.mkdir(parents=True, exist_ok=True)
        if not os.access(self.vault_path, os.W_OK):
            raise PermissionError(f"Obsidian vault is not writable: {self.vault_path}")

    def export_meeting(
        self,
        data: dict,
        title: str,
        timestamp: datetime,
        transcript: str = "",
    ) -> str:
        """Export meeting summary to markdown file."""
        meetings_dir = self.vault_path / self.output_folder
        meetings_dir.mkdir(parents=True, exist_ok=True)

        filename = self._sanitize_filename(
            f"{title} ({timestamp.strftime('%Y-%m-%d')})"
        )
        filepath = meetings_dir / f"{filename}.md"

        frontmatter = self._generate_meeting_frontmatter(data, timestamp)
        body = self._generate_meeting_body(data, title, timestamp, transcript)

        content = (
            f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}"
        )

        filepath.write_text(content)
        return str(filepath)

    def export_transcript(
        self,
        text: str,
        meeting_link: str,
        timestamp: datetime,
    ) -> str:
        """Export transcript to markdown file."""
        transcripts_dir = self.vault_path / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        title = Path(meeting_link).stem
        filename = self._sanitize_filename(
            f"{title} ({timestamp.strftime('%Y-%m-%d')})"
        )
        filepath = transcripts_dir / f"{filename}.md"

        frontmatter = self._generate_transcript_frontmatter(meeting_link, timestamp)
        body = self._generate_transcript_body(text, title, timestamp)

        content = (
            f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{body}"
        )

        filepath.write_text(content)
        return str(filepath)

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
        from jinja2 import Environment, BaseLoader, TemplateError

        meetings_dir = self.vault_path / self.output_folder
        meetings_dir.mkdir(parents=True, exist_ok=True)

        # Context: metadata first, then data fields (data overwrites on collision)
        context = {**metadata, **data}

        try:
            env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
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

    def _generate_meeting_frontmatter(
        self,
        data: dict,
        timestamp: datetime,
    ) -> dict:
        duration = data.get("duration_seconds", 0)
        duration_str = f"{duration // 60}m" if duration else "0m"

        return {
            "created": timestamp.isoformat(),
            "type": "meeting",
            "tags": self.tags,
            "duration": duration_str,
            "participants": data.get("participants", []),
        }

    def _generate_transcript_frontmatter(
        self,
        meeting_link: str,
        timestamp: datetime,
    ) -> dict:
        transcript_link = f"[[transcripts/{Path(meeting_link).stem}]]"
        return {
            "created": timestamp.isoformat(),
            "type": "transcript",
            "meeting": transcript_link,
        }

    def _generate_meeting_body(
        self,
        data: dict,
        title: str,
        timestamp: datetime,
        transcript: str = "",
    ) -> str:
        lines = [f"# {title}", ""]

        overview = data.get("overview", "")
        if overview:
            lines.extend(["## Overview", overview, ""])

        participants = data.get("participants", [])
        if participants:
            lines.append("## Participants")
            lines.extend([f"- {p}" for p in participants])
            lines.append("")

        key_points = data.get("key_points", [])
        if key_points:
            lines.append("## Key Points")
            lines.extend([f"{i}. {p}" for i, p in enumerate(key_points, 1)])
            lines.append("")

        action_items = data.get("action_items", [])
        if action_items:
            lines.append("## Action Items")
            for item in action_items:
                task = item.get("task", "")
                assignee = item.get("assignee", "")
                lines.append(f"- [ ] {task}{f' → @{assignee}' if assignee else ''}")
            lines.append("")

        if transcript:
            lines.extend(["## Transcript", "```", transcript, "```", ""])

        return "\n".join(lines).rstrip()

    def _generate_transcript_body(
        self,
        text: str,
        title: str,
        timestamp: datetime,
    ) -> str:
        lines = [f"# Transcript: {title}", ""]
        lines.append(text if text else "(No transcript)")
        return "\n".join(lines)

    def _sanitize_filename(self, name: str) -> str:
        import re

        name = re.sub(r'[:\\/\*\?"\|<>]', "-", name)
        name = re.sub(r"-+", "-", name)
        return name.strip("-")
