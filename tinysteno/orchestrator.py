"""LLM orchestration module for TinySteno."""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Optional

from openai import OpenAI

if TYPE_CHECKING:
    from tinysteno.personas import Persona

logger = logging.getLogger(__name__)

_TRANSCRIPT_MAX_CHARS = 15000
_TITLE_MAX_WORDS = 6
_TITLE_MAX_OVERVIEW_CHARS = 500
_LLM_TIMEOUT_SECONDS = 120.0


class Orchestrator:
    """Orchestrate LLM calls for transcript extraction and title generation."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=_LLM_TIMEOUT_SECONDS,
        )

    def summarize(self, transcript: str, persona: "Persona") -> dict:
        """Generate summary from transcript using persona's prompt and schema.

        Returns a dict with all schema field names as keys.
        Missing/invalid fields from the LLM default to "" (string) or [] (list).
        """
        user_message = self._build_user_message(transcript, persona)
        parsed: dict = {}

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": persona.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw_json = response.choices[0].message.content
            parsed = self._parse_json(raw_json)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")

        result: dict = {}
        for field_name, field_def in persona.schema.items():
            value = parsed.get(field_name)
            if field_def["type"] == "string":
                if isinstance(value, str):
                    result[field_name] = value
                else:
                    if value is not None:
                        logger.warning(
                            f"Field '{field_name}': expected string, got "
                            f"{type(value).__name__}; using empty string"
                        )
                    result[field_name] = ""
            else:  # list
                if isinstance(value, list):
                    result[field_name] = [str(item) for item in value]
                else:
                    if value is not None:
                        logger.warning(
                            f"Field '{field_name}': expected list, got "
                            f"{type(value).__name__}; using empty list"
                        )
                    result[field_name] = []

        return result

    def generate_title(self, field_value: str) -> Optional[str]:
        """Generate a short title from a field value string.

        Returns the generated title string, or None if the LLM call fails.
        """
        prompt = self._title_prompt(field_value)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Generate a brief title (max {_TITLE_MAX_WORDS} words, no special characters).",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            title = response.choices[0].message.content.strip()
            return self._clean_title(title)
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            return None

    def _build_user_message(self, transcript: str, persona: "Persona") -> str:
        """Build the LLM user message: JSON format instruction + transcript."""
        # Build JSON example showing expected shape
        json_example: dict = {}
        for field_name, field_def in persona.schema.items():
            if field_def["type"] == "string":
                json_example[field_name] = "string"
            else:
                json_example[field_name] = ["string", "..."]

        json_str = json.dumps(json_example, indent=2)

        # Build field descriptions
        descriptions = "\n".join(
            f"- {name}: {defn['description']}"
            for name, defn in persona.schema.items()
        )

        if len(transcript) > _TRANSCRIPT_MAX_CHARS:
            logger.warning(
                f"Transcript truncated from {len(transcript)} to "
                f"{_TRANSCRIPT_MAX_CHARS} characters for summarization."
            )
        truncated = transcript[:_TRANSCRIPT_MAX_CHARS]

        return (
            f"Return a JSON object with exactly these fields:\n"
            f"{json_str}\n\n"
            f"Field descriptions:\n{descriptions}\n\n"
            f"Transcript:\n{truncated}"
        )

    def _title_prompt(self, overview: str) -> str:
        return (
            f"Based on this content, generate a 3-6 word title "
            f"(no special chars): {overview[:_TITLE_MAX_OVERVIEW_CHARS]}"
        )

    def _parse_json(self, response: str) -> dict:
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", response.strip())
        stripped = re.sub(r"\n?```\s*$", "", stripped).strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            json_match = self._extract_json(stripped)
            if json_match:
                try:
                    return json.loads(json_match)
                except json.JSONDecodeError:
                    pass
            logger.warning(
                "Failed to parse LLM response as JSON; summary fields will be empty."
            )
            logger.debug("Raw LLM response: %r", response[:500])
            return {}

    def _extract_json(self, text: str) -> Optional[str]:
        # Greedy match: first { to last } — handles } inside string values
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else None

    def _clean_title(self, title: str) -> str:
        title = title.replace('"', "").replace("'", "").replace(":", "")
        title = re.sub(r"[^\w\s\-\.\(\)]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        words = title.split()[:_TITLE_MAX_WORDS]
        return " ".join(words)
