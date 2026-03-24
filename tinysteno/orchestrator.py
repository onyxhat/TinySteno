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

_CHUNK_SIZE_CHARS = 12000
_CHUNK_OVERLAP_CHARS = 500
_TITLE_MAX_WORDS = 6
_TITLE_MAX_OVERVIEW_CHARS = 500
_TAG_MAX_OVERVIEW_CHARS = 500
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

        Splits long transcripts into overlapping chunks, summarizes each chunk,
        then merges the partial results into a single deduplicated output.

        Returns a dict with all schema field names as keys.
        Missing/invalid fields from the LLM default to "" (string) or [] (list).
        """
        chunks = self._chunk_transcript(transcript)

        if len(chunks) == 1:
            return self._extract(self._build_user_message(transcript, persona), persona)

        logger.info(
            "Transcript split into %d chunks (%d total chars) for summarization.",
            len(chunks),
            len(transcript),
        )

        partials = []
        for i, chunk in enumerate(chunks):
            logger.debug("Summarizing chunk %d/%d", i + 1, len(chunks))
            result = self._extract(self._build_user_message(chunk, persona), persona)
            if any(v for v in result.values()):
                partials.append(result)

        if not partials:
            logger.error("All chunk summarizations failed; returning defaults.")
            return self._defaults(persona)

        if len(partials) == 1:
            return partials[0]

        logger.debug("Merging %d partial summaries.", len(partials))
        return self._extract(self._build_merge_message(partials, persona), persona)

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
                        "content": (
                            f"Generate a brief title "
                            f"(max {_TITLE_MAX_WORDS} words, no special characters)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            title = response.choices[0].message.content.strip()
            return self._clean_title(title)
        except Exception as e:
            logger.warning("Title generation failed: %s", e)
            return None

    def generate_tags(self, field_value: str) -> list:
        """Generate tags from a field value string.

        Returns a list of lowercase alphanumeric/underscore tags,
        or an empty list if the LLM call fails.
        """
        prompt = self._tags_prompt(field_value)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate 1-5 single-word tags that categorize the content. "
                            "Return only the tags as a comma-separated list, nothing else. "
                            "Use lowercase letters, digits, and underscores only. "
                            "Prefer single words; use underscores only if needed."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            return self._clean_tags(raw)
        except Exception as e:
            logger.warning("Tag generation failed: %s", e)
            return []

    def _chunk_transcript(self, transcript: str) -> list[str]:
        """Split transcript into overlapping chunks of _CHUNK_SIZE_CHARS.

        Overlap preserves context at chunk boundaries. Returns a single-element
        list if the transcript fits within one chunk.
        """
        if len(transcript) <= _CHUNK_SIZE_CHARS:
            return [transcript]

        chunks = []
        start = 0
        while start < len(transcript):
            end = start + _CHUNK_SIZE_CHARS
            chunks.append(transcript[start:end])
            if end >= len(transcript):
                break
            start = end - _CHUNK_OVERLAP_CHARS
        return chunks

    def _build_user_message(self, transcript: str, persona: "Persona") -> str:
        """Build the LLM user message: JSON format instruction + transcript."""
        json_example: dict = {}
        for field_name, field_def in persona.schema.items():
            if field_def["type"] == "string":
                json_example[field_name] = "string"
            else:
                json_example[field_name] = ["string", "..."]

        json_str = json.dumps(json_example, indent=2)
        descriptions = "\n".join(
            f"- {name}: {defn['description']}"
            for name, defn in persona.schema.items()
        )

        return (
            f"Return a JSON object with exactly these fields:\n"
            f"{json_str}\n\n"
            f"Field descriptions:\n{descriptions}\n\n"
            f"Transcript:\n{transcript}"
        )

    def _build_merge_message(self, partials: list[dict], persona: "Persona") -> str:
        """Build the LLM merge prompt for combining partial chunk results."""
        json_example: dict = {}
        for field_name, field_def in persona.schema.items():
            if field_def["type"] == "string":
                json_example[field_name] = "string"
            else:
                json_example[field_name] = ["string", "..."]

        json_str = json.dumps(json_example, indent=2)
        descriptions = "\n".join(
            f"- {name}: {defn['description']}"
            for name, defn in persona.schema.items()
        )

        return (
            f"The following are partial summaries extracted from sequential sections "
            f"of a single transcript. Merge them into one cohesive JSON result with "
            f"exactly these fields:\n"
            f"{json_str}\n\n"
            f"Field descriptions:\n{descriptions}\n\n"
            f"Merge rules:\n"
            f"- String fields: write a single comprehensive synthesis from all parts.\n"
            f"- List fields: combine all unique items; remove exact duplicates and "
            f"near-duplicates that convey the same point.\n"
            f"- Do not invent information not present in the partial summaries.\n\n"
            f"Partial summaries:\n{json.dumps(partials, indent=2)}"
        )

    def _extract(self, user_message: str, persona: "Persona") -> dict:
        """Call the LLM and validate/normalize the response against the persona schema."""
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
            logger.error("LLM call failed: %s", e)
        return self._validate(parsed, persona)

    def _validate(self, parsed: dict, persona: "Persona") -> dict:
        """Normalize a parsed LLM response against the persona schema."""
        result: dict = {}
        for field_name, field_def in persona.schema.items():
            value = parsed.get(field_name)
            if field_def["type"] == "string":
                if isinstance(value, str):
                    result[field_name] = value
                else:
                    if value is not None:
                        logger.warning(
                            "Field '%s': expected string, got %s; using empty string",
                            field_name,
                            type(value).__name__,
                        )
                    result[field_name] = ""
            else:  # list
                if isinstance(value, list):
                    result[field_name] = [str(item) for item in value]
                else:
                    if value is not None:
                        logger.warning(
                            "Field '%s': expected list, got %s; using empty list",
                            field_name,
                            type(value).__name__,
                        )
                    result[field_name] = []
        return result

    def _defaults(self, persona: "Persona") -> dict:
        """Return default empty values for all schema fields."""
        return {
            name: "" if defn["type"] == "string" else []
            for name, defn in persona.schema.items()
        }

    def _tags_prompt(self, overview: str) -> str:
        return (
            f"Based on this content, generate 1-5 single-word tags "
            f"(lowercase, alphanumeric and underscores only, comma-separated): "
            f"{overview[:_TAG_MAX_OVERVIEW_CHARS]}"
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

    def _clean_tags(self, raw: str) -> list:
        tags = []
        for part in raw.split(","):
            tag = part.strip().lower()
            tag = re.sub(r"\s+", "_", tag)          # spaces → underscores
            tag = re.sub(r"[^a-z0-9_]", "", tag)    # remove other invalid chars
            tag = re.sub(r"_+", "_", tag)            # collapse repeated underscores
            tag = tag.strip("_")                     # trim leading/trailing underscores
            if tag:
                tags.append(tag)
        return tags

    def _clean_title(self, title: str) -> str:
        title = title.replace('"', "").replace("'", "").replace(":", "")
        title = re.sub(r"[^\w\s\-\.\(\)]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        words = title.split()[:_TITLE_MAX_WORDS]
        return " ".join(words)
