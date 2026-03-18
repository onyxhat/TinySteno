"""LLM summarization module for TinySteno."""

import json
import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_TRANSCRIPT_MAX_CHARS = 15000
_TITLE_MAX_WORDS = 6
_TITLE_MAX_OVERVIEW_CHARS = 500
_LLM_TIMEOUT_SECONDS = 120.0


class MeetingData:
    """Container for meeting summary data."""

    def __init__(
        self,
        overview: str,
        participants: List[str],
        key_points: List[str],
        action_items: List[Dict[str, str]],
    ):
        self.overview = overview
        self.participants = participants
        self.key_points = key_points
        self.action_items = action_items

    def to_dict(self) -> dict:
        return {
            "overview": self.overview,
            "participants": self.participants,
            "key_points": self.key_points,
            "action_items": self.action_items,
        }


class Summarizer:
    """Summarize transcripts using OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=_LLM_TIMEOUT_SECONDS,
        )

    def summarize(self, transcript: str) -> MeetingData:
        """Generate meeting summary from transcript."""
        prompt = self._summary_prompt(transcript)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a meeting assistant that extracts structured information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            raw_json = response.choices[0].message.content
            parsed = self._parse_json(raw_json)

            return MeetingData(
                overview=parsed.get("overview", ""),
                participants=parsed.get("participants", []),
                key_points=parsed.get("key_points", []),
                action_items=parsed.get("action_items", []),
            )
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return MeetingData(
                overview=f"Summary unavailable - {e}",
                participants=[],
                key_points=[],
                action_items=[],
            )

    def generate_title(self, overview: str) -> str:
        """Generate a short title from overview."""
        prompt = self._title_prompt(overview)

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
            logger.warning(f"Title generation failed, using default: {e}")
            return "Meeting"

    def _summary_prompt(self, transcript: str) -> str:
        if len(transcript) > _TRANSCRIPT_MAX_CHARS:
            logger.warning(
                f"Transcript truncated from {len(transcript)} to {_TRANSCRIPT_MAX_CHARS} "
                "characters for summarization."
            )
        return f"""Analyze this meeting transcript and extract the following as JSON:
- overview: A brief summary paragraph
- participants: List of names mentioned
- key_points: List of 3-7 major discussion points
- action_items: List of objects with "task" and "assignee" fields

Transcript:
{transcript[:_TRANSCRIPT_MAX_CHARS]}
"""

    def _title_prompt(self, overview: str) -> str:
        return f"Based on this overview, generate a 3-6 word title (no special chars): {overview[:_TITLE_MAX_OVERVIEW_CHARS]}"

    def _parse_json(self, response: str) -> dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = self._extract_json(response)
            if json_match:
                try:
                    return json.loads(json_match)
                except json.JSONDecodeError:
                    pass
            logger.warning(
                "Failed to parse LLM response as JSON; summary fields will be empty."
            )
            return {}

    def _extract_json(self, text: str) -> Optional[str]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else None

    def _clean_title(self, title: str) -> str:
        title = title.replace('"', "").replace("'", "").replace(":", "")
        title = re.sub(r"[^\w\s\-\.\(\\)]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        words = title.split()[:_TITLE_MAX_WORDS]
        return " ".join(words)
