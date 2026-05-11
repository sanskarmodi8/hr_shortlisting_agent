"""
JD Parser Agent
───────────────
Takes raw job description text and extracts structured requirements
using Claude claude-sonnet-4-20250514 with JSON-mode output and Pydantic validation.
"""
from __future__ import annotations

import json
import logging
import os

import anthropic

from .models import JDRequirements

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


_SYSTEM_PROMPT = """\
You are a senior HR analyst specialising in technical recruiting.

Your task: extract structured requirements from a job description.

Rules:
1. Return ONLY valid JSON — no markdown fences, no preamble, no explanation.
2. Be granular about skills: list individual technologies (e.g. "LangChain", "Pydantic"), not 
   vague categories (e.g. "AI frameworks").
3. If a field cannot be determined, use an empty list or 0 for numeric fields.
4. For domain, pick the single most relevant domain string.

Output schema (strict):
{
  "title": "string",
  "required_skills": ["string"],
  "preferred_skills": ["string"],
  "min_experience_years": 0,
  "education_requirement": "string",
  "domain": "string",
  "key_responsibilities": ["string"],
  "nice_to_have": ["string"]
}
"""


def parse_jd(jd_text: str) -> JDRequirements:
    """
    Parse a job description string into a structured JDRequirements object.

    Raises
    ------
    ValueError  if the LLM returns invalid JSON or fails Pydantic validation.
    """
    client = _get_client()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Parse this job description into structured requirements:\n\n{jd_text}",
            }
        ],
    )

    raw = response.content[0].text.strip()

    # Defensive: strip accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("JD parser received invalid JSON: %s", raw[:200])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e

    try:
        return JDRequirements(**data)
    except Exception as e:
        logger.error("Pydantic validation failed for JD: %s", data)
        raise ValueError(f"JD validation failed: {e}") from e
