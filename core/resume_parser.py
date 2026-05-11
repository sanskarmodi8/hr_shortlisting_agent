"""
Resume Parser
─────────────
1. Extracts raw text from PDF or DOCX.
2. Runs PII masking + prompt injection sanitisation.
3. Calls LLM to extract a structured CandidateProfile.

Security note: we send only the masked + sanitised text to the LLM.
The original text never leaves the local process in plain form.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import anthropic
import fitz  # PyMuPDF
from docx import Document

from .models import CandidateProfile
from security.pii_detector import PIIDetector
from security.sanitizer import InputSanitizer

logger = logging.getLogger(__name__)

_pii = PIIDetector()
_san = InputSanitizer()
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages).strip()


def _extract_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── LLM profile extraction ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert resume parser for technical roles.

Extract structured information from the resume text below.
Return ONLY valid JSON — no markdown, no extra text.

Output schema (strict):
{
  "name": "full name or 'Unknown' if not found",
  "extracted_skills": ["list every specific technical skill mentioned"],
  "experience_years": 0.0,
  "experience_domains": ["primary domain(s) of work experience"],
  "education": "highest degree + field + institution",
  "certifications": ["cert name"],
  "projects": ["Project Title: one-line description of what was built and tech used"]
}

Rules:
- extracted_skills must be granular: "PyTorch", "FastAPI", not "deep learning frameworks".
- If a skill is implied by a project (e.g. project uses LangChain), include that skill.
- experience_years: sum of all work/internship durations in years (0 if student with no experience).
- projects: include personal, academic, and internship projects.
"""


def parse_resume(file_path: str) -> tuple[CandidateProfile, str]:
    """
    Parse a resume file.

    Returns
    -------
    profile      : CandidateProfile   – structured candidate data
    masked_text  : str                – the sanitised text sent to LLM (for audit)
    """
    raw_text = extract_text(file_path)

    # ── Security pipeline ──────────────────────────────────────────────────
    masked_text, pii_count, pii_map = _pii.mask_pii(raw_text)
    clean_text, was_injected, flagged = _san.sanitize(masked_text)
    fenced_text = _san.wrap_in_fence(clean_text)

    if was_injected:
        logger.warning(
            "Injection attempt in %s: %s", Path(file_path).name, flagged[:3]
        )

    # ── LLM profile extraction ─────────────────────────────────────────────
    client = _get_client()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Extract the candidate profile from this resume:\n\n{fenced_text}",
            }
        ],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Resume parser invalid JSON for %s", file_path)
        raise ValueError(f"LLM returned invalid JSON: {e}") from e

    profile = CandidateProfile(
        name=data.get("name", "Unknown"),
        email_masked=_pii.get_masked_email(pii_map),
        phone_masked=_pii.get_masked_phone(pii_map),
        extracted_skills=data.get("extracted_skills", []),
        experience_years=float(data.get("experience_years", 0)),
        experience_domains=data.get("experience_domains", []),
        education=data.get("education", ""),
        certifications=data.get("certifications", []),
        projects=data.get("projects", []),
        raw_text_length=len(raw_text),
        pii_fields_masked=pii_count,
    )

    return profile, clean_text
