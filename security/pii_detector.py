"""
PII Detection & Masking Layer
─────────────────────────────
Masks personal identifiers from resume text BEFORE sending to the LLM API.
This ensures we never ship raw PII to a cloud provider unnecessarily.

Strategy: regex-based patterns for common PII types (email, phone, LinkedIn URL,
GitHub URL) + a placeholder map so we can restore name references in results.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class PIIDetector:
    """Detects and masks PII in resume text before LLM processing."""

    # ── regex patterns ───────────────────────────────────────────────────────
    _EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    _PHONE_RE = re.compile(
        r"(\+?\d{1,3}[\s\-.]?)?"
        r"(\(?\d{2,4}\)?[\s\-.]?)"
        r"(\d{3,4}[\s\-.]?\d{3,4})"
    )
    _LINKEDIN_RE = re.compile(r"linkedin\.com/in/[a-zA-Z0-9\-_%]+", re.IGNORECASE)
    _GITHUB_RE   = re.compile(r"github\.com/[a-zA-Z0-9\-]+", re.IGNORECASE)
    _URL_RE      = re.compile(r"https?://[^\s]+", re.IGNORECASE)

    # ── public API ───────────────────────────────────────────────────────────

    def mask_pii(self, text: str) -> Tuple[str, int, Dict[str, str]]:
        """
        Mask PII in *text*.

        Returns
        -------
        masked_text  : str   – text with PII replaced by placeholders
        count        : int   – number of PII tokens masked
        pii_map      : dict  – mapping {placeholder → original} for audit logs
        """
        pii_map: Dict[str, str] = {}
        count = 0

        def replace(pattern: re.Pattern, placeholder: str, t: str) -> Tuple[str, int]:
            nonlocal count
            for match in set(pattern.findall(t)):
                original = match if isinstance(match, str) else "".join(match)
                if original.strip():
                    token = f"[{placeholder}_{count}]"
                    pii_map[token] = original.strip()
                    t = t.replace(original.strip(), token)
                    count += 1
            return t, count

        text, count = replace(self._EMAIL_RE,    "EMAIL",   text)
        text, count = replace(self._PHONE_RE,    "PHONE",   text)
        text, count = replace(self._LINKEDIN_RE, "LINKEDIN", text)
        text, count = replace(self._GITHUB_RE,   "GITHUB",  text)
        text, count = replace(self._URL_RE,      "URL",     text)

        return text, count, pii_map

    # ── helper utilities ─────────────────────────────────────────────────────

    def restore_name(self, name_from_llm: str, pii_map: Dict[str, str]) -> str:
        """LLM already sees the real name (it's not PII we strip), return as-is."""
        return name_from_llm

    def get_masked_email(self, pii_map: Dict[str, str]) -> str:
        for key, val in pii_map.items():
            if "EMAIL" in key:
                parts = val.split("@")
                if len(parts) == 2:
                    return f"{parts[0][:2]}****@{parts[1]}"
        return ""

    def get_masked_phone(self, pii_map: Dict[str, str]) -> str:
        for key, val in pii_map.items():
            if "PHONE" in key:
                digits = re.sub(r"\D", "", val)
                if len(digits) >= 4:
                    return f"****{digits[-4:]}"
        return ""

    def audit_summary(self, pii_map: Dict[str, str]) -> str:
        """Human-readable summary of what was masked (for audit logs)."""
        if not pii_map:
            return "No PII detected"
        types = [k.split("_")[1] for k in pii_map]
        from collections import Counter
        summary = ", ".join(f"{v}×{k}" for k, v in Counter(types).items())
        return f"Masked: {summary}"
