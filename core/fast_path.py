"""
Fast Path — BM25 Keyword Pre-Screening
────────────────────────────────────────
Mirrors the two-path architecture from the Leave Policy Agent:

  Fast path  →  BM25 keyword overlap, zero LLM cost  (<100 ms)
  Full path  →  Embedding similarity + LLM scoring

Decision boundary:
  overlap < REJECT_THRESHOLD  →  instant "No Hire", skip LLM entirely
  overlap ≥ ACCEPT_THRESHOLD  →  flag as likely strong candidate, still run full path
  in between                  →  full path

This cuts LLM API cost significantly on large batches (e.g. 200-resume HR runs).
"""
from __future__ import annotations

import re
import string
from typing import List

from rank_bm25 import BM25Okapi

# ── Thresholds ──────────────────────────────────────────────────────────────
REJECT_THRESHOLD = 0.15   # < 15% overlap → immediate No Hire
ACCEPT_THRESHOLD = 0.60   # ≥ 60% overlap → likely strong, still run full path

# ── Common stopwords (avoid polluting BM25 corpus) ──────────────────────────
_STOPWORDS = {
    "and", "or", "the", "a", "an", "with", "in", "of", "to", "for",
    "on", "at", "is", "are", "be", "as", "by", "from", "have", "has",
    "we", "our", "you", "your", "team", "role", "work", "working",
    "strong", "good", "excellent", "preferred", "required", "must",
    "experience", "knowledge", "understanding", "ability", "skills",
    "using", "use", "able", "will", "can", "may", "should",
}


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in _STOPWORDS and len(w) > 2]


def compute_bm25_overlap(
    jd_skills: List[str],
    candidate_skills: List[str],
    candidate_full_text: str,
) -> float:
    """
    Compute a normalised BM25 overlap score [0, 1].

    Strategy:
    - Build a BM25 corpus from the candidate's skill list + resume text.
    - Query it with all required JD skills.
    - Normalise by the max possible score.
    """
    if not jd_skills:
        return 0.5  # no requirements → neutral score

    # Build corpus: one "document" per candidate skill + one for full text
    corpus_docs = [_tokenise(skill) for skill in candidate_skills]
    corpus_docs.append(_tokenise(candidate_full_text[:3000]))  # first 3k chars of resume

    # Filter empty docs
    corpus_docs = [d for d in corpus_docs if d]
    if not corpus_docs:
        return 0.0

    bm25 = BM25Okapi(corpus_docs)

    # Score each required JD skill against the corpus
    scores = []
    for skill in jd_skills:
        query = _tokenise(skill)
        if query:
            doc_scores = bm25.get_scores(query)
            scores.append(float(max(doc_scores)))

    if not scores:
        return 0.0

    # Normalise: count how many JD skills got a non-zero score
    matched = sum(1 for s in scores if s > 0.01)
    overlap = matched / len(jd_skills)
    return round(overlap, 4)


def should_fast_path_reject(overlap: float) -> bool:
    """Return True if the candidate should be rejected without LLM scoring."""
    return overlap < REJECT_THRESHOLD


def fast_path_score(overlap: float) -> float:
    """
    Map BM25 overlap to a raw skills score (0–10) for fast-path rejected candidates.
    We use a conservative estimate — not a full LLM score.
    """
    return round(overlap * 10 * 0.6, 2)  # max 6/10 from fast path alone
