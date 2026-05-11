"""
Embedding Utilities — Semantic Skill Matching
──────────────────────────────────────────────
Uses sentence-transformers (all-MiniLM-L6-v2) to compute cosine similarity
between the JD's required skills and the candidate's extracted skills.

This catches semantic matches that BM25 misses:
  JD requires "vector database"  →  candidate has "Qdrant", "FAISS" → high similarity
  JD requires "LLM integration"  →  candidate has "LangChain", "Anthropic SDK"

Model is loaded once and cached to avoid repeated disk I/O.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """Load model once and cache it for the process lifetime."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer model (first run may take ~30s)…")
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning("sentence-transformers not installed; embeddings disabled.")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_skill_embedding_similarity(
    jd_skills: List[str],
    candidate_skills: List[str],
) -> float:
    """
    Compute semantic similarity between two skill lists.

    Algorithm:
    1. Encode each JD skill into a vector.
    2. Encode the entire candidate skill list into a single vector (mean pooling).
    3. For each JD skill, find the max cosine similarity against all candidate skill vectors.
    4. Return the mean of those per-JD-skill max similarities.

    Returns a score in [0, 1].
    """
    model = _load_model()
    if model is None or not jd_skills or not candidate_skills:
        return 0.5  # neutral fallback

    try:
        jd_embeddings = model.encode(jd_skills, convert_to_numpy=True)
        cand_embeddings = model.encode(candidate_skills, convert_to_numpy=True)

        per_skill_max = []
        for jd_vec in jd_embeddings:
            sims = [cosine_similarity(jd_vec, cand_vec) for cand_vec in cand_embeddings]
            per_skill_max.append(max(sims))

        return round(float(np.mean(per_skill_max)), 4)

    except Exception as e:
        logger.error("Embedding similarity failed: %s", e)
        return 0.5


def blend_skill_score(bm25_overlap: float, embedding_sim: float) -> float:
    """
    Blend BM25 and embedding similarity into a final normalised skills score.

    Weights:  40% BM25 (exact keyword match)  +  60% embeddings (semantic match)
    Returns value in [0, 1] — multiply by 10 for rubric score.
    """
    blended = 0.40 * bm25_overlap + 0.60 * embedding_sim
    return round(min(blended, 1.0), 4)
