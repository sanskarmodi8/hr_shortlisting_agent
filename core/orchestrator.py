"""
Pipeline Orchestrator
─────────────────────
Coordinates the full shortlisting pipeline:

  1. Parse JD
  2. For each resume:
     a. Parse PDF/DOCX → CandidateProfile
     b. Fast path: BM25 skill overlap check
        → If overlap < 15%: instant No Hire (zero LLM cost)
        → Else: continue to full path
     c. Full path: embedding similarity + LLM scoring
     d. Bias detection flag
  3. Rank all candidates by weighted_total
  4. Return ShortlistReport

Architecture note:
  The two-path design is borrowed from the Leave Policy Agent pattern.
  ~20-40% of resumes in a typical batch will be fast-path rejected,
  saving meaningful LLM API cost on large runs.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from .jd_parser import parse_jd
from .resume_parser import parse_resume
from .fast_path import compute_bm25_overlap, should_fast_path_reject, fast_path_score
from .embeddings import compute_skill_embedding_similarity, blend_skill_score
from .scoring_agent import score_candidate
from .models import (
    CandidateResult,
    CandidateProfile,
    DimensionScore,
    JDRequirements,
    ShortlistReport,
)

logger = logging.getLogger(__name__)

# ── Bias detection heuristics ─────────────────────────────────────────────────
# Flag if a candidate's name or text contains demographic signals that could
# introduce unconscious bias into manual review.
import re
_BIAS_SIGNAL_PATTERNS = [
    re.compile(r"\b(mr|mrs|ms|miss|dr|prof)\.?\s+", re.IGNORECASE),
    re.compile(r"\b(he/him|she/her|they/them)\b", re.IGNORECASE),
]


def _check_bias_signals(text: str, name: str) -> tuple[bool, str]:
    """Heuristically flag potential demographic signals for human review."""
    for pattern in _BIAS_SIGNAL_PATTERNS:
        if pattern.search(text) or pattern.search(name):
            return True, "Demographic signal detected — ensure blind evaluation."
    return False, ""


def _make_fast_path_result(
    profile: CandidateProfile,
    file_name: str,
    bm25_overlap: float,
    jd: JDRequirements,
) -> CandidateResult:
    """Build a minimal CandidateResult for fast-path rejected candidates."""
    skill_score = fast_path_score(bm25_overlap)
    dim = lambda score, weight, justification: DimensionScore(
        score=score,
        weight=weight,
        justification=justification,
        confidence=0.6,
        evidence=[],
    )
    result = CandidateResult(
        candidate_id=profile.candidate_id,
        file_name=file_name,
        profile=profile,
        skills_match=dim(
            skill_score,
            0.30,
            f"Fast-path: BM25 skill overlap {bm25_overlap:.0%} — below threshold.",
        ),
        experience_relevance=dim(3.0, 0.25, "Not evaluated — fast-path rejection."),
        education_certs=dim(3.0, 0.15, "Not evaluated — fast-path rejection."),
        project_portfolio=dim(3.0, 0.20, "Not evaluated — fast-path rejection."),
        communication_quality=dim(3.0, 0.10, "Not evaluated — fast-path rejection."),
        skill_gaps=jd.required_skills,
        key_strengths=[],
        fast_path_screened=True,
        bm25_overlap_score=bm25_overlap,
    )
    result.weighted_total = result.compute_weighted_total()
    result.recommendation = result.derive_recommendation()
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    jd_text: str,
    resume_paths: List[str],
    progress_callback=None,  # optional Streamlit progress hook
) -> ShortlistReport:
    """
    Run the full HR shortlisting pipeline.

    Parameters
    ----------
    jd_text         : raw job description text
    resume_paths    : list of absolute paths to PDF/DOCX resume files
    progress_callback : optional callable(step: int, total: int, message: str)
    """
    total_steps = len(resume_paths) + 1  # +1 for JD parsing
    step = 0

    def _progress(msg: str):
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(step, total_steps, msg)
        logger.info("[%d/%d] %s", step, total_steps, msg)

    # ── 1. Parse JD ────────────────────────────────────────────────────────
    _progress("Parsing job description…")
    jd: JDRequirements = parse_jd(jd_text)
    logger.info("JD parsed: %s | required skills: %d", jd.title, len(jd.required_skills))

    results: List[CandidateResult] = []
    fast_path_count = 0
    total_cost = 0.0
    total_time = 0.0

    # ── 2. Process each resume ─────────────────────────────────────────────
    for resume_path in resume_paths:
        file_name = Path(resume_path).name
        t_start = time.perf_counter()

        try:
            _progress(f"Processing {file_name}…")

            # Parse resume
            profile, masked_text = parse_resume(resume_path)

            # Fast path check
            bm25_overlap = compute_bm25_overlap(
                jd.required_skills, profile.extracted_skills, masked_text
            )

            if should_fast_path_reject(bm25_overlap):
                logger.info(
                    "FAST PATH REJECT: %s (BM25 overlap %.1f%%)",
                    file_name, bm25_overlap * 100,
                )
                result = _make_fast_path_result(profile, file_name, bm25_overlap, jd)
                fast_path_count += 1

            else:
                # Full path: embedding + LLM
                embedding_sim = compute_skill_embedding_similarity(
                    jd.required_skills, profile.extracted_skills
                )
                hybrid_score = blend_skill_score(bm25_overlap, embedding_sim)

                result = score_candidate(
                    jd=jd,
                    profile=profile,
                    resume_text=masked_text,
                    file_name=file_name,
                    hybrid_skills_score=hybrid_score,
                    bm25_overlap=bm25_overlap,
                    embedding_sim=embedding_sim,
                )

            # Bias detection
            bias_flag, bias_note = _check_bias_signals(masked_text, profile.name)
            result.bias_flag = bias_flag
            result.bias_note = bias_note

        except Exception as e:
            logger.error("Failed to process %s: %s", file_name, e)
            continue

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        result.processing_time_ms = round(elapsed_ms, 1)
        total_cost += result.estimated_cost_usd
        total_time += elapsed_ms
        results.append(result)

    # ── 3. Rank and build report ───────────────────────────────────────────
    results.sort(key=lambda r: r.weighted_total, reverse=True)

    shortlisted = sum(1 for r in results if r.recommendation in ("Strong Hire", "Hire"))
    maybe_count  = sum(1 for r in results if r.recommendation == "Maybe")
    rejected     = sum(1 for r in results if r.recommendation == "No Hire")

    report = ShortlistReport(
        job_title=jd.title,
        jd_requirements=jd,
        total_candidates=len(results),
        shortlisted=shortlisted,
        maybe_count=maybe_count,
        rejected=rejected,
        fast_path_filtered=fast_path_count,
        total_estimated_cost_usd=round(total_cost, 5),
        total_processing_time_ms=round(total_time, 1),
        results=results,
    )

    logger.info(
        "Pipeline complete: %d candidates | %d shortlisted | $%.4f total cost",
        len(results), shortlisted, total_cost,
    )
    return report
