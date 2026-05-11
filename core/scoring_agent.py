"""
LLM Scoring Agent
─────────────────
Scores a candidate across all 5 rubric dimensions using Claude claude-sonnet-4-20250514.
Output is validated through Pydantic — no raw string parsing anywhere downstream.

Prompt design principles:
- System prompt defines the role and output contract.
- User prompt injects JD + candidate data as clearly labelled sections.
- We request JSON only, and strip/validate defensively.
- Scores are grounded by asking the model to cite specific evidence from the resume.
"""
from __future__ import annotations

import json
import logging
import os

import anthropic

from .models import CandidateProfile, CandidateResult, DimensionScore, JDRequirements

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None

# Approximate cost for claude-sonnet-4 (input/output per million tokens)
_COST_PER_1M_INPUT  = 3.00   # USD
_COST_PER_1M_OUTPUT = 15.00  # USD


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


_SYSTEM_PROMPT = """\
You are a senior technical recruiter with deep expertise in AI/ML engineering roles.

Your task: evaluate a candidate against a job description using the exact rubric below.
Return ONLY valid JSON — no markdown, no preamble, no commentary.

Rubric dimensions and weights:
  skills_match        (30%): 0 = <30% skills match | 5 = 50-70% match | 10 = >85% match
  experience_relevance (25%): 0 = unrelated domain | 5 = adjacent domain | 10 = exact domain & seniority
  education_certs      (15%): 0 = doesn't meet minimum | 5 = meets minimum | 10 = exceeds + extra certs
  project_portfolio    (20%): 0 = no evidence | 5 = 1-2 generic projects | 10 = strong relevant portfolio
  communication_quality (10%): 0 = poor structure/grammar | 5 = adequate | 10 = crisp, structured, impactful

Scoring rules:
1. Be calibrated: 10/10 is rare. Most good candidates score 6–8.
2. For each dimension, cite 1–3 specific pieces of evidence from the resume.
3. Justify every score in one concise sentence.
4. skill_gaps: list required skills from the JD that the candidate demonstrably lacks.
5. key_strengths: top 3 differentiating strengths of this candidate for this role.

Output schema (strict — do not deviate):
{
  "skills_match":          {"score": 0.0, "justification": "...", "confidence": 0.8, "evidence": ["..."]},
  "experience_relevance":  {"score": 0.0, "justification": "...", "confidence": 0.8, "evidence": ["..."]},
  "education_certs":       {"score": 0.0, "justification": "...", "confidence": 0.8, "evidence": ["..."]},
  "project_portfolio":     {"score": 0.0, "justification": "...", "confidence": 0.8, "evidence": ["..."]},
  "communication_quality": {"score": 0.0, "justification": "...", "confidence": 0.8, "evidence": ["..."]},
  "skill_gaps":            ["missing skill 1", "missing skill 2"],
  "key_strengths":         ["strength 1", "strength 2", "strength 3"]
}
"""


def _build_user_prompt(
    jd: JDRequirements,
    profile: CandidateProfile,
    resume_text: str,
    hybrid_skills_score: float,
) -> str:
    return f"""
JOB REQUIREMENTS
────────────────
Title: {jd.title}
Domain: {jd.domain}
Required Skills: {', '.join(jd.required_skills)}
Preferred Skills: {', '.join(jd.preferred_skills)}
Min Experience: {jd.min_experience_years} years
Education: {jd.education_requirement}
Key Responsibilities: {'; '.join(jd.key_responsibilities[:5])}

CANDIDATE PROFILE (structured)
───────────────────────────────
Name: {profile.name}
Extracted Skills: {', '.join(profile.extracted_skills)}
Experience: {profile.experience_years} years in {', '.join(profile.experience_domains)}
Education: {profile.education}
Certifications: {', '.join(profile.certifications) or 'None listed'}
Projects: {'; '.join(profile.projects[:6])}

[HYBRID SCORING SIGNAL]
BM25 + Embedding skills overlap pre-score: {hybrid_skills_score:.2f}/1.0
(Use this as an anchor for skills_match scoring, but apply your own judgment.)

RESUME FULL TEXT
────────────────
{resume_text[:4000]}
"""


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  / 1_000_000 * _COST_PER_1M_INPUT
        + output_tokens / 1_000_000 * _COST_PER_1M_OUTPUT
    )


def score_candidate(
    jd: JDRequirements,
    profile: CandidateProfile,
    resume_text: str,
    file_name: str,
    hybrid_skills_score: float = 0.5,
    bm25_overlap: float = 0.0,
    embedding_sim: float = 0.0,
) -> CandidateResult:
    """
    Score a candidate using the LLM and return a validated CandidateResult.
    """
    client = _get_client()
    user_prompt = _build_user_prompt(jd, profile, resume_text, hybrid_skills_score)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
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
        logger.error("Scoring agent invalid JSON for %s: %s", file_name, raw[:200])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e

    cost = _estimate_cost(
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    # ── Build dimension scores ────────────────────────────────────────────
    def _dim(key: str, weight: float) -> DimensionScore:
        d = data.get(key, {})
        return DimensionScore(
            score=float(d.get("score", 5)),
            weight=weight,
            justification=d.get("justification", ""),
            confidence=float(d.get("confidence", 0.8)),
            evidence=d.get("evidence", []),
        )

    result = CandidateResult(
        candidate_id=profile.candidate_id,
        file_name=file_name,
        profile=profile,
        skills_match=_dim("skills_match", 0.30),
        experience_relevance=_dim("experience_relevance", 0.25),
        education_certs=_dim("education_certs", 0.15),
        project_portfolio=_dim("project_portfolio", 0.20),
        communication_quality=_dim("communication_quality", 0.10),
        skill_gaps=data.get("skill_gaps", []),
        key_strengths=data.get("key_strengths", []),
        bm25_overlap_score=bm25_overlap,
        embedding_similarity=embedding_sim,
        estimated_cost_usd=cost,
    )

    result.weighted_total = result.compute_weighted_total()
    result.recommendation = result.derive_recommendation()
    return result
