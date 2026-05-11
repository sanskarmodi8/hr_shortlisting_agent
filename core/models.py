"""
Core Pydantic models for the HR Shortlisting Agent.
All LLM outputs are validated through these models — zero raw string parsing.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, computed_field


# ─────────────────────────────────────────────
#  Job Description
# ─────────────────────────────────────────────

class JDRequirements(BaseModel):
    title: str
    required_skills: List[str]
    preferred_skills: List[str]
    min_experience_years: float
    education_requirement: str
    domain: str                         # e.g. "AI/ML", "Backend Engineering"
    key_responsibilities: List[str]
    nice_to_have: List[str] = []


# ─────────────────────────────────────────────
#  Candidate Profile (extracted from resume)
# ─────────────────────────────────────────────

class CandidateProfile(BaseModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    email_masked: str = ""              # e.g.  s****@gmail.com
    phone_masked: str = ""              # e.g.  +91-XXXXXX4567
    extracted_skills: List[str] = []
    experience_years: float = 0.0
    experience_domains: List[str] = []
    education: str = ""
    certifications: List[str] = []
    projects: List[str] = []
    raw_text_length: int = 0
    pii_fields_masked: int = 0


# ─────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────

class DimensionScore(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    weight: float                       # e.g. 0.30
    justification: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    evidence: List[str] = []            # direct quotes / proof from resume

    @computed_field  # type: ignore[misc]
    @property
    def weighted_score(self) -> float:
        return round(self.score * self.weight, 3)


class CandidateResult(BaseModel):
    candidate_id: str
    file_name: str
    profile: CandidateProfile

    # Five mandatory rubric dimensions
    skills_match: DimensionScore        # weight 0.30
    experience_relevance: DimensionScore  # weight 0.25
    education_certs: DimensionScore     # weight 0.15
    project_portfolio: DimensionScore   # weight 0.20
    communication_quality: DimensionScore  # weight 0.10

    # Aggregates
    weighted_total: float = 0.0
    recommendation: Literal["Strong Hire", "Hire", "Maybe", "No Hire"] = "Maybe"
    skill_gaps: List[str] = []
    key_strengths: List[str] = []

    # Architecture / observability metadata
    fast_path_screened: bool = False    # True = skipped LLM for this candidate
    bm25_overlap_score: float = 0.0
    embedding_similarity: float = 0.0
    processing_time_ms: float = 0.0
    estimated_cost_usd: float = 0.0

    # Bias detection (ethical AI layer)
    bias_flag: bool = False
    bias_note: str = ""

    # Human-in-the-loop override tracking
    overridden: bool = False
    override_reason: Optional[str] = None
    override_by: Optional[str] = None
    original_total: Optional[float] = None

    timestamp: datetime = Field(default_factory=datetime.now)

    def compute_weighted_total(self) -> float:
        return round(
            self.skills_match.score * 0.30
            + self.experience_relevance.score * 0.25
            + self.education_certs.score * 0.15
            + self.project_portfolio.score * 0.20
            + self.communication_quality.score * 0.10,
            2,
        )

    def derive_recommendation(self) -> Literal["Strong Hire", "Hire", "Maybe", "No Hire"]:
        t = self.weighted_total
        if t >= 7.5:
            return "Strong Hire"
        elif t >= 6.0:
            return "Hire"
        elif t >= 4.0:
            return "Maybe"
        else:
            return "No Hire"


# ─────────────────────────────────────────────
#  Final Report
# ─────────────────────────────────────────────

class ShortlistReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    job_title: str
    jd_requirements: JDRequirements
    generated_at: datetime = Field(default_factory=datetime.now)

    total_candidates: int
    shortlisted: int            # Strong Hire + Hire
    maybe_count: int
    rejected: int
    fast_path_filtered: int     # candidates screened without LLM

    total_estimated_cost_usd: float
    total_processing_time_ms: float

    results: List[CandidateResult]  # sorted by weighted_total desc
