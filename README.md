# HR Shortlisting Agent 🎯

> **AI Enablement Internship — Task 1 Submission**  
> Two-path agentic architecture · Hybrid BM25 + Semantic Embeddings + LLM scoring · Production-grade security · Human-in-the-loop

---

## Overview

This agent assists HR teams in evaluating candidates efficiently. It ingests a Job Description (JD) alongside a batch of resumes (PDF/DOCX), then produces a ranked shortlist with a transparent scoring rubric explaining every score — across all 5 mandatory dimensions.

**What makes this submission different from a basic LLM wrapper:**

- **Two-path architecture** — mirrors production agentic system design. BM25 pre-screening filters obvious mismatches at zero LLM cost before the expensive semantic path runs.
- **Hybrid scoring signal** — BM25 keyword overlap + sentence-transformer embeddings are computed first, giving the LLM an anchor score and reducing hallucination in the skills dimension.
- **Production security layer** — PII masking before any cloud API call, prompt injection detection with line-level quarantine, input sanitisation, and structured Pydantic output validation throughout.
- **Ethical AI flag** — demographic signal detection for bias-aware human review.
- **Full observability** — per-candidate API cost tracking, processing time, BM25/embedding scores, PII mask count.
- **Human-in-the-loop override** — audit-logged score adjustments with mandatory reason field.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       INPUT LAYER                                │
│  Job Description (TXT/PDF)  +  Resumes (PDF/DOCX)               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYER                                 │
│  1. PII Masking (email, phone, LinkedIn, GitHub, URLs)           │
│     → Masked text sent to LLM; originals stay local              │
│  2. Prompt Injection Sanitisation                                │
│     → Line-level detection + quarantine of injection attempts    │
│  3. Input fence wrapping (data vs instruction boundary)          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
           ┌──────────┼──────────────────┐
           │          │                  │
           ▼          ▼                  │
    ┌─────────────────────────┐          │
    │      JD PARSER          │          │
    │  Claude claude-sonnet-4 → Pydantic │
    │  JDRequirements (strict  JSON)     │
    └────────────┬────────────┘          │
                 │                       │
                 │        ┌──────────────┘
                 │        │  For each resume:
                 │        ▼
                 │  ┌─────────────────────────────────────────────┐
                 │  │           RESUME PARSER                      │
                 │  │  PyMuPDF/python-docx → text extraction       │
                 │  │  Claude claude-sonnet-4 → CandidateProfile (Pydantic)   │
                 │  └─────────────────────┬───────────────────────┘
                 │                        │
                 │                        ▼
                 │  ┌─────────────────────────────────────────────┐
                 │  │         FAST PATH (BM25)                     │
                 │  │  rank-bm25 keyword overlap check             │
                 │  │                                              │
                 │  │  overlap < 15%  →  instant No Hire           │
                 │  │                   zero LLM cost ⚡            │
                 │  │  overlap ≥ 15%  →  continue to full path     │
                 │  └─────────────────────┬───────────────────────┘
                 │                        │ (≥15% only)
                 │                        ▼
                 │  ┌─────────────────────────────────────────────┐
                 │  │         FULL PATH                            │
                 │  │  Embedding similarity (all-MiniLM-L6-v2)    │
                 │  │  BM25 + embedding → hybrid_score anchor      │
                 │  │                                              │
                 │  │  Claude claude-sonnet-4 → 5-dimension scores │
                 │  │  Pydantic validation on every output         │
                 │  └─────────────────────┬───────────────────────┘
                 │                        │
                 │                        ▼
                 │  ┌─────────────────────────────────────────────┐
                 │  │      BIAS DETECTION + RESULT ASSEMBLY        │
                 │  │  Demographic signal flag                     │
                 │  │  Weighted total computation                  │
                 │  │  Recommendation derivation                   │
                 │  └─────────────────────┬───────────────────────┘
                 │                        │
                 └────────────────────────┘
                                          │
                                          ▼
                 ┌────────────────────────────────────────────────┐
                 │           RANKED SHORTLIST REPORT               │
                 │   HTML (visual) + JSON (API-ready)              │
                 └────────────────────────────────────────────────┘
                                          │
                                          ▼
                 ┌────────────────────────────────────────────────┐
                 │         HUMAN-IN-THE-LOOP OVERRIDE              │
                 │   HR adjusts scores with mandatory reason       │
                 │   All overrides audit-logged                    │
                 └────────────────────────────────────────────────┘
```

---

## Tech Stack & Decision Log

### LLM: Claude claude-sonnet-4 (`claude-sonnet-4-20250514`)

| Factor | Decision |
|---|---|
| **Model** | Claude claude-sonnet-4 (`claude-sonnet-4-20250514`) |
| **Provider** | Anthropic |
| **Why this over GPT-4o** | Native JSON-mode reliability; strong instruction-following for structured rubric output; competitive cost |
| **Context window** | 200K tokens — handles long resumes and JDs with ease |
| **Tool calling** | Not needed for this task (LLM used for parsing + scoring only) |
| **Cost** | $3/M input, $15/M output — mitigated by fast-path filtering |

### Agent Framework: Custom Two-Path Orchestrator (no heavyweight framework)

Deliberate choice not to use LangChain/CrewAI for the core loop:
- **Transparency** — the pipeline is explicit Python; evaluators can trace every step
- **No abstraction overhead** — direct Anthropic SDK calls with full control over prompts
- **Pydantic-native** — output validation is cleaner without LangChain's output parsers
- The two-path pattern (fast path + full path) is borrowed from production agentic system design

### Embeddings: `all-MiniLM-L6-v2` (sentence-transformers)
- Lightweight, runs locally, no API cost
- Sufficient for skill-level semantic matching
- Loaded once and LRU-cached for the process lifetime

### Resume Parsing: PyMuPDF + python-docx + LLM extraction
- PyMuPDF handles multi-column PDFs better than pdfplumber
- LLM extraction used for structured fields (skills, projects) because regex fails on varied resume formats

---

## Scoring Rubric

| Dimension | Weight | 0 – Poor | 5 – Average | 10 – Excellent |
|---|---|---|---|---|
| Skills Match | **30%** | <30% skills match | 50–70% match | >85% match |
| Experience Relevance | **25%** | Unrelated domain | Adjacent domain | Exact domain & seniority |
| Education & Certs | **15%** | Doesn't meet minimum | Meets minimum | Exceeds + extra certs |
| Project / Portfolio | **20%** | No evidence | 1–2 generic projects | Strong relevant portfolio |
| Communication Quality | **10%** | Poor structure | Adequate clarity | Crisp, structured, impactful |

Recommendation thresholds: **≥7.5** = Strong Hire · **≥6.0** = Hire · **≥4.0** = Maybe · **<4.0** = No Hire

---

## Security Mitigations

| Risk | Mitigation | Implementation |
|---|---|---|
| **Prompt Injection** | Regex-based detection of 10+ injection patterns; flagged lines quarantined and replaced; text fenced with data-boundary markers | `security/sanitizer.py` — `InputSanitizer.sanitize()` |
| **PII in LLM Calls** | Email, phone, LinkedIn, GitHub, URLs masked with placeholders before any API call; originals never leave local process | `security/pii_detector.py` — `PIIDetector.mask_pii()` |
| **API Key Exposure** | `python-dotenv` + `.env.example`; `.env` in `.gitignore`; env var only (`ANTHROPIC_API_KEY`); never hardcoded | `.env.example`, `python-dotenv` |
| **Hallucinated Scores** | Pydantic v2 validation on all LLM outputs; `ge=0, le=10` bounds on every score; evidence field forces model to cite resume content | `core/models.py` — `DimensionScore` |
| **Unauthorised Access** | API key required in sidebar before any pipeline run; no unauthenticated endpoints exposed | `app.py` — key check before `run_btn` |
| **Bias / Demographic** | Demographic signal detector flags candidates for blind human review without blocking pipeline | `core/orchestrator.py` — `_check_bias_signals()` |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Anthropic API key ([get one here](https://console.anthropic.com))

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/hr-shortlisting-agent
cd hr-shortlisting-agent

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here

# 5. Run the tests to verify everything works
python -m pytest tests/ -v

# 6a. Launch the Streamlit UI
streamlit run app.py

# 6b. OR run the CLI
python run.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/
```

---

## Running Tests

```bash
python -m pytest tests/ -v

# Expected output (no LLM calls required — all unit tests are local):
# tests/test_agent.py::TestPIIDetector::test_masks_email               PASSED
# tests/test_agent.py::TestPIIDetector::test_masks_phone               PASSED
# tests/test_agent.py::TestInputSanitizer::test_detects_ignore_instructions  PASSED
# tests/test_agent.py::TestInputSanitizer::test_clean_resume_not_flagged     PASSED
# tests/test_agent.py::TestFastPath::test_high_overlap_not_rejected          PASSED
# tests/test_agent.py::TestFastPath::test_zero_match_rejected                PASSED
# ... (21 tests total)
```

> All unit tests run without an API key — they test the security layer, fast path, and Pydantic models locally.

---

## Project Structure

```
hr-shortlisting-agent/
├── app.py                        # Streamlit UI
├── run.py                        # CLI entry point
├── requirements.txt
├── .env.example
│
├── core/
│   ├── models.py                 # Pydantic data models (single source of truth)
│   ├── jd_parser.py              # JD → JDRequirements (LLM)
│   ├── resume_parser.py          # PDF/DOCX → CandidateProfile (LLM)
│   ├── fast_path.py              # BM25 pre-screening (zero LLM cost)
│   ├── embeddings.py             # Semantic skill similarity (local model)
│   ├── scoring_agent.py          # 5-dimension LLM scorer
│   └── orchestrator.py           # Main pipeline coordinator
│
├── security/
│   ├── pii_detector.py           # PII masking before LLM calls
│   └── sanitizer.py              # Prompt injection detection + sanitisation
│
├── output/
│   ├── report_generator.py       # HTML + JSON report export
│   └── templates/report.html     # Jinja2 report template (dark theme)
│
├── tests/
│   └── test_agent.py             # 21 unit tests (no API key required)
│
└── sample_data/
    ├── sample_jd.txt             # AI Enablement Intern JD
    └── resumes/                  # 5 test resumes (mix of strong/partial/weak + injection test)
```

---

## Sample Output

The HTML report includes:
- Summary stats bar (total / shortlisted / maybe / rejected / fast-screened / API cost)
- Per-candidate score grid (5 dimensions with colour-coded bars)
- Dimension justifications with evidence citations from the resume
- Skill gaps + key strengths
- Fast-path, bias, and override banners where applicable
- Full observability metadata (time, cost, BM25 score, embedding similarity, PII count)

---

## Design Decisions & Trade-offs

**Why two paths instead of scoring every candidate with the LLM?**  
In a real HR batch of 200+ resumes, ~30% will be completely off-domain (wrong field, wrong level). Running a full LLM scoring call on a civil engineer applying for an AI role wastes tokens and time. The BM25 fast path catches these at <100ms and zero API cost. The threshold (15%) is deliberately low to avoid false rejections — it only catches candidates with essentially zero skill overlap.

**Why not use LangChain agents?**  
For a deterministic pipeline with known steps (parse → screen → score → rank → report), a ReAct loop adds complexity without benefit. Direct SDK calls with Pydantic validation are more transparent and debuggable. If the pipeline were extended to handle multi-turn HR queries or tool use (e.g. "look up this candidate on LinkedIn"), an agent framework would make sense.

**Why mask PII before the LLM call?**  
Even though the LLM doesn't retain conversation history, sending raw PII to cloud APIs unnecessarily violates data minimisation principles (GDPR Article 5). The masked text contains all the information the scorer needs (skills, experience, projects) without transmitting contact details.

---

## Potential Extensions
- LinkedIn profile JSON ingestion (RapidAPI scraper)
- LangSmith / Langfuse tracing for LLM observability
- SQLite audit database for persistent override logs
- Batch caching (LangChain SQLite cache) to avoid re-scoring identical resumes
- Confidence-weighted re-scoring for low-confidence dimension scores

---