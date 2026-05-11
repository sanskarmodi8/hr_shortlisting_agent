"""
HR Shortlisting Agent — Streamlit UI
──────────────────────────────────────
Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import run_pipeline
from core.models import CandidateResult, ShortlistReport
from output.report_generator import generate_html_report, generate_json_report

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Shortlisting Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #6c8ef7, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
  }
  .sub-title { color: #7b7f96; font-size: 14px; margin-bottom: 24px; }

  .metric-card {
    background: #161820; border: 1px solid #2a2d3e; border-radius: 10px;
    padding: 16px 20px; text-align: center;
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; line-height: 1; }
  .metric-card .lbl { font-size: 11px; color: #7b7f96; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

  .rec-badge {
    padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; display: inline-block;
  }
  .stAlert > div { border-radius: 8px; }

  div[data-testid="stExpander"] { border: 1px solid #2a2d3e !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "report" not in st.session_state:
    st.session_state.report: ShortlistReport | None = None
if "overrides" not in st.session_state:
    st.session_state.overrides: dict = {}  # candidate_id → {reason, by, new_total}


# ── Helpers ───────────────────────────────────────────────────────────────────

def rec_color(rec: str) -> str:
    return {
        "Strong Hire": "#4ade80",
        "Hire":        "#6c8ef7",
        "Maybe":       "#fbbf24",
        "No Hire":     "#f87171",
    }.get(rec, "#7b7f96")


def score_bar(score: float, width: int = 120) -> str:
    pct = int(score / 10 * 100)
    color = (
        "#4ade80" if score >= 7.5 else
        "#6c8ef7" if score >= 5.0 else
        "#fbbf24" if score >= 3.0 else
        "#f87171"
    )
    return (
        f'<div style="background:#1e2130;border-radius:4px;height:6px;width:{width}px">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px"></div>'
        f'</div>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Set ANTHROPIC_API_KEY in .env or enter here.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.markdown("### 📋 Job Description")
    jd_input_method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

    jd_text = ""
    if jd_input_method == "Paste text":
        jd_text = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="Senior AI Engineer required...",
        )
    else:
        jd_file = st.file_uploader("Upload JD (TXT or PDF)", type=["txt", "pdf"])
        if jd_file:
            if jd_file.name.endswith(".pdf"):
                import fitz
                doc = fitz.open(stream=jd_file.read(), filetype="pdf")
                jd_text = "\n".join(page.get_text() for page in doc)
            else:
                jd_text = jd_file.read().decode("utf-8", errors="ignore")
            st.success(f"JD loaded ({len(jd_text)} chars)")

    st.divider()
    st.markdown("### 📄 Resumes")
    resume_files = st.file_uploader(
        "Upload resumes (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    st.divider()
    run_btn = st.button("🚀 Run Shortlisting", use_container_width=True, type="primary")
    
    if st.session_state.report:
        st.divider()
        st.markdown("### 📥 Export")
        col1, col2 = st.columns(2)
        with col1:
            html_path = "/tmp/shortlist_report.html"
            generate_html_report(st.session_state.report, html_path)
            with open(html_path, "rb") as f:
                st.download_button("HTML Report", f, "shortlist_report.html", "text/html", use_container_width=True)
        with col2:
            json_path = "/tmp/shortlist_report.json"
            generate_json_report(st.session_state.report, json_path)
            with open(json_path, "rb") as f:
                st.download_button("JSON Export", f, "shortlist_report.json", "application/json", use_container_width=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">HR Shortlisting Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Two-Path Architecture · Hybrid BM25 + Embeddings + LLM · '
    'PII-Safe · Human-in-the-Loop</div>',
    unsafe_allow_html=True,
)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("⚠️ Please enter your Anthropic API key.")
        st.stop()
    if not jd_text.strip():
        st.error("⚠️ Please provide a job description.")
        st.stop()
    if not resume_files:
        st.error("⚠️ Please upload at least one resume.")
        st.stop()

    # Save uploaded files to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        resume_paths = []
        for f in resume_files:
            path = os.path.join(tmpdir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            resume_paths.append(path)

        progress_bar = st.progress(0, text="Starting pipeline…")
        status_text  = st.empty()

        def update_progress(step, total, msg):
            pct = int(step / total * 100)
            progress_bar.progress(pct, text=msg)
            status_text.caption(f"Step {step}/{total}: {msg}")

        try:
            with st.spinner("Running shortlisting pipeline…"):
                report = run_pipeline(
                    jd_text=jd_text,
                    resume_paths=resume_paths,
                    progress_callback=update_progress,
                )
            st.session_state.report = report
            st.session_state.overrides = {}
            progress_bar.progress(100, text="✅ Pipeline complete!")
            status_text.empty()
            st.success(f"✅ Processed {report.total_candidates} candidates in {report.total_processing_time_ms:.0f}ms")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()


# ── Results ───────────────────────────────────────────────────────────────────
report: ShortlistReport | None = st.session_state.report

if report is None:
    st.info("👆 Upload a JD and resumes in the sidebar, then click **Run Shortlisting**.")
    
    # Show architecture diagram
    with st.expander("ℹ️ How it works — Architecture", expanded=False):
        st.markdown("""
```
Input: JD + Resumes (PDF/DOCX)
           │
           ▼
┌──────────────────────┐
│   Security Layer     │  PII masking + prompt injection sanitisation
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    JD Parser         │  Claude claude-sonnet-4 → JDRequirements (Pydantic)
└──────────┬───────────┘
           │
    ┌──────┴──────┐  For each resume
    │             │
    ▼             │
┌─────────┐       │
│Fast Path│ BM25  │  <15% skill overlap → instant No Hire, zero LLM cost
│  (<100ms)       │
└────┬────┘       │
     │            │
     │ ≥15%       │
     ▼            │
┌─────────────────┴──┐
│    Full Path        │  Embedding similarity (sentence-transformers)
│                     │  + LLM scoring (Claude claude-sonnet-4)
│                     │  + Pydantic output validation
└────────┬────────────┘
         │
         ▼
┌────────────────────┐
│  Bias Detection    │  Flag demographic signals for human review
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Ranked Report     │  HTML + JSON output
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  Human Override    │  HR adjusts scores with reason (audit logged)
└────────────────────┘
```
        """)
    st.stop()


tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Ranked Candidates", "🔍 Candidate Detail", "✏️ Human Override"])


# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, str(report.total_candidates), "Total",          "#e8eaf0"),
        (c2, str(report.shortlisted),      "Shortlisted",    "#4ade80"),
        (c3, str(report.maybe_count),      "Maybe",          "#fbbf24"),
        (c4, str(report.rejected),         "Rejected",       "#f87171"),
        (c5, str(report.fast_path_filtered),"Fast Screened", "#fb923c"),
        (c6, f"${report.total_estimated_cost_usd:.4f}", "API Cost", "#6c8ef7"),
    ]
    for col, val, lbl, color in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val" style="color:{color}">{val}</div>'
                f'<div class="lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        # Recommendation donut
        rec_counts = {
            "Strong Hire": sum(1 for r in report.results if r.recommendation == "Strong Hire"),
            "Hire":        sum(1 for r in report.results if r.recommendation == "Hire"),
            "Maybe":       sum(1 for r in report.results if r.recommendation == "Maybe"),
            "No Hire":     sum(1 for r in report.results if r.recommendation == "No Hire"),
        }
        fig = go.Figure(go.Pie(
            labels=list(rec_counts.keys()),
            values=list(rec_counts.values()),
            hole=0.55,
            marker_colors=["#4ade80", "#6c8ef7", "#fbbf24", "#f87171"],
            textfont_size=12,
        ))
        fig.update_layout(
            title="Recommendation Breakdown",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaf0",
            showlegend=True,
            height=320,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Score distribution bar chart
        names  = [r.profile.name[:18] for r in report.results]
        scores = [r.weighted_total for r in report.results]
        colors = [rec_color(r.recommendation) for r in report.results]

        fig2 = go.Figure(go.Bar(
            x=scores, y=names, orientation="h",
            marker_color=colors, text=[f"{s:.1f}" for s in scores],
            textposition="outside", textfont_color="#e8eaf0",
        ))
        fig2.update_layout(
            title="Candidate Scores",
            xaxis=dict(range=[0, 10.5], gridcolor="#2a2d3e", color="#7b7f96"),
            yaxis=dict(autorange="reversed", color="#7b7f96"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaf0",
            height=320,
            margin=dict(t=50, b=20, l=20, r=60),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Cost efficiency callout
    if report.fast_path_filtered > 0:
        saved_pct = report.fast_path_filtered / report.total_candidates * 100
        st.info(
            f"⚡ **Fast-path efficiency:** {report.fast_path_filtered} of {report.total_candidates} "
            f"candidates ({saved_pct:.0f}%) were screened without any LLM calls, "
            f"saving significant API cost on this batch."
        )


# ── Tab 2: Ranked candidates table ────────────────────────────────────────────
with tab2:
    # Build DataFrame
    rows = []
    for i, r in enumerate(report.results, 1):
        rows.append({
            "Rank":          i,
            "Name":          r.profile.name,
            "Skills":        r.skills_match.score,
            "Experience":    r.experience_relevance.score,
            "Education":     r.education_certs.score,
            "Projects":      r.project_portfolio.score,
            "Communication": r.communication_quality.score,
            "Total":         r.weighted_total,
            "Recommendation":r.recommendation,
            "Fast Path":     "⚡" if r.fast_path_screened else "",
            "Bias Flag":     "⚠" if r.bias_flag else "",
            "Cost ($)":      f"{r.estimated_cost_usd:.5f}",
        })

    df = pd.DataFrame(rows)

    def _style_row(row):
        color_map = {
            "Strong Hire": "background-color: rgba(74,222,128,0.08)",
            "Hire":        "background-color: rgba(108,142,247,0.08)",
            "Maybe":       "background-color: rgba(251,191,36,0.06)",
            "No Hire":     "background-color: rgba(248,113,113,0.06)",
        }
        style = color_map.get(row["Recommendation"], "")
        return [style] * len(row)

    styled = df.style.apply(_style_row, axis=1).format({
        "Skills": "{:.1f}", "Experience": "{:.1f}", "Education": "{:.1f}",
        "Projects": "{:.1f}", "Communication": "{:.1f}", "Total": "{:.2f}",
    })

    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

    # Download as CSV
    csv = df.to_csv(index=False)
    st.download_button("⬇ Download as CSV", csv, "shortlist.csv", "text/csv")


# ── Tab 3: Candidate detail ────────────────────────────────────────────────────
with tab3:
    candidate_names = [
        f"#{i} {r.profile.name} ({r.recommendation})"
        for i, r in enumerate(report.results, 1)
    ]
    selected = st.selectbox("Select candidate", candidate_names)
    idx = int(selected.split("#")[1].split(" ")[0]) - 1
    r: CandidateResult = report.results[idx]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        color = rec_color(r.recommendation)
        st.markdown(
            f"<h2 style='color:{color};margin:0'>{r.weighted_total:.1f} / 10</h2>"
            f"<p style='color:#7b7f96;margin:0'>{r.recommendation}</p>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f"**Name:** {r.profile.name}")
        st.markdown(f"**File:** {r.file_name}")
        st.markdown(f"**Experience:** {r.profile.experience_years} yrs")
        st.markdown(f"**Education:** {r.profile.education}")
    with col3:
        st.markdown(f"**Processing:** {r.processing_time_ms:.0f}ms")
        st.markdown(f"**API cost:** ${r.estimated_cost_usd:.5f}")
        st.markdown(f"**BM25 overlap:** {r.bm25_overlap_score:.0%}")
        st.markdown(f"**PII fields masked:** {r.profile.pii_fields_masked}")

    st.markdown("---")

    # Radar / spider chart of dimensions
    dims = ["Skills Match", "Experience", "Education", "Projects", "Communication"]
    scores = [
        r.skills_match.score,
        r.experience_relevance.score,
        r.education_certs.score,
        r.project_portfolio.score,
        r.communication_quality.score,
    ]
    fig3 = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=dims + [dims[0]],
        fill="toself",
        fillcolor="rgba(108,142,247,0.15)",
        line_color="#6c8ef7",
        name=r.profile.name,
    ))
    fig3.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], color="#7b7f96", gridcolor="#2a2d3e"),
            angularaxis=dict(color="#e8eaf0"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf0",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

    if not r.fast_path_screened:
        col_s, col_g = st.columns(2)
        with col_s:
            st.markdown("**✅ Key Strengths**")
            for s in (r.key_strengths or ["None identified"]):
                st.markdown(f"- {s}")
        with col_g:
            st.markdown("**❌ Skill Gaps**")
            for g in (r.skill_gaps or ["No significant gaps"]):
                st.markdown(f"- {g}")

        st.markdown("---")
        st.markdown("**Dimension Justifications**")
        dim_data = [
            ("Skills Match (30%)",   r.skills_match),
            ("Experience (25%)",     r.experience_relevance),
            ("Education (15%)",      r.education_certs),
            ("Projects (20%)",       r.project_portfolio),
            ("Communication (10%)",  r.communication_quality),
        ]
        for name, dim in dim_data:
            with st.expander(f"{name}  —  Score: {dim.score:.1f}/10  (confidence: {dim.confidence:.0%})"):
                st.markdown(f"**Justification:** {dim.justification}")
                if dim.evidence:
                    st.markdown("**Evidence from resume:**")
                    for ev in dim.evidence:
                        st.markdown(f"  - _{ev}_")

    if r.bias_flag:
        st.warning(f"⚠ Bias signal: {r.bias_note}")
    if r.overridden:
        st.info(f"✏ This score was overridden. Original: {r.original_total:.1f} → {r.weighted_total:.1f}. Reason: {r.override_reason}")


# ── Tab 4: Human override ──────────────────────────────────────────────────────
with tab4:
    st.markdown(
        "Adjust scores for candidates where you have additional context not captured by the agent. "
        "All overrides are logged with a mandatory reason for audit compliance."
    )
    st.markdown("---")

    for r in report.results:
        with st.expander(
            f"{'⭐' if r.recommendation == 'Strong Hire' else '✓' if r.recommendation == 'Hire' else '?'} "
            f"{r.profile.name}  —  Current score: **{r.weighted_total:.1f}**  ({r.recommendation})"
        ):
            new_score = st.slider(
                "Override total score",
                min_value=0.0, max_value=10.0,
                value=float(r.weighted_total),
                step=0.5,
                key=f"slider_{r.candidate_id}",
            )
            reason = st.text_input(
                "Reason for override (required)",
                key=f"reason_{r.candidate_id}",
                placeholder="e.g. Additional portfolio review revealed strong relevant work",
            )
            reviewer = st.text_input(
                "Reviewer name",
                key=f"reviewer_{r.candidate_id}",
                placeholder="HR Manager name",
            )

            if st.button("Apply Override", key=f"apply_{r.candidate_id}"):
                if not reason.strip():
                    st.error("Please provide a reason for the override.")
                else:
                    # Record override
                    r.original_total = r.weighted_total if not r.overridden else r.original_total
                    r.weighted_total = new_score
                    r.recommendation = r.derive_recommendation()
                    r.overridden = True
                    r.override_reason = reason
                    r.override_by = reviewer or "HR"

                    # Re-sort report results
                    report.results.sort(key=lambda x: x.weighted_total, reverse=True)
                    report.shortlisted = sum(1 for x in report.results if x.recommendation in ("Strong Hire", "Hire"))

                    st.success(f"✅ Override applied. New score: {new_score:.1f} ({r.recommendation})")
                    st.rerun()

    if any(r.overridden for r in report.results):
        st.markdown("---")
        st.markdown("**Override Audit Log**")
        override_data = []
        for r in report.results:
            if r.overridden:
                override_data.append({
                    "Candidate": r.profile.name,
                    "Original Score": r.original_total,
                    "New Score": r.weighted_total,
                    "New Recommendation": r.recommendation,
                    "Reason": r.override_reason,
                    "Reviewer": r.override_by,
                    "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M"),
                })
        st.dataframe(pd.DataFrame(override_data), use_container_width=True, hide_index=True)
