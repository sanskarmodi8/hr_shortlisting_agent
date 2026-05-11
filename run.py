"""
CLI Entry Point
───────────────
Run the shortlisting pipeline from the command line without the Streamlit UI.

Usage:
  python run.py --jd sample_data/sample_jd.txt \
                --resumes sample_data/resumes/ \
                --output results/

  python run.py --jd sample_data/sample_jd.txt \
                --resumes sample_data/resumes/arjun_sharma_resume.txt \
                          sample_data/resumes/priya_mehta_resume.txt
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


def _collect_resumes(paths: list[str]) -> list[str]:
    """Accept individual files or directories."""
    result = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            result.extend(str(f) for f in path.glob("*") if f.suffix.lower() in (".pdf", ".docx", ".txt"))
        elif path.is_file():
            result.append(str(path))
        else:
            logger.warning("Path not found: %s", p)
    return result


def main():
    parser = argparse.ArgumentParser(description="HR Shortlisting Agent CLI")
    parser.add_argument("--jd",       required=True, help="Path to job description (TXT/PDF)")
    parser.add_argument("--resumes",  required=True, nargs="+", help="Resume files or directory")
    parser.add_argument("--output",   default="results", help="Output directory (default: results/)")
    parser.add_argument("--no-html",  action="store_true", help="Skip HTML report generation")
    parser.add_argument("--no-json",  action="store_true", help="Skip JSON report generation")
    args = parser.parse_args()

    # ── Read JD ────────────────────────────────────────────────────────────
    jd_path = Path(args.jd)
    if not jd_path.exists():
        logger.error("JD file not found: %s", jd_path)
        sys.exit(1)

    if jd_path.suffix.lower() == ".pdf":
        import fitz
        doc = fitz.open(str(jd_path))
        jd_text = "\n".join(page.get_text() for page in doc)
    else:
        jd_text = jd_path.read_text(encoding="utf-8", errors="ignore")

    logger.info("JD loaded: %s (%d chars)", jd_path.name, len(jd_text))

    # ── Collect resumes ────────────────────────────────────────────────────
    resume_paths = _collect_resumes(args.resumes)
    if not resume_paths:
        logger.error("No resume files found.")
        sys.exit(1)
    logger.info("Found %d resume(s)", len(resume_paths))

    # ── Run pipeline ───────────────────────────────────────────────────────
    from core.orchestrator import run_pipeline

    def progress(step, total, msg):
        print(f"  [{step:2d}/{total}] {msg}")

    print("\n━━━ HR Shortlisting Agent ━━━")
    print(f"Resumes: {len(resume_paths)}")
    print("Running pipeline...\n")

    report = run_pipeline(jd_text, resume_paths, progress_callback=progress)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n━━━ Results (Report ID: RPT-{report.report_id}) ━━━")
    print(f"{'Rank':<5} {'Name':<25} {'Total':>6} {'Recommendation':<15} {'Fast?':<6} {'Cost':>8}")
    print("─" * 70)

    for i, r in enumerate(report.results, 1):
        fast = "⚡" if r.fast_path_screened else ""
        print(
            f"{i:<5} {r.profile.name:<25} {r.weighted_total:>6.2f} "
            f"{r.recommendation:<15} {fast:<6} ${r.estimated_cost_usd:>7.5f}"
        )

    print("─" * 70)
    print(
        f"\nTotal: {report.total_candidates} | "
        f"Shortlisted: {report.shortlisted} | "
        f"Maybe: {report.maybe_count} | "
        f"Rejected: {report.rejected} | "
        f"Fast-path: {report.fast_path_filtered}"
    )
    print(f"Total API cost: ${report.total_estimated_cost_usd:.5f}")
    print(f"Total time:     {report.total_processing_time_ms:.0f}ms\n")

    # ── Write reports ──────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    from output.report_generator import generate_html_report, generate_json_report

    if not args.no_html:
        html_path = str(out_dir / f"shortlist_{timestamp}.html")
        generate_html_report(report, html_path)
        print(f"📄 HTML report: {html_path}")

    if not args.no_json:
        json_path = str(out_dir / f"shortlist_{timestamp}.json")
        generate_json_report(report, json_path)
        print(f"📋 JSON report: {json_path}")

    print()


if __name__ == "__main__":
    main()
