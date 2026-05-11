"""
HTML Report Generator
─────────────────────
Renders a ShortlistReport into a self-contained HTML file using Jinja2.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from core.models import ShortlistReport

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_html_report(report: ShortlistReport, output_path: str) -> str:
    """
    Render the shortlist report to an HTML file.

    Parameters
    ----------
    report      : ShortlistReport
    output_path : destination file path (e.g. "shortlist_report.html")

    Returns
    -------
    output_path : str – confirmed path of the written file
    """
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
    template = env.get_template("report.html")

    html = template.render(report=report)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


def generate_json_report(report: ShortlistReport, output_path: str) -> str:
    """Export report as JSON for downstream processing / API consumption."""
    data = report.model_dump(mode="json")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return str(out)
