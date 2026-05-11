# Lazy imports — models are always safe; orchestrator imports anthropic
# which requires the package to be installed. Tests import submodules directly.
from .models import JDRequirements, CandidateProfile, CandidateResult, ShortlistReport


def run_pipeline(*args, **kwargs):
    """Lazy-load orchestrator so tests that don't need the LLM don't require anthropic."""
    from .orchestrator import run_pipeline as _run
    return _run(*args, **kwargs)


__all__ = [
    "JDRequirements",
    "CandidateProfile",
    "CandidateResult",
    "ShortlistReport",
    "run_pipeline",
]
