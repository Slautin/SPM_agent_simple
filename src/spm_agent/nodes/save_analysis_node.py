from spm_agent.config import records_dir
from spm_agent.states.image_analysis_state import AnalysisState
from spm_agent.utils.io_utils import save_json


async def save_analysis_node(state: AnalysisState) -> AnalysisState:
    """Snapshot the finished AnalysisState — one JSON per measurement."""
    kind = state.get("kind", "unknown")
    idx  = state.get("scan_index", 0)
    save_json(records_dir() / f"scan_{idx:02d}_{kind}.json", dict(state))
    return {} # type: ignore