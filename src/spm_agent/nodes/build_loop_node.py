from spm_agent.config import LOOPS_DIR
import shutil

from spm_agent.states.image_analysis_state import AnalysisState
from spm_agent.utils.loops_utils import segment_sspfm_loops

async def build_loop_node(state: AnalysisState) -> AnalysisState:
    """
    Segment the SS-PFM bias waveform into in-field / out-of-field loops.
    Deterministic: output fully defined by the input file.
    Expects 'file_channels' with a Bias channel (guaranteed by the router).
    """
    if LOOPS_DIR.exists():          # clean previous run
        shutil.rmtree(LOOPS_DIR)
    LOOPS_DIR.mkdir(parents=True, exist_ok=True)

    loops = segment_sspfm_loops(
        channels=state["file_channels"],   # type: ignore[typeddict-item]
        out_dir=LOOPS_DIR,
    )

    return {"loops": loops}#type: ignore