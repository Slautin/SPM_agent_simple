from spm_agent.config import loops_dir#LOOPS_DIR
import shutil

from spm_agent.states.image_analysis_state import AnalysisState
from spm_agent.utils.loops_utils import segment_sspfm_loops

async def build_loop_node(state: AnalysisState) -> AnalysisState:
    """
    Segment the SS-PFM bias waveform into in-field / out-of-field loops.
    Deterministic: output fully defined by the input file.
    Expects 'file_channels' with a Bias channel (guaranteed by the router).
    """
    # if LOOPS_DIR.exists():          # clean previous run
    #     shutil.rmtree(LOOPS_DIR)
    # LOOPS_DIR.mkdir(parents=True, exist_ok=True)
    root = loops_dir()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"loop_{len([p for p in root.iterdir() if p.is_dir()]):03d}"
    dest.mkdir()

    loops = segment_sspfm_loops(
        channels=state["file_channels"],   # type: ignore[typeddict-item]
        out_dir=dest,
    )

    return {"loops": loops}#type: ignore