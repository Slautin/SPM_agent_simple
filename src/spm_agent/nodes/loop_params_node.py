# nodes/loop_params_node.py
import numpy as np

from spm_agent.states.image_analysis_state import AnalysisState
from spm_agent.schemas.loops_params import LoopParams
from spm_agent.utils.loops_utils import extract_loop_params

from spm_agent.config import LOOPS_DIR


async def loop_params_node(state: AnalysisState) -> AnalysisState:
    """
    Extract hysteresis parameters from the segmented loops.
    Deterministic: no LLM calls. Runs on the X (and Y) quadrature,
    off-field and on-field separately.
    """
    loops = state["loops"]   # type: ignore[typeddict-item]

    if "X" not in loops["loops"]:
        raise ValueError(
            "No X channel in segmented loops - cannot extract parameters. "
            f"Available: {list(loops['loops'].keys())}")

    has_y = "Y" in loops["loops"]
    params: dict[str, LoopParams] = {}

    for branch in ("off", "on"):
        bias = np.load(loops[f"bias_{branch}_path"])
        x = np.load(loops["loops"]["X"][f"{branch}_path"])
        y = np.load(loops["loops"]["Y"][f"{branch}_path"]) if has_y else None

        params[branch] = LoopParams(**extract_loop_params(
            bias, x, y,
            fig_path=str(LOOPS_DIR / f"loop_{branch}_annotated.png"),
            units=loops["loops"]["X"]["units"],
        ))

    return {"loop_params": params}   # type: ignore[return-value]