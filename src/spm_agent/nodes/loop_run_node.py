from pathlib import Path

from spm_agent.mcp.spm_session import call
from spm_agent.schemas.loop_plan import loop_plan_diff
from spm_agent.states.pfm_experiment_state import PFMExperimentState

# schema field -> MCP argument name. Kept next to the tool name they belong to.
# LoopSettings field  ->  MCP argument name
LOOP_ARGS = {"v_dc_max_v":       "v_dc_max",
             "frequency_hz":     "loop_frequency",
             "var0_phase":       "var0_loop_phase",     # was var0_phase
             "var1_pulsetime_s": "var1_pulsetime_s",    # was var1_pulsetime
             "n_cycles":         "n_cycles"}


async def loop_run_node(state: PFMExperimentState) -> PFMExperimentState:
    """Apply the plan and measure one hysteresis loop at the current tip position.
    The location is never sent — move_tip put the probe there already."""
    pending = state["pending"]
    plan    = pending.get("params")                                   # type: ignore
    live    = state.get("instrument_state")
    if plan is None or live is None:
        raise RuntimeError("run_loop requires pending['params'] and instrument_state")

    ls = plan.loop_settings
    args = {arg: getattr(ls, f) for f, arg in LOOP_ARGS.items()}      # all five, explicitly

    diff = loop_plan_diff(plan, live)
    secs = ls.n_cycles / ls.frequency_hz if ls.frequency_hz else float("nan")
    print(f"[run_loop] starting — v_dc_max {ls.v_dc_max_v:.2f} V, "
          f"{ls.frequency_hz:.3f} Hz, {ls.n_cycles} cycles, "
          f"pulse {ls.var1_pulsetime_s*1e3:.1f} ms (~{secs:.0f} s); "
          f"changes={diff or 'none'}")

    data = await call("pfm_measure_hysteresis_loop", args)
    print(f"[run_loop] payload keys: {sorted(data)}")                 # tighten _resolve_path after the first real loop

    path =  data.get("path")
    if not Path(path).exists():
        raise RuntimeError(f"loop returned an unreadable path: {path!r}")

    print(f"[run_loop] done -> {path}")
    return {"pending": {**pending, "file_path": path}}                # type: ignore