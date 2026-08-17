# nodes/full_exp_analysis_node.py
from spm_agent.graphs.analysis_graph import build_analysis_graph
from spm_agent.states.pfm_experiment_state import PFMExperimentState
from spm_agent.states.image_analysis_state import AnalysisState

_analysis_graph = build_analysis_graph()    

def _last_scan_index(recs) -> int:
    return max((r["scan_index"] for r in recs if r["kind"] == "image"), default=-1)



async def full_exp_analysis_node(state: PFMExperimentState) -> PFMExperimentState:
    """Analyze the just-acquired file and pack it into an ExperimentalRecord.
    Consumes `pending`: verifies intent vs outcome, attributes the record, clears it."""
    recs    = state.get("experimental_records", [])
    pending = state["pending"]                    # must exist — acquisition ran before us

    payload: AnalysisState = {
        "file_path": pending["file_path"],
        "experiment_tasks": state.get("experiment_tasks", []),
        "scan_index": _last_scan_index(recs) + 1,
    }
    if state.get("experiment_context"):
        payload["experiment_context"] = state["experiment_context"]

    out = await _analysis_graph.ainvoke(payload)

    # intent vs outcome — fail loudly on mismatch
    expected = "image" if pending["kind"] == "scan" else "loop"
    if out["kind"] != expected:
        raise RuntimeError(f"ordered {pending['kind']!r} but file is {out['kind']!r}: "
                           f"{pending['file_path']}")

    record = {**out,
              "instrument_params": state.get("instrument_state"),
              "requested_params":   pending.get("params"),
              "decision_index":     pending["decision_index"]}
    
    if out["kind"] == "image":
        record["scan_index"] = _last_scan_index(recs) + 1     # new scan
    else:
        record["scan_index"] = _last_scan_index(recs)          # parent scan
        record["pixel_yx"]   = pending["pixel_yx"]             # the executed pick

    return {"experimental_records": recs + [record],
            "pending": None}                                   # consumed — cycle complete