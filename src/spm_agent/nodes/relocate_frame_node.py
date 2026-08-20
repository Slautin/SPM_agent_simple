import json
from math import hypot

from spm_agent.config import decisions_dir, MOVE_TOLERANCE_PX, SCAN_TRAVEL_LIMIT_M
from spm_agent.mcp.spm_session import call
from spm_agent.nodes.sync_status_node import sync_status_node
from spm_agent.states.pfm_experiment_state import PFMExperimentState
from spm_agent.utils.relocate_utils import pick_relocation


async def relocate_frame_node(state: PFMExperimentState) -> PFMExperimentState:
    """Step the scan frame onto the nearest unexplored neighbour, then re-read the
    instrument. Deterministic: the decision asks to relocate, it never says where.
    SIDE EFFECT: moves the scan centre.

    Runs BEFORE scan_plan, so the plan sees the new centre as the live one and locks
    onto it, and run_scan's achieved-vs-live centre check needs no special case."""
    pending = state["pending"]
    if pending["kind"] != "scan":                                        # type: ignore
        raise RuntimeError(
            f"relocate_frame called for kind={pending['kind']!r}")       # type: ignore

    decision = state["decision_records"][-1]
    if decision.frame_action != "relocate":
        raise RuntimeError(
            f"relocate_frame called for frame_action={decision.frame_action!r}")

    live = state.get("instrument_state")
    if live is None:
        raise RuntimeError("relocate_frame requires instrument_state — run sync_status first")

    visited = [r["instrument_params"].scan_settings
               for r in state.get("experimental_records", []) if r["kind"] == "image"]
    if not visited:
        raise RuntimeError("relocate_frame requires a scanned frame to step from")

    ss = live.scan_settings
    centre, rejected = pick_relocation(ss, visited, SCAN_TRAVEL_LIMIT_M)
    if centre is None:
        raise RuntimeError(
            f"nowhere left to relocate: no unexplored {ss.scan_size_m*1e6:.2f} um frame "
            f"within +/-{SCAN_TRAVEL_LIMIT_M*1e6:.0f} um "
            f"({rejected['explored']} candidate(s) already scanned, "
            f"{rejected['out_of_range']} outside the travel limit, "
            f"{len(visited)} frame(s) scanned so far)")
    x_m, y_m = centre

    print(f"[relocate] ({ss.x_scan_center_m*1e6:+.3f}, {ss.y_scan_center_m*1e6:+.3f}) -> "
          f"({x_m*1e6:+.3f}, {y_m*1e6:+.3f}) um — one {ss.scan_size_m*1e6:.2f} um frame "
          f"away, borders touching; skipped {rejected['explored']} explored, "
          f"{rejected['out_of_range']} out of range")

    await call("pfm_set_scan_center", {"x_offset": float(x_m), "y_offset": float(y_m)})

    out = await sync_status_node(state)          # calibration-freshness guard + readback
    got = out["instrument_state"].scan_settings                          # type: ignore
    tol = MOVE_TOLERANCE_PX * ss.scan_size_m / ss.pixels
    err = hypot(got.x_scan_center_m - x_m, got.y_scan_center_m - y_m)
    if err > tol:
        raise RuntimeError(
            f"scan centre landed {err*1e9:.1f} nm from the target — more than "
            f"{MOVE_TOLERANCE_PX} px ({tol*1e9:.1f} nm)")

    info = {"from_m":       [ss.x_scan_center_m, ss.y_scan_center_m],
            "to_m":         [x_m, y_m],
            "achieved_m":   [got.x_scan_center_m, got.y_scan_center_m],
            "scan_size_m":  ss.scan_size_m,
            "travel_limit_m": SCAN_TRAVEL_LIMIT_M,
            "frames_scanned": len(visited),
            "candidates_rejected": rejected}
    d = decisions_dir() / f"decision_{pending['decision_index']:02d}"     # type: ignore
    d.mkdir(parents=True, exist_ok=True)
    (d / "relocate.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"[relocate] achieved ({got.x_scan_center_m*1e6:+.3f}, "
          f"{got.y_scan_center_m*1e6:+.3f}) um, error {err*1e9:.1f} nm")
    return out
