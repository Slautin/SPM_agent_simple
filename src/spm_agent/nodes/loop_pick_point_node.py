import json

import numpy as np

from spm_agent.config import (decisions_dir, PICK_PATCH_PX, PICK_BORDER_PX,
                              PICK_PENALTY_RADIUS_PX, PICK_PENALTY_FLOOR)
from spm_agent.states.pfm_experiment_state import PFMExperimentState
from spm_agent.utils.decision_utils import resolve_criterion, all_criteria, find_criterion
from spm_agent.utils.pick_utils import (patch_mean, strategy_score,
                                        spatial_penalty, border_mask)


async def loop_pick_point_node(state: PFMExperimentState) -> PFMExperimentState:
    """Turn (criterion, strategy) into one pixel on the current scan.
    Deterministic: the decision never proposes a location."""
    pending = state["pending"]
    if pending["kind"] != "loop":                                   # type: ignore
        raise RuntimeError(f"pick_point called for kind={pending['kind']!r}")  # type: ignore

    decision = state["decision_records"][-1]
    recs     = state.get("experimental_records", [])
    scans    = [r for r in recs if r.get("kind") == "image"]
    if not scans:
        raise RuntimeError("pick_point requires an analysed scan")
    scan = scans[-1]

    name = resolve_criterion(decision, all_criteria(scan))
    if name is None:
        raise RuntimeError(
            f"target_criterion {decision.target_criterion!r} matches no criterion on "
            f"Scan_{scan['scan_index']}: {all_criteria(scan)}")

    imap, k = find_criterion(scan, name)          # the entry carries its own safety map
    phi     = np.load(imap["components_path"])[k]
    safety  = np.load(imap["safety_map_path"])
    shape   = phi.shape

    loops  = [r for r in recs if r.get("kind") == "loop"
              and r.get("scan_index") == scan["scan_index"]]
    points = [tuple(int(v) for v in r["pixel_yx"]) for r in loops]
    phi_s  = patch_mean(phi, PICK_PATCH_PX)
    measured = [float(phi_s[y, x]) for y, x in points]

    strategy = decision.target_strategy or "max"
    score = (strategy_score(phi, measured, strategy)
             * safety
             * spatial_penalty(shape, points, PICK_PENALTY_RADIUS_PX, PICK_PENALTY_FLOOR))
    score[~border_mask(shape, PICK_BORDER_PX)] = 0.0
    smooth = patch_mean(score, PICK_PATCH_PX)

    if not np.isfinite(smooth).any() or smooth.max() <= 0:
        raise RuntimeError("no admissible pixel: safety and the border margin exclude "
                           "the whole frame")

    y, x = (int(v) for v in np.unravel_index(int(np.argmax(smooth)), shape))

    info = {"criterion": name, "requested_criterion": decision.target_criterion,
            "strategy": strategy, "task": imap["experiment_task"], "pixel_yx": [y, x],
            "phi_at_pick":    round(float(phi_s[y, x]), 4),
            "safety_at_pick": round(float(safety[y, x]), 4),
            "phi_measured_before": [round(v, 4) for v in measured]}
    d = decisions_dir() / f"decision_{pending['decision_index']:02d}"   # type: ignore
    d.mkdir(parents=True, exist_ok=True)
    (d / "pick.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"[pick] {name}/{strategy} -> px({y},{x}) | phi {info['phi_at_pick']}, "
          f"safety {info['safety_at_pick']} | previously {info['phi_measured_before']}")

    return {"pending": {**pending, "pixel_yx": (y, x)}}                # type: ignore