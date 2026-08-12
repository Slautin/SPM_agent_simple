from math import isclose
from pathlib import Path

from spm_agent.mcp.spm_session import call
from spm_agent.schemas.scan_plan import plan_diff
from spm_agent.states.pfm_experiment_state import PFMExperimentState

# schema field -> MCP argument name. Kept next to the tool names they belong to.
EXC_ARGS  = {"drive_amplitude_v": "v_ac_v",
             "dart_width_hz":     "f_dart_width_hz",
             "dart_igain":        "dart_igain"}
SCAN_ARGS = {"pixels":            "n_points",
             "scan_rate_hz":      "scan_rate_hz",
             "scan_size_m":       "scan_size_m"}

TUNE_PROBE = True          # retune the contact resonance before every scan


def _verify(data: dict, plan, live, rel_tol: float = 1e-3) -> None:
    """Achieved vs requested. NOTE: the returned 'ScanRate' is the reciprocal of the
    status field's scan_rate_hz (s/line vs lines/s) — never compare them directly."""
    problems = []

    size, pts = data.get("ScanSize"), data.get("PointsLines")
    if size is not None and not isclose(size, plan.scan_settings.scan_size_m, rel_tol=rel_tol):
        problems.append(f"scan_size_m requested {plan.scan_settings.scan_size_m:.6g}, "
                        f"got {size:.6g}")
    if pts is not None and int(pts) != plan.scan_settings.pixels:
        problems.append(f"pixels requested {plan.scan_settings.pixels}, got {int(pts)}")

    for key, want in (("x_scan_center_m", live.scan_settings.x_scan_center_m),
                      ("y_scan_center_m", live.scan_settings.y_scan_center_m)):
        got = data.get(key)
        if got is not None and not isclose(got, want, rel_tol=rel_tol, abs_tol=1e-9):
            problems.append(f"{key} moved: expected {want:.6g}, scanned at {got:.6g}")

    if problems:
        raise RuntimeError("scan did not match the plan:\n  - " + "\n  - ".join(problems))


async def scan_run_node(state: PFMExperimentState) -> PFMExperimentState:
    """Apply the plan and acquire one DART-PFM image.
    The frame centre is never sent: under 'hold' and centred zooms it is whatever the
    instrument already has."""
    pending = state["pending"]
    plan    = pending.get("params")                                    # type: ignore
    live    = state.get("instrument_state")
    if plan is None or live is None:
        raise RuntimeError("run_scan requires pending['params'] and instrument_state")

    diff = plan_diff(plan, live)
    exc  = {arg: diff[f] for f, arg in EXC_ARGS.items()  if f in diff}
    scan = {arg: diff[f] for f, arg in SCAN_ARGS.items() if f in diff}

    if exc:
        await call("pfm_set_excitation_params", exc)
        print(f"[run_scan] excitation set: {exc}")

    ss = plan.scan_settings
    print(f"[run_scan] starting — {ss.scan_size_m*1e6:.2f} um @ {ss.pixels} px, "
          f"{ss.scan_rate_hz:.3f} Hz (~{ss.pixels/ss.scan_rate_hz:.0f} s); "
          f"args={scan or 'instrument defaults'}")

    data = await call("pfm_run_dart_pfm_scan", {**scan, "tune_probe": TUNE_PROBE})

    path = data.get("path")
    if not path or not Path(path).exists():
        raise RuntimeError(f"scan returned an unreadable path: {path!r}")
    _verify(data, plan, live)

    print(f"[run_scan] done — tuned={data.get('tuned')} -> {path}")
    return {"pending": {**pending, "file_path": path}}                 # type: ignore