from math import hypot

from spm_agent.config import IMAGE_ROW0_AT_TOP, MOVE_TOLERANCE_PX
from spm_agent.mcp.spm_session import call
from spm_agent.states.pfm_experiment_state import PFMExperimentState
from spm_agent.utils.pick_utils import pixel_to_frame_xy


async def move_tip_node(state: PFMExperimentState) -> PFMExperimentState:
    """Move the probe to the picked pixel, in absolute scan-frame coordinates.
    SIDE EFFECT: moves the tip."""
    pending = state["pending"]
    if pending["kind"] != "loop":                                    # type: ignore
        raise RuntimeError(f"move_tip called for kind={pending['kind']!r}")  # type: ignore
    if "pixel_yx" not in pending:                                    # type: ignore
        raise RuntimeError("move_tip requires pixel_yx — pick_point must run first")

    scans = [r for r in state.get("experimental_records", []) if r.get("kind") == "image"]
    if not scans:
        raise RuntimeError("move_tip requires an analysed scan to convert pixels")
    ss = scans[-1]["instrument_params"].scan_settings      # the grid the pixel refers to

    row, col = pending["pixel_yx"]                                    # type: ignore
    x_m, y_m = pixel_to_frame_xy(ss, row, col, IMAGE_ROW0_AT_TOP)
    px_m = ss.scan_size_m / ss.pixels

    args = {"x_m": float(x_m), "y_m": float(y_m)}
    #cal = state.get("scanner_calibrations")
    # if cal is not None:                       # direct move; without these it detours via centre
    #     args["x_scanner_offset_m"] = cal.x_scanner_offset_m
    #     args["y_scanner_offset_m"] = cal.y_scanner_offset_m

    print(f"[move_tip] px({row},{col}) -> ({x_m*1e6:+.4f}, {y_m*1e6:+.4f}) um on a "
          f"{ss.scan_size_m*1e6:.2f} um frame ({px_m*1e9:.1f} nm/px)")

    data = await call("pfm_move_tip", args)                # raises on ScanAngle != 0, readback fail

    err = hypot(data.get("x_error_m") or 0.0, data.get("y_error_m") or 0.0)
    if err > MOVE_TOLERANCE_PX * px_m:
        raise RuntimeError(
            f"move_tip landed {err*1e9:.1f} nm from the target — more than "
            f"{MOVE_TOLERANCE_PX} px ({MOVE_TOLERANCE_PX*px_m*1e9:.1f} nm)")

    print(f"[move_tip] achieved ({data['x_m']*1e6:+.4f}, {data['y_m']*1e6:+.4f}) um, "
          f"error {err*1e9:.1f} nm, feedback on")
    return {}