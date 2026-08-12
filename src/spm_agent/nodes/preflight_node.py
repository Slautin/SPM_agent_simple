# src/spm_agent/nodes/preflight_node.py
import json
import time

from spm_agent.config import run_dir
from spm_agent.mcp.spm_session import call
from spm_agent.states.pfm_experiment_state import PFMExperimentState

def _fmt(v, scale=1.0, unit="", fmt=".1f"):
    """Status fields are None when unreadable — never let a print kill the run."""
    return "n/a" if v is None else f"{v*scale:{fmt}}{unit}"

async def preflight_node(state: PFMExperimentState) -> PFMExperimentState:
    """Read-only readiness check. Runs BEFORE the first tip motion (calibration).
    Fails the run rather than driving an instrument in an unknown state."""
    raw = await call("pfm_get_experiment_status")

    session = {
        "run_dir":              str(run_dir()),
        "started_ts":           time.time(),
        "instrument_directory": raw["directory"],       # where .ibw files land
    }

    (run_dir() / "session.json").write_text(
        json.dumps({**session, "status_at_start": raw}, indent=2), encoding="utf-8"
        )

    print(f"[preflight] dir {raw.get('directory')!r} | "
          f"frame {_fmt(raw.get('scan_size_m'), 1e6, ' um')} @ {raw.get('n_points')} px | "
          f"f_DART {_fmt(raw.get('f_dart_hz'), 1e-3, ' kHz')} | "
          f"V_ac {raw.get('v_ac_v')} V | feedback {raw.get('feedback_on')}")
    if raw.get("warnings"):
        print(f"[preflight] WARNINGS: {'; '.join(raw['warnings'])}")
        
    return {"session_info": session}      # type: ignore