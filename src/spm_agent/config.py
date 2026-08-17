from pathlib import Path
import shutil

import anthropic
import httpx
from langgraph.types import RetryPolicy


PROJECT_ROOT = Path().resolve().parents[0]

#schifireaders_mcp command path
_scifi_cmd = shutil.which("scifireaders_mcp")

if _scifi_cmd is None:
    # Windows fallback: entry-point executable is usually in .venv/Scripts
    _scifi_cmd = Path(".venv") / "Scripts" / "scifireaders_mcp.exe"

SCIFIREADERS_MCP_COMMAND = str(_scifi_cmd)

#spm_mcp configuration
SPM_MCP_SERVER_CONFIG = {
        "spm": {
            "transport": "streamable-http",
            "url": "http://10.46.217.255:8000/mcp"#"http://10.128.35.95:8000/mcp",
        },
    }

#cache dir for readfile node
CASHE_DIR = (
    PROJECT_ROOT / "src" / "spm_agent" / ".cashe" / "current"
)

#segmentation/importance config

SEG_DIR = Path("./seg_proto").resolve()
SEG_DIR.mkdir(parents=True, exist_ok=True)

SEG_MODEL = "claude-sonnet-4-6"
SEG_MAX_TOKENS = 4096
SEG_VISION_IN_LOOP = True
SEG_MAX_SUPERSTEPS = 40 #for the importance map: SEG_MAX_SUPERSTEPSx2 

#channel reccomendation model
CHAN_MODEL      = "claude-sonnet-4-6"   # ablation arm; was ChatOpenAI("gpt-5.4")
CHAN_MAX_TOKENS = 4096


#decision loop
MAX_TOTAL_DECISIONS = 4      # hard safety cap, never reached in a sane experiment

#scanning bounds
LOCKED_SCAN_ALWAYS = ("x_scan_center_m", "y_scan_center_m")   # location is never the plan's
LOCKED_SCAN_HOLD   = ("scan_size_m",)                         # hold additionally freezes size
LOCKED_EXC_ALWAYS  = ("drive_frequency_hz",)                  # set by the probe tune

LOCK_REASON = {
    "x_scan_center_m":    "the frame centre is computed, never proposed",
    "y_scan_center_m":    "the frame centre is computed, never proposed",
    "scan_size_m":        "frame_action='hold' keeps the current frame",
    "drive_frequency_hz": "it is set by the probe tune, not by the plan",
}

SCAN_SIZES_PX = (64, 128, 256, 512)

SCAN_BOUNDS = {              # command limits; the instrument read-models carry none
    "scan_rate_hz":      (0.1, 1.5),
    "drive_amplitude_v": (0.05, 5.0),
    "dart_width_hz":     (2.0e3, 3.0e4),
    "dart_igain":        (50.0, 1500.),
    "scan_size_m":       (1.0e-7, 3.0e-5),
}

#loop point picking
PICK_PATCH_PX          = 3      # neighbourhood a single measurement effectively samples
PICK_BORDER_PX         = 5      # frame edge is not measurable
PICK_PENALTY_RADIUS_PX = 5     # keep loops apart
PICK_PENALTY_FLOOR     = 0.05   # residual desirability at an already-measured pixel

#sandbox
from spm_agent.sandbox import sandbox_python

SANDBOX_PY = str(sandbox_python())
IMPORTANCE_DIR = CASHE_DIR.parent / "importance"
LOOPS_DIR = CASHE_DIR.parent / "loops" 


#new dir structure
from datetime import datetime

# --- run-scoped artifact directories ---
RUNS_ROOT = PROJECT_ROOT / "src" / "runs"

_run_dir: Path | None = None

def new_run(tag: str = "") -> Path:
    """Start a new experiment run. Call once, at experiment start."""
    global _run_dir
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    _run_dir = RUNS_ROOT / name
    _run_dir.mkdir(parents=True, exist_ok=True)
    return _run_dir

def run_dir() -> Path:
    """Current run directory (auto-creates one if new_run was never called)."""
    return _run_dir if _run_dir is not None else new_run()

def importance_dir() -> Path: return run_dir() / "importance"
def loops_dir() -> Path:      return run_dir() / "loops"
def decisions_dir() -> Path:  return run_dir() / "decisions"
def seg_dir() -> Path:  return run_dir() / "segmentation"
def criteria_dir() -> Path: return run_dir() / "criteria_seg"
def channels_dir() -> Path: return run_dir() / "channels"
def records_dir() -> Path: return run_dir() / "records"


#anthropic retrypolicy
def _transient(exc: Exception) -> bool:
    """Retry network-level failures only — never our own guards, validation errors
    or InstrumentError, which mean the run should stop and be looked at."""
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError,
                            anthropic.APIConnectionError, anthropic.RateLimitError,
                            anthropic.InternalServerError))


LLM_RETRY = RetryPolicy(max_attempts=4, initial_interval=2.0, backoff_factor=2.0,
                        max_interval=30.0, jitter=True, retry_on=_transient)