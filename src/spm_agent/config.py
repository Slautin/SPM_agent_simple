from pathlib import Path
import shutil


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
            "url": "http://10.128.35.95:8000/mcp",
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
SEG_MAX_TOKENS = 2046#4096
SEG_VISION_IN_LOOP = True
SEG_MAX_SUPERSTEPS = 30 #for the importance map: SEG_MAX_SUPERSTEPSx2 


#decision loop
MAX_TOTAL_DECISIONS = 4      # hard safety cap, never reached in a sane experiment


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