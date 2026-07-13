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
SEG_MAX_TOKENS = 2048
SEG_VISION_IN_LOOP = True
SEG_MAX_SUPERSTEPS = 30 #for the importance map: SEG_MAX_SUPERSTEPSx2 


#sandbox
from spm_agent.sandbox import sandbox_python

SANDBOX_PY = str(sandbox_python())

IMPORTANCE_DIR = CASHE_DIR.parent / "importance"

LOOPS_DIR = CASHE_DIR.parent / "loops" 
