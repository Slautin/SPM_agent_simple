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
            "transport": "http",
            "url": "http://localhost:8766/mcp",
        },
    }

#cache dir for readfile node
CASHE_DIR = (
    PROJECT_ROOT / "src" / "spm_agent" / ".cashe" / "current"
)

