from pathlib import Path

PROJECT_ROOT = Path().resolve().parents[0]

SCIFIREADERS_MCP_COMMAND = (
    PROJECT_ROOT / ".venv" / "bin" / "scifireaders-mcp"
)

CASHE_DIR = (
    PROJECT_ROOT / "src" / "spm_agent" / ".cashe" / "current"
)