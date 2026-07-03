"""
Provision and manage the ISOLATED Python interpreter used to execute
LLM-generated analysis code (the `run_python` sandbox).


The importance-map analyst runs model-written code. That code must NOT be able
to import `spm_agent` (hence the instrument MCP client) or otherwise reach
hardware. Interpreter holds ONLY numpy/scipy/scikit-image/matplotlib and NOT this package.

One-time setup:
    python -m spm_agent.sandbox            # create ~/.spm_agent/sandbox + self-check
    python -m spm_agent.sandbox --force    # rebuild from scratch
Programmatic:
    from spm_agent.sandbox import ensure_sandbox
    SANDBOX_PY = str(ensure_sandbox())
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# Sandbox venv lives OUTSIDE the package (override with an env var).
SANDBOX_DIR = Path(
    os.environ.get("SPM_AGENT_SANDBOX_DIR", Path.home() / ".spm_agent" / "sandbox")
)

# Pinned dependency list shipped inside the package.
SANDBOX_REQ = Path(__file__).parent / "sandbox-requirements.txt"


def sandbox_python() -> Path:
    """Path to the sandbox interpreter (does not guarantee it exists yet)."""
    if os.name == "nt":  # Windows (e.g. the instrument PC)
        return SANDBOX_DIR / "Scripts" / "python.exe"
    return SANDBOX_DIR / "bin" / "python"


def _assert_isolated(py: Path) -> None:
    """Fail loudly if the sandbox can import this package (isolation broken)."""
    r = subprocess.run([str(py), "-c", "import spm_agent"], capture_output=True, text=True)
    if r.returncode == 0:
        raise RuntimeError(
            f"Sandbox at {py} is NOT isolated: 'import spm_agent' succeeded. "
            "Rebuild with system_site_packages=False and do not install this package into it."
        )


def ensure_sandbox(force: bool = False) -> Path:
    """
    Create the isolated venv if needed, install pinned deps, verify isolation,
    and return the interpreter path. Idempotent.
    """
    py = sandbox_python()

    if force and SANDBOX_DIR.exists():
        shutil.rmtree(SANDBOX_DIR)

    if not py.exists():
        print(f"[sandbox] creating isolated venv at {SANDBOX_DIR} ...")
        # system_site_packages=False is CRITICAL: no inheritance of the main env,
        # so spm_agent and the instrument client are NOT visible in here.
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(SANDBOX_DIR)

        print(f"[sandbox] installing pinned deps from {SANDBOX_REQ.name} ...")
        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(SANDBOX_REQ)], check=True)

    _assert_isolated(py)   # tested invariant on every call
    print(f"[sandbox] ready + isolated: {py}")
    return py


if __name__ == "__main__":
    ensure_sandbox(force="--force" in sys.argv)