# src/spm_agent/tools/python_backend.py
import base64, subprocess
from pathlib import Path
from langchain_core.tools import tool
import shutil

MAX_FIG_PX = 2048


def _fig_b64(path, max_px: int = MAX_FIG_PX) -> str:
    """Figures are re-sent on every later turn — shrink what the model sees.
    The full-resolution copy still goes to archive_dir."""
    import io
    from PIL import Image
    im = Image.open(path)
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px))
    if im.mode == "RGBA":
        im = im.convert("RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


class LocalBackend:
    """Run ONE code cell in the isolated interpreter, in a persistent workdir.
    Fresh process each call (state carried via files). Returns stdout + new PNGs."""

    def __init__(self, workdir: Path, python_exe: str, timeout: int = 30, archive_dir=None):
        self.wd, self.py, self.timeout = Path(workdir), str(python_exe), timeout
        self.archive_dir = Path(archive_dir) if archive_dir else None
        self._step = 0


    def execute(self, code: str) -> dict:
        (self.wd / "cell.py").write_text(code)                  # 1. write the model's code
        before = set(self.wd.glob("*.png"))                     #    note existing figures
        try:                                                    # 2. run it — isolated + bounded
            p = subprocess.run([self.py, "cell.py"], cwd=self.wd,
                               timeout=self.timeout, capture_output=True, text=True)
            out = (p.stdout + p.stderr)[-4000:]                 #    stdout, or traceback on error
        except subprocess.TimeoutExpired:
            out = f"ERROR: exceeded {self.timeout}s, killed."
        new = sorted(set(self.wd.glob("*.png")) - before)       # 3. figures it just saved
        if self.archive_dir is not None:                      # NEW: keep every figure
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            for f in new:
                shutil.copy(f, self.archive_dir / f"step{self._step:02d}_{f.name}")
        self._step += 1
        #images = [base64.b64encode(f.read_bytes()).decode() for f in new]
        images = [_fig_b64(f) for f in new]
        return {"stdout": out or "(no output)", "images": images}


def make_run_python(backend: LocalBackend):
    @tool
    def run_python(code: str) -> list:
        """Execute Python in the sandbox (numpy/scipy/skimage/matplotlib). The working dir
        persists across calls. Read inputs and write outputs (importance_map.npy, figures)
        as files there. Save a matplotlib PNG to SEE a result — it's returned as an image."""
        r = backend.execute(code)                               # OUR subprocess runs the code
        blocks = [{"type": "text", "text": r["stdout"]}]        # numbers/errors → model
        for b64 in r["images"]:                                 # figures → model (vision)
            blocks.append({"type": "image_url",
                           "image_url": {"url": f"data:image/png;base64,{b64}"}})
        return blocks
    return run_python

def extract_run_python_code(messages) -> str:
    cells = [call["args"].get("code", "")
             for m in messages
             for call in (getattr(m, "tool_calls", None) or [])
             if call["name"] == "run_python"]
    return "\n\n# --- next cell ---\n".join(cells)