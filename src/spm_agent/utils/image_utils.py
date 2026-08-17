from pathlib import Path
import os
import numpy as np
from PIL import Image as PILImage
import io

from spm_agent.config import CASHE_DIR

import base64
from pathlib import Path

def get_channel_stats(array) -> dict:
    """
    Get basic image stats
    """
    return {
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "p01": float(np.nanpercentile(array, 1)),
        "p99": float(np.nanpercentile(array, 99)),
    }

def save_preview(channel) -> dict:
    """
    Save a PNG preview to the cache for later vision-model review.
    """
    from spm_agent.config import CASHE_DIR
    CASHE_DIR.mkdir(parents=True, exist_ok=True)
    
    title = channel['title']
    im_path = os.path.join(CASHE_DIR, f'{title}.png')

    data = np.array(channel['data'])
    finite = data[np.isfinite(data)]
    lo, hi = np.percentile(finite, (1, 99)) if finite.size else (0.0, 1.0)  # robust contrast
    if hi <= lo:                          # flat-image / zero-range guard
        hi = lo + 1e-12

    norm_data = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    norm_data = np.nan_to_num(norm_data, nan=0.0)   # non-scanned / NaN -> black

    image = PILImage.fromarray((norm_data * 255).astype(np.uint8))

    try:
        image.save(im_path, format="PNG")

        return {
            "ok": True,
            "path": str(im_path),
            "error": None,
        }
    
    except Exception as exc:
        return {
            "ok": False,
            "path": str(im_path),
            "error": str(exc),
        }

def image_path_to_data_url(path: str | Path) -> str:
    path = Path(path)

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/png;base64,{b64}"

def render_overlay_datauri(view01, mask, max_px=256) -> str:   # CHANGED: takes the view
    rgb = (np.stack([view01] * 3, -1) * 255).astype(np.uint8)
    rgb[mask] = (0.4 * rgb[mask] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    img = PILImage.fromarray(rgb); img.thumbnail((max_px, max_px))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def save_mask_and_overlay(view01, mask, task, out_dir):        # CHANGED: view01 param name
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mask_path, overlay_path = out_dir / f"{task}_mask.npy", out_dir / f"{task}_overlay.png"
    np.save(mask_path, mask.astype(np.uint8))
    rgb = (np.stack([view01] * 3, -1) * 255).astype(np.uint8)
    rgb[mask] = (0.4 * rgb[mask] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    PILImage.fromarray(rgb).save(overlay_path)
    return str(mask_path), str(overlay_path)

def norm01(a):                       # NaN-aware min-max → [0,1]
    a = a.astype(np.float32)
    f = a[np.isfinite(a)]
    if f.size == 0: return np.zeros_like(a)
    lo, hi = float(f.min()), float(f.max())
    return np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)

def to_view01(raw, p=(1, 99)):        # robust display image (what the model sees)
    f = raw[np.isfinite(raw)]
    lo, hi = np.percentile(f, p) if f.size else (0.0, 1.0)
    if hi <= lo: hi = lo + 1e-12
    return np.nan_to_num(np.clip((raw - lo) / (hi - lo), 0, 1))


