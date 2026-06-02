from pathlib import Path
import os
import numpy as np
from PIL import Image as PILImage

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
    norm_data = (data - data.min())/np.ptp(data)
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