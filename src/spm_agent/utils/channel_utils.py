from spm_agent.states.image_analysis_state import Channel
from spm_agent.utils.image_utils import image_path_to_data_url
from spm_agent.config import CASHE_DIR

import os
import numpy as np

def channel_to_text_block(channel_id: str, channel: Channel) -> dict:
    stats = channel.get("stats", "")
    return {
        "type": "text",
        "text": (
            f"Channel ID: {channel_id}\n"
            f"Data type: {channel.get('data_type')}\n"
            f"Title: {channel.get('title')}\n"
            f"Units: {channel.get('units')}\n"
            f"Shape: {str(channel.get('shape'))}\n"
            "Basic statistics:\n"
            f"  min: {stats.get('min')}\n"
            f"  max: {stats.get('max')}\n"
            f"  mean: {stats.get('mean')}\n"
            f"  std: {stats.get('std')}\n"
            f"  p01: {stats.get('p01')}\n"
            f"  p99: {stats.get('p99')}\n\n"
            f"The next image is the preview for {channel_id}."
        ),
    }

def channel_to_image_block(channel: Channel) -> dict:
    image_url = image_path_to_data_url(channel['preview_path'])
    return {
        "type": "image_url",
        "image_url": {
            'url': image_url
            }
    }

def save_array(channel) -> dict:
    """Cashe RAW channel array for later analysis on image segmentation"""

    CASHE_DIR.mkdir(parents=True, exist_ok=True)

    title = channel['title']
    arr_path = os.path.join(CASHE_DIR, f'{title}.npy')

    data = np.asarray(channel['data'], dtype=np.float32)
    try:
        np.save(arr_path, data)
        return {"ok": True, "path": str(arr_path), "error": None}
    except Exception as exc:
        return {"ok": False, "path": str(arr_path), "error": str(exc)}


def load_array(path: str) -> np.ndarray:
    """Load a cached raw channel array."""
    return np.load(path)

    


