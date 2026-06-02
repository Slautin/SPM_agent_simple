from spm_agent.states.image_analysis_state import Channel
from spm_agent.utils.image_utils import image_path_to_data_url

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
