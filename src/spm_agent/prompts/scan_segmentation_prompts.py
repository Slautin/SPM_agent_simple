import numpy as np

from langchain_core.messages import SystemMessage, HumanMessage
from spm_agent.tools.segmentation_toolbox import SegSession
from spm_agent.utils.image_utils import render_overlay_datauri

IMAGE_SEGMENTATION_SYSTEM_PROMPT = (
        "You are an SPM/PFM segmentation agent doing BASIC image analysis. You get ONE channel "
        "image and ONE task: produce a binary mask using the tools. Compose primitives freely.\n"
        # "- Thin BOUNDARY features (domain walls): compute_gradient_magnitude then threshold ('above').\n"
        # "- REGION features (domains): smooth, threshold ('otsu'), clean with morphology.\n"
        # "- threshold_image(units='absolute') thresholds RAW physical values — prefer it when you want "
        # "a reproducible, file-independent mask.\n"
        "- Check mask_summary. Use <12 ops, then STOP and give "
        "one short paragraph: method, parameters, why."
    )

def build_segmentation_system_message(vision_in_loop: bool) -> SystemMessage:
    txt = IMAGE_SEGMENTATION_SYSTEM_PROMPT
    
    if vision_in_loop:
        txt += "\n- You may call show_overlay() to SEE the current mask; use it sparingly."
    return SystemMessage(content=txt)

def build_segmentation_human_message(task: str, session: SegSession) -> HumanMessage:   # CHANGED: takes session
    raw = session.raw; f = raw[np.isfinite(raw)]
    stats = (f"shape={raw.shape}, mean={f.mean():.3g}, std={f.std():.3g}, "
             f"p1={np.percentile(f,1):.3g}, p99={np.percentile(f,99):.3g}") if f.size else "no finite data"
    uri = render_overlay_datauri(session.view, np.zeros_like(session.mask))  # show the robust view
    return HumanMessage(content=[
        {"type": "text", "text": f"Segmentation task: {task}\n"
                                 f"Channel statistics (raw units): {stats}\n"
                                 f"Produce a binary mask of this feature using the tools."},
        {"type": "image_url", "image_url": {"url": uri}}])