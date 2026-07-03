from langchain_core.messages import SystemMessage, HumanMessage
from spm_agent.utils.image_utils import image_path_to_data_url

def _fmt(v):
    return f"{v:.3g}" if isinstance(v, (int, float)) else str(v)

IMPORTANCE_MAP_SYSTEM_PROMPT = (
    "You are a SPM/PFM data analyst deciding which locations are most promising for the next "
    "local measurements, for a given experimental task.\n\n"

    "Work like an analyst in a notebook using ONE tool: run_python "
    "(numpy / scipy / scikit-image / matplotlib). Write code, read the printed numbers and the "
    "figures it returns, and iterate. Each call is a FRESH process; the working directory "
    "persists — re-load inputs each call and save intermediates to disk to reuse them. To SEE a "
    "result, save a matplotlib PNG in the working directory; it is returned to you as an image.\n\n"

    "Inspect data through shapes, summary statistics, and figures — never print whole arrays, "
    "it wastes the context.\n\n"

    "Inputs (channel arrays as float physical data, and segmentation masks) are given as file "
    "paths in the next message, with a note on what each mask represents. Load them with numpy. "
    "Decide the scoring criteria yourself from the task and the data; there is no fixed recipe.\n\n"

    "Deliverable — when confident, produce exactly:\n"
    "  1. importance_map.npy : a 2D float array on the SAME pixel grid as the inputs, finite "
    "values 0-100, higher = more worth measuring for THIS task. Save it in the working directory.\n"
    "  2. a short paragraph naming the criteria you chose and why they serve the task.\n\n"

    "IMAGE ANALYSIS ONLY, in pixel space. Do not output coordinates, rank points, or use physical/"
    "microscope units — produce only the map. Compute everything from the data; do not fabricate."
)

def build_importance_system_message() -> SystemMessage:
    return SystemMessage(content=IMPORTANCE_MAP_SYSTEM_PROMPT)

def build_importance_human_message(experiment_task, channels, segmentation_results) -> HumanMessage:
    grid = next(iter(channels.values())).get("shape") if channels else "?"

    ch_lines = []
    for cid, ch in channels.items():
        s = ch.get("stats", {})
        ch_lines.append(
            f"  '{cid}': {ch.get('title')} [{ch.get('units')}]  "
            f"min={_fmt(s.get('min'))} max={_fmt(s.get('max'))} mean={_fmt(s.get('mean'))} "
            f"p01={_fmt(s.get('p01'))} p99={_fmt(s.get('p99'))}\n"
            f"      file: {ch.get('array_path')}"
        )

    mask_lines = []
    for stask, seg in segmentation_results.items():
        mask_lines.append(
            f"  '{stask}' — channel '{seg.get('channel')}', coverage {seg.get('coverage')}\n"
            f"      file: {seg.get('mask_path')}"
        )

    text = (
        f"Experiment task: {experiment_task}\n\n"
        f"All arrays share the pixel grid {grid}. Load them with numpy from the paths below.\n\n"
        "Channels (float, physical units):\n" + "\n".join(ch_lines) + "\n\n"
        "Segmentation masks (what each represents):\n" + "\n".join(mask_lines) + "\n\n"
        "Produce importance_map.npy for the task above. The overlays that follow "
        "(mask over channel) are for visual context."
    )

    content = [{"type": "text", "text": text}]
    for seg in segmentation_results.values():                 # overlays → vision grounding
        if seg.get("overlay_path"):
            content.append({"type": "text",
                "text": f"Overlay — mask '{stask}' (red) over channel '{seg.get('channel')}':"})
            content.append({"type": "image_url",
                            "image_url": {"url": image_path_to_data_url(seg["overlay_path"])}}) # type: ignore
    return HumanMessage(content=content) # type: ignore