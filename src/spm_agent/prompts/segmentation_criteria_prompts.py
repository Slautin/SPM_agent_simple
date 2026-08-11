from langchain_core.messages import SystemMessage, HumanMessage
from spm_agent.utils.image_utils import image_path_to_data_url
from spm_agent.utils.message_utils import with_context

def _fmt(v):
    return f"{v:.3g}" if isinstance(v, (int, float)) else str(v)

CRITERIA_SEG_SYSTEM_PROMPT = (
    "You are a SPM/PFM data analyst. For a given experimental task, first DEFINE explicit "
    "segmentation criteria, then APPLY them to segment the scan into physically meaningful "
    "region classes.\n\n"

    "Work like an analyst in a notebook using ONE tool: run_python "
    "(numpy / scipy / scikit-image / matplotlib). Each call is a FRESH process; the working "
    "directory persists — re-load inputs each call, save intermediates. To SEE a result, save "
    "a matplotlib PNG; it is returned to you as an image. Inspect data via shapes, statistics "
    "and figures — never print whole arrays.\n\n"

    "Phase 1 — CRITERIA: Identify the region classes relevant to the task and define measurable "
    "segmentation criteria for each. Criteria must be grounded in the observed data and have a "
    "physically meaningful interpretation consistent with the measurement modality. Verify all "
    "thresholds or rules from the data rather than assuming them; clearly label any empirical heuristics.\n\n"

    "Phase 2 — SEGMENT: implement exactly the rules from Phase 1. Classes must be mutually "
    "exclusive (resolve overlaps by stated priority). Visually verify with an overlay figure "
    "before finalizing.\n\n"

    "Deliverable — when confident, produce exactly:\n"
    "  1. criteria.json : {\"task\": str, \"classes\": [{\"id\": int>0, \"name\", \"definition\"}], "
    "\"criteria\": [{\"name\", \"channels\": [ids], \"rule\": readable expression as implemented, "
    "\"rationale\": physical mechanism}]}\n"
    "  2. labels.npy : int16 (H, W) on the SAME pixel grid; 0 = unassigned, values = class ids. "
    "Every declared class must be present; no undeclared labels.\n"
    "  3. a short paragraph: the criteria chosen and why they serve the task.\n\n"

    "IMAGE ANALYSIS ONLY, in pixel space. No coordinates, no physical units in outputs. "
    "Figures are expensive: dpi <= 150.\n"
)

def build_criteria_system_message(ctx=None) -> SystemMessage:
    system_message = with_context(CRITERIA_SEG_SYSTEM_PROMPT, ctx)
    return SystemMessage(content=system_message)

def build_criteria_human_message(experiment_task, channels, segmentation_results) -> HumanMessage:
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
        f"Produce criteria.json and labels.npy (grid {grid}) for the task above."
    )

    content = [{"type": "text", "text": text}]
    for seg in segmentation_results.values():                 # overlays → vision grounding
        if seg.get("overlay_path"):
            content.append({"type": "text",
                "text": f"Overlay — mask '{stask}' (red) over channel '{seg.get('channel')}':"})
            content.append({"type": "image_url",
                            "image_url": {"url": image_path_to_data_url(seg["overlay_path"])}}) # type: ignore
    return HumanMessage(content=content) # type: ignore