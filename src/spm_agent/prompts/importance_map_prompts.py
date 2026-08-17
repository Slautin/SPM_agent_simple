from langchain_core.messages import SystemMessage, HumanMessage
from spm_agent.utils.image_utils import image_path_to_data_url
from spm_agent.utils.message_utils import with_context

def _fmt(v):
    return f"{v:.3g}" if isinstance(v, (int, float)) else str(v)

IMPORTANCE_MAP_SYSTEM_PROMPT = (
    "You are an SPM/PFM data analyst. Given an experimental task and a scan, decide WHERE the next "
    "point measurements should be taken, and express that as scoring maps on the scan's pixel grid.\n\n"

    "TOOL\n"
    "run_python (numpy / scipy / scikit-image / matplotlib). Each call is a FRESH process; the "
    "working directory persists. In your first build call save the shared preprocessing (loaded "
    "channels, normalizations, the artifact-band mask) to prep.npz and load it in later cells "
    "instead of re-deriving it; rebuild it if its inputs change. Save a PNG to see a result; it is "
    "returned to you as an image. Inspect data through shapes, statistics and figures — never "
    "print whole arrays.\n\n"

    "WHAT TO PRODUCE\n"
    "1-4 TASK criteria — each float32 (H, W) in [0,1], higher = more informative for THIS task. "
    "Construct the smallest non-redundant set that comprehensively represents the scientific task."
    "Add criteria only when they capture additional task-relevant physical information; "
    "do not generate criteria merely to reach a target number"
    "Each must rest on a physical mechanism: what causes the contrast, and why it matters for this "
    "task. Check the mechanism against the measurement physics of the technique and discard what "
    "it contradicts; label heuristics as such. Criteria must not duplicate one another, and they "
    "must NOT mask or zero anything — hazards are the safety map's job, and the two are "
    "multiplied afterwards, so masking in both applies the penalty twice.\n"
    "ONE SAFETY map — higher = more likely to succeed. Low ONLY where the instrument cannot "
    "measure: contact or tracking loss, artifact bands, scratches, debris, etc. Weak signal alone is "
    "not unsafe and can be related to real physics; confirm failure before suppressing."
    "A clean scan stays near 1. State the numeric thresholds you used in its rationale.\n\n"

    "DELIVERABLES (in the working directory)\n"
    "  components.npy  : float32 (K, H, W), the task criteria, finite, in [0,1]\n"
    "  components.json : {\"names\": [K snake_case], \"rationale\": [K one sentence, each stating "
    "the physical mechanism]} — same order as the array\n"
    "  safety.npy      : float32 (H, W), finite, in [0,1]\n"
    "  safety.json     : {\"name\": str, \"rationale\": str}\n"
    "  a short closing paragraph: the criteria you chose and why they serve the task.\n\n"

    "RULES\n"
    "  - Save the deliverables as soon as they are valid, then improve them in place. Never hold "
    "results until the end.\n"
    "  - Do NOT combine the maps into a final score; that is computed outside.\n"
    "  - Pixel space only: no coordinates, no ranked points, no physical units in the outputs.\n"
    "  - Channels may contain NaN — handle it; the saved maps must be finite.\n"
    "  - Arrays are row-0-at-top; use matplotlib's default origin so your figures match the "
    "previews you were given.\n"
    "  - Compute everything from the data. Never fabricate a number, threshold or mechanism.\n"
)

def build_importance_system_message(ctx=None) -> SystemMessage:
    return SystemMessage(content=with_context(IMPORTANCE_MAP_SYSTEM_PROMPT, ctx))


def _prior_lines(recs, max_chars=420) -> list[str]:
    """Distil the channel-recommendation report into a few actionable lines."""
    if not recs:
        return []
    out, warn = [], list(recs.global_warnings)
    for r in recs.task_recommendation:
        why = " ".join((r.reasoning or "").split())[:max_chars]
        if r.feasible:
            sec = f" (+{', '.join(r.secondary_channels)})" if r.secondary_channels else ""
            out.append(f"  FEASIBLE   {r.task}: '{r.primary_channel}'{sec}, conf {r.confidence:.2f} — {why}")
        else:
            out.append(f"  not found  {r.task}: {why}")
        warn += list(r.warnings)
    if warn:
        out.append("  warnings: " + "; ".join(dict.fromkeys(warn))[:400])
    return out

def _known_features(recs, max_chars: int = 360) -> str:
    """Block 3: what the recommendation step established. Compact by design."""
    if not recs:
        return ""
    found, missing, lines = [], [], []
    for r in recs.task_recommendation:
        why = " ".join((r.reasoning or "").split())[:max_chars]
        if r.feasible:
            found.append(f"  {r.task}: '{r.primary_channel}' (conf {r.confidence:.2f}) — {why}")
        else:
            missing.append(r.task)
    lines += found
    if missing:
        lines.append("  not found in this scan: " + "; ".join(missing))
    warn = list(recs.global_warnings) + [w for r in recs.task_recommendation for w in r.warnings]
    if warn:
        lines.append("  warnings: " + "; ".join(dict.fromkeys(warn))[:300])
    return "\n".join(lines)


def build_importance_human_message(experiment_task, channels, recs=None,
                                   preview_grid_path=None, safety_reuse_path=None) -> HumanMessage:
    grid = next(iter(channels.values())).get("shape") if channels else "?"

    ch_lines = []
    for cid, ch in channels.items():
        s = ch.get("stats", {})
        ch_lines.append(
            f"  '{cid}': {ch.get('title')} [{ch.get('units')}]  "
            f"p01={_fmt(s.get('p01'))} p99={_fmt(s.get('p99'))} "
            f"mean={_fmt(s.get('mean'))} std={_fmt(s.get('std'))}\n"
            f"      {ch.get('array_path')}")

    known = _known_features(recs)
    known_block = (
        "[3] KNOWN FEATURES — from a separate channel-recommendation step, which judged which\n"
        "    classical features are findable here. Evidence, not instruction: every channel above\n"
        "    is available to you, and you may contradict it if the data says otherwise.\n"
        + known + "\n\n") if known else ""

    safety_block = (
        f"[4] SAFETY — already computed for this scan ({safety_reuse_path}). "
        "Produce TASK criteria only.\n\n") if safety_reuse_path else ""

    text = (
        f"[1] TASK\n{experiment_task}\n\n"
        f"[2] CHANNELS — all arrays share the pixel grid {grid}; load them with numpy.\n"
        + "\n".join(ch_lines) + "\n\n"
        + known_block + safety_block
        + "The image below is a labelled contact sheet of all channels, same orientation as the "
          "arrays.")

    content = [{"type": "text", "text": text}]
    if preview_grid_path:
        content.append({"type": "image_url",
                        "image_url": {"url": image_path_to_data_url(preview_grid_path)}}) # type: ignore
    return HumanMessage(content=content)   # type: ignore
