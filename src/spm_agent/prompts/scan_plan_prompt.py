from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from spm_agent.utils.channel_utils import channel_to_text_block, channel_to_image_block
from spm_agent.utils.message_utils import with_context

MAX_SCAN_HISTORY = 4        # older scans drop out; the recent ones carry the signal
MAX_WARNINGS     = 6
MAX_PREVIEWS     = 3

SCAN_PLAN_SYSTEM_PROMPT = (
    "You are the microscopist configuring the next DART-PFM scan. A separate decision step "
    "has already chosen WHAT to measure and WHERE; you choose acquisition parameters to measure it.\n\n"

    "You receive: the decision and its reasoning, the current instrument settings, and for "
    "each previous scan what was requested, what the instrument actually ran with, and how "
    "that scan turned out — channel statistics, channel-recommendation warnings, and "
    "segmentation coverage. Preview images of the most recent scan are attached.\n\n"

    "Return the parameters you want to use for the next scan. Restate every field, copying "
    "unchanged the ones you do not intend to move. Locked fields are listed in the input; a "
    "plan that changes them is rejected.\n\n"

    "Your levers, and what they trade:\n"
    "  drive_amplitude_v — PFM response scales with it. Too low leaves the amplitude in the "
    "noise floor; too high can perturb or write domains while imaging.\n"
    "  pixels — resolution against time. Scan time = pixels / scan_rate_hz seconds.\n"
    "  scan_rate_hz — too fast and the Z ans DART feedback cannot track (streaks, smeared "
    "topography); too slow and experiment will be too long and expensive.\n"
    "  dart_width_hz — the two drive frequencies straddle the contact resonance by this "
    "width. Too narrow loses tracking on rough topography; too wide costs sensitivity.\n"
    "  dart_igain — DART frequency-tracking gain: too low lags the resonance, too high "
    "rings.\n\n"

    "Rules:\n"
    "- Diagnose only from the evidence given; do not assume defects that are not visible in "
    "the statistics, warnings or previews.\n"
    "- Change only what your diagnosis justifies. An unchanged plan is a valid answer when "
    "the previous acquisition was sound.\n"
    "- If a previous plan already tried a remedy and the outcome did not improve, do not "
    "repeat it — try a different lever or state that the limit is elsewhere.\n"
    "- Never propose the frame centre: the location is set elsewhere. Propose scan_size_m "
    "only when the decision's frame_action asks for a zoom; otherwise copy it unchanged.\n"
    "- In `reasoning`, say what each changed value is expected to fix."
)

def build_scan_plan_system_message(ctx: str | None = None) -> SystemMessage:
    return SystemMessage(content=with_context(SCAN_PLAN_SYSTEM_PROMPT, ctx))

def build_scan_plan_human_message(state, locked: tuple = ()) -> HumanMessage:
    """Deterministic input: decision brief -> current settings -> per-scan
    (requested -> achieved -> outcome) -> previews of the most recent scan."""
    decision = state["decision_records"][-1]
    live     = state["instrument_state"]
    recs     = state.get("experimental_records", [])
    scans    = [r for r in recs if r.get("kind") == "image"]
    tasks    = state.get("experiment_tasks", [])

    shown  = scans[-MAX_SCAN_HISTORY:]
    hidden = len(scans) - len(shown)

    PERMISSION = {
        "hold":     "The frame is unchanged. Set only acquisition quality parameters.",
        "zoom_in":  "Propose a SMALLER scan_size_m . The centre is "
                    "fixed, so the new frame is the middle of the current one.",
        "zoom_out": "Propose a LARGER scan_size_m. The centre is "
                    "fixed, so the current frame stays in the middle.",
    }

    text = [
        "EXPERIMENT TASK: " + ("; ".join(tasks) if tasks else "(none given)"),
        "",
        f"DECISION: action={decision.action}, frame_action={decision.frame_action}",
        f"      understanding: {decision.understanding}",
        f"      open questions: {decision.open_questions}",
        f"      reasoning: {decision.reasoning}",
        "",
        "CURRENT INSTRUMENT SETTINGS",
        f"      frame     : {_frame_line(live.scan_settings)}",
        f"      excitation: {_exc_line(live.pfm_excitation)}",
        f"      contact   : setpoint {live.contact_feedback.setpoint_v:.3f} V, "
        f"igain {live.contact_feedback.gain:g}   (reported only, not settable)",
        "",
        f"PREVIOUS SCANS (requested -> achieved -> outcome)"
        + (f"   [{hidden} earlier scan(s) omitted]" if hidden > 0 else ""),
    ]
    text += [_scan_block(r) for r in shown] or ["  none"]

    if locked:
        text += ["",
                 f"FRAME: {PERMISSION.get(decision.frame_action, '')}",
                 "LOCKED — copy these unchanged from CURRENT INSTRUMENT SETTINGS: "
                 + ", ".join(locked)]

    content = [{"type": "text", "text": "\n".join(text)}]

    if scans:                    # previews: latest scan only — the channel cache is overwritten
        latest = scans[-1]
        chans  = latest.get("file_channels", {}) or {}
        heads  = [c for c in _primary_channels(latest)
                  if chans[c].get("preview_path") and Path(chans[c]["preview_path"]).exists()]
        if heads:
            content.append({"type": "text",
                            "text": f"Preview images of the most recent scan "
                                    f"(Scan_{latest.get('scan_index', '?')}) follow."})
            for ch_id in heads:
                content.append(channel_to_text_block(ch_id, chans[ch_id]))
                content.append(channel_to_image_block(chans[ch_id]))

    return HumanMessage(content=content)     # type: ignore

# --------------------------- auxilary functions for human message---------------------------

def _frame_line(ss) -> str:
    px_nm = ss.scan_size_m / ss.pixels * 1e9 if ss.pixels else float("nan")
    secs  = ss.pixels / ss.scan_rate_hz if ss.scan_rate_hz else float("nan")
    return (f"{ss.scan_size_m*1e6:.2f} um @ {ss.pixels} px ({px_nm:.1f} nm/px), "
            f"centre ({ss.x_scan_center_m*1e6:+.3f}, {ss.y_scan_center_m*1e6:+.3f}) um, "
            f"{ss.scan_rate_hz:.3f} Hz (~{secs:.0f} s per image)")


def _exc_line(pe) -> str:
    return (f"v_ac {pe.drive_amplitude_v:.3f} V, f_DART {pe.drive_frequency_hz/1e3:.1f} kHz, "
            f"width {pe.dart_width_hz/1e3:.1f} kHz, dart_igain {pe.dart_igain:g}")


def _stats_line(ch_id, ch) -> str:
    s = ch.get("stats") or {}
    if s.get("ok") is False:
        return f"      {ch_id}: unusable ({s.get('error')})"
    g = lambda k: "n/a" if s.get(k) is None else f"{s[k]:.3g}"
    return (f"      {ch_id} [{ch.get('units')}]: p01 {g('p01')}, p99 {g('p99')}, "
            f"mean {g('mean')}, std {g('std')}")


def _primary_channels(record, limit=MAX_PREVIEWS) -> list[str]:
    chans = record.get("file_channels", {}) or {}
    rec   = record.get("channel_recommendations")
    out: list[str] = []
    for r in (rec.task_recommendation if rec else []):
        if r.feasible and r.primary_channel in chans and r.primary_channel not in out:
            out.append(r.primary_channel)
    return (out or list(chans))[:limit]


def _rec_lines(record) -> list[str]:
    rec = record.get("channel_recommendations")
    if not rec:
        return []
    per = "; ".join(
        f"{r.task.replace(' segmentation', '').replace(' detection', '')}"
        f" -> {r.primary_channel or 'none'} ({r.confidence:.2f})"
        for r in rec.task_recommendation if r.feasible)
    lines = [f"      channel recs (overall {rec.overall_confidence:.2f}): {per or 'none feasible'}"]
    warn = list(rec.global_warnings)
    for r in rec.task_recommendation:
        warn += list(r.warnings)
    if warn:
        lines.append("      warnings: " + "; ".join(warn[:MAX_WARNINGS]))
    return lines


def _seg_line(record) -> str | None:
    seg = record.get("segmentation_results") or {}
    if not seg:
        return None
    return "      segmentation: " + "; ".join(
        f"'{t}': {r['coverage']:.0%}, {r['n_regions']} regions" for t, r in seg.items())


def _scan_block(record) -> str:
    req = record.get("requested_params")          # ScanPlan of that scan, if stored
    ip  = record.get("instrument_params")         # post-acquisition snapshot
    lines = [f"  Scan_{record.get('scan_index', '?')}"]

    lines.append("      requested: " + (
        f"{_frame_line(req.scan_settings)} | {_exc_line(req.pfm_excitation)}"
        if req is not None else "(first scan — operator settings adopted)"))
    lines.append("      achieved : " + (
        f"{_frame_line(ip.scan_settings)} | {_exc_line(ip.pfm_excitation)}"
        if ip is not None else "n/a"))
    if req is not None and getattr(req, "diagnosis", None):
        lines.append(f"      said then: {req.diagnosis}")

    chans = record.get("file_channels", {}) or {}
    for ch_id in _primary_channels(record):
        lines.append(_stats_line(ch_id, chans[ch_id]))
    lines += _rec_lines(record)
    seg = _seg_line(record)
    if seg:
        lines.append(seg)
    return "\n".join(lines)


