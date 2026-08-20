from langchain_core.messages import SystemMessage, HumanMessage
from spm_agent.utils.message_utils import with_context

MAX_LOOP_HISTORY = 5

LOOP_PLAN_SYSTEM_PROMPT = (
    "You are the microscopist configuring the next SS-PFM hysteresis loop. The decision "
    "step has chosen to measure a loop and a deterministic pick has already fixed WHERE; "
    "you choose the switching waveform.\n\n"

    "You receive: the decision and its reasoning, where the tip is going and why, the "
    "current waveform settings, and for loops already measured on this scan what "
    "waveform was requested, what the instrument ran with, and how that loop turned out — "
    "its quality class, listed issues, and extracted parameters.\n\n"

    "Return the waveform you want. Restate every field, copying unchanged the ones you do "
    "not intend to move. Be reasonable and do not change params without need.\n\n"

    "Your levers, and what they trade:\n"
    "  v_dc_max_v — peak switching bias. Too low and the loop never saturates or does not "
    "switch at all; too high risks dielectric breakdown, electrochemical damage and more. "
    "Raise it only when the evidence says the loop is unsaturated or not switching.\n"
    "  frequency_hz — sweep rate. Slower sweeps give the polarization time to respond and "
    "reduce apparent coercive voltage inflation; faster sweeps cost less time.\n"
    "  var1_pulsetime_s — duration of each bias step. Longer settling suppresses transient "
    " contributions but reduce the gap length for the off-field signal acquiring.\n"
    "  n_cycles — averaging. More cycles reduce noise but costs time.\n"
    "  var0_phase — waveform phase offset; do not change it without significant reasons.\n\n"

    "Diagnostic notes:\n"
    "- 'unsaturated' means the branches never flatten at the extremes. \n"
    "- 'no_switching' after an adequate bias usually means the site is not ferroelectric, "
    "or contact is poor — do not simply keep raising the bias.\n"
    "- Each loop characterizes its own site; loops at different sites are not repeats.\n\n"

    "Rules:\n"
    "- Diagnose only from the loops listed. Do not assume behaviour that was not measured.\n"
    "- Change only what your diagnosis justifies. An unchanged waveform is a valid answer.\n"
    "- Never propose a location: that is fixed before you are called."
)


def build_loop_plan_system_message(ctx: str | None = None) -> SystemMessage:
    return SystemMessage(content=with_context(LOOP_PLAN_SYSTEM_PROMPT, ctx))


def _wave_line(ls) -> str:
    return (f"v_dc_max {ls.v_dc_max_v:.2f} V, freq {ls.frequency_hz:.3f} Hz, "
            f"pulse {ls.var1_pulsetime_s*1e3:.1f} ms, phase {ls.var0_phase:.2f}, "
            f"{ls.n_cycles} cycles")


def _loop_block(label, rec) -> str:
    req = rec.get("requested_params")
    ip  = rec.get("instrument_params")
    rv  = rec.get("loop_review")
    off = (rec.get("loop_params") or {}).get("off_field")

    lines = [f"  Loop {label} at px{tuple(rec.get('pixel_yx', ()))}"]
    lines.append("      requested: " + (
        _wave_line(req.loop_settings) if req is not None
        else "(first loop — operator settings adopted)"))
    if ip is not None:
        lines.append(f"      achieved : {_wave_line(ip.loop_settings)}")
    if req is not None and getattr(req, "diagnosis", None):
        lines.append(f"      said then: {req.diagnosis}")
    if rv is not None:
        fe = "ferroelectric" if rv.is_ferroelectric_like else "NOT ferroelectric"
        lines.append(f"      outcome  : {rv.loop_quality}, {fe}")
        if rv.issues:
            lines.append("      issues   : " + "; ".join(rv.issues[:4]))
    if off is not None:
        lines.append(f"      params   : Vc {off.v_c_rising:+.2f}/{off.v_c_falling:+.2f} V, "
                     f"width {off.loop_width_v:+.2f} V, height {off.loop_height_m:.3g}")
    return "\n".join(lines)


def build_loop_plan_human_message(state, pick_info: dict | None = None) -> HumanMessage:
    """Deterministic input: decision brief -> target site -> current waveform ->
    per-loop (requested -> achieved -> outcome)."""
    decision = state["decision_records"][-1]
    live     = state["instrument_state"]
    recs     = state.get("experimental_records", [])
    scans    = [r for r in recs if r.get("kind") == "image"]
    scan_idx = scans[-1]["scan_index"] if scans else 0
    loops    = [r for r in recs if r.get("kind") == "loop"
                and r.get("scan_index") == scan_idx]
    shown    = loops[-MAX_LOOP_HISTORY:]
    pending = state["pending"]

    text = [
        "EXPERIMENT TASK: " + "; ".join(state.get("experiment_tasks", [])
                                        or ["(none given)"]),
        "",
        f"DECISION: measure a loop serving '{decision.target_criterion}' "
        f"({decision.target_strategy})",
        f"      understanding: {decision.understanding}",
        f"      reasoning: {decision.reasoning}",
    ]

    text += [
        "",
        f"TARGET SITE (already fixed — not yours to choose): "
        f"px{tuple(pending['pixel_yx'])}, serving '{decision.target_criterion}' "
        f"({decision.target_strategy})",
    ]
    
    text += [
        "",
        "CURRENT WAVEFORM",
        f"      {_wave_line(live.loop_settings)}",
        "",
        f"LOOPS ALREADY MEASURED ON Scan_{scan_idx} (requested -> achieved -> outcome)"
        + (f"   [{len(loops) - len(shown)} earlier omitted]" if len(loops) > len(shown) else ""),
    ]
    text += [_loop_block(f"{scan_idx}.{i}", r)
             for i, r in enumerate(shown, len(loops) - len(shown) + 1)] or ["  none"]
    

    return HumanMessage(content="\n".join(text))