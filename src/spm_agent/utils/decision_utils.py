import difflib
import json
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
from pathlib import Path
#from spm_agent.utils.image_utils import image_path_to_data_url
from spm_agent.schemas.experimental_decision import ExperimentDecision

from typing import get_args
from spm_agent.schemas.loop_review import LoopReview

def resolve_criterion(decision: ExperimentDecision, names: list[str]) -> str | None:
    """Validated criterion name, or None -> deterministic weighted-map fallback (log it)."""
    if decision.action != "loop":
        return None
    t = (decision.target_criterion or "").strip()
    if t in names:
        return t
    close = difflib.get_close_matches(t, names, n=1, cutoff=0.6)
    return close[0] if close else None

#digest

def _fmt(v, unit=""):
    return "n/a" if v is None else f"{v:+.3g}{unit}"


def _phi(components, y, x, patch=3):
    """Per-component patch means at a pixel — same convention as the penalty."""
    y0, x0 = max(y - patch, 0), max(x - patch, 0)
    return components[:, y0:y+patch+1, x0:x+patch+1].mean(axis=(1, 2))

def _frame_line(scan):
    ip = scan.get("instrument_params")
    if ip is None:
        return "  frame: n/a"
    ss = ip.scan_settings
    return (f"  frame: {ss.scan_size_m*1e6:.2f} um @ {ss.pixels} px "
            f"({ss.scan_size_m/ss.pixels*1e9:.1f} nm/px), "
            f"centre ({ss.x_scan_center_m*1e6:+.3f}, {ss.y_scan_center_m*1e6:+.3f}) um")

def _criteria_of(scan):
    """(names, components, rationale) concatenated across all task entries."""
    names, rats, arrs = [], [], []
    for imap in scan.get("importance_maps", []):
        m = imap["components_meta"]
        names += list(m.names); rats += list(m.rationale)
        arrs.append(np.load(imap["components_path"]))
    return names, np.concatenate(arrs, axis=0), rats

# def _sampling_block(loops, names, components, patch=3):
#     if not loops:
#         return "  sampling per criterion: no loops yet"
#     phis = np.array([_phi(components, *r["pixel_yx"], patch) for r in loops])   # (L, K)
#     lines = []
#     for k, n in enumerate(names):
#         v = np.sort(phis[:, k])
#         hi, lo = int((v > 0.67).sum()), int((v < 0.33).sum())
#         lines.append(f"    - {n}: high {hi}, mid {len(v)-hi-lo}, low {lo}"
#                      f"   [{', '.join(f'{x:.2f}' for x in v)}]")
#     return ("  sampling per criterion (each loop's score; low <0.33, high >0.67):\n"
#             + "\n".join(lines))

def _map_bands(phi, lo=0.33, hi=0.67):
    """Share of the frame in each band — the same bands the measured line uses, so the
    two read against each other."""
    v = phi.ravel()
    return (f"{(v < lo).mean():.0%} low / "
            f"{((v >= lo) & (v <= hi)).mean():.0%} mid / "
            f"{(v > hi).mean():.0%} high")


def _sampling_block(loops, names, components, patch=3):
    """Per criterion: what the frame contains, and where the loops went."""
    phis = (np.array([_phi(components, *r["pixel_yx"], patch) for r in loops])
            if loops else None)
    lines = []
    for k, n in enumerate(names):
        lines.append(f"    - {n}")
        lines.append(f"        frame   : {_map_bands(components[k])}")
        if phis is None:
            lines.append("        measured: no loops yet")
            continue
        v = np.sort(phis[:, k])
        hi, lo = int((v > 0.67).sum()), int((v < 0.33).sum())
        lines.append(f"        measured: {lo} low / {len(v)-hi-lo} mid / {hi} high"
                     f"   [{', '.join(f'{x:.2f}' for x in v)}]")
    return ("  per criterion — 'frame' is the share of pixels in each band, 'measured' "
            "is where the loops landed (low <0.33, high >0.67):\n" + "\n".join(lines))

def _loop_line(label, rec, names, components, patch=3):
    py, px = rec["pixel_yx"]
    phi = _phi(components, py, px, patch)
    at = ", ".join(f"{n} {p:.2f}" for n, p in zip(names, phi))
    rv  = rec["loop_review"]
    off = rec["loop_params"]["off_field"]
    on  = rec["loop_params"].get("on_field")
    fe  = "ferroelectric" if rv.is_ferroelectric_like else "NOT ferroelectric"
    line = (f"  {label} px({py},{px}) | at: {at}\n"
            f"     off: {rv.loop_quality}, {fe} | Vc {_fmt(off.v_c_rising,'V')}/"
            f"{_fmt(off.v_c_falling,'V')} | width {_fmt(off.loop_width_v,'V')} | "
            f"imprint {_fmt(off.imprint_v,'V')} | height {off.loop_height_m:.3g}\n")
    if on:
        line += f"     on:  height {on.loop_height_m:.3g}, offset {_fmt(on.response_offset_m)}\n"
    if rv.issues:
        line += f"     issues: {'; '.join(rv.issues)}\n"
    return line + f"     \"{rv.interpretation}\""


def _scan_block(scan, loops, patch=3, current=False):
    entries = scan.get("importance_maps", [])
    names, comps, rats = _criteria_of(scan)
    si = scan["scan_index"]

    parts = [f"Scan_{si}{' (current)' if current else ''}", _frame_line(scan)]

    if len(entries) == 1:                       # single task — no need to name it per line
        parts.append("  criteria (name — rationale):")
        parts += [f"    - {n} — {r}" for n, r in zip(names, rats)]
    else:
        for imap in entries:
            m = imap["components_meta"]
            parts.append(f"  criteria for '{imap['experiment_task']}':")
            parts += [f"    - {n} — {r}" for n, r in zip(m.names, m.rationale)]

    for imap in entries:                        # what the deterministic exclusion actually is
        sm = imap.get("safety_meta")
        if sm is not None:
            parts.append(f"  safety: {sm.name} — {sm.rationale}")

    parts.append(_sampling_block(loops, names, comps, patch))
    parts.append(f"  loops ({len(loops)}):")
    parts += [_loop_line(f"{si}.{i}", r, names, comps, patch)
              for i, r in enumerate(loops, 1)] or ["  none"]
    return "\n".join(parts)

def build_decision_digest(state, patch=3) -> str:
    recs  = state.get("experimental_records", [])
    scans = [r for r in recs if r["kind"] == "image"]

    def loops_of(s):
        return [r for r in recs
                if r["kind"] == "loop" and r["scan_index"] == s["scan_index"]]

    last  = scans[-1]
    tasks = [imap["experiment_task"] for imap in last.get("importance_maps", [])]
    header = ("EXPERIMENT TASK: " + tasks[0] if len(tasks) == 1
              else "EXPERIMENT TASKS:\n  - " + "\n  - ".join(tasks))

    blocks = [_scan_block(s, loops_of(s), patch, current=(s is last)) for s in scans]

    return (f"{header}\n\nMEASURED RESULTS\n\n" + "\n\n".join(blocks)
            + f"\n\nCRITERIA (current scan, use exact names): "
              f"{', '.join(all_criteria(last))}")

    return text#content

def all_criteria(scan) -> list[str]:
    """Every criterion name on this scan, across all task entries."""
    return [n for imap in scan.get("importance_maps", [])
              for n in imap["components_meta"].names]


def find_criterion(scan, name: str):
    """(entry, index within that entry) for a criterion name.
    The entry also carries the safety map that belongs with that criterion."""
    for imap in scan.get("importance_maps", []):
        names = list(imap["components_meta"].names)
        if name in names:
            return imap, names.index(name)
    return None, None


# --------------------------- summary digest ---------------------------

_POOLED = (("v_c_rising", "V"), ("v_c_falling", "V"), ("loop_width_v", "V"),
           ("imprint_v", "V"), ("loop_height_m", ""), ("loop_area_per_cycle", ""))

QUALITY_CLASSES = get_args(LoopReview.model_fields["loop_quality"].annotation)


def _population_block(loops) -> str:
    """Cross-scan aggregates. The per-loop lines already carry every value; this is the
    population view, the one thing the model cannot reliably assemble from a list.
    n= is reported because the Vc fields are Optional — a loop that never switched
    legitimately has none, and that absence is itself a finding."""
    if not loops:
        return "  no loops measured"

    revs = [r["loop_review"] for r in loops]
    fe   = sum(r.is_ferroelectric_like for r in revs)
    qual = {q: sum(r.loop_quality == q for r in revs) for q in QUALITY_CLASSES}
    lines = [f"  {len(loops)} loop(s) | ferroelectric-like {fe}/{len(loops)} | quality "
             + ", ".join(f"{k} {n}" for k, n in qual.items() if n)]
    
    for field, unit in _POOLED:
        got = [v for v in (getattr(r["loop_params"]["off_field"], field, None)
                           for r in loops) if v is not None]
        if not got:
            lines.append(f"    {field}: not extractable on any loop")
            continue
        lines.append(f"    {field}: median {np.median(got):+.3g}{unit}, "
                     f"range [{min(got):+.3g}, {max(got):+.3g}]{unit}, "
                     f"n={len(got)}/{len(loops)}")
    return "\n".join(lines)


def _trail_block(decisions, recs) -> str:
    """What was asked for, in order, and what it produced. build_decision_digest omits
    this on purpose — decide reasons fresh — but the campaign narrative lives here."""
    produced = {}
    for r in recs:
        produced.setdefault(r.get("decision_index"), []).append(r)

    lines = []
    for i, d in enumerate(decisions):
        what = ({"scan": f"scan/{d.frame_action}",
                 "loop": f"loop {d.target_criterion}/{d.target_strategy}"}
                .get(d.action, "stop"))
        made = ", ".join(f"Scan_{r['scan_index']}" if r["kind"] == "image"
                         else f"loop on Scan_{r['scan_index']} px{tuple(r['pixel_yx'])}"
                         for r in produced.get(i, []))
        lines.append(f"  D{i:02d} {what} -> {made or 'nothing'}\n      {d.reasoning}")
    return "\n".join(lines)


def _coverage_line(scans) -> str:
    """Where the campaign actually looked — a real limitation once relocate moves the
    frame, and invisible from any single scan block."""
    fr = [s["instrument_params"].scan_settings for s in scans]
    xs = [f.x_scan_center_m for f in fr]
    ys = [f.y_scan_center_m for f in fr]
    hw = [f.scan_size_m / 2 for f in fr]
    return (f"  {len(scans)} scan(s) spanning "
            f"x [{min(x-h for x, h in zip(xs, hw))*1e6:+.2f}, "
            f"{max(x+h for x, h in zip(xs, hw))*1e6:+.2f}] um, "
            f"y [{min(y-h for y, h in zip(ys, hw))*1e6:+.2f}, "
            f"{max(y+h for y, h in zip(ys, hw))*1e6:+.2f}] um")


def build_summary_digest(state, patch=3) -> str:
    """The decision digest plus what only matters once the run is over: where it
    looked, the decision trail, and the pooled loop statistics. No CRITERIA trailer —
    nothing is being chosen next."""
    recs      = state.get("experimental_records", [])
    scans     = [r for r in recs if r["kind"] == "image"]
    loops     = [r for r in recs if r["kind"] == "loop"]
    decisions = state.get("decision_records", [])
    tasks     = state.get("experiment_tasks", [])

    def loops_of(s):
        return [r for r in loops if r["scan_index"] == s["scan_index"]]

    parts = ["TASKS:\n  - " + "\n  - ".join(tasks) if tasks else "TASKS: (none given)",
             "",
             "CAMPAIGN",
             _coverage_line(scans) if scans else "  no scans",
             f"  {len(loops)} loop(s) over {len(decisions)} decision(s)",
             "",
             "DECISION TRAIL (what was asked, and what it produced)",
             _trail_block(decisions, recs),
             "",
             "LOOP POPULATION (all scans pooled; off-field parameters)",
             _population_block(loops),
             "",
             "MEASURED RESULTS",
             ""]
    return "\n".join(parts) + "\n\n".join(_scan_block(s, loops_of(s), patch) for s in scans)