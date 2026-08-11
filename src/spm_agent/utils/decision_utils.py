import difflib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from spm_agent.utils.image_utils import image_path_to_data_url
from spm_agent.schemas.experimental_decision import ExperimentDecision

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


def _loop_line(i, rec, names, components, patch=3):
    py, px = rec["pixel_yx"]
    phi = _phi(components, py, px, patch)
    at = ", ".join(f"{n} {p:.2f}" for n, p in zip(names, phi))
    rv  = rec["loop_review"]
    off = rec["loop_params"]["off_field"]
    on  = rec["loop_params"].get("on_field")
    fe  = "ferroelectric" if rv.is_ferroelectric_like else "NOT ferroelectric"
    line = (f"  {i}. px({py},{px}) | at: {at}\n"
            f"     off: {rv.loop_quality}, {fe} | Vc {_fmt(off.v_c_rising,'V')}/"
            f"{_fmt(off.v_c_falling,'V')} | width {_fmt(off.loop_width_v,'V')} | "
            f"imprint {_fmt(off.imprint_v,'V')} | height {off.loop_height_m:.3g}\n")
    if on:
        line += f"     on:  height {on.loop_height_m:.3g}, offset {_fmt(on.response_offset_m)}\n"
    if rv.issues:
        line += f"     issues: {'; '.join(rv.issues)}\n"
    return line + f"     \"{rv.interpretation}\""


def _scan_block(scan, loops, task_idx=0, patch=3, current=False):
    imap = scan["importance_maps"][task_idx]
    meta = imap["components_meta"]                      # single source of truth
    components = np.load(imap["components_path"])
    crit = "\n".join(f"    - {n} ({w:.2f}) — {r}"
                     for n, w, r in zip(meta.names, meta.weights, meta.rationale))
    seg = "; ".join(f"'{t}': {r['coverage']:.0%}, {r['n_regions']} regions"
                    for t, r in scan["segmentation_results"].items())
    head = (f"Scan_{scan['scan_index']}{' (current)' if current else ''}\n"
            f"  criteria (weight — rationale):\n{crit}\n"
            f"  segmentation: {seg}\n"
            f"  loops ({len(loops)}):")
    lines = [_loop_line(i, r, meta.names, components, patch)
             for i, r in enumerate(loops, 1)] or ["  none"]
    return "\n".join([head] + lines)


def render_map_figure(imp_map, loops, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(imp_map, cmap="viridis")
    for i, rec in enumerate(loops, 1):
        py, px = rec["pixel_yx"]
        ax.plot(px, py, "o", ms=10, mfc="none", mec="red", mew=2)
        ax.annotate(str(i), (px, py), textcoords="offset points",
                    xytext=(6, 6), color="red", fontsize=12, fontweight="bold")
    ax.set_title("Importance map + measured points")
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def render_loop_gallery(loops, out_path, ncols=3):
    items = [(i, r["loop_params"]["off_field"].annotated_path)
             for i, r in enumerate(loops, 1)]
    items = [(i, p) for i, p in items if p]
    if not items:
        return None
    n = len(items); nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
    for ax in axes.flat: ax.axis("off")
    for ax, (i, p) in zip(axes.flat, items):
        ax.imshow(plt.imread(p)); ax.set_title(f"loop {i}", fontsize=11)
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


def build_decision_digest(state, work_dir, task_idx=0, patch=3):
    """Deterministic digest of the whole experiment -> multimodal message content."""
    work_dir = Path(work_dir)
    recs  = state.get("experimental_records", [])
    scans = [r for r in recs if r["kind"] == "image"]

    def loops_of(s):
        return [r for r in recs
                if r["kind"] == "loop" and r["scan_index"] == s["scan_index"]]

    task = scans[-1]["experiment_tasks"][task_idx]
    blocks = [_scan_block(s, loops_of(s), task_idx, patch, current=(s is scans[-1]))
              for s in scans]

    last, last_loops = scans[-1], loops_of(scans[-1])
    meta = last["importance_maps"][task_idx]["components_meta"]
    text = (f"EXPERIMENT TASK: {task}\n\nMEASURED RESULTS\n\n"
            + "\n\n".join(blocks)
            + f"\n\nCRITERIA (current scan, use exact names): {', '.join(meta.names)}")

    map_png = str(work_dir / "digest_map.png")
    render_map_figure(np.load(last["importance_maps"][task_idx]["importance_map_path"]),
                      last_loops, map_png)
    gallery_png = render_loop_gallery(last_loops, str(work_dir / "digest_loops.png"))

    content = [{"type": "text", "text": text},
               {"type": "text", "text": "Current scan: importance map with measured points:"},
               {"type": "image_url", "image_url": {"url": image_path_to_data_url(map_png)}}]
    if gallery_png:
        content += [{"type": "text", "text": "Current scan's loops (numbers match the list):"},
                    {"type": "image_url", "image_url": {"url": image_path_to_data_url(gallery_png)}}]
    return content