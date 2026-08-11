import numpy as np

import os
from matplotlib.figure import Figure

def segment_sspfm_loops(channels: dict, out_dir, skip_frac: float = 0.2,
                        exclude=("Bias", "Raw", "ZSnsr", "Defl", "Phas2"), add_xy: bool = True) -> dict:
    """
    Segment an SS-PFM (DART) measurement into in-field / out-of-field loops.

    channels : 'file_channels' dict from state (needs 'array_path', 'units');
               must contain a Bias channel.
    out_dir  : directory for the .npy loops and the overview .png
    skip_frac: fraction of samples dropped at the start of each bias segment
               (transient after the bias step)

    Saves per channel: {title}_on.npy, {title}_off.npy
    Saves bias axes:   bias_on.npy / bias_off.npy (x-axis of the loops)
    Saves overview:    loops_overview.png
    """
    os.makedirs(out_dir, exist_ok=True)

    bias_key = next(k for k, c in channels.items() if c["title"].startswith("Bias"))
    bias = np.load(channels[bias_key]["array_path"]).ravel()

    # 1) keep only acquired samples (NaN padding at start/end)
    finite = np.isfinite(bias)
    bias_f = bias[finite]

    # 2) run-length encode constant-bias segments
    change = np.where(np.abs(np.diff(bias_f)) > 1e-9)[0] + 1
    bounds = np.concatenate([[0], change, [len(bias_f)]])

    # 3) classify segments; off-field inherits the preceding pulse bias
    tol = 1e-3 * np.nanmax(np.abs(bias_f))
    on_slices, off_slices, on_bias, off_bias = [], [], [], []
    last_pulse = None
    for s, e in zip(bounds[:-1], bounds[1:]):
        v = bias_f[s]
        s_eff = s + max(1, int(np.ceil((e - s) * skip_frac)))
        if s_eff >= e:
            continue
        if np.abs(v) > tol:                      # on-field segment
            on_slices.append((s_eff, e)); on_bias.append(v); last_pulse = v
        elif last_pulse is not None:             # off-field, after a pulse
            off_slices.append((s_eff, e)); off_bias.append(last_pulse)

    bias_on = np.asarray(on_bias, dtype=np.float32)
    bias_off = np.asarray(off_bias, dtype=np.float32)
    np.save(os.path.join(out_dir, "bias_on.npy"), bias_on)
    np.save(os.path.join(out_dir, "bias_off.npy"), bias_off)

    # 4) average each segment for every channel
    loops = {}
    for k, ch in channels.items():
        title = ch["title"]
        if any(title.startswith(x) for x in exclude):
            continue
        y = np.load(ch["array_path"]).ravel()
        if y.size != bias.size:
            continue
        y = y[finite]
        y_on = np.array([np.nanmean(y[s:e]) for s, e in on_slices], dtype=np.float32)
        y_off = np.array([np.nanmean(y[s:e]) for s, e in off_slices], dtype=np.float32)
        p_on = os.path.join(out_dir, f"{title}_on.npy")
        p_off = os.path.join(out_dir, f"{title}_off.npy")
        np.save(p_on, y_on); np.save(p_off, y_off)
        loops[title] = {"on_path": p_on, "off_path": p_off,
                        "units": ch.get("units", ""), "n_points": int(y_on.size)}

    # ---- 4a) collect raw traces: measured + derived X/Y ----------------
    traces = {}   # title -> (raw_array, units)
    for k, ch in channels.items():
        title = ch["title"]
        if any(title.startswith(x) for x in exclude):
            continue
        y = np.load(ch["array_path"]).ravel()
        if y.size != bias.size:
            continue
        traces[title] = (y, ch.get("units", ""))

    if add_xy and "Amp" in traces and "Phase" in traces:
        a, units = traces["Amp"]
        phi = np.deg2rad(traces["Phase"][0])
        traces["X"] = (a * np.cos(phi), units)   # piezoresponse
        traces["Y"] = (a * np.sin(phi), units)   # quadrature

    # ---- 4b) segment-average every trace -------------------------------
    loops = {}
    for title, (y, units) in traces.items():
        y = y[finite]
        y_on = np.array([np.nanmean(y[s:e]) for s, e in on_slices], dtype=np.float32)
        y_off = np.array([np.nanmean(y[s:e]) for s, e in off_slices], dtype=np.float32)
        p_on = os.path.join(out_dir, f"{title}_on.npy")
        p_off = os.path.join(out_dir, f"{title}_off.npy")
        np.save(p_on, y_on); np.save(p_off, y_off)
        loops[title] = {"on_path": p_on, "off_path": p_off,
                        "units": units, "n_points": int(y_on.size)}

    # 5) overview figure: rows = channels, cols = in-field / out-of-field
    fig = Figure(figsize=(8, 2.4 * len(loops)), dpi=120)
    axs = fig.subplots(len(loops), 2, squeeze=False, sharex=True)
    for r, (title, info) in enumerate(loops.items()):
        for c, (x, suffix) in enumerate([(bias_on, "on"), (bias_off, "off")]):
            y = np.load(info[f"{suffix}_path"])
            axs[r][c].plot(x, y, ".-", lw=0.7, ms=2)
            axs[r][c].set_ylabel(f"{title} [{info['units']}]", fontsize=8)
            axs[r][c].grid(True, alpha=0.3)
            if r == 0:
                axs[r][c].set_title(["in-field", "out-of-field"][c])
    for c in range(2):
        axs[-1][c].set_xlabel("Bias [V]")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "loops_overview.png")
    fig.savefig(fig_path)

    return {"bias_on_path": os.path.join(out_dir, "bias_on.npy"),
            "bias_off_path": os.path.join(out_dir, "bias_off.npy"),
            "loops": loops, "overview_path": fig_path,
            "n_pulses": int(bias_on.size)}

def classify_measurement(arrays: dict[str, np.ndarray]) -> str:
    """Classify a measurement as 'image', 'loop' (SS-PFM), or 'spectrum'.
    arrays: {title: np.ndarray} of the file's channels."""
    if any(a.ndim >= 2 and min(a.shape) > 1 for a in arrays.values()):
        return "image"

    bias = next((a for t, a in arrays.items() if t.startswith("Bias")), None)
    if bias is None:
        return "spectrum"

    b = bias.ravel()
    b = b[np.isfinite(b)]
    if b.size < 100 or np.max(np.abs(b)) < 1e-6:
        return "spectrum"

    change = np.where(np.abs(np.diff(b)) > 1e-9)[0] + 1
    bounds = np.concatenate([[0], change, [len(b)]])
    lens = np.diff(bounds)
    vals = b[bounds[:-1]]

    dwell  = (lens >= 3).mean()                        # piecewise-constant
    zero   = np.abs(vals) < 1e-3 * np.abs(b).max()     # off-field segments
    altern = (np.diff(zero.astype(int)) != 0).mean()   # on/off interleaving

    if len(lens) >= 20 and dwell > 0.8 and 0.2 < zero.mean() < 0.8 and altern > 0.8:
        return "loop"
    return "spectrum"

def extract_loop_params(bias, x, y=None, rotate=True, grid_n=201,
                        fig_path=None, units="") -> dict:
    """
    Extract hysteresis parameters from one SS-PFM loop (bias, x [, y]).

    bias, x, y : 1D arrays in acquisition order (multi-cycle OK).
                 x, y = lock-in quadratures; passing y enables phase
                 rotation and the quadrature QC metric.
    rotate     : rotate (x,y) by the PCA angle phi0 so the switching
                 signal lands entirely in x' (instrument-phase independent).

    Coercive voltages = steepest zero crossings of (x - response_offset)
    on the cycle-averaged rising/falling branches.
    """
    bias = np.asarray(bias, float); x = np.asarray(x, float)
    out = {}

    # 0) quadrature rotation
    if rotate and y is not None:
        y = np.asarray(y, float)
        xc, yc = x - x.mean(), y - y.mean()
        phi0 = 0.5 * np.arctan2(2 * (xc * yc).mean(), (xc**2 - yc**2).mean())
        xr = x * np.cos(phi0) + y * np.sin(phi0)
        yr = -x * np.sin(phi0) + y * np.cos(phi0)
        out["phase_offset_deg"] = float(np.degrees(phi0))
        out["quadrature_residual"] = float(np.std(yr) / max(np.std(xr), 1e-30))
        x = xr
    else:
        out["phase_offset_deg"] = None
        out["quadrature_residual"] = None

    # 1) branch split by envelope sweep direction
    g = np.gradient(bias)
    rising, falling = g > 0, g < 0
    n_cycles = max(1, int(np.round((np.diff(np.sign(g)) != 0).sum() / 2)))
    out["n_cycles"] = n_cycles

    # 2) cycle-average each branch on a common bias grid (binned means)
    grid = np.linspace(bias.min(), bias.max(), grid_n)
    def branch_avg(mask):
        b_, x_ = bias[mask], x[mask]
        idx = np.digitize(b_, grid)
        xa = np.full(grid_n, np.nan)
        for i in range(grid_n):
            sel = idx == i
            if sel.any():
                xa[i] = x_[sel].mean()
        ok = np.isfinite(xa)
        return np.interp(grid, grid[ok], xa[ok])
    x_rise, x_fall = branch_avg(rising), branch_avg(falling)

    # rms scatter of points around their averaged branch (noise metric)
    res = np.concatenate([x[rising] - np.interp(bias[rising], grid, x_rise),
                          x[falling] - np.interp(bias[falling], grid, x_fall)])
    out["branch_rms_noise"] = float(np.sqrt(np.mean(res**2)))

    # 3) saturation & vertical offset (outer 15% of bias range)
    hi, lo = grid > 0.85 * grid.max(), grid < 0.85 * grid.min()
    sat_at_vplus = float(np.nanmean(np.concatenate([x_rise[hi], x_fall[hi]])))
    sat_at_vminus = float(np.nanmean(np.concatenate([x_rise[lo], x_fall[lo]])))
    v_offset = 0.5 * (sat_at_vplus + sat_at_vminus)
    out.update(sat_at_vplus_m=sat_at_vplus, sat_at_vminus_m=sat_at_vminus,
               response_offset_m=float(v_offset))

    # 4) coercive voltages: steepest zero crossing per branch
    def crossing(xb):
        s = xb - v_offset
        idx = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
        if idx.size == 0:
            return None
        i = idx[np.argmax(np.abs(np.diff(xb))[idx])]
        return float(grid[i] - s[i] * (grid[i+1] - grid[i]) / (s[i+1] - s[i]))
    vc_rise, vc_fall = crossing(x_rise), crossing(x_fall)
    out["v_c_rising"] = vc_rise
    out["v_c_falling"] = vc_fall
    if vc_rise is not None and vc_fall is not None:
        out["imprint_v"] = float((vc_rise + vc_fall) / 2)
        out["loop_width_v"] = float(abs(vc_rise - vc_fall))
    else:
        out["imprint_v"] = out["loop_width_v"] = None

    # 5) remnant response: branch values at V = 0
    r_rise = float(np.interp(0.0, grid, x_rise))
    r_fall = float(np.interp(0.0, grid, x_fall))
    out.update(remnant_rising_m=r_rise,
               remnant_falling_m=r_fall,
               loop_height_m=float(abs(r_fall - r_rise)))

    # 6) area + rotation direction (signed shoelace, time-ordered path)
    signed = 0.5 * float(np.sum(bias * np.roll(x, -1) - np.roll(bias, -1) * x))
    out["loop_area_per_cycle"] = float(abs(signed) / n_cycles)
    out["direction"] = "ccw" if signed > 0 else "cw"

    out["annotated_path"] = None
    if fig_path is not None:
        _save_annotated_loop(fig_path, bias, x, grid, x_rise, x_fall, out, units)
        out["annotated_path"] = str(fig_path)

    return out

def _save_annotated_loop(fig_path, bias, x, grid, x_rise, x_fall, out, units=""):
    """Annotated loop preview for vision-model review."""
    fig = Figure(figsize=(7, 5), dpi=120)
    ax = fig.add_subplot()

    ax.plot(bias, x, ".", ms=2.5, alpha=0.35, color="gray", label="data")
    ax.plot(grid, x_rise, "-", lw=2, color="tab:blue", label="rising branch")
    ax.plot(grid, x_fall, "-", lw=2, color="tab:orange", label="falling branch")

    off = out["response_offset_m"]
    ax.axhline(off, ls=":", c="k", lw=0.8)
    ax.axvline(0, ls=":", c="k", lw=0.8)

    for vc, c, nm in [(out["v_c_rising"], "tab:blue", "Vc↑"),
                      (out["v_c_falling"], "tab:orange", "Vc↓")]:
        if vc is not None:
            ax.axvline(vc, ls="--", c=c, lw=1.2)
            ax.annotate(f"{nm} {vc:+.2f} V", (vc, off), fontsize=8,
                        rotation=90, va="bottom", ha="right", color=c)

    ax.plot([0, 0], [out["remnant_rising_m"], out["remnant_falling_m"]],
            "rs", ms=7, zorder=5, label="remnants @ 0 V")

    fmt = lambda v, p="{:+.3g}": "n/a" if v is None else p.format(v)
    stats = (f"imprint    {fmt(out['imprint_v'], '{:+.2f}')} V\n"
             f"width      {fmt(out['loop_width_v'], '{:.2f}')} V\n"
             f"height     {out['loop_height_m']:.3g} {units}\n"
             f"area/cycle {out['loop_area_per_cycle']:.3g}\n"
             f"direction  {out['direction']}, {out['n_cycles']} cycles\n"
             f"SNR        {out['loop_height_m']/max(out['branch_rms_noise'],1e-30):.1f}\n"
             f"quad.res.  {fmt(out['quadrature_residual'], '{:.3f}')}")
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=8,
            family="monospace", va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="0.7"))

    ax.set_xlabel("Bias [V]")
    ax.set_ylabel(f"X' [{units}]" if units else "X'")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_path)