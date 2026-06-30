import numpy as np
from dataclasses import dataclass, field

from scipy import ndimage as ndi 
from langchain_core.tools import tool

from spm_agent.utils.image_utils import norm01, to_view01, render_overlay_datauri

def _otsu(a01: np.ndarray) -> float:
    """Automatic threshold (Otsu) on a [0,1] image, pure numpy (no skimage)."""
    h, _ = np.histogram(a01[np.isfinite(a01)], bins=256, range=(0, 1))
    h = h.astype(float); total = h.sum()
    if total == 0:
        return 0.5
    p = h / total
    omega = np.cumsum(p)                                   # class-0 weight
    mu = np.cumsum(p * ((np.arange(256) + 0.5) / 256))     # class-0 mean
    mu_t = mu[-1]
    denom = omega * (1 - omega); denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom            # between-class variance
    return float((np.argmax(sigma_b2) + 0.5) / 256)

@dataclass
class SegSession:
    """In-memory scratch buffer for ONE task. NOT a LangGraph state — it just
    holds the pixel data and is shared with the tools via a Python closure.
    The agent only ever sees numbers (and, if enabled, the overlay image)."""
    raw: np.ndarray                       # CHANGED: physical units, may hold NaN (absolute thresholds)
    view: np.ndarray
    work: np.ndarray                       # current working image (filters applied here)
    mask: np.ndarray                       # current boolean mask (the result so far)
    ops:  list = field(default_factory=list)   # ordered log of tool calls = the "program"
    want_view: bool = False                # set True when the agent calls show_overlay()

    @classmethod
    def from_raw(cls, raw):           # CHANGED: was from_image
        raw = raw.astype(np.float32)
        work = np.nan_to_num(norm01(raw), nan=0.0)
        return cls(raw=raw, view=to_view01(raw), work=work.copy(),
                   mask=np.zeros(raw.shape, dtype=bool))

    def mask_stats(self):
        """Return (n_connected_regions, coverage_fraction) of the current mask."""
        _, n = ndi.label(self.mask)
        return int(n), float(self.mask.mean())
    

    
def build_segmentation_tools(session: SegSession, vision_in_loop: bool) -> list:
    """Return the agent's instrument: small, orthogonal image-analysis ops.
    Each tool CLOSES OVER `session` (that's how it reaches the pixels without
    passing them through the LLM state), mutates work/mask, logs to ops,
    and returns NUMBERS ONLY."""

    @tool
    def describe_image() -> dict:
        """Intensity statistics of the current working image (shape, min/max/mean/std, p10/p90)."""
        w = session.work
        return {"shape": list(w.shape), "min": float(w.min()), "max": float(w.max()),
                "mean": float(w.mean()), "std": float(w.std()),
                "p10": float(np.percentile(w, 10)), "p90": float(np.percentile(w, 90))}

    @tool
    def smooth_image(sigma: float) -> dict:
        """Gaussian blur the working image to suppress noise. sigma in pixels (~1-3)."""
        session.work = norm01(ndi.gaussian_filter(session.work, sigma=float(sigma)))
        session.ops.append({"op": "smooth", "sigma": sigma})
        return {"std": float(session.work.std())}

    @tool
    def compute_gradient_magnitude() -> dict:
        """Replace the working image with its Sobel gradient magnitude.
        Use for THIN BOUNDARY features (domain walls, grain boundaries)."""
        gx = ndi.sobel(session.work, axis=0); gy = ndi.sobel(session.work, axis=1)
        session.work = norm01(np.hypot(gx, gy))
        session.ops.append({"op": "gradient_magnitude"})
        return {"mean": float(session.work.mean()), "max": float(session.work.max())}

    @tool
    def clean_mask(operation: str, size: int = 2) -> dict:
        """Morphological cleanup of the mask.
        operation: 'open' (drop specks) | 'close' (fill holes) | 'dilate' | 'erode'. size = element size."""
        st = np.ones((int(size), int(size)), dtype=bool)
        fn = {"open": ndi.binary_opening, "close": ndi.binary_closing,
              "dilate": ndi.binary_dilation, "erode": ndi.binary_erosion}[operation]
        session.mask = fn(session.mask, structure=st)
        session.ops.append({"op": "morphology", "operation": operation, "size": size})
        n, cov = session.mask_stats()
        return {"n_regions": n, "coverage": round(cov, 4)}

    @tool
    def remove_small_regions(min_pixels: int) -> dict:
        """Drop connected components smaller than min_pixels from the mask."""
        lbl, n = ndi.label(session.mask)
        if n:
            sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
            keep = [i + 1 for i, s in enumerate(sizes) if s >= int(min_pixels)]
            session.mask = np.isin(lbl, keep)
        session.ops.append({"op": "remove_small_regions", "min_pixels": min_pixels})
        n2, cov = session.mask_stats()
        return {"n_regions": n2, "coverage": round(cov, 4)}

    @tool
    def mask_summary() -> dict:
        """Current mask: region count, coverage, largest-region size in pixels."""
        lbl, n = ndi.label(session.mask)
        largest = int(ndi.sum(np.ones_like(lbl), lbl,
                      index=np.arange(1, n + 1)).max()) if n else 0
        return {"n_regions": int(n), "coverage": round(float(session.mask.mean()), 4),
                "largest_region_px": largest}

    @tool
    def threshold_image(method: str, value: float = 0.0, polarity: str = "above",
                        units: str = "normalized") -> dict:
        """Set the mask by thresholding.
        method: 'otsu' (auto) | 'percentile' (value=0-100) | 'absolute' (value in image units).
        units: 'normalized' = run on the processed [0,1] image;
               'absolute'   = run on RAW physical values → reproducible across files."""
        src = session.raw if units == "absolute" else session.work        # NEW
        if method == "otsu":
            t = _otsu(session.work)                                        # otsu only on normalized
        elif method == "percentile":
            t = float(np.nanpercentile(src, value))
        else:
            t = float(value)
        m = (src >= t) if polarity == "above" else (src <= t)
        session.mask = m & np.isfinite(src)                               # NaN never in mask
        session.ops.append({"op": "threshold", "method": method, "value": value,
                            "polarity": polarity, "units": units, "t": round(t, 4)})
        n, cov = session.mask_stats()
        return {"threshold": round(t, 4), "n_regions": n, "coverage": round(cov, 4)}

    @tool
    def reset_working_image() -> dict:
        """Restore the working image from raw and clear the mask."""
        session.work = np.nan_to_num(_norm01(session.raw), nan=0.0)        # CHANGED
        session.mask = np.zeros_like(session.mask)
        session.ops.append({"op": "reset"})
        return {"ok": True}

    tools = [describe_image, smooth_image, compute_gradient_magnitude, threshold_image,
             clean_mask, remove_small_regions, mask_summary, reset_working_image]

    # show_overlay is the ONLY tool that triggers an image injection (Cell 8).
    # It's added only in vision mode so numbers-only runs stay a clean ablation.
    if vision_in_loop:
        @tool
        def show_overlay() -> list:
            """Render the current mask over the channel image and return it so you can SEE it.
            Use when numeric feedback is ambiguous and you must visually confirm the mask
            matches the target feature before choosing the next operation."""
            session.ops.append({"op": "show_overlay"})
            n, cov = session.mask_stats()
            return [                                   # Claude reads images in tool results
                {"type": "text", "text": f"Overlay rendered. coverage={cov:.3f}, regions={n}."},
                {"type": "image_url",
                 "image_url": {"url": render_overlay_datauri(session.view, session.mask)}},
            ]
        tools.append(show_overlay)

    return tools