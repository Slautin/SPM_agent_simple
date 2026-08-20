import numpy as np
from scipy.ndimage import uniform_filter, distance_transform_edt


def patch_mean(a, patch_px: int) -> np.ndarray:
    """A pixel is only as good as its neighbourhood — matches _phi in decision_utils."""
    return uniform_filter(np.asarray(a, float), size=2 * patch_px + 1, mode="nearest")


def strategy_score(phi: np.ndarray, measured: list[float], strategy: str) -> np.ndarray:
    """Desire map in [0,1] from one criterion map and how it has been sampled."""
    if strategy == "max":
        return phi
    if strategy == "min":
        return 1.0 - phi
    if strategy == "diverse":
        if not measured:                      # nothing to be far from -> behave as max
            return phi
        d = np.abs(phi[None] - np.asarray(measured, float)[:, None, None]).min(axis=0)
        return d / (d.max() + 1e-12)
    raise ValueError(f"unknown strategy {strategy!r}")


def spatial_penalty(shape, points, radius_px: int, floor: float) -> np.ndarray:
    """1.0 far from measured points, `floor` at them, linear in between."""
    if not points:
        return np.ones(shape)
    seeds = np.ones(shape, bool)
    for y, x in points:
        seeds[y, x] = False
    return floor + (1.0 - floor) * np.clip(
        distance_transform_edt(seeds) / radius_px, 0.0, 1.0)


def border_mask(shape, margin_px: int) -> np.ndarray:
    """False within margin_px of the frame edge."""
    m = np.zeros(shape, bool)
    m[margin_px:shape[0] - margin_px, margin_px:shape[1] - margin_px] = True
    return m

def pixel_to_frame_xy(scan_settings, row: int, col: int, row0_at_top: bool = True):
    """Centre of pixel (row, col) in scan-frame metres, on the frame that pixel
    grid belongs to. Pass that scan's own ScanSettings — never the live one."""
    ss = scan_settings
    L, N = ss.scan_size_m, ss.pixels
    if not (0 <= row < N and 0 <= col < N):
        raise ValueError(f"pixel ({row},{col}) outside a {N}x{N} grid")

    dx = (col + 0.5) / N * L - L / 2
    dy = (row + 0.5) / N * L - L / 2
    return (ss.x_scan_center_m + dx,
            ss.y_scan_center_m + (-dy if row0_at_top else dy))