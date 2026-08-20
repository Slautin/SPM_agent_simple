from math import hypot

# All coordinates are scan-frame metres — the same system as ScanSettings.x/y_scan_center_m.


def footprint_overlap(cx: float, cy: float, size_m: float, ss,
                      rel_tol: float = 1e-6) -> bool:
    """Does a candidate frame share area with an already-scanned one? Frames are
    square and axis-aligned (ScanAngle is 0 throughout). Borders that merely TOUCH
    are not an overlap — that contact is exactly the placement relocate aims for."""
    reach = (size_m + ss.scan_size_m) / 2
    tol   = rel_tol * reach
    return (abs(cx - ss.x_scan_center_m) < reach - tol and
            abs(cy - ss.y_scan_center_m) < reach - tol)


def within_travel(cx: float, cy: float, size_m: float, limit_m: float) -> bool:
    """The whole frame, not just its centre, must stay inside +/- limit_m."""
    half = size_m / 2
    return abs(cx) + half <= limit_m and abs(cy) + half <= limit_m


def _ring(r: int) -> list[tuple[int, int]]:
    """Lattice cells at Chebyshev distance r from the origin, nearest first.
    Ring 1 is the eight frames whose border touches the current one; the four
    edge-sharing ones sort ahead of the four that meet only at a corner."""
    cells = [(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1)
             if max(abs(i), abs(j)) == r]
    return sorted(cells, key=lambda c: (hypot(*c), c))


def pick_relocation(live_ss, visited, limit_m
                    ) -> tuple[tuple[float, float] | None, dict[str, int]]:
    """Centre of the nearest unexplored frame, or None if there is none left.

    The search walks a lattice anchored on the current frame and stepping by one
    frame size, so any cell it returns has its border against the frame it stepped
    from. Rings are searched outward, so a neighbour is preferred over a distant
    gap, and the first admissible cell wins. `visited` are the ScanSettings of
    every frame scanned so far — a candidate overlapping any of them is not
    unexplored, which is what makes this work after a zoom left frames of mixed
    size behind. The returned counts say why candidates were dropped."""
    step     = live_ss.scan_size_m
    cx0, cy0 = live_ss.x_scan_center_m, live_ss.y_scan_center_m
    rejected = {"explored": 0, "out_of_range": 0}

    for r in range(1, int(limit_m / step) + 2):      # far enough to leave the fuse
        for i, j in _ring(r):
            cx, cy = cx0 + i * step, cy0 + j * step
            if not within_travel(cx, cy, step, limit_m):
                rejected["out_of_range"] += 1
                continue
            if any(footprint_overlap(cx, cy, step, ss) for ss in visited):
                rejected["explored"] += 1
                continue
            return (cx, cy), rejected
    return None, rejected
