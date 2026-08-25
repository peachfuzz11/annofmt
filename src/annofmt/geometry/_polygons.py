"""Pure-python polygon helpers shared by RBBox and Segm.

No external dependencies. All functions treat polygons as sequences of
``(x, y)`` tuples and never mutate their inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]


def signed_area(polygon: Polygon) -> float:
    """Signed shoelace area (positive for counter-clockwise winding)."""
    total = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def shoelace_area(polygon: Polygon) -> float:
    """Absolute area of a simple polygon."""
    return abs(signed_area(polygon))


def clip_convex(subject: Polygon, clip: Polygon) -> list[Point]:
    """Clip ``subject`` against the convex polygon ``clip`` (Sutherland-Hodgman).

    Returns the vertices of the intersection polygon; empty when disjoint.
    """
    output: list[Point] = [(p[0], p[1]) for p in subject]
    clip_pts: list[Point] = [(p[0], p[1]) for p in clip]
    if signed_area(clip_pts) < 0:
        clip_pts.reverse()
    n = len(clip_pts)
    for i in range(n):
        if not output:
            return []
        ax, ay = clip_pts[i]
        bx, by = clip_pts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        input_pts = output
        output = []
        prev = input_pts[-1]
        prev_side = ex * (prev[1] - ay) - ey * (prev[0] - ax)
        for cur in input_pts:
            cur_side = ex * (cur[1] - ay) - ey * (cur[0] - ax)
            if (cur_side >= 0) != (prev_side >= 0):
                t = prev_side / (prev_side - cur_side)
                output.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if cur_side >= 0:
                output.append(cur)
            prev, prev_side = cur, cur_side
    return output


def convex_hull(points: Sequence[Point]) -> list[Point]:
    """Convex hull via Andrew's monotone chain, counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def min_area_rect(points: Sequence[Point]) -> tuple[Point, float, float, float]:
    """Minimum-area enclosing rectangle of ``points`` via rotating calipers.

    Returns ``(center, width, height, angle_rad)`` where the rectangle is
    centered at ``center``, rotated by ``angle_rad``.
    """
    hull = convex_hull(list(points))
    if len(hull) < 3:
        raise ValueError("min-area rect requires at least 3 distinct points")
    best: tuple[float, Point, float, float, float] | None = None
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        vx, vy = -uy, ux
        us = [px * ux + py * uy for px, py in hull]
        vs = [px * vx + py * vy for px, py in hull]
        w = max(us) - min(us)
        h = max(vs) - min(vs)
        area = w * h
        if best is None or area < best[0]:
            cu = (max(us) + min(us)) / 2.0
            cv = (max(vs) + min(vs)) / 2.0
            center: Point = (cu * ux + cv * vx, cu * uy + cv * vy)
            angle = math.atan2(uy, ux)
            best = (area, center, w, h, angle)
    assert best is not None
    return best[1], best[2], best[3], best[4]
