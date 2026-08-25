from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence
from dataclasses import FrozenInstanceError
from numbers import Real
from typing import cast

from annofmt.geometry._polygons import min_area_rect, signed_area
from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox

Index = int
Point2D = tuple[Index, Index]
Rings = tuple[tuple[Point2D, ...], ...]


def _to_index(value: object, name: str) -> Index:
    """Coerce a coordinate to an integer pixel index.

    Accepts anything ``operator.index()`` supports (int, numpy integers, ...)
    and real numbers with an integral value. Everything else is rejected:
    ``Segm`` works on indices only.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer index, got bool {value!r}")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        pass
    if isinstance(value, Real):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        if value.is_integer():
            return int(value)
    raise ValueError(f"{name} must be an integer pixel index, got {value!r}")


def _normalize_rings(indices: Iterable[Iterable[Sequence[float]]]) -> Rings:
    materialized = [tuple(part) for part in indices]
    if not materialized:
        raise ValueError("segmentation needs at least one ring")
    first_ring = materialized[0]
    if first_ring and isinstance(first_ring[0], Real):
        flat_ring = cast(tuple[Sequence[float], ...], tuple(materialized))
        rings: list[tuple[Sequence[float], ...]] = [flat_ring]
    else:
        rings = materialized
    normalized: list[tuple[Point2D, ...]] = []
    for ring in rings:
        points: list[Point2D] = []
        for point in ring:
            if len(point) != 2:
                raise ValueError(f"each point needs exactly 2 coordinates, got {point!r}")
            points.append((_to_index(point[0], "x"), _to_index(point[1], "y")))
        if len(points) < 3:
            raise ValueError(f"each ring needs at least 3 points, got {len(points)}")
        normalized.append(tuple(points))
    return tuple(normalized)


def _rasterize(rings: Sequence[Sequence[Point2D]], height: int, width: int) -> bytearray:
    """Even-odd scanline fill of all rings combined into a row-major grid.

    Pixels are unit squares centered at ``(col + 0.5, row + 0.5)``; rings may
    overlap or contain holes regardless of winding order.
    """
    grid = bytearray(height * width)
    edges: list[tuple[float, float, float, float]] = []
    for ring in rings:
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % n]
            if y1 != y2:
                edges.append((float(x1), float(y1), float(x2), float(y2)))
    for row in range(height):
        yc = row + 0.5
        crossings: list[float] = []
        for ex1, ey1, ex2, ey2 in edges:
            if (ey1 <= yc < ey2) or (ey2 <= yc < ey1):
                t = (yc - ey1) / (ey2 - ey1)
                crossings.append(ex1 + t * (ex2 - ex1))
        if not crossings:
            continue
        crossings.sort()
        base = row * width
        for k in range(0, len(crossings) - 1, 2):
            start = max(0, math.ceil(crossings[k] - 0.5))
            stop = min(width, math.ceil(crossings[k + 1] - 0.5))
            for col in range(start, stop):
                grid[base + col] = 1
    return grid


def _runs_from_grid(grid: bytearray, height: int, width: int) -> tuple[Index, ...]:
    """COCO-style column-major run-length encoding.

    Counts alternate zeros/ones starting with zeros; a fully empty mask
    encodes as ``(height * width,)``.
    """
    runs: list[Index] = []
    current_value = 0
    count = 0
    for col in range(width):
        for row in range(height):
            value = grid[row * width + col]
            if value == current_value:
                count += 1
            else:
                runs.append(count)
                current_value = value
                count = 1
    runs.append(count)
    return tuple(runs)


def _grid_from_runs(runs: Sequence[Index], height: int, width: int) -> bytearray:
    """Inverse of :func:`_runs_from_grid`: scatter column-major counts into a
    row-major grid."""
    grid = bytearray(height * width)
    pos = 0
    value = 0
    for count in runs:
        length = _to_index(count, "run length")
        if length < 0:
            raise ValueError(f"run lengths must be non-negative, got {length}")
        if value:
            for k in range(pos, min(pos + length, len(grid))):
                row = k % height
                col = k // height
                grid[row * width + col] = 1
        pos += length
        value ^= 1
    total = sum(_to_index(run, "run length") for run in runs)
    if total != len(grid):
        raise ValueError(f"RLE counts sum to {total}, expected {height * width}")
    return grid


def _trace_loops(grid: bytearray, height: int, width: int) -> Rings:
    """Extract boundary loops of the filled region as lattice polygons.

    Boundary edges are directed consistently around filled cells, so outer
    boundaries and holes become separately wound loops that compose correctly
    under the even-odd fill convention. Collinear vertices are collapsed.
    """

    def filled(row: int, col: int) -> bool:
        return 0 <= row < height and 0 <= col < width and grid[row * width + col] == 1

    outgoing: dict[Point2D, list[Point2D]] = {}
    for row in range(height):
        for col in range(width):
            if not filled(row, col):
                continue
            if not filled(row - 1, col):
                outgoing.setdefault((col, row), []).append((col + 1, row))
            if not filled(row, col + 1):
                outgoing.setdefault((col + 1, row), []).append((col + 1, row + 1))
            if not filled(row + 1, col):
                outgoing.setdefault((col + 1, row + 1), []).append((col, row + 1))
            if not filled(row, col - 1):
                outgoing.setdefault((col, row + 1), []).append((col, row))

    def turn_rank(origin: Point2D, incoming: tuple[int, int], target: Point2D) -> tuple[int, int]:
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        straightness = dx * incoming[0] + dy * incoming[1]
        cross = incoming[0] * dy - incoming[1] * dx
        return (-straightness, cross)

    loops: list[tuple[Point2D, ...]] = []
    while outgoing:
        start = next(iter(outgoing))
        loop: list[Point2D] = [start]
        current = start
        incoming: tuple[int, int] | None = None
        while True:
            candidates = outgoing.get(current)
            if not candidates:
                break
            if incoming is None or len(candidates) == 1:
                nxt = candidates.pop()
            else:
                direction = incoming
                origin = current
                nxt = max(candidates, key=lambda pt: turn_rank(origin, direction, pt))
                candidates.remove(nxt)
            if not candidates:
                del outgoing[current]
            incoming = (nxt[0] - current[0], nxt[1] - current[1])
            current = nxt
            if current == start:
                break
            loop.append(current)
        simplified: list[Point2D] = []
        m = len(loop)
        for i in range(m):
            prev_pt, pt, next_pt = loop[i - 1], loop[i], loop[(i + 1) % m]
            if (pt[0] - prev_pt[0]) * (next_pt[1] - pt[1]) != (pt[1] - prev_pt[1]) * (next_pt[0] - pt[0]):
                simplified.append(pt)
        if len(simplified) >= 3:
            loops.append(tuple(simplified))
    return tuple(loops)


class Segm(BBox):
    """Immutable segmentation over integer pixel indices.

    Storage mirrors :class:`BBox`: the center/extent slots hold the enclosing
    box of the indices, plus one extra slot with the polygon rings themselves.
    The corner properties therefore work fully inherited.

    Coordinates are integer indices only; fractional values are rejected.
    ``indices`` holds one or more rings that combine under an even-odd rule,
    so a ring nested inside another acts as a hole regardless of winding. A
    single flat ring may also be passed directly to the constructor.
    """

    __slots__ = ("indices",)

    indices: Rings

    def __init__(self, indices: Iterable[Iterable[Sequence[float]]]) -> None:
        rings = _normalize_rings(indices)
        xs = [point[0] for ring in rings for point in ring]
        ys = [point[1] for ring in rings for point in ring]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        object.__setattr__(self, "x", (x_min + x_max) / 2)
        object.__setattr__(self, "y", (y_min + y_max) / 2)
        object.__setattr__(self, "w", x_max - x_min)
        object.__setattr__(self, "h", y_max - y_min)
        object.__setattr__(self, "indices", rings)

    @classmethod
    def from_polygon(cls, points: Iterable[Sequence[float]]) -> Segm:
        """Single-ring segmentation."""
        return Segm((tuple(points),))

    @classmethod
    def from_bbox(cls, bbox: BBox) -> Segm:
        """Exact rectangle ring covering the box (floored min corners,
        ceiled max corners)."""
        x_min = math.floor(bbox.x_min)
        y_min = math.floor(bbox.y_min)
        x_max = math.ceil(bbox.x_max)
        y_max = math.ceil(bbox.y_max)
        return Segm.from_polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

    @classmethod
    def from_rbbox(cls, rbbox: RBBox) -> Segm:
        """Ring over the four rotated-box corners, rounded to the lattice."""
        corners = rbbox.corners()
        return Segm.from_polygon(tuple((int(round(x)), int(round(y))) for x, y in corners))

    @classmethod
    def from_rle(cls, runs: Sequence[Index], height: Index, width: Index) -> Segm:
        """Decode COCO-style column-major run-length counts into rings."""
        height_i = _to_index(height, "height")
        width_i = _to_index(width, "width")
        if height_i <= 0 or width_i <= 0:
            raise ValueError("height and width must be positive")
        grid = _grid_from_runs(runs, height_i, width_i)
        loops = _trace_loops(grid, height_i, width_i)
        if not loops:
            raise ValueError("mask is empty, no rings to trace")
        return Segm(loops)

    def to_rle(self, height: Index, width: Index) -> tuple[Index, ...]:
        """Encode as COCO-style column-major run-length counts on a raster of
        the given size (even-odd fill)."""
        height_i = _to_index(height, "height")
        width_i = _to_index(width, "width")
        if height_i <= 0 or width_i <= 0:
            raise ValueError("height and width must be positive")
        grid = _rasterize(self.indices, height_i, width_i)
        return _runs_from_grid(grid, height_i, width_i)

    @property
    def area(self) -> float:
        """Sum of signed ring areas; oppositely wound inner rings subtract.

        Rings produced by :meth:`from_rle` carry consistent winding. For
        hand-built multi-ring geometries this equals the covered area only
        when rings do not overlap unless they are wound as holes.
        """
        return abs(sum(signed_area(ring) for ring in self.indices))

    def contains_point(self, x: Index, y: Index) -> bool:
        """Even-odd containment test at an integer index point."""
        xi = _to_index(x, "x")
        yi = _to_index(y, "y")
        inside = False
        for ring in self.indices:
            j = len(ring) - 1
            for i in range(len(ring)):
                x1, y1 = ring[j]
                x2, y2 = ring[i]
                if (y1 > yi) != (y2 > yi):
                    t = (yi - y1) / (y2 - y1)
                    if xi < x1 + t * (x2 - x1):
                        inside = not inside
                j = i
        return inside

    def translate(self, dx: float, dy: float) -> Segm:
        dxi = _to_index(dx, "dx")
        dyi = _to_index(dy, "dy")
        shifted = tuple(tuple((px + dxi, py + dyi) for px, py in ring) for ring in self.indices)
        return Segm(shifted)

    def scale(self, factor_w: float, factor_h: float) -> Segm:
        """Scale the indices about the center and round back to integers."""
        scaled_box = BBox(self.x, self.y, self.w, self.h).scale(factor_w, factor_h)
        cx, cy = self.x, self.y
        fx = scaled_box.w / self.w if self.w else factor_w
        fy = scaled_box.h / self.h if self.h else factor_h
        rings = tuple(
            tuple(
                (int(round(cx + (px - cx) * fx)), int(round(cy + (py - cy) * fy)))
                for px, py in ring
            )
            for ring in self.indices
        )
        return Segm(rings)

    def as_rbbox(self) -> RBBox:
        """Lossy minimum-area enclosing rotated box (rotating calipers)."""
        (cx, cy), w, h, angle = min_area_rect([point for ring in self.indices for point in ring])
        return RBBox(cx, cy, w, h, angle)

    def to_bbox(self) -> BBox:
        """Plain axis-aligned :class:`BBox` of the enclosing extent."""
        return BBox(self.x, self.y, self.w, self.h)

    def iou(self, other: object) -> float:
        """IoU against another :class:`Segm`, rasterized on the combined
        extent of both geometries.

        Raises :class:`TypeError` for any other type, including plain
        :class:`BBox`.
        """
        if type(other) is not Segm:
            raise TypeError(f"Segm.iou supports Segm only, got {type(other).__name__}")
        height = max(pt[1] for segm in (self, other) for ring in segm.indices for pt in ring) + 1
        width = max(pt[0] for segm in (self, other) for ring in segm.indices for pt in ring) + 1
        grid_a = _rasterize(self.indices, height, width)
        grid_b = _rasterize(other.indices, height, width)
        intersection = sum(a & b for a, b in zip(grid_a, grid_b, strict=True))
        union = sum(a | b for a, b in zip(grid_a, grid_b, strict=True))
        if union == 0:
            return 0.0
        return intersection / union

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __eq__(self, other: object) -> bool:
        if type(other) is not Segm:
            return NotImplemented
        return self.indices == other.indices

    def __hash__(self) -> int:
        return hash((type(self), self.indices))

    def __repr__(self) -> str:
        return f"Segm(x={self.x}, y={self.y}, w={self.w}, h={self.h}, indices={self.indices})"
