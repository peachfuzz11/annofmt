from __future__ import annotations

import dataclasses
import math

import pytest

from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox
from annofmt.geometry.segm import Segm


class TestConstruction:
    def test_center_format_and_derived_corners(self):
        bbox = BBox(2, 3, 2, 2)
        assert (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max) == (1, 2, 3, 4)

    def test_zero_size_allowed(self):
        degenerate = BBox(1, 1, 0, 0)
        assert degenerate.area == 0

    def test_negative_extent_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            BBox(0, 0, -1, 2)

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            BBox(0, 0, float("nan"), 1)
        with pytest.raises(ValueError, match="finite"):
            BBox(0, float("inf"), 1, 1)

    def test_value_equality_and_hash(self):
        assert BBox(2, 3, 2, 2) == BBox(2, 3, 2, 2)
        assert len({BBox(2, 3, 2, 2), BBox(2, 3, 2, 2)}) == 1

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            BBox(2, 3, 2, 2).x = 0


class TestFormats:
    def test_from_x1y1wh(self):
        assert BBox.from_x1y1wh(1, 2, 4, 6) == BBox(3, 5, 4, 6)
        assert (BBox.from_x1y1wh(1, 2, 4, 6).x_min, BBox.from_x1y1wh(1, 2, 4, 6).y_min) == (1, 2)

    def test_from_x1y1x2y2_normalizes(self):
        assert BBox.from_x1y1x2y2(4, 6, 1, 2) == BBox(2.5, 4, 3, 4)

    def test_resolve_formats(self):
        expected = BBox(2.5, 4, 3, 4)
        assert BBox.resolve(x_min=1, y_min=2, x_max=4, y_max=6) == expected
        assert BBox.resolve(x1=1, y1=2, x2=4, y2=6) == expected
        assert BBox.resolve(**expected.to_xywh()) == expected
        assert BBox.resolve(x1=1, y1=2, w=3, h=4) == expected

    def test_resolve_unknown_format(self):
        with pytest.raises(ValueError, match="Could not format"):
            BBox.resolve(a=1, b=2)

    def test_to_dicts_round_trip(self):
        bbox = BBox(2.5, 4, 3, 4)
        assert BBox.resolve(**bbox.to_x_min_y_min_x_max_y_max()) == bbox
        assert BBox.resolve(**bbox.to_x1y1x2y2()) == bbox
        assert BBox.resolve(**bbox.to_x1x2y1y2()) == bbox
        assert BBox.resolve(**bbox.to_xywh()) == bbox


class TestOperations:
    def test_translate(self):
        assert BBox(2, 3, 2, 2).translate(10, 20) == BBox(12, 23, 2, 2)

    def test_scale_about_center(self):
        scaled = BBox(2, 3, 2, 2).scale(2, 3)
        assert scaled == BBox(2, 3, 4, 6)
        assert (scaled.x_min, scaled.y_min) == (0, 0)

    def test_scale_validates_factors(self):
        bbox = BBox(2, 3, 2, 2)
        with pytest.raises(ValueError, match="positive"):
            bbox.scale(0, 1)
        with pytest.raises(ValueError, match="positive"):
            bbox.scale(1, -1)
        with pytest.raises(ValueError, match="finite"):
            bbox.scale(float("nan"), 1)

    def test_reference(self):
        frame = BBox(5, 5, 10, 10)
        inner = BBox(7, 8, 2, 4)
        relative = inner.reference(frame)
        assert (relative.x_min, relative.y_min) == (6, 6)
        assert (relative.w, relative.h) == (2, 4)

    def test_overlaps_and_intersection(self):
        a = BBox(1, 1, 2, 2)
        b = BBox(2, 2, 2, 2)
        assert a.overlaps(b)
        assert a.get_intersection(b) == BBox(1.5, 1.5, 1, 1)
        far = BBox(50, 50, 1, 1)
        assert not a.overlaps(far)
        assert a.get_intersection(far) is None
        touching = BBox(2.5, 1.5, 1, 1)
        assert not a.overlaps(touching)

    def test_overlaps_accepts_subclasses_via_bounds(self):
        rotated = RBBox(2, 2, 2, 2, math.radians(45))
        assert BBox(2, 2, 2, 2).overlaps(rotated)

    def test_iou_known_values(self):
        assert BBox(1, 1, 2, 2).iou(BBox(1, 1, 2, 2)) == pytest.approx(1.0)
        assert BBox(1, 1, 2, 2).iou(BBox(20, 20, 2, 2)) == 0.0
        assert BBox(1, 1, 2, 2).iou(BBox(2, 1, 2, 2)) == pytest.approx(1 / 3)

    def test_iou_degenerate_zero_union(self):
        degenerate = BBox(1, 1, 0, 0)
        assert degenerate.iou(BBox(1, 1, 0, 0)) == 0.0


class TestStrictTyping:
    def test_iou_rejects_subclasses(self):
        with pytest.raises(TypeError, match="plain BBox"):
            BBox(1, 1, 2, 2).iou(RBBox(1, 1, 2, 2))
        with pytest.raises(TypeError, match="plain BBox"):
            BBox(1, 1, 2, 2).iou(Segm.from_polygon([(0, 0), (2, 0), (2, 2)]))

    def test_to_bbox_returns_self(self):
        bbox = BBox(2, 3, 2, 2)
        assert bbox.to_bbox() is bbox
