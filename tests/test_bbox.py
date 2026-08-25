from __future__ import annotations

import dataclasses
import math

import pytest

from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox
from annofmt.geometry.segm import Segm


class TestConstruction:
    def test_normalizes_swapped_corners(self):
        assert BBox(3, 4, 1, 2) == BBox(1, 2, 3, 4)

    def test_value_equality_and_hash(self):
        assert BBox(1, 2, 3, 4) == BBox(1, 2, 3, 4)
        assert len({BBox(1, 2, 3, 4), BBox(1, 2, 3, 4)}) == 1

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            BBox(0, 0, float("nan"), 1)
        with pytest.raises(ValueError, match="finite"):
            BBox(0, 0, float("inf"), 1)

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            BBox(1, 2, 3, 4).x_min = 0

    def test_dimensions(self):
        bbox = BBox(1, 2, 5, 12)
        assert bbox.width == 4
        assert bbox.height == 10
        assert bbox.area == 40


class TestFormats:
    def test_from_x1y1wh(self):
        assert BBox.from_x1y1wh(1, 2, 3, 4) == BBox(1, 2, 4, 6)

    def test_from_x1y1x2y2(self):
        assert BBox.from_x1y1x2y2(4, 6, 1, 2) == BBox(1, 2, 4, 6)

    def test_from_xywh(self):
        assert BBox.from_xywh(2, 3, 4, 6) == BBox(0, 0, 4, 6)

    def test_from_x1y1wh_negative_extent(self):
        assert BBox.from_x1y1wh(4, 6, -3, -4) == BBox(1, 2, 4, 6)

    def test_to_dicts_round_trip(self):
        bbox = BBox(1, 2, 3, 4)
        assert BBox.resolve(**bbox.to_x_min_y_min_x_max_y_max()) == bbox
        assert BBox.resolve(**{"x1": 1, "y1": 2, "x2": 3, "y2": 4}) == bbox
        assert BBox.resolve(**bbox.to_xywh()) == bbox
        assert BBox.resolve(x1=1, y1=2, w=2, h=2) == bbox

    def test_resolve_unknown_format(self):
        with pytest.raises(ValueError, match="Could not format"):
            BBox.resolve(a=1, b=2)

    def test_yolo_round_trip(self):
        bbox = BBox.from_yolo(0.5, 0.5, 0.2, 0.4)
        yolo = bbox.to_yolo()
        assert yolo["cx"] == pytest.approx(0.5)
        assert yolo["cy"] == pytest.approx(0.5)
        assert yolo["w"] == pytest.approx(0.2)
        assert yolo["h"] == pytest.approx(0.4)
        assert BBox.from_yolo(**yolo).to_yolo() == pytest.approx(yolo)


class TestOperations:
    def test_translate(self):
        assert BBox(1, 2, 3, 4).translate(10, 20) == BBox(11, 22, 13, 24)

    def test_scale_about_center(self):
        assert BBox(0, 0, 2, 2).scale(2, 3) == BBox(-1, -2, 3, 4)

    def test_scale_validates_factors(self):
        bbox = BBox(0, 0, 2, 2)
        with pytest.raises(ValueError, match="positive"):
            bbox.scale(0, 1)
        with pytest.raises(ValueError, match="positive"):
            bbox.scale(1, -1)
        with pytest.raises(ValueError, match="finite"):
            bbox.scale(float("nan"), 1)

    def test_reference(self):
        assert BBox(5, 5, 7, 9).reference(BBox(5, 5, 10, 10)) == BBox(0, 0, 2, 4)

    def test_overlaps_and_intersection(self):
        a, b = BBox(0, 0, 2, 2), BBox(1, 1, 3, 3)
        assert a.overlaps(b)
        assert a.get_intersection(b) == BBox(1, 1, 2, 2)
        c = BBox(10, 10, 11, 11)
        assert not a.overlaps(c)
        assert a.get_intersection(c) is None
        assert not a.overlaps(BBox(2, 0, 3, 1))

    def test_iou_known_values(self):
        assert BBox(0, 0, 2, 2).iou(BBox(0, 0, 2, 2)) == pytest.approx(1.0)
        assert BBox(0, 0, 2, 2).iou(BBox(5, 5, 6, 6)) == 0.0
        assert BBox(0, 0, 2, 2).iou(BBox(1, 0, 3, 2)) == pytest.approx(1 / 3)

    def test_iou_degenerate_zero_union(self):
        degenerate = BBox(1, 1, 1, 1)
        assert degenerate.iou(degenerate) == 0.0

    def test_bounds_and_as_bbox_identity(self):
        bbox = BBox(1, 2, 3, 4)
        assert bbox.bounds() is bbox
        assert bbox.as_bbox() is bbox


class TestCrossType:
    def test_iou_delegates_to_rbbox_exactly(self):
        rbbox = RBBox(1.0, 1.0, 2.0, 2.0, math.radians(30))
        bbox = BBox(0, 0, 2, 2)
        assert bbox.iou(rbbox) == pytest.approx(rbbox.iou(bbox))

    def test_iou_rejects_segm(self):
        segm = Segm.from_polygon([(0, 0), (2, 0), (2, 2)])
        with pytest.raises(TypeError, match="Segm"):
            BBox(0, 0, 2, 2).iou(segm)
