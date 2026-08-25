from __future__ import annotations

import dataclasses
import math

import pytest

from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox
from annofmt.geometry.segm import Segm


class TestConstruction:
    def test_angle_defaults_to_zero(self):
        assert RBBox(1, 2, 3, 4).angle == 0.0

    def test_angle_canonicalized(self):
        assert RBBox(0, 0, 2, 2, math.radians(190)).angle == pytest.approx(math.radians(-170))
        assert RBBox(0, 0, 2, 2, math.radians(-190)).angle == pytest.approx(math.radians(170))
        assert RBBox(0, 0, 2, 2, 3 * math.pi).angle == pytest.approx(-math.pi)

    def test_from_degrees_and_accessors(self):
        rbbox = RBBox.from_degrees(1, 2, 3, 4, 90)
        assert rbbox.angle_rad == pytest.approx(math.pi / 2)
        assert rbbox.angle_deg == pytest.approx(90.0)

    def test_negative_extent_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            RBBox(0, 0, -1, 2)
        with pytest.raises(ValueError, match="non-negative"):
            RBBox(0, 0, 1, -2)

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            RBBox(0, 0, float("nan"), 2)

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            RBBox(0, 0, 1, 1).cx = 5

    def test_equality_and_hash(self):
        a = RBBox.from_degrees(0, 0, 2, 2, 45)
        b = RBBox.from_degrees(0, 0, 2, 2, 45)
        assert a == b
        assert len({a, b}) == 1


class TestGeometry:
    def test_area_independent_of_angle(self):
        assert RBBox.from_degrees(0, 0, 3, 5, 37).area == pytest.approx(15.0)

    def test_corners_axis_aligned(self):
        corners = RBBox(0, 0, 4, 2).corners()
        assert set(corners) == {(2, 1), (-2, 1), (-2, -1), (2, -1)}

    def test_corners_rotated_45(self):
        side = math.sqrt(2)
        corners = RBBox(0, 0, side, side, math.pi / 4).corners()
        expected = {(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)}
        for corner in corners:
            assert any(
                math.isclose(corner[0], x, abs_tol=1e-9) and math.isclose(corner[1], y, abs_tol=1e-9)
                for x, y in expected
            )

    def test_as_bbox_unrotated_is_identity(self):
        assert RBBox(2, 3, 4, 6).as_bbox() == BBox(0, 0, 4, 6)

    def test_as_bbox_45_degree_square(self):
        side = math.sqrt(2)
        assert RBBox(0, 0, side, side, math.pi / 4).as_bbox() == BBox(-1, -1, 1, 1)

    def test_bounds_matches_as_bbox(self):
        rbbox = RBBox.from_degrees(1, 1, 10, 3, 20)
        assert rbbox.bounds() == rbbox.as_bbox()

    def test_rotate_wraps(self):
        assert RBBox(0, 0, 2, 2).rotate(370, unit="deg").angle_deg == pytest.approx(10.0)

    def test_rotate_invalid_unit(self):
        with pytest.raises(ValueError, match="unit"):
            RBBox(0, 0, 2, 2).rotate(10, unit="gradians")

    def test_translate_and_scale_preserve_angle(self):
        rbbox = RBBox.from_degrees(1, 2, 4, 8, 30)
        moved = rbbox.translate(5, 5).scale(2, 2)
        assert (moved.cx, moved.cy) == (6, 7)
        assert moved.w == pytest.approx(8)
        assert moved.h == pytest.approx(16)
        assert moved.angle == rbbox.angle


class TestIoU:
    def test_identical(self):
        rbbox = RBBox.from_degrees(0, 0, 4, 4, 17)
        assert rbbox.iou(rbbox) == pytest.approx(1.0)

    def test_disjoint(self):
        a = RBBox(0, 0, 2, 2)
        b = RBBox(100, 100, 2, 2)
        assert a.iou(b) == 0.0

    def test_axis_aligned_half_shift(self):
        assert RBBox(0, 0, 2, 2).iou(RBBox(1, 0, 2, 2)) == pytest.approx(1 / 3)

    def test_unit_squares_45_degrees(self):
        a = RBBox(0, 0, 1, 1, 0.0)
        b = RBBox(0, 0, 1, 1, math.pi / 4)
        assert a.iou(b) == pytest.approx(math.sqrt(2) / 2, rel=1e-6)

    def test_accepts_bbox_exactly(self):
        rbbox = RBBox(1, 1, 2, 2, 0.0)
        assert rbbox.iou(BBox(1, 0, 3, 2)) == pytest.approx(1 / 3)

    def test_rejects_segm(self):
        segm = Segm.from_polygon([(0, 0), (2, 0), (2, 2)])
        with pytest.raises(TypeError, match="Segm"):
            RBBox(0, 0, 2, 2).iou(segm)


class TestConversions:
    def test_no_bbox_to_rbbox_conversion(self):
        assert not hasattr(BBox, "as_rbbox")
        assert not hasattr(BBox, "to_rbbox")
