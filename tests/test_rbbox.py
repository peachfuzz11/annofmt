from __future__ import annotations

import math

import pytest

from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox
from annofmt.geometry.segm import Segm


class TestConstruction:
    def test_angle_defaults_to_zero(self):
        assert RBBox(1, 2, 3, 4).a == 0.0

    def test_angle_canonicalized(self):
        assert RBBox(0, 0, 2, 2, math.radians(190)).a == pytest.approx(math.radians(-170))
        assert RBBox(0, 0, 2, 2, math.radians(-190)).a == pytest.approx(math.radians(170))
        assert RBBox(0, 0, 2, 2, 3 * math.pi).a == pytest.approx(-math.pi)

    def test_from_degrees_and_accessors(self):
        rbbox = RBBox.from_degrees(1, 2, 3, 4, 90)
        assert rbbox.angle_rad == pytest.approx(math.pi / 2)
        assert rbbox.angle_deg == pytest.approx(90.0)

    def test_negative_extent_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            RBBox(0, 0, -1, 2)

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            RBBox(0, 0, float("nan"), 2)

    def test_frozen(self):
        with pytest.raises(AttributeError):
            RBBox(0, 0, 1, 1).x = 5

    def test_equality_uses_all_five_fields(self):
        base = RBBox.from_degrees(0, 0, 2, 2, 45)
        assert base == RBBox.from_degrees(0, 0, 2, 2, 45)
        assert base != RBBox.from_degrees(0, 0, 2, 2, 46)
        assert len({base, RBBox.from_degrees(0, 0, 2, 2, 45)}) == 1

    def test_not_equal_to_bbox_with_same_bounds(self):
        axis = BBox(1, 1, 2, 2)
        rotated = RBBox(1, 1, 2, 2, 0.0)
        assert rotated != axis
        assert axis != rotated


class TestDerivedCorners:
    def test_unrotated_matches_plain_bbox(self):
        rbbox = RBBox(2, 3, 4, 6)
        plain = BBox(2, 3, 4, 6)
        assert (rbbox.x_min, rbbox.y_min, rbbox.x_max, rbbox.y_max) == (
            plain.x_min,
            plain.y_min,
            plain.x_max,
            plain.y_max,
        )

    def test_rotated_corners_are_enclosing(self):
        side = math.sqrt(2)
        rbbox = RBBox(0, 0, side, side, math.pi / 4)
        assert (rbbox.x_min, rbbox.y_min, rbbox.x_max, rbbox.y_max) == (-1, -1, 1, 1)

    def test_area_is_exact_not_enclosing(self):
        side = math.sqrt(2)
        rbbox = RBBox(0, 0, side, side, math.pi / 4)
        assert rbbox.area == pytest.approx(2.0)
        assert rbbox.to_bbox().area == pytest.approx(4.0)


class TestGeometry:
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

    def test_from_corners_rectangle(self):
        ring = ((0, 0), (2, 0), (2, 2), (0, 2))
        rbbox = RBBox.from_corners(*ring)
        assert rbbox == RBBox(1, 1, 2, 2, 0.0)

    def test_from_corners_rotated_diamond(self):
        diamond = ((1, 0), (2, 1), (1, 2), (0, 1))
        rbbox = RBBox.from_corners(*diamond)
        assert rbbox.x == pytest.approx(1)
        assert rbbox.y == pytest.approx(1)
        assert rbbox.w == pytest.approx(math.sqrt(2))
        assert rbbox.h == pytest.approx(math.sqrt(2))
        assert rbbox.angle_deg == pytest.approx(45)

    def test_from_corners_round_trip(self):
        original = RBBox.from_degrees(3, -2, 10, 4, 25)
        rebuilt = RBBox.from_corners(*original.corners())
        assert rebuilt.w == pytest.approx(original.w)
        assert rebuilt.h == pytest.approx(original.h)
        assert rebuilt.x == pytest.approx(original.x)
        assert rebuilt.y == pytest.approx(original.y)
        rebuilt_corners = sorted(rebuilt.corners())
        original_corners = sorted(original.corners())
        for (rx, ry), (ox, oy) in zip(rebuilt_corners, original_corners, strict=True):
            assert rx == pytest.approx(ox)
            assert ry == pytest.approx(oy)

    def test_from_corners_rejects_non_rectangle(self):
        with pytest.raises(ValueError, match="rectangle"):
            RBBox.from_corners((0, 0), (2, 0), (2, 2), (0.5, 2))

    def test_rotate_wraps(self):
        assert RBBox(0, 0, 2, 2).rotate(370, unit="deg").angle_deg == pytest.approx(10.0)

    def test_rotate_invalid_unit(self):
        with pytest.raises(ValueError, match="unit"):
            RBBox(0, 0, 2, 2).rotate(10, unit="gradians")

    def test_translate_and_scale_preserve_rotation(self):
        rbbox = RBBox.from_degrees(1, 2, 4, 8, 30)
        moved = rbbox.translate(5, 5).scale(2, 2)
        assert (moved.x, moved.y) == (6, 7)
        assert moved.w == pytest.approx(8)
        assert moved.h == pytest.approx(16)
        assert moved.a == rbbox.a


class TestIoU:
    def test_identical(self):
        rbbox = RBBox.from_degrees(0, 0, 4, 4, 17)
        assert rbbox.iou(rbbox) == pytest.approx(1.0)

    def test_disjoint(self):
        a = RBBox(0, 0, 2, 2)
        b = RBBox(100, 100, 2, 2)
        assert a.iou(b) == 0.0

    def test_axis_aligned_half_shift(self):
        assert RBBox(1, 1, 2, 2).iou(RBBox(2, 1, 2, 2)) == pytest.approx(1 / 3)

    def test_unit_squares_45_degrees(self):
        a = RBBox(0, 0, 1, 1, 0.0)
        b = RBBox(0, 0, 1, 1, math.pi / 4)
        assert a.iou(b) == pytest.approx(math.sqrt(2) / 2, rel=1e-6)

    def test_rejects_everything_else(self):
        with pytest.raises(TypeError, match="RBBox only"):
            RBBox(1, 1, 2, 2).iou(BBox(1, 1, 2, 2))
        with pytest.raises(TypeError, match="RBBox only"):
            RBBox(1, 1, 2, 2).iou(Segm.from_polygon([(0, 0), (2, 0), (2, 2)]))


class TestToBBox:
    def test_unrotated_is_value_equal_to_plain_box(self):
        assert RBBox(2, 3, 4, 6).to_bbox() == BBox(2, 3, 4, 6)
        assert isinstance(RBBox(2, 3, 4, 6).to_bbox(), BBox)
        assert type(RBBox(2, 3, 4, 6).to_bbox()) is BBox

    def test_rotated_45_degree_square(self):
        side = math.sqrt(2)
        bbox = RBBox(0, 0, side, side, math.pi / 4).to_bbox()
        assert bbox == BBox(0, 0, 2, 2)
