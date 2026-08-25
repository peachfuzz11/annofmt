from __future__ import annotations

import random

import pytest

from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox
from annofmt.geometry.segm import Segm

SQUARE = ((0, 0), (2, 0), (2, 2), (0, 2))
L_SHAPE = ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))


class TestConstruction:
    def test_flat_polygon_is_wrapped(self):
        segm = Segm([SQUARE])
        assert len(segm.indices) == 1
        assert segm.indices[0] == SQUARE

    def test_explicit_rings(self):
        segm = Segm([SQUARE, L_SHAPE])
        assert len(segm.indices) == 2

    def test_inherited_xywh_is_enclosing_box(self):
        segm = Segm.from_polygon(SQUARE)
        assert (segm.x, segm.y, segm.w, segm.h) == (1, 1, 2, 2)
        assert (segm.x_min, segm.y_min, segm.x_max, segm.y_max) == (0, 0, 2, 2)

    def test_fractional_coordinates_rejected(self):
        with pytest.raises(ValueError, match="integer pixel index"):
            Segm.from_polygon([(0.5, 0), (2, 0), (2, 2)])

    def test_integral_floats_accepted(self):
        segm = Segm.from_polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)])
        assert segm.indices[0][0] == (0, 0)

    def test_bools_rejected(self):
        with pytest.raises(ValueError, match="bool"):
            Segm.from_polygon([(True, 0), (2, 0), (2, 2)])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="at least one ring"):
            Segm([])

    def test_degenerate_ring_rejected(self):
        with pytest.raises(ValueError, match="at least 3"):
            Segm.from_polygon([(0, 0), (1, 1)])

    def test_bad_point_shape_rejected(self):
        with pytest.raises(ValueError, match="exactly 2"):
            Segm.from_polygon([(0, 0, 0), (2, 0), (2, 2)])

    def test_frozen_and_hashable(self):
        a = Segm.from_polygon(SQUARE)
        b = Segm.from_polygon(list(SQUARE))
        assert a == b
        assert len({a, b}) == 1
        with pytest.raises(AttributeError):
            a.indices = ()

    def test_equality_uses_indices_not_bounds(self):
        square = Segm.from_polygon(SQUARE)
        same_bounds_other_shape = Segm.from_polygon(L_SHAPE)
        assert square != same_bounds_other_shape


class TestBBoxConversions:
    def test_from_bbox_floors_and_ceils(self):
        segm = Segm.from_bbox(BBox(2.5, 4, 3, 4))
        assert segm.indices == (((1, 2), (4, 2), (4, 6), (1, 6)),)

    def test_to_bbox_enclosing(self):
        segm = Segm.from_polygon(L_SHAPE)
        bbox = segm.to_bbox()
        assert type(bbox) is BBox
        assert bbox == BBox(1, 1, 2, 2)

    def test_lattice_round_trip_is_stable(self):
        bbox = BBox(2.5, 4, 3, 4)
        segm = Segm.from_bbox(bbox)
        assert Segm.from_bbox(segm.to_bbox()).indices == segm.indices

    def test_from_rbbox_corners(self):
        rbbox = RBBox(1.5, 1.5, 3, 3, 0.0)
        segm = Segm.from_rbbox(rbbox)
        assert segm.to_bbox() == BBox(1.5, 1.5, 3, 3)


class TestAreaAndContainment:
    def test_area_square(self):
        assert Segm.from_polygon(SQUARE).area == pytest.approx(4.0)

    def test_area_with_hole_subtracts(self):
        outer = ((0, 0), (3, 0), (3, 3), (0, 3))
        hole = ((1, 1), (1, 2), (2, 2), (2, 1))
        donut = Segm([outer, hole])
        assert donut.area == pytest.approx(8.0)
        assert not donut.contains_point(1, 1)
        assert donut.contains_point(0, 0)
        assert donut.contains_point(2, 1)

    def test_contains_point_interior_exterior(self):
        segm = Segm.from_polygon(L_SHAPE)
        assert segm.contains_point(0, 0)
        assert segm.contains_point(1, 0)
        assert not segm.contains_point(2, 2)
        assert not segm.contains_point(-1, 0)


class TestRLE:
    def test_full_square_encoding(self):
        runs = Segm.from_polygon(SQUARE).to_rle(2, 2)
        assert runs == (0, 4)

    def test_decode_full_square(self):
        segm = Segm.from_rle((0, 4), 2, 2)
        assert segm.to_bbox() == BBox(1, 1, 2, 2)

    def test_l_shape_known_runs(self):
        runs = Segm.from_polygon(L_SHAPE).to_rle(2, 2)
        assert runs == (0, 3, 1)

    def test_rle_size_mismatch_rejected(self):
        with pytest.raises(ValueError, match="sum to"):
            Segm.from_rle((0, 5), 2, 2)

    def test_empty_mask_rejected_on_decode(self):
        with pytest.raises(ValueError, match="empty"):
            Segm.from_rle((16,), 4, 4)

    @pytest.mark.parametrize(
        "geometry",
        [
            Segm.from_polygon(SQUARE),
            Segm.from_polygon(L_SHAPE),
            Segm([(0, 0), (3, 0), (3, 3), (0, 3)]),
        ],
        ids=["square", "l-shape", "outer-only"],
    )
    def test_rle_round_trip_preserves_mask(self, geometry):
        expected = geometry.to_rle(8, 8)
        decoded = Segm.from_rle(expected, 8, 8)
        assert decoded.to_rle(8, 8) == expected

    def test_donut_round_trip(self):
        outer = ((0, 0), (4, 0), (4, 4), (0, 4))
        hole = ((1, 1), (1, 3), (3, 3), (3, 1))
        donut = Segm([outer, hole])
        runs = donut.to_rle(5, 5)
        restored = Segm.from_rle(runs, 5, 5)
        assert restored.to_rle(5, 5) == runs
        assert sum(runs[1::2]) == 12

    def test_random_blob_round_trips(self):
        rng = random.Random(42)
        for _ in range(25):
            x0 = rng.randrange(0, 12)
            y0 = rng.randrange(0, 12)
            x1 = min(19, x0 + rng.randrange(1, 8))
            y1 = min(19, y0 + rng.randrange(1, 8))
            cut_x = rng.randrange(x0, x1 + 1) if x1 > x0 else x0
            blob = ((x0, y0), (cut_x, y0), (cut_x, y1), (x1, y1), (x1, y0 + (y1 - y0) // 2), (x0, y0 + (y1 - y0) // 2))
            segm = Segm.from_polygon(blob)
            runs = segm.to_rle(20, 20)
            assert Segm.from_rle(runs, 20, 20).to_rle(20, 20) == runs


class TestIoU:
    def test_identical(self):
        segm = Segm.from_polygon(SQUARE)
        assert segm.iou(segm) == pytest.approx(1.0)

    def test_disjoint(self):
        a = Segm.from_polygon(SQUARE)
        b = Segm.from_polygon(((10, 10), (12, 10), (12, 12), (10, 12)))
        assert a.iou(b) == 0.0

    def test_half_shift(self):
        a = Segm.from_polygon(SQUARE)
        b = Segm.from_polygon(((1, 0), (3, 0), (3, 2), (1, 2)))
        assert a.iou(b) == pytest.approx(1 / 3)

    def test_with_hole(self):
        solid = Segm.from_polygon(((0, 0), (4, 0), (4, 4), (0, 4)))
        holed = Segm([((0, 0), (4, 0), (4, 4), (0, 4)), ((1, 1), (1, 3), (3, 3), (3, 1))])
        assert solid.iou(holed) == pytest.approx(12 / 16)

    def test_rejects_everything_else(self):
        with pytest.raises(TypeError, match="Segm only"):
            Segm.from_polygon(SQUARE).iou(BBox(1, 1, 2, 2))
        with pytest.raises(TypeError, match="Segm only"):
            Segm.from_polygon(SQUARE).iou(RBBox(1, 1, 2, 2))


class TestMinAreaRect:
    def test_axis_aligned_rect_recovers_exactly(self):
        segm = Segm.from_polygon(((0, 0), (4, 0), (4, 2), (0, 2)))
        rbbox = segm.as_rbbox()
        assert rbbox.area == pytest.approx(8.0)
        enclosing = rbbox.to_bbox()
        for x, y in segm.indices[0]:
            assert enclosing.x_min <= x <= enclosing.x_max
            assert enclosing.y_min <= y <= enclosing.y_max

    def test_diamond_tighter_than_bbox(self):
        diamond = Segm.from_polygon(((2, 0), (4, 2), (2, 4), (0, 2)))
        rbbox = diamond.as_rbbox()
        assert rbbox.area < diamond.to_bbox().area
        assert rbbox.area == pytest.approx(8.0)


class TestOperations:
    def test_translate_shifts_indices_and_center(self):
        moved = Segm.from_polygon(SQUARE).translate(10, 20)
        assert moved.indices[0] == ((10, 20), (12, 20), (12, 22), (10, 22))
        assert (moved.x, moved.y) == (11, 21)

    def test_scale_rescales_indices(self):
        grown = Segm.from_polygon(SQUARE).scale(2, 2)
        assert grown.area == pytest.approx(16.0)
        assert grown.indices[0] == ((-1, -1), (3, -1), (3, 3), (-1, 3))
