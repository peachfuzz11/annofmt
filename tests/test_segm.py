from __future__ import annotations

import dataclasses
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
        assert len(segm.parts) == 1
        assert segm.parts[0] == SQUARE

    def test_explicit_parts(self):
        segm = Segm([SQUARE, L_SHAPE])
        assert len(segm.parts) == 2

    def test_fractional_coordinates_rejected(self):
        with pytest.raises(ValueError, match="integer pixel index"):
            Segm.from_polygon([(0.5, 0), (2, 0), (2, 2)])

    def test_integral_floats_accepted(self):
        segm = Segm.from_polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)])
        assert segm.parts[0][0] == (0, 0)

    def test_bools_rejected(self):
        with pytest.raises(ValueError, match="bool"):
            Segm.from_polygon([(True, 0), (2, 0), (2, 2)])

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
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.parts = ()

    def test_empty_segm_allowed(self):
        assert Segm([]).parts == ()


class TestBBoxConversions:
    def test_from_bbox_floors_and_ceils(self):
        segm = Segm.from_bbox(BBox(0.2, 1.7, 3.9, 2.1))
        assert segm.parts == (((0, 1), (4, 1), (4, 3), (0, 3)),)

    def test_as_bbox_enclosing(self):
        segm = Segm.from_polygon(L_SHAPE)
        assert segm.as_bbox() == BBox(0, 0, 2, 2)

    def test_lattice_round_trip_is_stable(self):
        bbox = BBox(1.0, 2.0, 4.0, 6.0)
        segm = Segm.from_bbox(bbox)
        assert Segm.from_bbox(segm.as_bbox()).parts == segm.parts

    def test_as_bbox_of_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Segm([]).as_bbox()

    def test_from_rbbox_corners(self):
        rbbox = RBBox(1.5, 1.5, 3, 3, 0.0)
        segm = Segm.from_rbbox(rbbox)
        assert segm.as_bbox() == BBox(0, 0, 3, 3)


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
        runs = Segm.from_bbox(BBox(0, 0, 2, 2)).to_rle(2, 2)
        assert runs == (0, 4)

    def test_decode_full_square(self):
        segm = Segm.from_rle((0, 4), 2, 2)
        assert segm.as_bbox() == BBox(0, 0, 2, 2)

    def test_l_shape_known_runs(self):
        runs = Segm.from_polygon(L_SHAPE).to_rle(2, 2)
        assert runs == (0, 3, 1)

    def test_empty_mask_encoding(self):
        assert Segm([]).to_rle(3, 5) == (15,)

    def test_rle_size_mismatch_rejected(self):
        with pytest.raises(ValueError, match="sum to"):
            Segm.from_rle((0, 5), 2, 2)

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
        a = Segm.from_polygon(((0, 0), (2, 0), (2, 2), (0, 2)))
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

    def test_empty(self):
        assert Segm([]).iou(Segm.from_polygon(SQUARE)) == 0.0

    def test_rejects_other_types(self):
        with pytest.raises(TypeError, match="BBox"):
            Segm.from_polygon(SQUARE).iou(BBox(0, 0, 2, 2))


class TestMinAreaRect:
    def test_axis_aligned_rect_recovers_exactly(self):
        segm = Segm.from_polygon(((0, 0), (4, 0), (4, 2), (0, 2)))
        rbbox = segm.as_rbbox()
        assert rbbox.area == pytest.approx(8.0)
        enclosing = rbbox.as_bbox()
        for x, y in segm.parts[0]:
            assert enclosing.x_min <= x <= enclosing.x_max
            assert enclosing.y_min <= y <= enclosing.y_max

    def test_diamond_tighter_than_bbox(self):
        diamond = Segm.from_polygon(((2, 0), (4, 2), (2, 4), (0, 2)))
        rbbox = diamond.as_rbbox()
        assert rbbox.area < diamond.as_bbox().area
        assert rbbox.area == pytest.approx(8.0)

    def test_translate_scale(self):
        segm = Segm.from_polygon(SQUARE)
        moved = segm.translate(10, 20)
        assert moved.as_bbox() == BBox(10, 20, 12, 22)
        grown = segm.scale(3, 2)
        assert grown.area == pytest.approx(4 * 3 * 2)
