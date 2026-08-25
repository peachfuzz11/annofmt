from __future__ import annotations

import dataclasses

import pytest

from annofmt import Annotation, Tag
from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox
from annofmt.geometry.segm import Segm


def test_tag_score_validated():
    with pytest.raises(ValueError, match="score"):
        Tag("cat", score=1.5)
    with pytest.raises(ValueError, match="score"):
        Tag("cat", score=-0.1)
    assert Tag("cat").score == 1.0


def test_annotation_holds_any_geometry():
    geometries = [
        BBox(0, 0, 2, 2),
        RBBox.from_degrees(1, 1, 4, 2, 30),
        Segm.from_polygon([(0, 0), (2, 0), (2, 2)]),
    ]
    for geometry in geometries:
        annotation = Annotation(geometry)
        assert annotation.bbox == geometry.as_bbox()


def test_has_label():
    annotation = Annotation(BBox(0, 0, 1, 1), (Tag("cat"), Tag("blurry", score=0.3)))
    assert annotation.has_label("cat")
    assert annotation.has_label("cat", "dog")
    assert not annotation.has_label("dog")


def test_add_tag_returns_new_instance():
    tag = Tag("cat")
    original = Annotation(BBox(0, 0, 1, 1))
    updated = original.add_tag(tag)
    assert updated.tags == (tag,)
    assert original.tags == ()
    assert updated is not original


def test_annotation_rejects_non_tags():
    with pytest.raises(TypeError, match="Tag"):
        Annotation(BBox(0, 0, 1, 1), ("cat",))


def test_frozen():
    annotation = Annotation(BBox(0, 0, 1, 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        annotation.geometry = RBBox(0, 0, 1, 1)
