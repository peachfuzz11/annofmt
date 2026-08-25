"""annofmt: immutable geometry types and utilities for computer-vision annotations."""

from annofmt.annotation import Annotation
from annofmt.geometry import BBox, Geometry, RBBox, Segm
from annofmt.tag import Tag

__all__ = ["Annotation", "BBox", "Geometry", "RBBox", "Segm", "Tag"]

__version__ = "0.1.0"
