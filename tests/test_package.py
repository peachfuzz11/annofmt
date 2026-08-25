from __future__ import annotations

import annofmt
from annofmt.annotation import Annotation
from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox
from annofmt.geometry.segm import Segm
from annofmt.tag import Tag


def test_submodule_imports():
    for cls in (BBox, RBBox, Segm, Annotation, Tag):
        assert cls is not None


def test_package_inits_are_empty():
    import pathlib

    package_root = pathlib.Path(annofmt.__file__).parent
    init = (package_root / "__init__.py").read_text().strip()
    geometry_init = (package_root / "geometry" / "__init__.py").read_text().strip()
    assert init == ""
    assert geometry_init == ""
