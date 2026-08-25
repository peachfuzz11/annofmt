from __future__ import annotations

import annofmt


def test_public_api_exports():
    for name in annofmt.__all__:
        assert getattr(annofmt, name) is not None
    assert "BBox" in annofmt.__all__
    assert "RBBox" in annofmt.__all__
    assert "Segm" in annofmt.__all__


def test_version():
    assert annofmt.__version__ == "0.1.0"
