# annofmt

Immutable geometry types and utilities for computer-vision annotations.
Zero dependencies, fully typed, Python 3.12+.

## Install

```bash
uv add annofmt
# or
pip install annofmt
# or straight from the repo
uv add git+https://github.com/peachfuzz11/annofmt.git
```

## Quickstart

```python
from annofmt.annotation import Annotation
from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox
from annofmt.geometry.segm import Segm
from annofmt.tag import Tag

bbox = BBox(x=30, y=50, w=40, h=20)   # center + extent
bbox.x_min                            # derived corner values
bbox.to_xywh()

# Normalized coordinates in [0, 1] relative to image dimensions
bbox_normalized = bbox.normalize(Height, Width)
# Denormalize back to pixel coordinates
bbox_denormalized = bbox_normalized.denormalize(Height, Width)

rotated = RBBox.from_degrees(x=30, y=50, w=40, h=20, degrees=25)   # same xywh, plus `a`
rotated.corners()
rotated.to_bbox()                     # lossy enclosing axis-aligned box

segm = Segm.from_polygon([(0, 0), (9, 0), (9, 9), (5, 5), (0, 9)])
segm.indices                          # integer pixel-index rings
segm.to_rle(height=10, width=10)      # COCO-style column-major run-length counts
Segm.from_rle(runs, 10, 10)           # ...and back to rings

annotation = Annotation(geometry=segm).add_tag(Tag("person", score=0.98))
annotation.bbox                       # works for every geometry type
```

All types are immutable: operations return new instances and never mutate
the receiver.

## Geometry types

`BBox` is the base class; storage is always center-plus-extent (`x`, `y`,
`w`, `h`).

| Type     | Extra storage | Description                                       |
| -------- | ------------- | ------------------------------------------------- |
| `BBox`   | —             | axis-aligned box; corners are derived properties   |
| `RBBox`  | `a`           | adds rotation in radians; corners become the enclosing box |
| `Segm`   | `indices`     | polygon rings over integer pixel indices; `x/y/w/h` hold the enclosing box |

## Conversion matrix

| From \ To | `BBox`                           | `RBBox`                                 | `Segm`                                              |
| --------- | -------------------------------- | --------------------------------------- | --------------------------------------------------- |
| `BBox`    | —                                | unavailable (rotation cannot be invented)| exact rectangle ring (`Segm.from_bbox`)             |
| `RBBox`   | enclosing box (`to_bbox`, lossy) | —                                       | corner ring (`Segm.from_rbbox`, rounded to lattice) |
| `Segm`    | enclosing box (`to_bbox`, lossy) | minimum-area rect (`as_rbbox`, lossy)   | —                                                   |

IoU is defined **only between two geometries of the same type** and raises
`TypeError` otherwise. Convert explicitly first when a cross-type comparison
is really wanted (e.g. compare every geometry's `to_bbox()`).

## Conventions

- **Angles**: stored in radians, canonically wrapped to `[-pi, pi)`. Build
  with `from_degrees`, read via `.angle_deg` / `.angle_rad`.
- **Indices**: `Segm` coordinates are integer pixel indices; fractional input
  is rejected. Vertices sit on the integer lattice.
- **Rasterization**: pixels are unit squares centered at `(col + 0.5,
  row + 0.5)`; rings combine under an even-odd rule, so nested rings act as
  holes regardless of winding order.
- **RLE**: COCO-style column-major counts starting with zeros;
  `to_rle(height, width)` / `from_rle(runs, height, width)` round-trip.
- **Immutability**: every operation yields a new instance.
