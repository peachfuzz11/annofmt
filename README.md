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

# Build from whichever coordinate layout you already have
BBox.from_xywh(30, 50, 40, 20)        # center x/y + w/h (native, YOLO-style)
BBox.from_xyxy(10, 40, 50, 60)        # corners: x_min, y_min, x_max, y_max (Pascal VOC)
BBox.from_xxyy(10, 50, 40, 60)        # x range then y range: x1, x2, y1, y2
BBox.from_ltwh(10, 40, 40, 20)        # top-left corner + w/h (COCO)

# ...and read any layout back out as a plain tuple
bbox.to_xywh()                        # (x, y, w, h)
bbox.to_xyxy()                        # (x_min, y_min, x_max, y_max)
bbox.to_xxyy()                        # (x_min, x_max, y_min, y_max)
bbox.to_ltwh()                        # (x_min, y_min, w, h)

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

## Coordinate layouts

`BBox` stores center-plus-extent, but you rarely have data in that form.
The `from_*` classmethods accept the common layouts and the `to_*` methods
return them as plain tuples:

| Layout      | Constructor         | Aliases                        | Meaning                              |
| ----------- | ------------------- | ------------------------------ | ------------------------------------ |
| `xywh`      | `BBox.from_xywh`    | `from_cxcywh`                  | center x, center y, width, height    |
| `xyxy`      | `BBox.from_xyxy`    | `from_x1y1x2y2`                | x_min, y_min, x_max, y_max (Pascal VOC) |
| `xxyy`      | `BBox.from_xxyy`    | `from_x1x2y1y2`                | x_min, x_max, y_min, y_max           |
| `ltwh`      | `BBox.from_ltwh`    | `from_x1y1wh`                  | left, top, width, height (COCO)      |

`from_xyxy` / `from_xxyy` are order-independent (corners may be passed in
either order). The classmethods are inherited by `RBBox` (angle defaults to
`0`). Every constructor and accessor takes / preserves `meta`.

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
