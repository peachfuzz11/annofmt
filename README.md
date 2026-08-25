# annofmt

Immutable geometry types and utilities for computer-vision annotations.
Zero dependencies, fully typed, Python 3.12+.

## Install

```bash
uv add annofmt
# or
pip install annofmt
```

## Quickstart

```python
from annofmt import Annotation, BBox, RBBox, Segm, Tag

bbox = BBox(10, 20, 50, 90)
bbox.to_yolo()                      # normalized center format, no image size needed
BBox.from_yolo(0.5, 0.5, 0.2, 0.4)  # stays normalized; use .scale(w, h) for pixels

rotated = RBBox.from_degrees(cx=30, cy=50, w=40, h=10, degrees=25)
rotated.as_bbox()                   # lossy enclosing axis-aligned box

segm = Segm.from_polygon([(0, 0), (9, 0), (9, 9), (5, 5), (0, 9)])
segm.to_rle(height=10, width=10)    # COCO-style column-major run-length counts
Segm.from_rle(runs, 10, 10)         # ...and back to lattice polygons

annotation = Annotation(geometry=segm).add_tag(Tag("person", score=0.98))
annotation.bbox                     # works for every geometry type
```

All types are frozen dataclasses: operations return new instances and never
mutate the receiver.

## Geometry types

| Type      | Coordinates            | Description                              |
| --------- | ---------------------- | ---------------------------------------- |
| `BBox`    | continuous floats      | axis-aligned bounding box                |
| `RBBox`   | continuous floats      | rotated box (center + extent + angle)    |
| `Segm`    | integer pixel indices  | one or more polygon rings, even-odd fill |

`RBBox` stores angles canonically in radians in `[-pi, pi)`; construct with
`RBBox(...)`, `RBBox.from_degrees(...)` and read via `.angle_rad` /
`.angle_deg`.

## Conversion matrix

| From \ To | `BBox`                            | `RBBox`                                | `Segm`                                              |
| --------- | --------------------------------- | -------------------------------------- | --------------------------------------------------- |
| `BBox`    | —                                 | unavailable (no rotation to recover)   | exact rectangle ring (`Segm.from_bbox`)             |
| `RBBox`   | enclosing box (`as_bbox`, lossy)  | —                                      | corner ring (`Segm.from_rbbox`, rounded to lattice) |
| `Segm`    | enclosing box (`as_bbox`, lossy)  | minimum-area rect (`as_rbbox`, lossy)  | —                                                   |

Notes:

- `RBBox -> BBox` exists because the enclosing axis-aligned box is well
  defined; `BBox -> RBBox` does not, because rotation cannot be invented.
- IoU is exact within a family (`BBox`/`BBox`, `RBBox`/`RBBox`,
  `Segm`/`Segm`); `RBBox` also accepts `BBox` exactly (a `BBox` is a rotated
  box at angle 0), and `BBox.iou(rbbox)` delegates accordingly. Other
  combinations raise `TypeError` instead of silently approximating.

## Conventions

- **Indices**: `Segm` coordinates are integer pixel indices. Fractional input
  is rejected. Vertices sit on the integer lattice.
- **Rasterization**: pixels are unit squares centered at `(col + 0.5,
  row + 0.5)`; rings combine under an even-odd rule, so nested rings act as
  holes regardless of winding order.
- **RLE**: COCO-style column-major counts starting with zeros;
  `to_rle(height, width)` / `from_rle(runs, height, width)` round-trip.
- **YOLO**: `from_yolo`/`to_yolo` deal purely in normalized center-format
  coordinates and never touch image sizes; mapping between normalized space
  and pixels is orthogonal via `scale(width_factor, height_factor)`.
- **Immutability**: every geometry operation yields a new instance.

## Migrating from the pre-release `annotations` package

- The distribution and import name changed to `annofmt`.
- `Tag.prob` is now `Tag.score` and validated to `[0, 1]`.
- `scale()` takes `(factor_w, factor_h)` (was `(factor_h, factor_w)`) and
  validates its factors instead of silently ignoring zeros.
- `Annotation` holds any `Geometry` under `.geometry`; `.bbox` remains as a
  convenience property.
- All classes are now immutable dataclasses with value equality and hashing.
