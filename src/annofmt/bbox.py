import warnings


class BBox:
    def __init__(self, x, y, w, h, meta=None):
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.meta = meta or {}

    @classmethod
    def from_xywh(cls, x, y, w, h, meta=None):
        """Build from center x, center y, width, height (the native storage format)."""
        return cls(x, y, w, h, meta=meta)

    @classmethod
    def from_x1y1x2y2(cls, x1, y1, x2, y2, meta=None):
        """Build from two corners: (x1, y1) top-left, (x2, y2) bottom-right.

        x1 is the left edge, y1 the top edge, x2 the right edge, y2 the bottom
        edge. Corners may be given in either order.
        """
        x_left, x_right = min(x1, x2), max(x1, x2)
        y_top, y_bottom = min(y1, y2), max(y1, y2)
        return cls((x_left + x_right) / 2, (y_top + y_bottom) / 2, x_right - x_left, y_bottom - y_top, meta=meta)

    @classmethod
    def from_x1x2y1y2(cls, x1, x2, y1, y2, meta=None):
        """Build from the x range then the y range: x1 left, x2 right, y1 top, y2 bottom.

        Same box as :meth:`from_x1y1x2y2`, only the argument order differs. Edges
        may be given in either order.
        """
        return cls.from_x1y1x2y2(x1, y1, x2, y2, meta=meta)

    @classmethod
    def from_x1y1wh(cls, x1, y1, w, h, meta=None):
        """Build from the top-left corner (x1, y1) plus width and height (COCO layout).

        x1 is the left edge, y1 the top edge.
        """
        return cls(x1 + w / 2, y1 + h / 2, w, h, meta=meta)

    def to_xywh(self):
        """Return (center x, center y, width, height)."""
        return (self.x, self.y, self.w, self.h)

    def to_x1y1x2y2(self):
        """Return the two corners (x1, y1, x2, y2): left, top, right, bottom edge."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_x1x2y1y2(self):
        """Return the x range then the y range (x1, x2, y1, y2): left, right, top, bottom edge."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def to_x1y1wh(self):
        """Return (x1, y1, width, height): top-left corner plus size (COCO layout)."""
        return (self.x_min, self.y_min, self.w, self.h)

    def normalize(self, W, H):
        nx = self.x / W
        ny = self.y / H
        nw = self.w / W
        nh = self.h / H
        if not (0 <= nx <= 1 and 0 <= ny <= 1 and 0 <= nw <= 1 and 0 <= nh <= 1):
            warnings.warn("BBox normalize: coordinates outside [0, 1] after normalization", RuntimeWarning, stacklevel=2)
        return BBox(nx, ny, nw, nh, self.meta)

    def denormalize(self, W, H):
        dx = self.x * W
        dy = self.y * H
        dw = self.w * W
        dh = self.h * H
        if not (0 <= dx <= W and 0 <= dy <= H and 0 <= dw <= W and 0 <= dh <= H):
            warnings.warn("BBox denormalize: coordinates outside [0, W] and [0, H]", RuntimeWarning, stacklevel=2)
        return BBox(dx, dy, dw, dh, self.meta)

    @property
    def x_min(self):
        return self.x - self.w / 2

    @property
    def y_min(self):
        return self.y - self.h / 2

    @property
    def x_max(self):
        return self.x + self.w / 2

    @property
    def y_max(self):
        return self.y + self.h / 2

    @property
    def area(self):
        return self.w * self.h

    def translate(self, dx, dy):
        return BBox(self.x + dx, self.y + dy, self.w, self.h, self.meta)

    def scale(self, factor_w, factor_h):
        return BBox(self.x, self.y, self.w * factor_w, self.h * factor_h, self.meta)

    def overlaps(self, other):
        return not (
            self.x_max <= other.x_min or self.x_min >= other.x_max or
            self.y_max <= other.y_min or self.y_min >= other.y_max
        )

    def get_intersection(self, other):
        if not self.overlaps(other):
            return None
        return BBox(
            (max(self.x_min, other.x_min) + min(self.x_max, other.x_max)) / 2,
            (max(self.y_min, other.y_min) + min(self.y_max, other.y_max)) / 2,
            min(self.x_max, other.x_max) - max(self.x_min, other.x_min),
            min(self.y_max, other.y_max) - max(self.y_min, other.y_min),
            self.meta,
        )

    def iou(self, other):
        if not isinstance(other, BBox):
            raise TypeError(f"Expected BBox, got {type(other).__name__}")
        intersection = self.get_intersection(other)
        if intersection is None:
            return 0.0
        union = self.area + other.area - intersection.area
        if union <= 0:
            return 0.0
        return intersection.area / union

    def to_bbox(self):
        return BBox(self.x, self.y, self.w, self.h, self.meta)

    def has_label(self, label):
        labels = self.meta.get("labels", [])
        if isinstance(labels, str):
            return labels == label
        return label in labels

    def __repr__(self):
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h}, meta={self.meta})"
