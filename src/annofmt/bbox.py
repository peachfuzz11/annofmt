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
        """Center x, center y, width, height (the native storage format)."""
        return cls(x, y, w, h, meta=meta)

    # Explicit-center alias for callers that spell it out.
    from_cxcywh = from_xywh

    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2, meta=None):
        """Two opposite corners: (x1, y1) and (x2, y2). Order-independent (Pascal VOC)."""
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        return cls((x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min, meta=meta)

    @classmethod
    def from_xxyy(cls, x1, x2, y1, y2, meta=None):
        """x range then y range: x1, x2, y1, y2. Order-independent."""
        return cls.from_xyxy(x1, y1, x2, y2, meta=meta)

    @classmethod
    def from_ltwh(cls, left, top, w, h, meta=None):
        """Top-left corner plus width, height (COCO)."""
        return cls(left + w / 2, top + h / 2, w, h, meta=meta)

    # Common spellings of the same layouts.
    from_x1y1x2y2 = from_xyxy
    from_x1x2y1y2 = from_xxyy
    from_x1y1wh = from_ltwh

    def to_xywh(self):
        """Center x, center y, width, height."""
        return (self.x, self.y, self.w, self.h)

    def to_xyxy(self):
        """Top-left and bottom-right corners: x_min, y_min, x_max, y_max."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_xxyy(self):
        """x range then y range: x_min, x_max, y_min, y_max."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def to_ltwh(self):
        """Top-left corner plus width, height: x_min, y_min, w, h."""
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
