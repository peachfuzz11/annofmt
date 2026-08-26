import math
from annofmt.bbox import BBox

class RBBox(BBox):
    def __init__(self, x, y, w, h, a=0.0, meta=None):
        super().__init__(x, y, w, h, meta)
        self.a = a

    @property
    def x_min(self):
        return self.x - self._half_enclosed_w

    @property
    def y_min(self):
        return self.y - self._half_enclosed_h

    @property
    def x_max(self):
        return self.x + self._half_enclosed_w

    @property
    def y_max(self):
        return self.y + self._half_enclosed_h

    @property
    def _half_enclosed_w(self):
        return (abs(self.w * math.cos(self.a)) + abs(self.h * math.sin(self.a))) / 2

    @property
    def _half_enclosed_h(self):
        return (abs(self.w * math.sin(self.a)) + abs(self.h * math.cos(self.a))) / 2

    @property
    def area(self):
        return self.w * self.h

    @property
    def corners(self):
        cos_a = math.cos(self.a)
        sin_a = math.sin(self.a)
        half_w = self.w / 2
        half_h = self.h / 2
        x1 = self.x + half_w * cos_a - half_h * sin_a
        y1 = self.y + half_w * sin_a + half_h * cos_a
        x2 = self.x - half_w * cos_a - half_h * sin_a
        y2 = self.y - half_w * sin_a + half_h * cos_a
        x3 = self.x - half_w * cos_a + half_h * sin_a
        y3 = self.y - half_w * sin_a - half_h * cos_a
        x4 = self.x + half_w * cos_a + half_h * sin_a
        y4 = self.y + half_w * sin_a - half_h * cos_a
        return (x1, y1, x2, y2, x3, y3, x4, y4)

    def translate(self, dx, dy):
        return RBBox(self.x + dx, self.y + dy, self.w, self.h, self.a, self.meta)

    def scale(self, factor_w, factor_h):
        return RBBox(self.x, self.y, self.w * factor_w, self.h * factor_h, self.a, self.meta)

    def to_bbox(self):
        return BBox(self.x, self.y, 2 * self._half_enclosed_w, 2 * self._half_enclosed_h, self.meta)

    def __repr__(self):
        return f"RBBox(x={self.x}, y={self.y}, w={self.w}, h={self.h}, a={self.a}, meta={self.meta})"
