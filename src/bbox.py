import typing


class BBox:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        super().__init__()
        x_min, y_min, x_max, y_max = map(float, (x_min, y_min, x_max, y_max))
        self._x_min = min(x_min, x_max)
        self._x_max = max(x_min, x_max)
        self._y_min = min(y_min, y_max)
        self._y_max = max(y_min, y_max)

    @property
    def x_min(self) -> float:
        return self._x_min

    @property
    def x_max(self) -> float:
        return self._x_max

    @property
    def y_min(self) -> float:
        return self._y_min

    @property
    def y_max(self) -> float:
        return self._y_max

    @classmethod
    def from_x1y1wh(cls, x1, y1, w, h):
        return cls(x1, y1, x1 + w, y1 + h)

    @classmethod
    def from_x1y1x2y2(cls, x1, y1, x2, y2):
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        return cls(x_min, y_min, x_max, y_max)

    @classmethod
    def from_xywh(cls, x, y, w, h):
        return cls(x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def to_xywh(self) -> dict:
        w, h = (self.x_max - self.x_min), (self.y_max - self.y_min)
        x, y = self.x_min + w / 2, self.y_min + h / 2
        return {"x": x, "y": y, "w": w, "h": h}

    def to_x1x2y1y2(self) -> dict:
        return {"x1": self.x_min, "x2": self.x_max, "y1": self.y_min, "y2": self.y_max}

    def to_x1y1x2y2(self) -> dict:
        return {"x1": self.x_min, "y1": self.y_min, "x2": self.x_max, "y2": self.y_max}

    def to_x_min_y_min_x_max_y_max(self) -> dict:
        return {"x_min": self.x_min, "y_min": self.y_min, "x_max": self.x_max, "y_max": self.y_max}

    def from_yolo(self, w, h):
        return BBox(self.x_min * w, self.y_min * h, self.x_max * w, self.y_max * h)

    @classmethod
    def resolve(cls, **kwargs) -> "BBox":
        if "x_min" in kwargs and "x_max" in kwargs and "y_min" in kwargs and "y_max" in kwargs:
            return cls(**kwargs)
        elif "x1" in kwargs and "x2" in kwargs and "y1" in kwargs and "y2" in kwargs:
            return cls.from_x1y1x2y2(**kwargs)
        elif "x" in kwargs and "y" in kwargs and "w" in kwargs and "h" in kwargs:
            return cls.from_xywh(**kwargs)
        elif "x1" in kwargs and "y1" in kwargs and "w" in kwargs and "h" in kwargs:
            return cls.from_x1y1wh(**kwargs)
        else:
            raise ValueError(f"Could not format input {kwargs}")

    def scale(self, factor_h: float, factor_w: float) -> "BBox":
        selector = self.to_xywh()
        if factor_h:
            selector["h"] *= factor_h
        if factor_w:
            selector["w"] *= factor_w
        resolved = BBox.resolve(**selector)
        return resolved

    def get_area(self):
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def get_overlap(self, other: "BBox") -> typing.Optional["BBox"]:
        if self.overlaps(other):
            x1 = max(self.x_min, other.x_min)
            y1 = max(self.y_min, other.y_min)
            x2 = min(self.x_max, other.x_max)
            y2 = min(self.y_max, other.y_max)
            return BBox.from_x1y1x2y2(x1=x1, y1=y1, x2=x2, y2=y2)
        return None

    def overlaps(self, other: "BBox") -> bool:
        return not (
                self.x_max <= other.x_min
                or self.x_min >= other.x_max
                or self.y_max <= other.y_min
                or self.y_min >= other.y_max
        )

    def iou(self, other: "BBox") -> float:
        intersection = self.get_intersection(other)
        if intersection:
            intersection_area = intersection.get_area()
            union_area = self.get_area() + other.get_area() - intersection_area
            return intersection_area / union_area
        return 0

    def get_intersection(self, other: "BBox") -> "BBox":
        return self.get_overlap(other)

    def intersects(self, other: "BBox") -> bool:
        return self.overlaps(other)

    def reference(self, other: "BBox"):
        x_min = self.x_min - other.x_min
        x_max = self.x_max - other.x_min
        y_min = self.y_min - other.y_min
        y_max = self.y_max - other.y_min
        return BBox(x_min, y_min, x_max, y_max)

    def __str__(self):
        return "BBox:" + str(self.to_x1x2y1y2())

    def __repr__(self):
        return "BBox:" + str(self.to_x1x2y1y2())
