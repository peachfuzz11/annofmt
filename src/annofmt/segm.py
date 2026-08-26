from annofmt.bbox import BBox

class Segm(BBox):
    def __init__(self, indices, meta=None):
        # Normalize indices to tuple of tuples
        first = indices[0] if indices else ()
        if isinstance(first, (list, tuple)) and len(first) == 2 and not isinstance(first[0], (list, tuple)):
            # List of points, wrap in single ring
            self.indices = (tuple(indices),)
        else:
            # List of rings
            self.indices = tuple(tuple(p) for p in indices)
        
        all_points = [p for ring in self.indices for p in ring]
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        # Call BBox __init__ with computed bounding box
        super().__init__(
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            max(xs) - min(xs),
            max(ys) - min(ys),
            meta
        )
        
    @property
    def all_points(self):
        return [p for ring in self.indices for p in ring]

    def translate(self, dx, dy):
        shifted = tuple(
            tuple((p[0] + dx, p[1] + dy) for p in ring)
            for ring in self.indices
        )
        return Segm(shifted, self.meta)

    def scale(self, factor_w, factor_h):
        cx, cy = self.x, self.y
        scaled = tuple(
            tuple((cx + (p[0] - cx) * factor_w, cy + (p[1] - cy) * factor_h) for p in ring)
            for ring in self.indices
        )
        return Segm(scaled, self.meta)

    def to_bbox(self):
        return BBox(self.x, self.y, self.w, self.h, self.meta)

    def iou(self, other):
        if not isinstance(other, Segm):
            raise TypeError(f"Expected Segm, got {type(other).__name__}")
        my_points = set(self.all_points)
        other_points = set(other.all_points)
        intersection = my_points & other_points
        union = my_points | other_points
        if len(union) == 0:
            return 0.0
        return len(intersection) / len(union)

    def __repr__(self):
        return f"Segm(indices={self.indices}, meta={self.meta})"
