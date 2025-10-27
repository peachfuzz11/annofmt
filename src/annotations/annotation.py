import typing

from annotations.bbox import BBox
from annotations.tag import Tag


class Annotation:

    def __init__(self, bbox: BBox, tags: typing.List[Tag]):
        self.bbox = bbox
        self.tags = tags

    def __str__(self):
        return f"Annotation:{self.bbox}, {self.tags}"

    def __repr__(self):
        return f"Annotation:{self.bbox}, {self.tags}"

    def has_label(self, *labels: str):
        return any(t.label in labels for t in self.tags)

    def add_tag(self, tag: Tag):
        self.tags.append(tag)
        return self
