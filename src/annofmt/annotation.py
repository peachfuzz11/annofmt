from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from annofmt.tag import Tag

if TYPE_CHECKING:
    from annofmt.geometry._base import Geometry
    from annofmt.geometry.bbox import BBox


@dataclass(frozen=True, slots=True)
class Annotation:
    """A single annotated object: one geometry plus any number of tags.

    Immutable; :meth:`add_tag` returns a new instance.
    """

    geometry: Geometry
    tags: tuple[Tag, ...] = ()

    def __post_init__(self) -> None:
        tags = tuple(self.tags)
        for tag in tags:
            if not isinstance(tag, Tag):
                raise TypeError(f"tags must be Tag instances, got {type(tag).__name__}")
        object.__setattr__(self, "tags", tags)

    @property
    def bbox(self) -> BBox:
        """Axis-aligned bounding box of the annotation's geometry."""
        return self.geometry.to_bbox()

    def has_label(self, *labels: str) -> bool:
        return any(tag.label in labels for tag in self.tags)

    def add_tag(self, tag: Tag) -> Annotation:
        return Annotation(self.geometry, self.tags + (tag,))
