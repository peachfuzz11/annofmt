from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tag:
    """A class label with an optional confidence score in ``[0, 1]``."""

    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        label = str(self.label)
        score = float(self.score)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be within [0, 1], got {score}")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "score", score)

    def __str__(self) -> str:
        return f"Tag:{self.label}"
