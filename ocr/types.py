from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from parser.models import OcrToken


@dataclass(slots=True)
class OcrResult:
    tokens: list[OcrToken]

    @property
    def texts(self) -> list[str]:
        return [t.text for t in self.tokens]

    @property
    def mean_confidence(self) -> float:
        if not self.tokens:
            return 0.0
        return sum(t.confidence for t in self.tokens) / len(self.tokens)


class OcrEngine(Protocol):
    def recognize(self, image) -> OcrResult: ...
