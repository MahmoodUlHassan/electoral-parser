from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PageType(str, Enum):
    HEADER = "header"
    MAP = "map"
    VOTER = "voter"
    SUPPLEMENT = "supplement"
    BLANK = "blank"
    UNKNOWN = "unknown"


class DetectionMethod(str, Enum):
    GRID = "grid"
    CONTOUR = "contour"


@dataclass(frozen=True, slots=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def clip(self, width: int, height: int) -> Box:
        x = max(0, self.x)
        y = max(0, self.y)
        return Box(x, y, max(0, min(width, self.x2) - x), max(0, min(height, self.y2) - y))


@dataclass(slots=True)
class OcrToken:
    text: str
    confidence: float
    bbox: list[list[float]]

    @property
    def cx(self) -> float:
        return sum(p[0] for p in self.bbox) / 4

    @property
    def cy(self) -> float:
        return sum(p[1] for p in self.bbox) / 4

    @property
    def x0(self) -> float:
        return min(p[0] for p in self.bbox)

    @property
    def y0(self) -> float:
        return min(p[1] for p in self.bbox)


@dataclass(slots=True)
class CardDetection:
    index: int
    box: Box
    occupied: bool
    method: DetectionMethod


@dataclass(slots=True)
class PageMeta:
    constituency_no: int | None = None
    constituency: str | None = None
    part_no: int | None = None
    section_no: int | None = None
    section: str | None = None
    page: int | None = None
    total_pages: int | None = None
    qualifying_date: str | None = None
    publication_date: str | None = None
    expected_total: int | None = None
    expected_male: int | None = None
    expected_female: int | None = None
    expected_others: int | None = None


@dataclass(slots=True)
class VoterRecord:
    serial_no: int | None
    epic: str | None
    name: str | None
    relation_type: str | None
    relative_name: str | None
    house_no: str | None
    age: int | None
    gender: str | None
    page: int
    part_no: int | None
    constituency: str | None
    section: str | None
    raw_ocr: list[str] = field(default_factory=list)
    tokens: list[OcrToken] = field(default_factory=list)
    confidence: float = 0.0
    crop_path: str | None = None
    valid: bool = True
    errors: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "serialNo": self.serial_no,
            "epic": self.epic,
            "name": self.name,
            "relationType": self.relation_type,
            "relativeName": self.relative_name,
            "houseNo": self.house_no,
            "age": self.age,
            "gender": self.gender,
            "page": self.page,
            "partNo": self.part_no,
            "constituency": self.constituency,
            "section": self.section,
        }

    def to_full_dict(self) -> dict[str, Any]:
        data = self.to_public_dict()
        data.update(
            {
                "rawOcr": self.raw_ocr,
                "confidence": round(self.confidence, 4),
                "boundingBoxes": [
                    {"text": t.text, "confidence": round(t.confidence, 4), "bbox": t.bbox}
                    for t in self.tokens
                ],
                "cropPath": self.crop_path,
                "valid": self.valid,
                "errors": self.errors,
            }
        )
        return data
