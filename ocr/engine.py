from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ocr.types import OcrResult
from parser.models import OcrToken

logger = logging.getLogger("electoral.ocr")


class PaddleOcrEngine:
    """PaddleOCR English recognizer with 2.x and 3.x result adapters."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._ocr: Any = None

    def _ensure(self) -> None:
        if self._ocr is not None:
            return
        import os

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x prefers keyword flags; 2.x used use_angle_cls.
        try:
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except TypeError:
            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=True)

    def recognize(self, image: np.ndarray) -> OcrResult:
        self._ensure()
        if image.size == 0:
            return OcrResult(tokens=[])
        raw = None
        if hasattr(self._ocr, "ocr"):
            try:
                raw = self._ocr.ocr(image)
            except Exception:
                logger.exception("PaddleOCR.ocr failed")
        if raw is None and hasattr(self._ocr, "predict"):
            try:
                raw = self._ocr.predict(image)
            except Exception:
                logger.exception("PaddleOCR.predict failed")
                raw = []
        return OcrResult(tokens=_parse_paddle_result(raw))


def _nonempty_seq(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return None if value.size == 0 else value
    try:
        if len(value) == 0:
            return None
    except TypeError:
        return None
    return value


def _parse_paddle_result(raw: Any) -> list[OcrToken]:
    tokens: list[OcrToken] = []
    if raw is None:
        return tokens

    # 3.x predict → list of dict-like result objects
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        item = raw[0]
        texts = _nonempty_seq(item.get("rec_texts")) or _nonempty_seq(item.get("rec_text")) or []
        scores = (
            _nonempty_seq(item.get("rec_scores"))
            or _nonempty_seq(item.get("rec_score"))
            or [1.0] * len(list(texts))
        )
        boxes = (
            _nonempty_seq(item.get("rec_polys"))
            or _nonempty_seq(item.get("dt_polys"))
            or _nonempty_seq(item.get("rec_boxes"))
            or []
        )
        n_boxes = len(boxes)
        for i, text in enumerate(texts):
            conf = float(scores[i]) if i < len(scores) else 0.0
            box = _as_quad(boxes[i] if i < n_boxes else [[0, 0], [0, 0], [0, 0], [0, 0]])
            if str(text).strip():
                tokens.append(OcrToken(text=str(text).strip(), confidence=conf, bbox=box))
        return tokens

    # Result objects with attributes (PaddleX)
    if isinstance(raw, list) and raw and hasattr(raw[0], "get") is False:
        first = raw[0]
        texts = getattr(first, "rec_texts", None) or getattr(first, "str", None)
        if texts:
            scores = getattr(first, "rec_scores", [1.0] * len(texts))
            boxes = getattr(first, "rec_polys", None) or getattr(first, "dt_polys", [])
            for i, text in enumerate(texts):
                conf = float(scores[i]) if i < len(scores) else 0.0
                box = _as_quad(boxes[i] if boxes is not None and i < len(boxes) else [[0, 0]] * 4)
                if str(text).strip():
                    tokens.append(OcrToken(text=str(text).strip(), confidence=conf, bbox=box))
            if tokens:
                return tokens

    # 2.x: [ [ [box, (text, conf)], ... ] ]
    lines = raw[0] if raw and isinstance(raw, list) and raw and raw[0] is not None else raw
    if not lines:
        return tokens
    for item in lines:
        if not item or len(item) < 2:
            continue
        box, payload = item[0], item[1]
        if isinstance(payload, (list, tuple)):
            text, conf = payload[0], float(payload[1])
        else:
            text, conf = str(payload), 1.0
        if str(text).strip():
            tokens.append(OcrToken(text=str(text).strip(), confidence=conf, bbox=_as_quad(box)))
    return tokens


def _as_quad(box: Any) -> list[list[float]]:
    arr = np.asarray(box, dtype=float).reshape(-1, 2)
    if arr.shape[0] == 2:
        x0, y0 = arr[0]
        x1, y1 = arr[1]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if arr.shape[0] >= 4:
        return [[float(x), float(y)] for x, y in arr[:4]]
    return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
