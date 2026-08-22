from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import numpy as np

from ocr.types import OcrResult
from parser.models import OcrToken
from parser.profile import PhaseProfiler

logger = logging.getLogger("electoral.ocr")

# Detail phases: CPU-time sum across calls/threads (can exceed wall OCR when parallel).
PHASE_DETECT = "OCR detect (sum)"
PHASE_REC = "OCR recognize (sum)"
PHASE_DOC = "OCR doc preprocess (sum)"
PHASE_ORIENT = "OCR textline orient (sum)"
PHASE_OTHER = "OCR other (sum)"

# Flat upright English rolls — tiny is default; small used for invalid-card retry.
OCR_MODEL_NAMES: dict[str, tuple[str, str]] = {
    "tiny": ("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
    "small": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    "medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}
DEFAULT_OCR_MODEL = "tiny"


class _TimedCall:
    """Wraps a callable and accumulates wall time into a profiler detail phase.

    PaddleX predictors often return generators; we materialize with list() so
    timing covers the actual inference, matching how OCRPipeline consumes them.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        phase: str,
        profiler: PhaseProfiler | None,
        lock: threading.Lock,
    ) -> None:
        self._fn = fn
        self._phase = phase
        self._profiler = profiler
        self._lock = lock

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            result = self._fn(*args, **kwargs)
            # Eagerly run generator predictors so timing includes inference.
            if result is not None and type(result).__name__ in {
                "generator",
                "list_iterator",
                "map",
            }:
                return list(result)
            if hasattr(result, "__iter__") and not isinstance(
                result, (list, tuple, dict, str, bytes, np.ndarray)
            ):
                # Avoid treating numpy arrays / already-materialized results as gens.
                try:
                    return list(result)
                except TypeError:
                    return result
            return result
        finally:
            elapsed = time.perf_counter() - t0
            if self._profiler is not None:
                with self._lock:
                    self._profiler.add_detail(self._phase, elapsed, count=1)


class PaddleOcrEngine:
    """PaddleOCR English recognizer with 2.x and 3.x result adapters."""

    def __init__(
        self,
        lang: str = "en",
        profiler: PhaseProfiler | None = None,
        *,
        model: str = DEFAULT_OCR_MODEL,
    ) -> None:
        self.lang = lang
        self.model = model if model in OCR_MODEL_NAMES else DEFAULT_OCR_MODEL
        self._ocr: Any = None
        self._profiler = profiler
        self._profile_lock = threading.Lock()
        self._wrapped = False

    def set_profiler(self, profiler: PhaseProfiler | None) -> None:
        self._profiler = profiler
        # Re-wrap if already initialized so new profiler receives events.
        if self._ocr is not None and profiler is not None:
            self._wrap_submodels()

    def _ensure(self) -> None:
        if self._ocr is not None:
            return
        import os

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        det_name, rec_name = OCR_MODEL_NAMES[self.model]
        # Skip all angle/doc classifiers — rolls are upright scans.
        try:
            self._ocr = PaddleOCR(
                text_detection_model_name=det_name,
                text_recognition_model_name=rec_name,
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except TypeError:
            # PaddleOCR 2.x API
            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=False)
        logger.info("OCR model=%s (%s / %s)", self.model, det_name, rec_name)
        if self._profiler is not None:
            self._wrap_submodels()

    def _inner_pipeline(self) -> Any | None:
        pipe = getattr(self._ocr, "paddlex_pipeline", None)
        if pipe is None:
            return None
        return getattr(pipe, "_pipeline", pipe)

    def _wrap_submodels(self) -> None:
        """Instrument det / rec / optional preprocess on the PaddleX OCR pipeline."""
        inner = self._inner_pipeline()
        if inner is None:
            logger.warning("no paddlex OCR pipeline; cannot split detect/recognize timing")
            return
        lock = self._profile_lock

        def wrap_attr(obj: Any, attr: str, phase: str) -> None:
            if not hasattr(obj, attr):
                return
            current = getattr(obj, attr)
            if isinstance(current, _TimedCall):
                current._profiler = self._profiler
                return
            setattr(obj, attr, _TimedCall(current, phase, self._profiler, lock))

        wrap_attr(inner, "text_det_model", PHASE_DETECT)
        wrap_attr(inner, "text_rec_model", PHASE_REC)

        # Optional stages (usually off with our flags).
        if getattr(inner, "use_doc_preprocessor", False) and hasattr(inner, "doc_preprocessor_pipeline"):
            wrap_attr(inner, "doc_preprocessor_pipeline", PHASE_DOC)
        if getattr(inner, "use_textline_orientation", False) and hasattr(inner, "textline_orientation_model"):
            wrap_attr(inner, "textline_orientation_model", PHASE_ORIENT)

        self._wrapped = True
        logger.debug("wrapped PaddleOCR submodels for timing")

    def recognize(self, image: np.ndarray) -> OcrResult:
        self._ensure()
        if image.size == 0:
            return OcrResult(tokens=[])
        t0 = time.perf_counter()
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
        # Attribute leftover wall time not captured by det/rec wrappers.
        if self._profiler is not None and self._wrapped:
            # Only meaningful when wrappers didn't fire (legacy path); skip otherwise.
            pass
        _ = t0  # kept for future leftover accounting if needed
        return OcrResult(tokens=_parse_paddle_result(raw))


class OcrEnginePool:
    """N Paddle engines for concurrent card OCR (one instance is not thread-safe)."""

    def __init__(
        self,
        size: int,
        *,
        lang: str = "en",
        model: str = DEFAULT_OCR_MODEL,
        profiler: PhaseProfiler | None = None,
        factory: Callable[[], PaddleOcrEngine] | None = None,
    ) -> None:
        from queue import Queue

        size = max(1, size)
        self._q: Queue[PaddleOcrEngine] = Queue()
        self._engines: list[PaddleOcrEngine] = []
        for _ in range(size):
            if factory is not None:
                eng = factory()
            else:
                eng = PaddleOcrEngine(lang=lang, profiler=profiler, model=model)
            eng._ensure()
            self._engines.append(eng)
            self._q.put(eng)

    def __len__(self) -> int:
        return len(self._engines)

    def recognize(self, image: np.ndarray) -> OcrResult:
        eng = self._q.get()
        try:
            return eng.recognize(image)
        finally:
            self._q.put(eng)

    def set_profiler(self, profiler: PhaseProfiler | None) -> None:
        for eng in self._engines:
            eng.set_profiler(profiler)


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
