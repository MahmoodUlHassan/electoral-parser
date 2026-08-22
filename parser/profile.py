from __future__ import annotations

import json
import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.table import Table

logger = logging.getLogger("electoral.profile")


@dataclass
class PhaseProfiler:
    """Accumulates wall-clock seconds per named pipeline phase.

    Detail phases (OCR detect/rec) are CPU-time sums and may exceed wall time
    when card OCR runs on multiple threads; they are excluded from TOTAL.
    """

    phases: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    details: dict[str, float] = field(default_factory=dict)
    detail_counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, name: str, seconds: float, *, count: int = 1) -> None:
        with self._lock:
            self.phases[name] = self.phases.get(name, 0.0) + seconds
            self.counts[name] = self.counts.get(name, 0) + count

    def add_detail(self, name: str, seconds: float, *, count: int = 1) -> None:
        with self._lock:
            self.details[name] = self.details.get(name, 0.0) + seconds
            self.detail_counts[name] = self.detail_counts.get(name, 0) + count

    @contextmanager
    def track(self, name: str, *, count: int = 1) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0, count=count)

    def total(self) -> float:
        with self._lock:
            return sum(self.phases.values())

    def rows(self) -> list[tuple[str, float]]:
        with self._lock:
            return sorted(self.phases.items(), key=lambda kv: (-kv[1], kv[0]))

    def detail_rows(self) -> list[tuple[str, float]]:
        with self._lock:
            return sorted(self.details.items(), key=lambda kv: (-kv[1], kv[0]))

    def to_dict(self) -> dict:
        total = self.total()
        with self._lock:
            phases = list(self.phases.items())
            counts = dict(self.counts)
            details = list(self.details.items())
            detail_counts = dict(self.detail_counts)
        phases.sort(key=lambda kv: (-kv[1], kv[0]))
        details.sort(key=lambda kv: (-kv[1], kv[0]))
        detail_total = sum(s for _, s in details)
        return {
            "totalSeconds": round(total, 4),
            "phases": [
                {
                    "name": name,
                    "seconds": round(sec, 4),
                    "count": counts.get(name, 0),
                    "pct": round(100.0 * sec / total, 1) if total > 0 else 0.0,
                }
                for name, sec in phases
            ],
            "ocrInternals": {
                "note": "CPU-time sums across calls/threads; can exceed wall OCR when parallel",
                "totalSeconds": round(detail_total, 4),
                "phases": [
                    {
                        "name": name,
                        "seconds": round(sec, 4),
                        "count": detail_counts.get(name, 0),
                        "pct": round(100.0 * sec / detail_total, 1) if detail_total > 0 else 0.0,
                    }
                    for name, sec in details
                ],
            },
        }

    def print_table(self, *, title: str = "Parse timing", console: Console | None = None) -> None:
        console = console or Console()
        total = self.total()
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("Phase")
        table.add_column("Seconds", justify="right")
        table.add_column("Count", justify="right")
        table.add_column("%", justify="right")
        with self._lock:
            phase_items = list(self.phases.items())
            counts = dict(self.counts)
            detail_items = list(self.details.items())
            detail_counts = dict(self.detail_counts)
        for name, sec in sorted(phase_items, key=lambda kv: (-kv[1], kv[0])):
            pct = (100.0 * sec / total) if total > 0 else 0.0
            table.add_row(name, f"{sec:.2f}", str(counts.get(name, 0)), f"{pct:.1f}")
        table.add_row("TOTAL", f"{total:.2f}", "", "100.0", style="bold")
        console.print(table)

        if not detail_items:
            return
        detail_total = sum(s for _, s in detail_items)
        detail = Table(
            title="OCR internals (CPU-sum; may exceed wall when parallel)",
            show_header=True,
            header_style="bold",
        )
        detail.add_column("Phase")
        detail.add_column("Seconds", justify="right")
        detail.add_column("Count", justify="right")
        detail.add_column("%", justify="right")
        for name, sec in sorted(detail_items, key=lambda kv: (-kv[1], kv[0])):
            pct = (100.0 * sec / detail_total) if detail_total > 0 else 0.0
            detail.add_row(name, f"{sec:.2f}", str(detail_counts.get(name, 0)), f"{pct:.1f}")
        detail.add_row("SUM", f"{detail_total:.2f}", "", "100.0", style="bold")
        console.print(detail)

    def write_json(self, path: Path, *, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if extra:
            payload = {**extra, **payload}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote profile %s", path)
