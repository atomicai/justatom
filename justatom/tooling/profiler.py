import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class MemoryReport:
    name: str
    duration_ms: float
    items_processed: int
    rss_start_mb: float | None = None
    rss_end_mb: float | None = None
    rss_delta_mb: float | None = None
    rss_peak_mb: float | None = None
    heap_peak_mb: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class MemoryProfilerSpan:
    def __init__(self, profiler: "MemoryProfiler", name: str, meta: dict[str, Any] | None = None):
        self._profiler = profiler
        self._name = name
        self._meta = meta or {}
        self._items_processed = 0
        self._start_ts: float | None = None
        self._rss_start_mb: float | None = None
        self._started_tracemalloc = False

    def __enter__(self) -> "MemoryProfilerSpan":
        if not self._profiler.enabled:
            return self

        self._start_ts = time.perf_counter()
        self._rss_start_mb = self._profiler._rss_mb() if self._profiler.track_rss else None

        if self._profiler.track_heap:
            if tracemalloc.is_tracing():
                tracemalloc.reset_peak()
            else:
                tracemalloc.start()
                self._started_tracemalloc = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._profiler.enabled:
            return

        duration_ms = 0.0
        if self._start_ts is not None:
            duration_ms = (time.perf_counter() - self._start_ts) * 1000.0

        rss_end_mb = self._profiler._rss_mb() if self._profiler.track_rss else None
        rss_peak_mb = rss_end_mb
        rss_delta_mb = None
        if self._rss_start_mb is not None and rss_end_mb is not None:
            rss_delta_mb = rss_end_mb - self._rss_start_mb

        heap_peak_mb = None
        if self._profiler.track_heap and tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            heap_peak_mb = peak / (1024.0 * 1024.0)

        if self._started_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()

        report = MemoryReport(
            name=self._name,
            duration_ms=duration_ms,
            items_processed=self._items_processed,
            rss_start_mb=self._rss_start_mb,
            rss_end_mb=rss_end_mb,
            rss_delta_mb=rss_delta_mb,
            rss_peak_mb=rss_peak_mb,
            heap_peak_mb=heap_peak_mb,
            meta=self._meta,
        )
        self._profiler._push_report(report)

    def tick(self, n: int = 1) -> None:
        if not self._profiler.enabled:
            return
        self._items_processed += n


class MemoryProfiler:
    def __init__(
        self,
        enabled: bool = False,
        *,
        track_rss: bool = True,
        track_heap: bool = True,
    ):
        self.enabled = enabled
        self.track_rss = track_rss
        self.track_heap = track_heap
        self._reports: list[MemoryReport] = []

    def span(self, name: str, **meta) -> MemoryProfilerSpan:
        return MemoryProfilerSpan(self, name=name, meta=meta)

    def report(self) -> MemoryReport | None:
        if len(self._reports) == 0:
            return None
        return self._reports[-1]

    def reports(self) -> list[MemoryReport]:
        return list(self._reports)

    def reset(self) -> None:
        self._reports.clear()

    def _push_report(self, report: MemoryReport) -> None:
        self._reports.append(report)
        logger.info(
            "MEMORY | {name} | items={items} | duration_ms={duration:.2f} | "
            "rss_delta_mb={rss_delta} | heap_peak_mb={heap_peak}",
            name=report.name,
            items=report.items_processed,
            duration=report.duration_ms,
            rss_delta=(None if report.rss_delta_mb is None else round(report.rss_delta_mb, 4)),
            heap_peak=(None if report.heap_peak_mb is None else round(report.heap_peak_mb, 4)),
        )

    @staticmethod
    def _rss_mb() -> float | None:
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return rss / (1024.0 * 1024.0)
            return rss / 1024.0
        except Exception:
            return None


__all__ = ["MemoryProfiler", "MemoryReport", "MemoryProfilerSpan"]