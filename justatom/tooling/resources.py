from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessRss:
    pid: int
    rss_kb: int
    command: str

    @property
    def rss_mb(self) -> float:
        return float(self.rss_kb) / 1024.0

    def compact(self) -> str:
        command = Path(str(self.command).strip()).name or str(self.command).strip() or "unknown"
        command = command.replace(" ", "_")
        return f"{int(self.pid)}:{command}:{self.rss_mb:.1f}MB"


def _run_ps(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["ps", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _parse_ps_line(line: str) -> ProcessRss | None:
    parts = line.split(None, 2)
    if len(parts) < 2:
        return None
    try:
        rss_kb = int(parts[0])
        pid = int(parts[1])
    except ValueError:
        return None
    command = parts[2] if len(parts) > 2 else "unknown"
    if rss_kb < 0 or pid <= 0:
        return None
    return ProcessRss(pid=pid, rss_kb=rss_kb, command=command)


def read_process_rss(pid: int) -> ProcessRss | None:
    if int(pid) <= 0:
        return None
    try:
        lines = _run_ps(["-o", "rss=", "-o", "pid=", "-o", "comm=", "-p", str(int(pid))])
    except Exception:
        return None
    for line in lines:
        parsed = _parse_ps_line(line)
        if parsed is not None:
            return parsed
    return None


def top_rss_processes(limit: int = 8) -> list[ProcessRss]:
    try:
        lines = _run_ps(["-axo", "rss=,pid=,comm="])
    except Exception:
        return []
    processes = [parsed for line in lines if (parsed := _parse_ps_line(line)) is not None]
    processes.sort(key=lambda proc: proc.rss_kb, reverse=True)
    return processes[: max(int(limit), 0)]


def format_resource_snapshot(
    label: str,
    *,
    self_process: ProcessRss | None,
    top_processes: list[ProcessRss],
) -> str:
    label = str(label).strip() or "snapshot"
    if self_process is None:
        self_part = "self_pid=NA self_rss_mb=NA"
    else:
        self_part = f"self_pid={int(self_process.pid)} self_rss_mb={self_process.rss_mb:.1f}"
    top_part = ", ".join(proc.compact() for proc in top_processes) or "none"
    return f"RSS {label}: {self_part} top=[{top_part}]"


def build_resource_snapshot(label: str, *, pid: int, top: int = 8) -> str:
    return format_resource_snapshot(
        label,
        self_process=read_process_rss(pid),
        top_processes=top_rss_processes(top),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a compact RSS memory snapshot.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args(argv)

    print(build_resource_snapshot(args.label, pid=args.pid, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
