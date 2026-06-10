import csv
from datetime import datetime
from pathlib import Path


class CSVLogger:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = self.file_path.exists() and self.file_path.stat().st_size > 0
        self._fieldnames: list[str] | None = None
        if self._initialized:
            with self.file_path.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                self._fieldnames = next(reader, None)

    def log_metrics(self, metrics: dict):
        row = {"timestamp": datetime.utcnow().isoformat(), **metrics}
        fieldnames = self._resolve_fieldnames(row)
        mode = "a" if self._initialized else "w"
        with self.file_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)

    def close_log(self):
        return None

    def _resolve_fieldnames(self, row: dict) -> list[str]:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            return self._fieldnames

        missing = [key for key in row if key not in self._fieldnames]
        if not missing:
            return self._fieldnames

        self._fieldnames = [*self._fieldnames, *missing]
        if self._initialized:
            self._rewrite_existing_rows()
        return self._fieldnames

    def _rewrite_existing_rows(self):
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            return
        with self.file_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with self.file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            writer.writerows(rows)
