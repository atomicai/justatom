from datetime import datetime

from pathlib import Path
import csv

class CSVLogger:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = self.file_path.exists() and self.file_path.stat().st_size > 0

    def log_metrics(self, metrics: dict):
        row = {"timestamp": datetime.utcnow().isoformat(), **metrics}
        fieldnames = list(row.keys())
        mode = "a" if self._initialized else "w"
        with self.file_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)

    def close_log(self):
        return None