import os
import subprocess
import sys
import unittest
from pathlib import Path

from justatom.tooling.resources import ProcessRss, format_resource_snapshot


class ResourceSnapshotTest(unittest.TestCase):
    def test_formats_self_and_top_processes(self):
        line = format_resource_snapshot(
            "justatom tune",
            self_process=ProcessRss(pid=123, rss_kb=1536, command="bash"),
            top_processes=[
                ProcessRss(pid=456, rss_kb=2048, command="python"),
                ProcessRss(pid=789, rss_kb=1024, command="weaviate"),
            ],
        )

        self.assertEqual(
            line,
            "RSS justatom tune: self_pid=123 self_rss_mb=1.5 " "top=[456:python:2.0MB, 789:weaviate:1.0MB]",
        )

    def test_cli_emits_snapshot_for_requested_pid(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "justatom.tooling.resources",
                "--label",
                "smoke",
                "--pid",
                str(os.getpid()),
                "--top",
                "1",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RSS smoke:", result.stdout)

    def test_pipeline_scripts_call_resource_snapshot_helper(self):
        pipeline = Path("scripts/run_pipeline.sh").read_text()
        benchmark = Path("scripts/run_benchmark.sh").read_text()

        self.assertIn("justatom.tooling.resources", pipeline)
        self.assertIn('log_rss "before $label"', pipeline)
        self.assertIn('log_rss "after $label"', pipeline)
        self.assertIn("justatom.tooling.resources", benchmark)


if __name__ == "__main__":
    unittest.main()
