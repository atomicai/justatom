import subprocess
import sys
from pathlib import Path


def test_api_import_does_not_require_llm_dependencies():
    script = """
import sys
sys.modules["openai"] = None
import justatom.api
assert "justatom.running.llm" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
