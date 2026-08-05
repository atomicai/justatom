import subprocess
import sys
from pathlib import Path


def test_running_mask_import_does_not_require_bertopic() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import builtins

original_import = builtins.__import__

def import_without_bertopic(name, *args, **kwargs):
    if name == "bertopic" or name.startswith("bertopic."):
        raise ModuleNotFoundError("bertopic intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_bertopic
from justatom.running.mask import ICLUSTERINGWrapperBackend

assert ICLUSTERINGWrapperBackend.__name__ == "ICLUSTERINGWrapperBackend"
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
