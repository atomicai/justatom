import os
from pathlib import Path


def bash_executable() -> str:
    configured = os.environ.get("JUSTATOM_TEST_BASH")
    if configured:
        return configured
    if os.name != "nt":
        return "bash"

    roots = {
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramW6432"),
        os.environ.get("ProgramFiles(x86)"),
        "C:/Program Files",
    }
    for root in filter(None, roots):
        candidate = Path(root) / "Git" / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)

    raise RuntimeError("Git Bash is required to run the shell contract tests on Windows")
