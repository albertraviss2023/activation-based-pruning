from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nbformat


def load_notebook_namespace(notebook_path: str | Path, stop_marker: str) -> SimpleNamespace:
    """Load code definitions from a notebook without executing its run section."""
    path = Path(notebook_path)
    nb = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)

    code = "\n\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    code = "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith("!")
    )

    if stop_marker in code:
        code = code.split(stop_marker)[0]

    ns: dict[str, object] = {}
    exec(compile(code, str(path), "exec"), ns)
    return SimpleNamespace(**ns)

