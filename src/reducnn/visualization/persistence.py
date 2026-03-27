import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _sanitize(name: str, max_len: int = 80) -> str:
    txt = re.sub(r"\s+", "_", str(name or "").strip().lower())
    txt = re.sub(r"[^a-zA-Z0-9_.-]", "", txt)
    txt = txt.strip("._-")
    if not txt:
        txt = "artifact"
    return txt[:max_len]


def _artifact_dir() -> Optional[Path]:
    raw = os.environ.get("REDUCNN_ARTIFACT_DIR", "").strip()
    if not raw:
        return None
    d = Path(raw)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mirror_dir() -> Optional[Path]:
    raw = os.environ.get("REDUCNN_ARTIFACT_MIRROR_DIR", "").strip()
    if not raw:
        return None
    d = Path(raw)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_prefix() -> str:
    run_id = os.environ.get("REDUCNN_RUN_ID", "").strip()
    if run_id:
        return _sanitize(run_id, max_len=64)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def persist_matplotlib_figure(fig: Any, stem_hint: str, ext: str = "png") -> Optional[str]:
    """Persists a matplotlib figure when REDUCNN_ARTIFACT_DIR is set.

    Returns the absolute output path on success, else None.
    """
    out_dir = _artifact_dir()
    if out_dir is None or fig is None:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = _sanitize(stem_hint or "plot")
    filename = f"{_run_prefix()}_{stamp}_{stem}.{ext}"
    out_path = out_dir / filename
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")

    mirror = _mirror_dir()
    if mirror is not None:
        try:
            mirror_path = mirror / filename
            mirror_path.write_bytes(out_path.read_bytes())
        except Exception:
            pass

    return str(out_path.resolve())


def persist_plotly_figure(fig: Any, stem_hint: str, kind: str = "html") -> Optional[str]:
    """Persists a Plotly figure when REDUCNN_ARTIFACT_DIR is set.

    - `kind="html"` writes standalone HTML.
    """
    out_dir = _artifact_dir()
    if out_dir is None or fig is None:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = _sanitize(stem_hint or "plotly")
    kind = str(kind).lower().strip()
    if kind != "html":
        kind = "html"

    filename = f"{_run_prefix()}_{stamp}_{stem}.{kind}"
    out_path = out_dir / filename
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)

    mirror = _mirror_dir()
    if mirror is not None:
        try:
            mirror_path = mirror / filename
            mirror_path.write_bytes(out_path.read_bytes())
        except Exception:
            pass

    return str(out_path.resolve())


def persist_json(payload: Any, stem_hint: str) -> Optional[str]:
    """Persists JSON payload when REDUCNN_ARTIFACT_DIR is set."""
    out_dir = _artifact_dir()
    if out_dir is None:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = _sanitize(stem_hint or "payload")
    filename = f"{_run_prefix()}_{stamp}_{stem}.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    mirror = _mirror_dir()
    if mirror is not None:
        try:
            mirror_path = mirror / filename
            mirror_path.write_bytes(out_path.read_bytes())
        except Exception:
            pass

    return str(out_path.resolve())
