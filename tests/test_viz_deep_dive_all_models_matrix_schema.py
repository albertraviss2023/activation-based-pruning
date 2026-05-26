from __future__ import annotations

import json
from pathlib import Path


def test_all_models_matrix_report_schema_if_present():
    report = Path("outputs/viz_deep_dive/agnostic_validation/all_models_matrix_latest.json")
    if not report.exists():
        return

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert "results" in payload
    assert isinstance(payload["results"], list)

    for backend_row in payload["results"]:
        assert "backend" in backend_row
        assert "adapter" in backend_row
        if backend_row.get("rows"):
            for row in backend_row["rows"]:
                assert "requested_model" in row
                assert "status" in row
