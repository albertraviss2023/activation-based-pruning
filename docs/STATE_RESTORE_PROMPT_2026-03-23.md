# State Restore Prompt (2026-03-23)

Use this in a fresh session to continue without re-analysis.

## Copy-Paste Prompt

```text
Continue work in:
C:/Users/alber/activation-based-pruning

Load context files first:
1) docs/VISUALIZATION_REVIEW_SUMMARY_2026-03-23.md
2) docs/SYSTEM_STATE_SNAPSHOT_2026-03-23.json

Context rules:
- Treat SYSTEM_STATE_SNAPSHOT_2026-03-23.json as source-of-truth.
- Keep visualization_deep_dive.ipynb backend-agnostic (PyTorch + Keras).
- Preserve leakage guard (VAL for pruning calibration, TEST for final evaluation only).
- Preserve deterministic checkpoint layout under saved_models/{baselines,pruned_raw,fine_tuned}.
- Preserve finetuned-load behavior into pruned architecture.
- Use phase manifests to verify artifact completeness.

First actions:
1) Summarize current state in 8-12 bullets.
2) List pending issues that block presentation quality.
3) Execute the top fix, then report changed files and validation run.
```

## Refresh Rule
After each major milestone, update all three files:
- `docs/VISUALIZATION_REVIEW_SUMMARY_<date>.md`
- `docs/SYSTEM_STATE_SNAPSHOT_<date>.json`
- `docs/STATE_RESTORE_PROMPT_<date>.md`
