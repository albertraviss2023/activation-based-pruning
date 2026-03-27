# State Restore Prompt (Methods) (2026-03-23)

Use this in a fresh session to continue without re-analysis.

## Copy-Paste Prompt

```text
Continue work in:
C:/Users/alber/activation-based-pruning

Load context files first:
1) docs/METHODS_FIDELITY_SUMMARY_2026-03-23.md
2) docs/SYSTEM_STATE_SNAPSHOT_METHODS_2026-03-23.json
3) docs/LITERATURE_FIDELITY_REPORT_v2.md

Context rules:
- Bundle only: apoz, mean_abs_act, l1_norm.
- All other methods (chip, l2, global methods) are custom-registration notebook methods.
- Do not rename a method to a literature name unless algorithm is faithful.
- Keep PyTorch/Keras scoring semantics aligned.

First actions:
1) Confirm bundled method-surface changes.
2) Implement faithful custom methods (NISP, SeNPIS, TIS, REPrune, ThiNet).
3) Run tests and update fidelity report to v3 with evidence.
```
