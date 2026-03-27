# State Restore Prompt (Reusable)

Use this prompt in a fresh session to restore working context quickly.

## Copy-Paste Prompt

```text
You are continuing work on the repository:
C:/Users/alber/activation-based-pruning

Load and use these documents as the current system state:
1) docs/VISUALIZATION_REVIEW_SUMMARY_2026-03-22.md
2) docs/SYSTEM_STATE_SNAPSHOT_2026-03-22.json

Task continuity requirements:
- Treat the snapshot as authoritative context from the previous session.
- Continue from the "first implementation targets" and "recommended_overhaul" phases.
- Prioritize visualization improvements for:
  1) candidate discovery (APoZ/L1/Taylor),
  2) animated pruning process (candidate vs pruned vs kept),
  3) final pruned architecture graph by model type (VGG/ResNet/DenseNet).
- Keep changes compatible with existing Presentation Mode notebook flow and exports.

Before coding:
- Summarize snapshot understanding in 8-12 bullets.
- Propose the next 3 concrete code edits in order.
Then implement immediately.
```

## Recommended Snapshot Refresh Rule
- After each major milestone, create:
  - `docs/SYSTEM_STATE_SNAPSHOT_<date>.json`
  - `docs/STATE_RESTORE_PROMPT_<date>.md`
- Keep the markdown review summary updated for humans, and the JSON snapshot updated for machine handoff.
