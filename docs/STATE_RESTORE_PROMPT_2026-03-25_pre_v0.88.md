# State Restore Prompt (Pre v0.88, 2026-03-25)

1. Load snapshot:
- `docs/SYSTEM_STATE_SNAPSHOT_2026-03-25_pre_v0.88.json`

2. Primary goals:
- Enforce artifact persistence for production experiment notebooks (`cifar*`, `cat_dog*`).
- Keep user workflow minimal with mandatory load-or-train baseline behavior.
- Keep output structure under `outputs/` and mirror artifact indexing under `saved_models/artifacts/...`.
- Validate production notebooks against current backend behavior.
- Bump package version to `0.88` and publish changelog/release notes.

3. If context resets:
- Reopen the snapshot JSON and this prompt first, then continue notebook + release tasks.
