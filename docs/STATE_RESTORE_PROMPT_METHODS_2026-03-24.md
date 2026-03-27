# State Restore Prompt (Methods, 2026-03-24)

Load this state before continuing work on custom registration experiments:

1. Read:
- `docs/METHODS_FIDELITY_SUMMARY_2026-03-24.md`
- `docs/SYSTEM_STATE_SNAPSHOT_METHODS_2026-03-24.json`

2. Primary context:
- Keras surgery parity fixes are in place in `src/reducnn/backends/keras_backend.py`.
- Custom registration notebook is user-friendly and outputs matrix CSV reports.
- Remaining productization gap: automatic Keras pretrained baseline bootstrap/save if missing.

3. Next expected task (if requested):
- add baseline bootstrap cell/utility for Keras that trains (or imports) and saves to
  `saved_models/baselines/keras/cifar-10/<model>/...` with timestamp naming.
