# Methods Fidelity Summary (2026-03-23)

## Current State
- Literature fidelity audit completed in `docs/LITERATURE_FIDELITY_REPORT_v2.md`.
- Bundled-vs-custom boundary clarified by user: package should bundle only `apoz`, `mean_abs_act`, and `l1_norm`.
- All other methods (e.g., `chip`, `custom_l2`, and advanced global methods) are to be exercised through custom registration notebook tests.
- Current custom notebook contains several method-name/algorithm mismatches for global methods.

## Locked User Requirements
- Keep only three bundled methods in package surface: `apoz`, `mean_abs_act`, `l1_norm`.
- Evaluate/implement other methods in custom registration notebook only.
- Achieve 100% literature fidelity for named methods.
- Ensure PyTorch/Keras method math is semantically aligned (no inconsistent aggregation logic).

## Known Gaps
- `custom_nisp`, `custom_senpis`, `custom_reprune`, `custom_thinet` are currently toyish vs literature.
- `custom_tis`, `custom_entropy`, `custom_class_entropy`, `custom_spectral_energy` are partial approximations.
- Global-method notebook loop currently runs with `scope="local"`.

## Next Work Block
1. Enforce bundled-method surface restriction in package codepaths.
2. Keep CHIP/L2 as custom registrations in notebook.
3. Implement faithful math/protocol for NISP/SeNPIS/TIS/REPrune/ThiNet.
4. Add backend-parity checks and rerun full validation.
