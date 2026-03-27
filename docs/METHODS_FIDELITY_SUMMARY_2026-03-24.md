# Methods + Backend Parity Summary (2026-03-24)

## Status
- Custom method registration notebook now runs with backend parity for requested matrix after Keras surgery fixes.
- Keras structural rebuild out-of-bounds issues were fixed in backend surgery logic.
- PyTorch and Keras adapter score dispatch both pass adapter config kwargs to registered methods.

## Key code changes
- `src/reducnn/backends/keras_backend.py`
  - safer residual mask harmonization by channel-size grouping
  - robust index sanitation for concat/add/conv/bn/dense during rebuild
- `src/reducnn/backends/torch_backend.py`
  - registered methods now receive adapter config kwargs
- `experiments_custom_method_registration_minimal.ipynb`
  - user-friendly preset workflow and robust per-method execution reporting

## Validation highlights
- Regression tests: `10 passed`
- Full requested matrix summary artifact:
  - `outputs/custom_method_matrix/matrix_all_requested_models_after_keras_fix_summary_20260324.csv`
  - all backend/model combinations report `11` method successes in selected post-fix runs
