# Literature Fidelity Report v2 (Method Audit)

Date: 2026-03-23
Scope: `reducnn` core methods + custom methods in `experiments_custom_method_registration_minimal.ipynb`

## 1) Sources Used

Literature source (provided by you):
- `Activation-based_pruning_for_CNNs.pdf`
  - Key definitions and equations used in this audit:
  - APoZ Eq. (4), Mean Activation Eq. (5), Entropy Eq. (8), HRank Eq. (9)-(10),
    NISP Eq. (15), SeNPIS Eq. (16)-(17), ThiNet Eq. (18), AOFP Eq. (19)
  - Method taxonomy and protocol guidance: pp. 10-22

Code audited:
- `src/reducnn/pruner/criteria.py`
- `src/reducnn/pruner/chip.py`
- `src/reducnn/backends/torch_backend.py`
- `src/reducnn/backends/keras_backend.py`
- `experiments_custom_method_registration_minimal.ipynb` (registration cell and run cells)

## 2) Fidelity Rubric

- `Faithful (100%)`: core objective + equation + key protocol steps are implemented.
- `High`: equation mostly correct, minor protocol simplifications.
- `Approximate`: same family/signal, but important algorithmic parts are missing.
- `Toyish`: mainly name-level alignment; method behavior is materially different from literature.

## 3) Core Package Methods (Bundled)

| Method | Literature expectation | Current implementation | Verdict |
|---|---|---|---|
| `apoz` | Fraction of zero post-ReLU activations (Eq. 4) | Computes `(1-APoZ)` as keep-importance using post-ReLU transform in both backends | High (not 100%) |
| `mean_abs_act` | Mean post-activation magnitude (Eq. 5 style) | Mean absolute activation per channel | High |
| `taylor` | First-order sensitivity `|A * dL/dA|` | Implemented in both backends with hooks/grad tape and channel reduction | High |
| `chip` | CHIP uses channel independence via nuclear-norm change / correlation structure (survey pp. 20-21) | Uses `1 - mean(abs(correlation))` only (`chip.py`) | Approximate |
| `l1_norm` | Classic structured magnitude pruning (Li et al.) uses filter norm | Uses mean absolute weight per filter (not sum) | Approximate |
| `l2_norm` | Structured magnitude variant | RMS over filter weights | Approximate |
| `variance_act` | Not a canonical method in cited ABP equations | Variance heuristic | Heuristic (not literature-bound) |
| `random` | Baseline only | Random scores | Baseline only |

### Evidence (core code)
- APoZ / Taylor / CHIP / mean-abs / variance paths:
  - `src/reducnn/backends/torch_backend.py:926-947`
  - `src/reducnn/backends/keras_backend.py:754-795`
- CHIP formula actually used:
  - `src/reducnn/pruner/chip.py:17, 55-59`
- Weight-norm criteria:
  - `src/reducnn/pruner/criteria.py:10-63`

## 4) Custom Registration Notebook Methods

Notebook methods audited:
- `custom_entropy`, `custom_hrank`, `custom_class_entropy`, `custom_spectral_energy`,
  `custom_nisp`, `custom_senpis`, `custom_tis`, `custom_reprune`, `custom_thinet`

### High-level finding
Several custom methods are currently **name-aligned but algorithmically simplified**. In particular, `custom_nisp`, `custom_senpis`, `custom_reprune`, and `custom_thinet` are not faithful to the literature procedures.

### Method-by-method verdict

| Custom method | Literature expectation | Current implementation | Verdict |
|---|---|---|---|
| `custom_entropy` | Entropy from activation distribution (survey Eq. 8; often GAP+histogram) | Histogram entropy on flattened channel activations | Approximate |
| `custom_hrank` | Average rank of feature maps across samples (Eq. 9-10 family) | Matrix-rank per sample/channel, averaged | High |
| `custom_class_entropy` | Class-conditional entropy pruning (classwise criterion) | Per-class entropy averaged, then global pruning | Approximate |
| `custom_spectral_energy` | LRMF-style frequency-domain criterion (DCT/median-representation protocol) | FFT total energy only | Approximate-to-Toyish |
| `custom_nisp` | Backward importance propagation from FRL (Eq. 15) | Uses Taylor contribution (`A*grad`) | Toyish |
| `custom_senpis` | SeNPIS: class-wise loss change under filter zeroing + attenuation | Classwise Taylor + variance-based boost | Toyish |
| `custom_tis` | TIS: class-aware thresholded Taylor-based binary importance aggregation | Continuous classwise Taylor sum, no threshold/max protocol | Approximate |
| `custom_reprune` | REPrune: kernel clustering + representative coverage optimization | Correlation redundancy score only | Toyish |
| `custom_thinet` | ThiNet: greedy next-layer reconstruction minimization + LS re-scaling (Eq. 18) | Next-layer weight magnitude proxy | Toyish |

### Critical protocol mismatch in notebook runs
Global methods are executed with `scope="local"` in both backend loops:
- `outputs/custom_methods_nb_cell_9.py:39`
- `outputs/custom_methods_nb_cell_11.py:33`

That alone prevents faithful evaluation of global formulations (NISP/SeNPIS/TIS/REPrune/ThiNet).

### Evidence (custom code)
- Core custom method definitions:
  - `outputs/custom_methods_nb_cell_7.py:362-526`
- NISP/SeNPIS/TIS logic:
  - `outputs/custom_methods_nb_cell_7.py:408-427, 493-512`
- REPrune/ThiNet logic:
  - `outputs/custom_methods_nb_cell_7.py:433-441, 518-526`

## 5) Toyish Implementations Requiring Correction

To meet your stated target ("100% fidelity"), these are mandatory to rewrite:

1. `custom_nisp`
- Must implement FRL-based importance initialization and recursive backward propagation with layer weights (Eq. 15 path), not Taylor proxy.

2. `custom_senpis`
- Must compute class-wise loss change from explicit filter ablation and apply similarity attenuation (SSIM/COSSIM behavior), then class aggregation (Eq. 16-17 intent).

3. `custom_reprune`
- Must implement kernel-level clustering and representative coverage selection (MCP-style objective), not plain correlation score.

4. `custom_thinet`
- Must implement reconstruction objective and greedy channel subset selection with least-squares refinement (Eq. 18 protocol).

5. `chip` (core package)
- Current implementation is a correlation heuristic. For strict fidelity, implement CHIP’s channel-independence behavior using the paper’s nuclear-norm change formulation.

## 6) Additional Non-trivial Gaps (Not Toyish, but not 100%)

1. `l1_norm` uses mean absolute weight instead of sum norm.
- For layer-local ranking this is often order-equivalent, but for strict mathematical fidelity, use exact paper norm convention.

2. APoZ/Mean Activation collection semantics.
- Currently uses fixed small calibration passes and explicit ReLU transform on feature tensors.
- For strict protocol fidelity, enforce post-nonlinearity tap points and document sample count/statistical stability.

3. `custom_tis` missing thresholded binary scoring behavior.
- Should include class-wise thresholding and per-class aggregation exactly as described.

## 7) Final Verdict

Current status against your "100% fidelity" requirement:
- Core bundled ABP methods: **partially faithful** (good engineering quality, but not fully paper-exact for CHIP and some protocol details).
- Custom notebook methods: **mixed**, with several **toyish** global-method substitutions.

Therefore, your concern is valid: we still have literature gaps, especially for the newly added global custom methods.

---

## 8) Immediate Correction Plan (for next implementation pass)

P0 (must-fix first):
1. Rewrite `custom_nisp` to true propagation formulation.
2. Rewrite `custom_senpis` with explicit class-wise ablation loss and attenuation.
3. Rewrite `custom_reprune` with clustering + representative coverage.
4. Rewrite `custom_thinet` with reconstruction-greedy objective and LS refinement.
5. Run these global methods with `scope="global"` in notebook loop for faithful evaluation.

P1:
1. Upgrade core `chip` from correlation heuristic to nuclear-norm-change implementation.
2. Tighten APoZ/Mean-Activation tap points to strict post-activation collection.
3. Align `l1_norm` implementation with exact norm convention used in target paper comparisons.

