# Module Documentation: `reducnn`

This document maps the public package structure for developers and thesis
readers who need to understand how ReduCNN is organized. It is intentionally
kept at module level; implementation details live in the source docstrings and
workflow-specific documents under `docs/`.

## Package Overview

ReduCNN is split into six main areas:

- `reducnn.core`: shared interfaces, decorators, storage helpers, and package
  exceptions.
- `reducnn.backends`: PyTorch and Keras adapters that implement model loading,
  training, evaluation, statistics, and structural surgery.
- `reducnn.pruner`: pruning method registration, scoring criteria, mask
  building, and structural pruning orchestration.
- `reducnn.analyzer`: validation and Pareto-style analysis utilities.
- `reducnn.visualization`: plotting and visual reporting helpers.
- `reducnn.engine`: high-level orchestration for train, prune, fine-tune, and
  report workflows.

## `reducnn.core`

### `adapter.py`

Defines the `FrameworkAdapter` abstract base class. Backends implement this
interface so the pruning engine can work with PyTorch or Keras without changing
experiment code.

Important responsibilities:

- construct or load supported architectures;
- train and evaluate models;
- compute model statistics such as accuracy, parameters, and FLOPs;
- collect method scores;
- apply structural pruning surgery;
- save and load checkpoints.

### `decorators.py`

Provides utility decorators for backend dispatch, timing, and consistent
pipeline logging.

### `storage.py`

Contains `CloudStorage`, a path helper for local and Google Colab workflows. It
keeps notebook code portable when artifacts need to be saved locally or on
Google Drive.

### `exceptions.py`

Defines project-specific exceptions such as `SurgeryError`,
`UnsupportedFrameworkError`, and `MethodRegistrationError`.

## `reducnn.backends`

### `torch_backend.py`

Implements PyTorch support through `PyTorchAdapter` and the structural pruning
logic needed for convolutional networks. The backend handles device placement,
evaluation, training, activation collection, FLOPs estimation, and physical
channel removal.

For residual architectures such as ResNet, structural surgery must preserve
shape compatibility across shortcut additions. The backend therefore tracks
dependencies between convolution, batch normalization, downstream convolution,
and residual branches.

### `keras_backend.py`

Implements Keras/TensorFlow support through `KerasAdapter`. The backend supports
model construction, evaluation, training, analytical statistics, activation
collection, and graph-aware model rebuilding after structural pruning.

### `factory.py`

Provides backend adapter discovery and creation helpers.

## `reducnn.pruner`

### `surgeon.py`

Defines `ReduCNNPruner`, the primary pruning entry point. It coordinates:

1. method scoring;
2. pruning-mask construction;
3. structural surgery through the active backend adapter.

The pruner supports local and global scopes, where local pruning selects
channels independently per layer and global pruning ranks candidate channels
across the eligible model region.

### `registry.py`

Implements the custom pruning method registry.

Public helpers:

- `register_method(name, framework="global", supported_scopes=None)`;
- `list_methods(framework=None, include_global=True)`;
- `list_method_names(framework=None, include_global=True)`;
- `get_method(name, framework)`;
- `call_score_fn(method_name, framework, kwargs)`.

The registry is intentionally open so experiment notebooks and files under
`custom_methods/` can add new scoring methods without editing the core package.

### `criteria.py`, `meta_criteria.py`, `chip.py`

Contain bundled pruning scores and literature-inspired criteria. Examples
include L1 norm, APoZ, mean activation, CHIP-style activation scoring, and
custom or meta-criteria used by experiments.

### `custom_method_tools.py`

Provides reusable helper functions passed to registered methods, including
activation collection, channel matrix conversion, weight norms, and other
scoring utilities.

### `mask_builder.py`

Builds boolean keep masks from score maps for local and global pruning. This is
where pruning ratio, scope, and per-layer channel counts become concrete masks
for structural surgery.

### `hybrid2.py`

Contains support code for hybrid and layerwise method selection experiments.
Thesis-specific LFPC objective experiments are documented separately in
`docs/OBJECTIVE_LFPC_EXPERIMENTS.md`.

## `reducnn.analyzer`

### `validator.py`

Compares pruning methods and validates score behavior. It is useful for checking
whether methods produce compatible score shapes and for method-agreement
analysis.

### `pareto.py`

Builds accuracy, compression, and runtime trade-off views across pruning
settings.

### `classifier.py`

Contains classifier-oriented analysis helpers used by reporting workflows.

## `reducnn.visualization`

### `stakeholder.py`

High-level plots for model compression summaries, layer sensitivity, and
training history.

### `research.py`

Research-facing diagnostic plots such as score distributions and rank
correlations.

### `animator.py`, `flow_animator.py`, `pruning_visualizer.py`, `persistence.py`

Utilities for animated or persisted pruning visualizations and report artifacts.

## `reducnn.engine`

### `orchestrator.py`

Defines `Orchestrator`, a convenience wrapper for end-to-end workflows:

1. build or load a baseline;
2. prune with a configured method and scope;
3. fine-tune the pruned model;
4. save checkpoints and metrics.

Use the lower-level adapters and `ReduCNNPruner` directly when an experiment
needs full control over scoring, timing, or custom reporting.

## Dataset And Architecture Notes

ReduCNN is intended to be dataset-agnostic. Dataset shape, class count, and
normalization should be supplied through config or inferred from loaders where
the backend supports it.

Architectures differ in pruning difficulty:

- VGG-style sequential models are easier to prune because channel dependencies
  mostly flow forward through convolution and batch-normalization chains.
- ResNet-style residual models require dependency-aware pruning so shortcut and
  residual branches remain shape-compatible.

These architectural differences matter in the objective LFPC experiments:
similar pruning ratios can produce different accuracy, FLOPs, parameter, and
runtime behavior depending on model family, dataset, scope, and selected
layerwise methods.
