"""Add explanatory markdown and code-cell comments to LFPC experiment notebooks.

The script is intentionally conservative: it does not reorder cells, remove
outputs, or rewrite experiment logic. It inserts one markdown explanation before
each code cell and one comment preamble inside each code cell. Re-running it is
safe because both insertions use stable markers.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat


DOC_MARKER = "<!-- reduc_nn_lfpc_cell_doc -->"
COMMENT_MARKER = "# [ReduCNN LFPC cell note]"


NOTEBOOK_GLOB = "experiments_lfpc_realistic_thresholds_enhanced_visuals*_registered_methods*.ipynb"


def notebook_context(path: Path) -> dict[str, str]:
    name = path.stem.lower()
    dataset = "Cats-vs-Dogs" if "cats_dogs" in name or "catdog" in name else "CIFAR-10"
    model = "ResNet18" if "resnet18" in name else "VGG16" if "vgg16" in name else "unknown model"
    if "objective_flops_accuracy" in name:
        objective = "FLOPs + Accuracy"
    elif "objective_time_accuracy" in name:
        objective = "Time + Accuracy"
    elif "all_three" in name:
        objective = "FLOPs + Time + Accuracy"
    else:
        objective = "notebook-defined objective"
    return {"dataset": dataset, "model": model, "objective": objective}


def compact_source(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(COMMENT_MARKER):
            continue
        lines.append(stripped)
        if len(lines) >= 8:
            break
    return " ".join(lines).lower()


def infer_cell_role(source: str) -> tuple[str, str, str]:
    text = compact_source(source)

    rules: list[tuple[tuple[str, ...], tuple[str, str, str]]] = [
        (
            ("import ", "from ", "warnings", "random", "seed"),
            (
                "Environment, imports, and reproducibility setup",
                "It loads Python packages, configures runtime helpers, and prepares deterministic or portable execution settings used by later cells.",
                "The notebook runtime has the libraries, paths, and helper functions needed by the experiment.",
            ),
        ),
        (
            ("objective_scenario", "optimized_objective_terms", "lfpc_target", "max_allowed_accuracy"),
            (
                "Objective and experiment configuration",
                "It declares the active optimization objective, accuracy/compression targets, pruning ratios, scopes, and reporting controls for this notebook context.",
                "All later discovery, benchmarking, and reporting cells use the same objective and context variables.",
            ),
        ),
        (
            ("run_metadata", "metadata", "experiment_registry", "manifest"),
            (
                "Run metadata and artifact traceability",
                "It records dataset, model, objective, timestamp, paths, and configuration values so analysis notebooks can recover the exact context.",
                "A metadata record or manifest is saved for later context-safe reporting.",
            ),
        ),
        (
            ("get_model", "baseline", "checkpoint", "force_retrain", "load_checkpoint"),
            (
                "Baseline model loading or training",
                "It builds the model, loads an existing baseline checkpoint when allowed, or trains/saves a baseline when the configured policy requires it.",
                "A baseline model and baseline accuracy metrics are available before pruning starts.",
            ),
        ),
        (
            ("dataloader", "dataset", "transforms", "train_loader", "val_loader", "test_loader"),
            (
                "Dataset and loader preparation",
                "It prepares dataset splits, preprocessing transforms, calibration loaders, and evaluation loaders for the active dataset.",
                "Training, calibration, validation, and test loaders are ready for scoring and benchmarking.",
            ),
        ),
        (
            ("register_method", "method_metadata", "custom_", "chip", "thinet", "nisp", "reprune"),
            (
                "Pruning method registration",
                "It registers built-in and custom scoring methods with the ReduCNN method registry, including their valid local or global scopes.",
                "All candidate singular and hybrid methods can be called consistently through the registry.",
            ),
        ),
        (
            ("score_map", "get_score_map", "score_timing", "methods scheduled", "cost normalization"),
            (
                "Candidate method scoring",
                "It evaluates pruning scores for each candidate method on the active calibration data and records scoring cost where needed.",
                "Layerwise score maps and method timing records are available for LFPC discovery and runtime-aware objectives.",
            ),
        ),
        (
            ("train_lfpc", "lfpc", "sampler", "alpha", "policy", "threshold"),
            (
                "LFPC layerwise policy discovery",
                "It searches over candidate pruning methods under the configured similarity thresholds and objective terms to select a method per layer.",
                "A discovered layerwise hybrid policy table and search history are produced for each scope and pruning ratio.",
            ),
        ),
        (
            ("fixed_stack", "run_pruned_policy", "hybrid", "structural_prune", "healing"),
            (
                "Fixed hybrid stack pruning benchmark",
                "It applies each discovered hybrid policy, builds masks, performs structural pruning, optionally heals the model, and evaluates final metrics.",
                "Hybrid stack rows contain accuracy delta, FLOPs reduction, parameter reduction, pruning time, and checkpoint provenance.",
            ),
        ),
        (
            ("singular", "single method", "singular_method", "re-prune", "reprune_singular"),
            (
                "Same-context singular method benchmark",
                "It loads cached singular pruned models when available, runs missing singular contexts when configured, and enforces dataset-model-scope-ratio alignment.",
                "Singular baseline metrics are available for fair comparison against hybrid stacks in the same context.",
            ),
        ),
        (
            ("phase 2", "phase 3", "pareto", "stability", "convergence"),
            (
                "Phase 2/3 benchmarking and stability diagnostics",
                "It standardizes metrics, selects best stacks, computes Pareto candidates, and summarizes policy stability across threshold settings.",
                "Context-segmented benchmark tables, Pareto plots, and stability artifacts are saved for reporting.",
            ),
        ),
        (
            ("top_stack", "same-scope", "same-context", "comparison", "win_rate"),
            (
                "Top-stack reporting and same-context comparisons",
                "It ranks hybrid stacks for the active objective and compares the top policies against singular methods with matching dataset, model, scope, and ratio.",
                "Notebook-local comparison plots and tables show accuracy, FLOPs, parameters, and pruning-time trade-offs.",
            ),
        ),
        (
            ("plot", "plt.", "figure", "imshow", "bar", "scatter", "display("),
            (
                "Visualization and notebook display",
                "It renders plots or tables that make the experiment results inspectable inside the notebook and reusable in thesis reporting.",
                "The notebook displays figures or tables and usually saves matching artifacts to the run output directory.",
            ),
        ),
        (
            ("to_csv", "to_json", "savefig", "torch.save", "write_text"),
            (
                "Artifact export",
                "It writes structured tables, figures, checkpoints, or audit files to the configured output directories.",
                "Downstream analysis notebooks can reload the saved artifacts without rerunning pruning.",
            ),
        ),
    ]

    for tokens, result in rules:
        if any(token in text for token in tokens):
            return result

    return (
        "Experiment helper or execution cell",
        "It defines helper functions or executes a notebook-specific step used by the LFPC pruning workflow.",
        "The following cells can reuse the variables, functions, or artifacts created here.",
    )


def markdown_for_cell(index: int, context: dict[str, str], source: str) -> str:
    title, how, expected = infer_cell_role(source)
    return (
        f"{DOC_MARKER}\n"
        f"### Code cell {index}: {title}\n\n"
        f"**Context.** This cell belongs to the `{context['objective']}` LFPC experiment "
        f"for `{context['dataset']}` with `{context['model']}`.\n\n"
        f"**What it does.** {title}.\n\n"
        f"**How it works.** {how}\n\n"
        f"**Expected result.** {expected}\n\n"
        "If this cell fails, inspect the active dataset/model/objective context and the "
        "artifact paths printed by the notebook before continuing."
    )


def comment_block_for_cell(source: str) -> str:
    title, how, expected = infer_cell_role(source)
    return (
        f"{COMMENT_MARKER}\n"
        f"# Purpose: {title}.\n"
        f"# Mechanics: {how}\n"
        f"# Expected result: {expected}\n"
    )


def add_comment_to_source(source: str) -> str:
    if COMMENT_MARKER in source:
        return source
    block = comment_block_for_cell(source)
    if source.startswith("%%"):
        lines = source.splitlines()
        if len(lines) == 1:
            return source + "\n" + block
        return "\n".join([lines[0], block, *lines[1:]])
    return block + "\n" + source


def already_documented_previous(cells: list, index: int) -> bool:
    if index == 0:
        return False
    prev = cells[index - 1]
    return prev.cell_type == "markdown" and DOC_MARKER in str(prev.source)


def document_notebook(path: Path) -> tuple[int, int]:
    nb = nbformat.read(path, as_version=4)
    context = notebook_context(path)
    new_cells = []
    docs_added = 0
    comments_added = 0
    code_counter = 0

    old_cells = list(nb.cells)
    for idx, cell in enumerate(old_cells):
        if cell.cell_type != "code":
            new_cells.append(cell)
            continue

        code_counter += 1
        if not already_documented_previous(old_cells, idx):
            new_cells.append(nbformat.v4.new_markdown_cell(markdown_for_cell(code_counter, context, cell.source)))
            docs_added += 1

        old_source = str(cell.source)
        cell.source = add_comment_to_source(old_source)
        if cell.source != old_source:
            comments_added += 1
        new_cells.append(cell)

    nb.cells = new_cells
    nbformat.write(nb, path)
    return docs_added, comments_added


def main() -> None:
    root = Path.cwd()
    notebooks = sorted(root.glob(NOTEBOOK_GLOB))
    if not notebooks:
        raise SystemExit(f"No notebooks matched {NOTEBOOK_GLOB}")

    total_docs = 0
    total_comments = 0
    for path in notebooks:
        docs_added, comments_added = document_notebook(path)
        total_docs += docs_added
        total_comments += comments_added
        print(f"{path.name}: added {docs_added} markdown cells, {comments_added} code comments")

    print(f"Done: {len(notebooks)} notebooks, {total_docs} markdown cells, {total_comments} code comments")


if __name__ == "__main__":
    main()
