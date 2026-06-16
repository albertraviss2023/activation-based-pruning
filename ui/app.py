from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from runner import DATASETS, MODELS, PruningJobConfig, available_methods, default_output_dir, estimate_pruning_plan, ensure_methods_registered, export_job_config, run_torch_job, runtime_status


st.set_page_config(page_title="ReduCNN Studio", page_icon="R", layout="wide")


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1180px;}
    .stButton>button {border-radius: 6px; height: 2.7rem; font-weight: 650;}
    .metric-card {
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
    }
    .subtle {color: #5b6575;}
    .runtime-strip {
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: #f8fafc;
        color: #1f2937;
        margin: 0.5rem 0 1.2rem 0;
    }
    .runtime-strip strong {font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ReduCNN Studio")
st.caption("Configure, prune, fine-tune, and inspect CNN compression runs from one clean workspace.")
runtime = runtime_status()
runtime_label = "CUDA GPU" if runtime["cuda_available"] else "CPU"
device_label = html.escape(str(runtime.get("device", "unknown")))
torch_label = html.escape(str(runtime.get("torch_version", "unknown")))
python_label = html.escape(str(runtime.get("python", "unknown")))
st.markdown(
    f"""
    <div class="runtime-strip">
        <strong>Runtime:</strong> {runtime_label}
        &nbsp; | &nbsp; <strong>Device:</strong> {device_label}
        &nbsp; | &nbsp; <strong>PyTorch:</strong> {torch_label}
        &nbsp; | &nbsp; <strong>Python:</strong> {python_label}
    </div>
    """,
    unsafe_allow_html=True,
)
if not runtime["cuda_available"]:
    st.info("For Colab GPU pruning, start this app inside a Colab GPU runtime. A local Docker container cannot use Colab CUDA.")

with st.sidebar:
    st.header("Experiment")
    backend = st.selectbox("Backend", ["pytorch"], format_func={"pytorch": "PyTorch"}.get)
    custom_methods_path = st.text_input("Custom methods path", value="custom_methods")
    method_load_report = ensure_methods_registered(custom_methods_path)
    method_catalog = available_methods("torch", custom_methods_path)
    dataset = st.selectbox("Dataset", list(DATASETS), format_func=lambda k: DATASETS[k]["label"])
    model = st.selectbox("Model", list(MODELS), format_func=lambda k: MODELS[k])
    method = st.selectbox(
        "Pruning method",
        list(method_catalog),
        format_func=lambda k: method_catalog[k]["label"],
    )
    scope = st.radio("Scope", ["local", "global"], horizontal=True)
    ratio = st.slider("Pruning ratio", 0.05, 0.90, 0.30, 0.05)

    st.header("Run Settings")
    smoke_mode = st.toggle("Smoke mode", value=True, help="Use synthetic data for a fast end-to-end check.")
    baseline_mode = st.selectbox(
        "Baseline source",
        ["auto_latest", "train_new", "load_checkpoint", "model_init"],
        format_func={
            "auto_latest": "Auto: latest saved, else train",
            "train_new": "Train new baseline",
            "load_checkpoint": "Load checkpoint",
            "model_init": "Use model initialization",
        }.get,
    )
    pretrained = st.toggle("ImageNet weights", value=False, help="Use ImageNet initialization where the selected model supports it.")
    baseline_checkpoint_path = st.text_input("Baseline checkpoint", value="")
    batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=8, step=1)
    calibration_batches = st.number_input("Calibration batches", min_value=1, max_value=500, value=2, step=1)
    epochs = st.number_input("Baseline epochs", min_value=0, max_value=100, value=0, step=1)
    finetune_epochs = st.number_input("Fine-tune epochs", min_value=1, max_value=100, value=1, step=1)
    learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=3e-4, step=1e-4, format="%.6f")
    cats_dogs_dir = st.text_input("Cat/Dog folder", value="data/cats_dogs/train")
    output_dir = st.text_input("Output folder", value=default_output_dir())
    st.header("Save Artifacts")
    save_baseline = st.toggle("Save baseline checkpoint", value=False)
    save_raw_pruned = st.toggle("Save raw pruned model", value=True)
    save_finetuned = st.toggle("Save fine-tuned model", value=True)
    save_plots = st.toggle("Save plots and tables", value=True)

config = PruningJobConfig(
    backend=backend,
    dataset=dataset,
    model=model,
    method=method,
    scope=scope,
    ratio=float(ratio),
    epochs=int(epochs),
    finetune_epochs=int(finetune_epochs),
    batch_size=int(batch_size),
    calibration_batches=int(calibration_batches),
    learning_rate=float(learning_rate),
    pretrained=bool(pretrained),
    baseline_mode=baseline_mode,
    baseline_checkpoint_path=baseline_checkpoint_path,
    save_baseline=bool(save_baseline),
    save_raw_pruned=bool(save_raw_pruned),
    save_finetuned=bool(save_finetuned),
    save_plots=bool(save_plots),
    cats_dogs_dir=cats_dogs_dir,
    custom_methods_path=custom_methods_path,
    smoke_mode=bool(smoke_mode),
    output_dir=output_dir,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("Pruning Plan")
    plan = estimate_pruning_plan(config)
    st.dataframe(pd.DataFrame([plan]).T.rename(columns={0: "Value"}), use_container_width=True)

    st.subheader("Method")
    st.markdown(f"**{method_catalog[method]['label']}**")
    st.write(method_catalog[method]["description"])

    st.subheader("Available Methods")
    methods_df = pd.DataFrame(
        [{"key": key, "method": meta["label"], "notes": meta["description"]} for key, meta in method_catalog.items()]
    )
    st.dataframe(methods_df, hide_index=True, use_container_width=True)

    st.subheader("Custom Method Modules")
    if method_load_report:
        st.dataframe(
            pd.DataFrame(
                [{"module": key, "status": value} for key, value in method_load_report.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("No custom method modules loaded from the configured path.")

with right:
    st.subheader("Run")
    st.write(
        "Use smoke mode for a quick wiring check. Disable it when you want to download CIFAR data or use your local Cat vs Dog folder."
    )
    run_target = st.radio("Run target", ["local", "export_colab_config"], format_func={"local": "Run here", "export_colab_config": "Export Colab job config"}.get)
    run = st.button("Start pruning run" if run_target == "local" else "Export job config", type="primary", use_container_width=True)

    if run:
        if run_target == "export_colab_config":
            path = export_job_config(config)
            st.success(f"Saved Colab job config: {path}")
        else:
            with st.status("Running ReduCNN pruning pipeline...", expanded=True) as status:
                st.write("Preparing data and model.")
                try:
                    summary = run_torch_job(config)
                    status.update(label="Run complete", state="complete")
                    st.session_state["last_summary"] = summary
                except Exception as exc:
                    status.update(label="Run failed", state="error")
                    st.exception(exc)

    summary = st.session_state.get("last_summary")
    if summary:
        st.subheader("Latest Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Param reduction", f"{summary['param_reduction_pct']:.1f}%")
        c2.metric("FLOP reduction", f"{summary['flop_reduction_pct']:.1f}%")
        c3.metric("Accuracy", f"{summary['final_accuracy_pct']:.2f}%")
        c4.metric("Accuracy delta", f"{summary['accuracy_delta_pct']:+.2f}%")
        artifacts = summary.get("artifacts", {})
        artifact_rows = [{"artifact": key, "path": value} for key, value in artifacts.items() if value]
        if artifact_rows:
            st.subheader("Saved Artifacts")
            st.dataframe(pd.DataFrame(artifact_rows), hide_index=True, use_container_width=True)
            plot_path = artifacts.get("layer_sensitivity_plot")
            if plot_path:
                try:
                    st.image(plot_path, caption="Layer sensitivity")
                except Exception:
                    pass
        st.json(summary)
