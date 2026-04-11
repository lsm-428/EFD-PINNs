"""
对比分析面板组件
===============

用于在 Streamlit 仪表板中进行多运行或多配置的对比分析。
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Import EFD3D modules
import sys

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.pinn_two_phase import TwoPhasePINN, PHYSICS
from src.models.aperture_model import EnhancedApertureModel
from src.config import CONFIG_PATH

# Color scheme
COMPARE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def compare_training_runs(run_dirs: List[str]) -> go.Figure:
    """Compare training curves from multiple runs.

    Args:
        run_dirs: List of run directory paths

    Returns:
        Plotly Figure with loss curves comparison
    """

    def parse_log_file(log_path):
        """Parse training log file."""
        data = {"epoch": [], "loss_total": [], "lr": []}
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if "Epoch" in line and "Loss:" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "Epoch":
                                data["epoch"].append(int(parts[i + 1]))
                            if p == "Loss:":
                                data["loss_total"].append(
                                    float(parts[i + 1].rstrip(","))
                                )
        except Exception as e:
            print(f"Warning: Failed to parse log: {e}")
        return data

    def get_run_info(run_dir):
        """Get run information."""
        info = {
            "id": Path(run_dir).name,
            "path": run_dir,
            "config": {},
            "final_loss": float("inf"),
        }
        config_path = Path(run_dir) / "config.json"
        if config_path.exists():
            info["config"] = json.loads(config_path.read_text())

        log_path = Path(run_dir) / "training.log"
        if log_path.exists():
            data = parse_log_file(log_path)
            if data and data["loss_total"]:
                info["metrics"] = data
                info["final_loss"] = data["loss_total"][-1]

        return info

    runs_data = [get_run_info(d) for d in run_dirs]

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Training Loss", "Final Loss Comparison"),
        horizontal_spacing=0.15,
    )

    # Loss curves
    for i, run in enumerate(runs_data):
        if "metrics" in run:
            m = run["metrics"]
            if "epoch" in m and "loss_total" in m:
                fig.add_trace(
                    go.Scatter(
                        x=m["epoch"],
                        y=m["loss_total"],
                        mode="lines",
                        name=run["id"][-12:],
                        line=dict(
                            color=COMPARE_COLORS[i % len(COMPARE_COLORS)], width=2
                        ),
                        legendgroup="loss",
                    ),
                    row=1,
                    col=1,
                )

    # Final loss bar chart
    ids = [r["id"][-12:] for r in runs_data]
    losses = [
        r["final_loss"] if r["final_loss"] != float("inf") else 0 for r in runs_data
    ]
    fig.add_trace(
        go.Bar(
            x=ids,
            y=losses,
            marker_color=COMPARE_COLORS[: len(runs_data)],
            name="Final Loss",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss L", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Run ID", row=1, col=2)
    fig.update_yaxes(title_text="Final Loss L", row=1, col=2)

    fig.update_layout(
        height=400,
        width=900,
        showlegend=True,
        legend=dict(x=0.65, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    )

    return fig


def compare_stage1_vs_stage2(model_path: str) -> go.Figure:
    """Compare Stage 1 (analytical) vs Stage 2 (PINN) predictions.

    Args:
        model_path: Path to trained model checkpoint

    Returns:
        Plotly Figure with Stage 1 vs Stage 2 comparison
    """
    # Load model
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = TwoPhasePINN()
    model.load_state_dict(state_dict)
    model.eval()

    # Load Stage 1
    stage1 = EnhancedApertureModel(config_path=str(CONFIG_PATH))

    # Test cases
    cases = [
        (0, 30, 0.02, "0V→30V"),
        (30, 0, 0.03, "30V→0V"),
        (0, 15, 0.02, "0V→15V"),
        (30, 15, 0.02, "30V→15V"),
    ]

    # Create subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=[case[3] for case in cases])

    for idx, (V_from, V_to, t_max, name) in enumerate(cases):
        times = np.linspace(0, t_max, 50)

        # Stage 1
        eta_s1 = [stage1.theta_eta_from_triad(V_from, V_to, t)[1] for t in times]

        # Stage 2 (PINN)
        Lx, Ly, Lz = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["Lz"]
        h_ink = PHYSICS["h_ink"]
        n = 500
        x = np.random.uniform(0, Lx, n)
        y = np.random.uniform(0, Ly, n)
        z = np.full(n, h_ink / 2)

        spatial = torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)

        # Collect tensors for Stage 2
        phi_tensors = []
        for t in times:
            inp = torch.cat(
                [
                    spatial,
                    torch.full((n, 1), float(V_from)),
                    torch.full((n, 1), float(V_to)),
                    torch.full((n, 1), t),
                ],
                dim=1,
            )

            with torch.no_grad():
                phi = model(inp)[:, 4]
            phi_tensors.append(phi)

        phi_tensors_np = [phi.cpu().numpy() for phi in phi_tensors]
        eta_pinn = [float((phi < 0.5).mean()) for phi in phi_tensors_np]

        # Plot Stage 1
        fig.add_trace(
            go.Scatter(
                x=times * 1000,
                y=np.array(eta_s1) * 100,
                mode="lines",
                name="Stage 1" if idx == 0 else None,
                line=dict(color="#1f77b4", dash="dash", width=2),
                showlegend=(idx == 0),
            ),
            row=(idx // 2) + 1,
            col=(idx % 2) + 1,
        )

        # Plot Stage 2
        fig.add_trace(
            go.Scatter(
                x=times * 1000,
                y=np.array(eta_pinn) * 100,
                mode="lines",
                name="Stage 2" if idx == 0 else None,
                line=dict(color="#ff7f0e", width=2),
                showlegend=(idx == 0),
            ),
            row=(idx // 2) + 1,
            col=(idx % 2) + 1,
        )

        fig.update_xaxes(title_text="Time (ms)", row=(idx // 2) + 1, col=(idx % 2) + 1)
        fig.update_yaxes(
            title_text="Aperture Ratio η (%)",
            range=[0, 100],
            row=(idx // 2) + 1,
            col=(idx % 2) + 1,
        )

    fig.update_layout(
        height=600,
        width=900,
        showlegend=True,
        legend=dict(x=0.5, y=1.05, orientation="h", bgcolor="rgba(255,255,255,0.8)"),
    )

    return fig


def analyze_volume_conservation(model_path: str) -> go.Figure:
    """Analyze volume conservation error across different voltages.

    Args:
        model_path: Path to trained model checkpoint

    Returns:
        Plotly Figure with volume conservation analysis
    """
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = TwoPhasePINN()
    model.load_state_dict(state_dict)
    model.eval()

    voltages = [0, 10, 20, 30]
    n = 32
    x = np.linspace(0, model.Lx, n)
    y = np.linspace(0, model.Ly, n)
    z = np.linspace(0, model.Lz, n)
    X, Y, Z = np.meshgrid(x, y, z)
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    initial_volume = model.Lx * model.Ly * PHYSICS["h_ink"]
    results = []

    for v in voltages:
        inp = torch.tensor(
            np.stack(
                [
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    np.full(len(coords), float(v)),
                    np.full(len(coords), float(v)),
                    np.full(len(coords), 0.02),
                ],
                axis=1,
            ),
            dtype=torch.float32,
        )

        with torch.no_grad():
            phi = model(inp)[:, 4].numpy()

        dV = model.Lx * model.Ly * model.Lz / (n**3)
        volume = np.sum(phi) * dV
        error = abs(volume - initial_volume) / initial_volume
        results.append({"Voltage": f"{v}V", "Error": error})

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[r["Voltage"] for r in results],
            y=[r["Error"] * 100 for r in results],
            marker_color="#1f77b4",
            name="Volume Error",
        )
    )

    # Add 1% threshold line
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#d62728",
        line_width=2,
        annotation_text="1% threshold",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Volume Conservation Error",
        xaxis_title="Voltage (V)",
        yaxis_title="Volume Error (%)",
        height=400,
        width=600,
        showlegend=False,
    )

    return fig


def render_compare_tab() -> None:
    """渲染对比分析标签页

    提供多运行对比功能，包括配置对比、性能对比和可视化对比。
    """
    st.subheader("📊 Model Comparison")

    # Define outputs/train directory
    train_dir = Path("outputs/train")

    # List available runs
    runs = []
    if train_dir.exists():
        runs = sorted(
            [
                d
                for d in train_dir.iterdir()
                if d.is_dir() and d.name.startswith("pinn_")
            ],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

    if not runs:
        st.info("No training runs found in outputs/train/. Train a model first.")
        return

    # Run selector
    run_options = [r.name for r in runs]
    run_paths = {r.name: str(r) for r in runs}

    selected_runs = st.multiselect(
        "Select runs to compare",
        options=run_options,
        help="Select 2 or more training runs to compare their performance",
    )

    if len(selected_runs) < 1:
        st.warning("Please select at least 1 run to compare.")
        return

    # Compare button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        compare_clicked = st.button("Compare Runs", type="primary")

    with col2:
        show_metrics = st.checkbox("Show Metrics Table", value=True)

    if compare_clicked:
        # Get full paths for selected runs
        selected_paths = [run_paths[name] for name in selected_runs]

        # Training curve comparison
        with st.spinner("Analyzing training runs..."):
            try:
                fig = compare_training_runs(selected_paths)
                st.plotly_chart(fig, use_container_width=True)

                # Metrics comparison table
                if show_metrics:
                    st.subheader("📈 Metrics Comparison")
                    metrics_data = _extract_run_metrics(selected_paths)
                    if metrics_data:
                        df = pd.DataFrame(metrics_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Could not extract metrics from selected runs.")

            except Exception as e:
                st.error(f"Error comparing runs: {e}")
                with st.expander("Error Details"):
                    import traceback

                    st.code(traceback.format_exc())


def _extract_run_metrics(run_paths: List[str]) -> List[Dict[str, Any]]:
    """Extract metrics from training runs for comparison table.

    Args:
        run_paths: List of run directory paths

    Returns:
        List of dictionaries containing run metrics
    """
    metrics_list = []

    for path in run_paths:
        run_dir = Path(path)
        metrics = {"Run ID": run_dir.name[-12:], "Path": str(run_dir)}

        # Parse config
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                metrics["Epochs"] = config.get("training", {}).get("epochs", "N/A")
                metrics["Batch Size"] = config.get("training", {}).get(
                    "batch_size", "N/A"
                )
                metrics["LR"] = config.get("training", {}).get("learning_rate", "N/A")
            except Exception:
                metrics["Epochs"] = "N/A"
                metrics["Batch Size"] = "N/A"
                metrics["LR"] = "N/A"

        # Parse training log
        log_path = run_dir / "training.log"
        if log_path.exists():
            try:
                losses = []
                with open(log_path, "r") as f:
                    for line in f:
                        if "Loss:" in line:
                            parts = line.split()
                            for i, p in enumerate(parts):
                                if p == "Loss:":
                                    losses.append(float(parts[i + 1].rstrip(",")))
                                    break
                if losses:
                    metrics["Final Loss"] = f"{losses[-1]:.4e}"
                    metrics["Min Loss"] = f"{min(losses):.4e}"
                    metrics["Total Epochs"] = str(len(losses))
            except Exception:
                metrics["Final Loss"] = "N/A"
                metrics["Min Loss"] = "N/A"
                metrics["Total Epochs"] = "N/A"

        metrics_list.append(metrics)

    return metrics_list


def select_runs_for_comparison(runs: List[Any], run_selector: Any) -> List[Any]:
    """选择要对比的训练运行

    Args:
        runs: 可用的训练运行列表
        run_selector: Streamlit 运行选择器状态

    Returns:
        选中的训练运行列表
    """
    pass


def render_comparison_summary(selected_runs: List[Any]) -> None:
    """渲染对比摘要信息

    Args:
        selected_runs: 选中的训练运行列表
    """
    pass


def render_loss_comparison(selected_runs: List[Any]) -> None:
    """渲染损失曲线对比图

    Args:
        selected_runs: 选中的训练运行列表
    """
    pass


def render_metrics_comparison(selected_runs: List[Any]) -> None:
    """渲染指标对比表

    Args:
        selected_runs: 选中的训练运行列表
    """
    pass


def render_config_comparison(selected_runs: List[Any]) -> None:
    """渲染配置对比表

    Args:
        selected_runs: 选中的训练运行列表
    """
    pass


def select_runs_for_comparison(runs: List[Any], run_selector: Any) -> List[Any]:
    """选择要对比的训练运行

    Args:
        runs: 可用的训练运行列表
        run_selector: Streamlit 运行选择器状态

    Returns:
        选中的训练运行列表
    """
    pass


def render_comparison_summary(selected_runs: List[Any]) -> None:
    """渲染对比摘要信息

    Args:
        selected_runs: 选中的训练运行列表
    """
    pass


def render_loss_comparison(selected_runs: List[Any]) -> None:
    """渲染损失曲线对比图

    Args:
        selected_runs: 选中的训练运行列表
    """
    pass


def render_metrics_comparison(selected_runs: List[Any]) -> None:
    """渲染指标对比表

    Args:
        selected_runs: 选中的训练运行列表
    """
    pass


def render_config_comparison(selected_runs: List[Any]) -> None:
    """渲染配置对比表

    Args:
        selected_runs: 选中的训练运行列表
    """
    pass
