#!/usr/bin/env python3
"""
Visualization and Output Module for Training Log Analysis

Provides functions for generating publication-quality plots and HTML reports.

Functions:
    - plot_loss_components(): Loss curves and fraction plots
    - plot_learning_curve(): Training dynamics with learning rate
    - generate_html_report(): Interactive HTML summary report

Usage:
    from src.dashboard.monitor.visualization_output import (
        plot_loss_components,
        plot_learning_curve,
        generate_html_report,
        LOSS_COLORS,
    )
"""

import os
from datetime import datetime
from typing import Dict, List, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Publication-Quality Plotting Settings
# =============================================================================

try:
    plt.style.use(["science", "ieee"])
except OSError:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

# Color schemes for different loss components (distinct colors)
LOSS_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


# =============================================================================
# Loss Visualization
# =============================================================================


def plot_loss_components(records: Dict[str, List[Any]], out_dir: str) -> None:
    """
    Plot loss components for publication.

    Generates:
    1. Log-scale loss curves (all components)
    2. Loss fraction stacked area plot

    Args:
        records: Parsed training records
        out_dir: Output directory for saving plots
    """
    epochs = np.array(records["epoch"])
    loss_total = np.array(records["loss_total"])

    # Identify loss components (exclude metadata)
    exclude = {"epoch", "stage", "loss_total", "lr"}
    component_names = [k for k in records.keys() if k not in exclude]

    # Figure 1: Loss curves (log scale)
    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.semilogy(
        epochs, loss_total, "k-", label=r"$\mathcal{L}_{total}$", linewidth=1.5
    )

    plot_count = 0
    for i, name in enumerate(component_names):
        values = np.array(records[name], dtype=float)
        if np.all(np.isnan(values)):
            continue
        valid = np.isfinite(values) & (values > 0)
        if not np.any(valid):
            continue
        ax1.semilogy(
            epochs[valid],
            values[valid],
            color=LOSS_COLORS[i],
            label=name,
            linewidth=1.2,
            alpha=0.85,
        )
        plot_count += 1

    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel(r"Loss $\mathcal{L}$", fontsize=10)
    ax1.set_title("(a) Training Loss Components", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7, ncol=1, framealpha=0.9)
    ax1.set_xlim((float(epochs[0]), float(epochs[-1])))
    fig1.tight_layout()

    # Save PNG + PDF
    output_path1_png = os.path.join(out_dir, "loss_components.png")
    output_path1_pdf = os.path.join(out_dir, "loss_components.pdf")
    fig1.savefig(output_path1_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig1.savefig(output_path1_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"  [OK] Loss curves: {output_path1_png}")

    # Figure 2: Loss fraction stacked area
    if plot_count > 0:
        comp_values = []
        comp_labels = []
        for i, name in enumerate(component_names):
            values = np.array(records[name], dtype=float)
            if np.all(np.isnan(values)):
                continue
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            comp_values.append(values)
            comp_labels.append(name)

        if comp_values:
            comp_array = np.stack(comp_values, axis=0)
            comp_array = np.maximum(comp_array, 0.0)
            sum_array = np.sum(comp_array, axis=0)
            sum_array = np.where(sum_array <= 0, 1.0, sum_array)
            frac_array = comp_array / sum_array

            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.stackplot(
                epochs,
                frac_array,
                labels=comp_labels,
                colors=LOSS_COLORS[: len(comp_labels)],
                alpha=0.85,
            )
            ax2.set_xlabel("Epoch", fontsize=10)
            ax2.set_ylabel("Fraction", fontsize=10)
            ax2.set_title(
                "(b) Relative Loss Contributions", fontsize=11, fontweight="bold"
            )
            ax2.set_ylim(0.0, 1.0)
            ax2.set_xlim((float(epochs[0]), float(epochs[-1])))
            ax2.legend(loc="upper right", fontsize=7, ncol=1, framealpha=0.9)
            fig2.tight_layout()

            output_path2_png = os.path.join(out_dir, "loss_fraction.png")
            output_path2_pdf = os.path.join(out_dir, "loss_fraction.pdf")
            fig2.savefig(
                output_path2_png, dpi=300, bbox_inches="tight", facecolor="white"
            )
            fig2.savefig(output_path2_pdf, bbox_inches="tight", facecolor="white")
            plt.close(fig2)
            print(f"  [OK] Loss fraction: {output_path2_png}")


def plot_learning_curve(records: Dict[str, List[Any]], out_dir: str) -> None:
    """
    Plot training loss and learning rate for publication.

    Args:
        records: Parsed training records
        out_dir: Output directory for saving plots
    """
    epochs = np.array(records["epoch"])
    loss_total = np.array(records["loss_total"])
    lr = np.array(records["lr"], dtype=float)

    # Skip if no LR data
    if np.all(np.isnan(lr)):
        print("  [SKIP] No learning rate data")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)

    # Top: loss curve
    ax1.semilogy(
        epochs, loss_total, color="#1f77b4", linewidth=1.5, label=r"$\mathcal{L}$"
    )
    ax1.set_ylabel(r"Loss $\mathcal{L}$", fontsize=10)
    ax1.set_title("Training Dynamics", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlim((float(epochs[0]), float(epochs[-1])))

    # Bottom: learning rate
    valid_lr = np.isfinite(lr)
    ax2.plot(epochs[valid_lr], lr[valid_lr], color="#ff7f0e", linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Learning Rate", fontsize=10)
    ax2.set_yscale("log")

    fig.tight_layout()

    # Save PNG + PDF
    output_path_png = os.path.join(out_dir, "learning_curve.png")
    output_path_pdf = os.path.join(out_dir, "learning_curve.pdf")
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Learning curve: {output_path_png}")


# =============================================================================
# HTML Report Generation
# =============================================================================


def generate_html_report(records: Dict[str, List[Any]], out_dir: str) -> None:
    """
    Generate interactive HTML report with training statistics.

    Args:
        records: Parsed training records
        out_dir: Output directory for saving report
    """
    print("\n[*] Generating interactive HTML report...")

    from src.dashboard.monitor.performance_metrics import summarize_tail

    epochs = np.array(records["epoch"])
    loss_total = np.array(records["loss_total"])

    best_idx = np.argmin(loss_total)
    final_loss = loss_total[-1]
    best_loss = loss_total[best_idx]
    best_epoch = epochs[best_idx]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EFD3D Training Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea20, #764ba220); padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .stat-label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #667eea; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .good {{ color: green; }}
        .warning {{ color: orange; }}
        .bad {{ color: red; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #999; font-size: 0.85em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>EFD3D Training Report</h1>
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">{len(epochs):,}</div><div class="stat-label">Total Epochs</div></div>
            <div class="stat-card"><div class="stat-value">{best_epoch:,}</div><div class="stat-label">Best Epoch</div></div>
            <div class="stat-card"><div class="stat-value">{final_loss:.2f}</div><div class="stat-label">Final Loss</div></div>
            <div class="stat-card"><div class="stat-value">{best_loss:.2f}</div><div class="stat-label">Best Loss</div></div>
        </div>
        <h2>Loss Components (Last 20 Epochs Median)</h2>
        <table><tr><th>Component</th><th>Value</th><th>Status</th></tr>
"""

    summary = summarize_tail(records, 20)
    for k, v in sorted(summary.items()):
        if k == "loss_total":
            continue
        status = "Good" if v < 0.5 else ("Medium" if v < 2.0 else "High")
        status_class = "good" if v < 0.5 else ("warning" if v < 2.0 else "bad")
        html_content += f"<tr><td>{k}</td><td>{v:.4f}</td><td class='{status_class}'>{status}</td></tr>\n"

    html_content += f"""</table>
        <div class="footer">
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
    </div>
</body>
</html>"""

    output_path = os.path.join(out_dir, "training_report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  [OK] HTML report: {output_path}")
