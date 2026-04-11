#!/usr/bin/env python3
"""
Log Parsing Module for Training Log Analysis

Provides functions for finding, parsing, and exporting training log data.

Functions:
    - find_log_path(): Locate training log files
    - parse_training_log(): Parse log and extract metrics
    - save_csv(): Export parsed data to CSV

Usage:
    from src.dashboard.monitor.log_parsing import find_log_path, parse_training_log, save_csv

    log_path, out_dir = find_log_path(model_dir)
    records = parse_training_log(log_path)
    save_csv(records, csv_path)
"""

import csv
import glob
import os
import re
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


# =============================================================================
# Log Parsing Patterns (Module-level for reuse)
# =============================================================================

TAG_MAP = {
    "LV": "low_voltage",
    "Vol": "volume_conservation",
    "θ": "contact_angle",
    "ηE": "eta_stage1_early",
    "φS": "phi_spatial",
    "C": "continuity",
    "IF": "interface",
    "ST": "surface_tension",
    "NS": "navier_stokes",
    "VOF": "vof",
}

PATTERN_MAIN = re.compile(
    r"Epoch\s+(\d+)\s+(?:\[S(\d)\]\s+)?\|\s+Loss:\s+([0-9.eE+\-]+)(.*)"
)
PATTERN_LR = re.compile(r"LR:\s*([0-9.eE+\-]+)")


# =============================================================================
# Log File Discovery
# =============================================================================


def find_log_path(model_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Find training log file.

    Args:
        model_dir: Specified model directory. If None, finds the latest.

    Returns:
        Tuple of (log_path, output_directory)

    Raises:
        FileNotFoundError: No training.log found

    Examples:
        >>> log_path, out_dir = find_log_path()
        >>> log_path, out_dir = find_log_path("outputs/train/pinn_20240101")
    """
    if model_dir is not None:
        log_path = os.path.join(model_dir, "training.log")
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"training.log not found in {model_dir}")
        return log_path, model_dir

    # Search for latest output directory
    # Supports both new and old directory structures:
    # - New: outputs/train/pinn_*
    # - Old: outputs_pinn_*
    # - LSTM: outputs_lstm_*

    # Try new structure first
    output_dirs = sorted(glob.glob("outputs/train/pinn_*"), reverse=True)
    for d in output_dirs:
        log_path = os.path.join(d, "training.log")
        if os.path.exists(log_path):
            return log_path, d

    # Try old structure
    output_dirs = sorted(glob.glob("outputs_pinn_*"), reverse=True)
    for d in output_dirs:
        log_path = os.path.join(d, "training.log")
        if os.path.exists(log_path):
            return log_path, d

    # Try LSTM structure
    lstm_dirs = sorted(glob.glob("outputs_lstm_*"), reverse=True)
    for d in lstm_dirs:
        log_path = os.path.join(d, "training.log")
        if os.path.exists(log_path):
            return log_path, d

    raise FileNotFoundError(
        "No training.log found in outputs/train/pinn_* or outputs_pinn_*"
    )


# =============================================================================
# Log Parsing
# =============================================================================


def parse_training_log(log_path: str) -> Dict[str, List[Any]]:
    """
    Parse training log file.

    Supports multiple log formats:
    - Epoch X [S1] | Loss: X.XX LV: X.XX Vol: X.XX ...
    - Epoch X | Loss: X.XX LR: X.XX

    Args:
        log_path: Path to log file

    Returns:
        Dictionary with parsed data, each key maps to a list

    Raises:
        RuntimeError: Unable to parse log file

    Examples:
        >>> records = parse_training_log("outputs/train/pinn_*/training.log")
        >>> records["epoch"]  # [1, 2, 3, ...]
        >>> records["loss_total"]  # [100.5, 95.2, ...]
    """
    pattern_main = PATTERN_MAIN
    pattern_lr = PATTERN_LR
    tag_map = TAG_MAP

    records: Dict[str, List[Any]] = {
        "epoch": [],
        "stage": [],
        "loss_total": [],
        "lr": [],
    }

    # Initialize all loss components
    for name in tag_map.values():
        records[name] = []

    line_count = 0
    matched_count = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            # Skip lines without loss information
            if "Loss:" not in line or "Epoch" not in line:
                continue

            m = pattern_main.search(line)
            if not m:
                continue

            matched_count += 1
            epoch = int(m.group(1))
            stage = int(m.group(2)) if m.group(2) else 1
            total_loss = float(m.group(3))
            tail = m.group(4) if m.group(4) else ""

            records["epoch"].append(epoch)
            records["stage"].append(stage)
            records["loss_total"].append(total_loss)

            # Extract learning rate
            lr_match = pattern_lr.search(tail)
            if lr_match:
                records["lr"].append(float(lr_match.group(1)))
            else:
                records["lr"].append(np.nan)

            # Extract loss components
            for tag, name in tag_map.items():
                pattern_tag = re.compile(rf"{re.escape(tag)}:\s*([0-9.eE+\-]+)")
                m_tag = pattern_tag.search(tail)
                if m_tag:
                    records[name].append(float(m_tag.group(1)))
                else:
                    records[name].append(np.nan)

    if matched_count == 0:
        raise RuntimeError(
            f"No valid epoch lines found in {log_path}\n"
            f"Total lines read: {line_count}\n"
            f"Expected format: 'Epoch X [S1] | Loss: X.XX ...'"
        )

    print(f"✅ Parsed {matched_count} training records (of {line_count} lines)")

    # Sort by epoch
    epochs = np.array(records["epoch"])
    order = np.argsort(epochs)
    for k, v in records.items():
        arr = np.array(v)
        records[k] = arr[order].tolist()

    return records


# =============================================================================
# CSV Export
# =============================================================================


def save_csv(records: Dict[str, List[Any]], csv_path: str) -> None:
    """
    Save parsed results to CSV file.

    Filters out columns that are entirely NaN.

    Args:
        records: Parsed data records
        csv_path: Output CSV file path

    Examples:
        >>> save_csv(records, "outputs/analysis/loss_breakdown.csv")
    """
    keys = list(records.keys())
    # Guard against empty records
    if not records or not keys:
        print("⚠️ Warning: records is empty, skipping CSV save")
        return
    length = len(records[keys[0]])

    # Filter out columns that are entirely NaN
    valid_keys = []
    for k in keys:
        values = np.array(records[k], dtype=float)
        if not np.all(np.isnan(values)):
            valid_keys.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(valid_keys)
        for i in range(length):
            row = [records[k][i] for k in valid_keys]
            writer.writerow(row)

    print(f"✅ CSV report saved: {csv_path}")
