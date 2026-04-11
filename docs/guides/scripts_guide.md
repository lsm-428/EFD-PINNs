# EFD3D Scripts Guide

**Last Updated**: 2026-03-12  
**Version**: v4.5  
**Total Scripts**: 1 main script + visualizer_3d module

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Current Scripts](#current-scripts)
4. [Dashboard Usage](#dashboard-usage)
5. [Visualizer 3D Module](#visualizer-3d-module)
6. [Removed Scripts](#removed-scripts)
7. [Usage Patterns](#usage-patterns)
8. [Output Reference](#output-reference)

---

## Overview

The EFD3D scripts module has been streamlined. Most CLI tools have been integrated into the interactive Streamlit Dashboard, providing a unified interface for all analysis tasks.

### Current Script Organization

```
scripts/
├── dashboard.py           # Streamlit Dashboard (main entry point)
│
└── visualizer_3d/         # 3D visualization module
    ├── visualizer_3d.py          # Main entry
    ├── visualization_core.py     # Core rendering
    ├── animation_utils.py        # Animation utilities
    └── interactive_features.py   # Interactive features
```

### Key Features

- **Streamlit Dashboard**: Interactive web UI consolidating most analysis tools
- **Auto-Detection**: Tools auto-detect latest models when paths not specified
- **Publication Quality**: IEEE-standard figures available via notebooks scripts
- **3D Visualization**: PyVista-based professional rendering

---

## Quick Start

### Basic Setup

```bash
# Activate environment
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows

# Navigate to project root
cd /home/scnu/Gitee/EFD3D
```

### Recommended Workflow

```bash
# Launch Dashboard (unified interface for all analysis)
streamlit run scripts/dashboard.py
```

The Dashboard provides access to:
- Training log analysis
- Model comparison
- Flow field visualization
- Performance benchmarking
- Stage 1 analytical demos

---

## Current Scripts

### 1. `dashboard.py` - Interactive Streamlit Dashboard

**Purpose**: Interactive web-based visualization and analysis interface

**File**: `scripts/dashboard.py`  
**Type**: Main entry point (Streamlit app)

#### Features

- Real-time model loading and inference
- Interactive parameter controls (voltage, time, z-plane)
- Live visualization updates
- Model comparison tools
- Downloadable figures

#### Tab Structure

| Tab | Functionality |
|-----|--------------|
| 🔬 Field Analysis | Flow field visualization (φ, velocity, pressure) |
| ⏱️ Performance | Performance benchmarking and profiling |
| 🔄 Compare | Model comparison tools |
| 📐 Stage 1 | Stage 1 analytical model demos |
| 📊 Training | Training log analysis and visualization |

#### Usage

```bash
# Standard launch (recommended)
streamlit run scripts/dashboard.py

# Custom port and host
streamlit run scripts/dashboard.py --port 8502 --server.address 0.0.0.0

# Headless mode (for remote access)
streamlit run scripts/dashboard.py --server.headless true
```

---

### 2. `visualizer_3d/` - 3D Visualization Module

**Purpose**: Professional 3D visualization of PINN predictions

**Location**: `scripts/visualizer_3d/`

#### Modules

| File | Description |
|------|------------|
| `visualizer_3d.py` | Main entry point with PixelVisualizer class |
| `visualization_core.py` | Core rendering functions |
| `animation_utils.py` | Animation generation utilities |
| `interactive_features.py` | Interactive visualization features |

#### Features

- 3D φ field rendering (isosurfaces, volumes)
- Interactive visualization with PyVista
- Animation generation
- Multiple visualization modes

#### Usage

```bash
# Run demo visualization
python -m scripts.visualizer_3d.visualizer_3d --demo

# Interactive 3D exploration
python -m scripts.visualizer_3d.visualizer_3d --interactive
```

#### Programmatic Usage

```python
from scripts.visualizer_3d.visualizer_3d import PixelVisualizer

viz = PixelVisualizer(model_path='outputs/train/pinn_*/best_model.pth')
viz.plot_aperture_3d(voltage=20, time=0.01)
viz.save('aperture_3d.png')
```

---

## Removed Scripts

The following scripts have been removed as their functionality is now available in the Dashboard or moved to the notebooks directory:

| Removed Script | Replacement |
|----------------|------------|
| `benchmark.py` | Dashboard tab: ⏱️ Performance |
| `compare.py` | Dashboard tab: 🔄 Compare |
| `analyze_log.py` | Dashboard tab: 📊 Training |
| `analyze_flow_field.py` | Dashboard tab: 🔬 Field Analysis |
| `stage1_demo.py` | Dashboard tab: 📐 Stage 1 |
| `generate.py` | (Removed - use Dashboard export features) |
| `ieee_figures.py` | (Removed - use Dashboard export features) |
| `verify_parameters.py` | (Simple utility, no replacement) |
| `test_all.py` | `pytest tests/` |
| `cli_utils.py` | (Inlined into scripts) |
| `constants.py` | (Inlined into scripts) |

### Migration

For all removed scripts, use the Dashboard as the primary interface:

```bash
streamlit run scripts/dashboard.py
```

For publication figure generation, use the Dashboard export features (available in the relevant tabs).

---

## Usage Patterns

### Pattern 1: Interactive Analysis

```bash
# Launch Dashboard for all analysis tasks
streamlit run scripts/dashboard.py
```

Navigate to the appropriate tab:
- 🔬 Field Analysis: Flow field visualization
- ⏱️ Performance: Benchmarking
- 🔄 Compare: Model comparison
- 📐 Stage 1: Analytical model demos
- 📊 Training: Training log analysis

### Pattern 2: 3D Visualization

```bash
# Run demo visualization
python -m scripts.visualizer_3d.visualizer_3d --demo

# Interactive 3D exploration
python -m scripts.visualizer_3d.visualizer_3d --interactive
```

### Pattern 3: Publication Figure Generation

```bash
# Generate publication figures using Dashboard export features
streamlit run scripts/dashboard.py
# Navigate to the desired tab and use the export/download options
```

### Pattern 4: Training Workflow

```bash
# 1. Train model
python train_two_phase.py --config config/v4.5-standard.json

# 2. Analyze via Dashboard
streamlit run scripts/dashboard.py  # Use 📊 Training tab

# 3. Generate publication figures using Dashboard export features
streamlit run scripts/dashboard.py  # Export from desired tabs
```

---

## Output Reference

### Dashboard Outputs

The Dashboard generates interactive visualizations in the browser. You can download figures as PNG from the UI.

### 3D Visualization Output

```
outputs/visualizations/   # visualizer_3d module
├── aperture_3d.png
├── phi_iso_surface.png
└── velocity_vectors_3d.png
```

### Using --output with Dashboard

The Dashboard provides export options in each tab for saving visualizations to custom locations.

---

*For the most current usage, run `streamlit run scripts/dashboard.py --help`.*

**Related Documentation**:
- [Usage Guide](usage.md) - General usage and quick start
- [Visualization Guide](visualization_guide.md) - Detailed visualization options
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [API Reference](../api/README.md) - Programmatic API documentation
