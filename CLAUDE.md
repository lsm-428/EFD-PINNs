# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🚀 Quick Start Commands

### Training
```bash
# Train with recommended configuration
uv run train_two_phase.py --config config/v4.5-standard.json

# Resume training from checkpoint
uv run train_two_phase.py --config config/v4.5-standard.json --resume_from outputs/train/pinn_YYYYMMDD_HHMMSS/best_model.pth
```

### Evaluation & Visualization
```bash
# Run evaluation and generate visualizations
uv run evaluate.py outputs/train/pinn_YYYYMMDD_HHMMSS/

# Launch interactive dashboard
uv run scripts/dashboard.py

# Run ablation study
uv run scripts/run_ablation.sh
```

### Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test module
uv run pytest tests/test_pinn_complete.py -v

# Run tests with detailed output
uv run pytest tests/ -v --tb=short

# Generate coverage report
uv run pytest tests/ --cov=src --cov-report=html
```

---

## 🏗️ Project Architecture

EFD3D is an industrial-grade Physics-Informed Neural Network (PINN) framework for 3D two-phase flow simulation in electrowetting applications.

### Core Architecture
- **Two-Stage Design**: Stage 1 (analytical contact angle model) + Stage 2 (PINN for full flow field)
- **6D Triad Input**: `[x, y, z, V_from, V_to, t_since]` - enables arbitrary voltage sequence simulation
- **Physics Constraints**: Navier-Stokes equations + VOF interface tracking + electrowetting forces
- **Hybrid Training**: Geometry → Kinematics → Full Physics stages

### Key Directories
```
EFD3D/
├── src/                           # Core source code (21 modules)
│   ├── models/                   # Neural network models
│   │   ├── pinn_two_phase.py     # Main PINN model (TwoPhasePINN + Trainer)
│   │   └── aperture_model.py     # Stage 1 analytical model
│   ├── physics/                  # Physics constraint engine
│   │   └── constraints.py        # Navier-Stokes, VOF, continuity equations
│   ├── training/                 # Training infrastructure
│   │   ├── scheduler.py          # Dynamic loss weight scheduling
│   │   ├── stabilizer.py         # NaN recovery, gradient clipping
│   │   └── components.py         # Training utilities
│   ├── config/                   # Configuration management
│   │   ├── __init__.py           # PHYSICS parameters export
│   │   └── physics_config.py     # Type-safe configuration
│   └── utils/                    # Utilities
│
├── config/                       # Training configurations
│   ├── v4.5-standard.json        # Recommended config (proven convergence)
│   └── device_calibrated_physics.json # Physics calibration
│
├── scripts/                      # User-facing tools
│   ├── dashboard.py              # Streamlit interactive dashboard
│   └── run_ablation.sh           # Ablation study script
│
├── tests/                        # Test suite (15 modules)
├── outputs/                      # Training outputs
│   └── train/                    # Model checkpoints and logs
│
└── docs/                         # Documentation
```

### Critical File Dependencies
- **Entry Points**: `train_two_phase.py`, `evaluate.py`, `scripts/dashboard.py`
- **Core Model**: `src/models/pinn_two_phase.py` (TwoPhasePINN class)
- **Physics Engine**: `src/physics/constraints.py` (PhysicsConstraints class)
- **Configuration**: `src/config/__init__.py` (PHYSICS parameters), `src/config/physics_config.py`
- **Training Control**: `src/training/scheduler.py`, `src/training/stabilizer.py`
- **Predictors**: `src/predictors/hybrid_predictor.py`, `src/predictors/pinn_aperture.py`

---

## ⚙️ Configuration System

### Physics Parameters
All physics parameters are centrally managed via the `PHYSICS` dictionary:
```python
from src.config import PHYSICS

# Device geometry
Lx = PHYSICS["Lx"]          # Pixel width: 174μm
dielectric_thickness = PHYSICS["dielectric_thickness"]  # 400nm

# Material properties
gamma = PHYSICS["gamma"]    # Surface tension: 0.015 N/m
rho_ink = PHYSICS["rho_ink"] # Ink density: 1000 kg/m³

# Initial conditions
theta0 = PHYSICS["theta0"]  # Initial contact angle: 120°
```

### Training Configuration
Primary config: `config/v4.5-standard.json`

Key parameters:
- `training.epochs`: Total training epochs
- `training.stage1_epochs`: Geometry stage epochs
- `training.stage2_epochs`: Kinematics stage epochs
- `physics.vof_weight`: VOF transport equation weight
- `physics.sharpening`: Interface sharpening coefficient

---

## 🧪 Testing Strategy

### Test Organization
```
tests/
├── test_pinn_complete.py              # End-to-end PINN training pipeline
├── test_physics_sanity.py             # Physics validation checks
├── test_vof_transport.py              # VOF equation verification
├── test_hybrid_predictor.py           # Stage 1+2 integration
├── test_enhanced_aperture_properties.py # Stage 1 model validation
├── test_dynamic_weights.py            # Loss weight scheduling
├── test_flow_solver_properties.py     # CFD solver verification
├── test_two_phase_data_generator.py   # Data generation tests
├── test_vof_3d.py                     # 3D VOF implementation
├── test_curvature_computation.py      # Curvature calculations
├── test_vof_sensitivity.py            # VOF sensitivity analysis
├── test_model_dimensions.py           # Model architecture validation
├── test_scripts_framework.py          # Script functionality
├── test_3d_visualization_properties.py # 3D visualization
└── test_code_changes.py               # Code modification tracking
```

### Common Test Patterns
- **Physics Validation**: Verify governing equation residuals < 1e-3
- **Conservation Laws**: Mass/volume conservation error < 1%
- **Boundary Conditions**: No-slip walls, interface continuity
- **Stage Integration**: Stage 1 → Stage 2 compatibility

---

## 📊 Development Workflows

### Training Pipeline
1. **Configuration**: Select/config training parameters in `config/`
2. **Training**: Run `train_two_phase.py` with chosen config
3. **Monitoring**: Use dashboard for real-time visualization
4. **Evaluation**: Run `evaluate.py` for detailed analysis
5. **Testing**: Verify with `test_pinn_complete.py`

### Model Development
1. **Physics Changes**: Modify `src/physics/constraints.py`
2. **Architecture Changes**: Update `src/models/pinn_two_phase.py`
3. **Training Logic**: Adjust `src/training/` components
4. **Configuration**: Update `config/v4.5-standard.json`
5. **Testing**: Add/modify tests in `tests/`

### Debugging Strategy
- **Training Instability**: Check `TrainingStabilizer` logs for NaN recovery
- **Physics Violations**: Use `test_physics_sanity.py` for equation validation
- **Performance Issues**: Profile with dashboard performance metrics
- **Interface Problems**: Validate with `test_vof_transport.py`

---

## 🔍 Key Code Navigation Tips

### Finding Physics Implementation
- Navier-Stokes: `src/physics/constraints.py` lines 120-180
- VOF Transport: `src/physics/constraints.py` lines 200-250
- Electrowetting Forces: `src/physics/constraints.py` lines 300-350

### Training Pipeline Flow
1. `train_two_phase.py` → `main()`
2. `src/models/pinn_two_phase.py` → `Trainer.__init__()`
3. `src/models/pinn_two_phase.py` → `Trainer.train()`
4. `src/training/scheduler.py` → `DynamicPhysicsWeightScheduler`

### Model Architecture
- Input Processing: `TwoPhasePINN.forward()` lines 150-200
- Dual-Branch MLP: `TwoPhasePINN.__init__()` lines 80-120
- Loss Computation: `PhysicsConstraints.compute_all_losses()`

---

## 📚 Documentation References

For detailed information, consult:
- **Quick Start**: `docs/guides/quickstart.md`
- **Physics Theory**: `docs/guides/physics_and_device_guide.md`
- **Configuration**: `docs/guides/configuration_guide.md`
- **Training Guide**: `docs/guides/training_guide.md`
- **API Reference**: `docs/api/README.md`

---

## 🛠️ Development Tools

### Environment Management
- **Package Manager**: `uv` (see `pyproject.toml`)
- **Python Version**: 3.12-3.13
- **GPU Acceleration**: CUDA 11.8 (PyTorch 2.7.1)

### Code Quality
- **Linting**: `ruff` (configured in `pyproject.toml`)
- **Formatting**: `black` (line length 88)
- **Testing**: `pytest` + `hypothesis`

### Output Structure
Training outputs are organized as:
```
outputs/train/pinn_YYYYMMDD_HHMMSS/
├── best_model.pth               # Best model weights
├── best_model_epoch_XXXXX.pth   # Epoch-specific checkpoints
├── training.log                 # Training progress log
├── interface_3d_steady.png      # 3D interface visualization
├── training_curve.png           # Loss curves
└── config.json                  # Training configuration snapshot
```

---

*Last updated: 2026-04-13 | Version: v4.5*
