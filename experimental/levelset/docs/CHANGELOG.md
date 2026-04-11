# Level Set 3D PINN Version History

This document chronicles the development, debugging, and optimization journey of the Level Set 3D PINN model from its inception (Baseline) to the current high-precision version (v5.5).

## v5.5: High-Precision Model (2026-01-24)
**Status:** 🏆 Highly Successful (Best Performing)

**Key Improvements:**
- **Data Scale**: Increased training data from 300k to **750k** samples (2.5x).
- **Voltage Density**: Increased sampling from 4 discrete points to **31 continuous points**.
- **Batch Size**: Increased to **4096** (128x larger), significantly improving gradient stability.
- **Constraints**: Full integration of Volume Conservation, Spatial Distribution, and Monotonicity constraints.

**Performance:**
- **30V Aperture**: **74.12%** (Significant improvement over v5.3's 68.58%).
- **Prediction Accuracy (MAE)**: **7.91%** (22% improvement).
- **Volume Error**: < 0.1% (Excellent physical conservation).
- **Loss**: Total loss reduced by 57% compared to v5.3.

---

## v5.4: Constraint Enhancement (2026-01-24)
**Status:** Feature Addition

**Key Changes:**
- **Volume Conservation (`volume_conservation`)**: Added constraint to prevent non-physical ink loss/gain.
- **Spatial Distribution (`psi_spatial`)**: Added variance/mean constraints to prevent mode collapse of the $\psi$ field.
- **Monotonicity (`aperture_monotonicity`)**: Added constraint to enforce strictly increasing aperture with voltage.
- **Strategy**: Ported successful constraints from the VOF method to Level Set.

---

## v5.3: Progressive Physics (2026-01-24)
**Status:** Stable Baseline

**Key Changes:**
- **Progressive Training**: Implemented a staged approach for `levelset_transport` weight ($10^{-8} \rightarrow 10^{-5} \rightarrow 10^{-3}$).
- **Stability**: Successfully balanced data loss and physics loss, eliminating training crashes.

**Performance:**
- **Symbol Accuracy**: Ink<0 reached ~88.3%.
- **30V Aperture**: ~68.58% (First version to achieve reasonable opening).

---

## v5.2: Critical Data Fixes (2026-01-23)
**Status:** Bug Fix

**Key Fixes:**
1.  **Normalization Bug**: Fixed `normalize_psi` scale from $5\mu m$ to $2\mu m$. Previously, small physical $\psi$ values were incorrectly normalized, causing high-voltage polar samples to be labeled as ink.
2.  **Sampling Range**: Expanded polar fluid sampling from `r_open * 0.8` to the entire pixel. Fixed the issue where the model learned "polar fluid = small center circle".

**Impact:**
- 30V Polar sample accuracy improved from 17.1% to **100%**.

---

## v5.1: Scientific Stabilization (2026-01-23)
**Status:** Stability Fix

**Key Changes:**
- **Disable Instability Source**: Completely disabled `levelset_transport` (weight=0.0) which was causing NaN errors.
- **Data Balancing**: Adjusted Z-axis sampling to be less concentrated at Z=0 (95% $\rightarrow$ 67%).
- **Conservative Hyperparameters**: Reduced learning rate ($10^{-4} \rightarrow 5 \times 10^{-5}$) and weights.

**Result:**
- Achieved stable training without NaN.
- **Limitation**: 30V aperture was very low (~15%) due to the data issues identified later in v5.2.

---

## v5.0: Initial Scientific Attempt (2026-01-19)
**Status:** ❌ Failed (Crashed)

**Issues:**
- **Instability**: Hardcoded `levelset_transport` weight (1.0-2.0) caused gradient explosion and NaN loss at Epoch 25400.
- **Sampling Bias**: Data overly concentrated at Z=0 (95%).
- **Symbol Constraint**: Incorrect implementation (amplitude term created a "gravity well").

---

## v2.0: Stable Training (2026-01-18)
**Status:** Stability Optimization

**Key Changes:**
- **5-Stage Training**: Expanded from 3 to 5 stages for smoother transition.
- **Electrostatic Warm-up**: Implemented gradual ramp-up of electrostatic weight ($0.001 \rightarrow 0.002 \rightarrow 0.005$) to prevent instability.
- **Batch Size**: Increased from 32 to 256.
- **Scheduler**: Switched to `MultiStepLR` to reduce learning rate at stage transitions.

**Configuration**: `stable_training_v2.json`

---

## v1.0: First Physics Implementation (Fix v1)
**Status:** Feature Implementation

**Key Changes:**
- **Physics Introduction**: First version to introduce `contact_angle` and `electrostatic` forces (enabled in Stage 2).
- **Sign Constraint**: Introduced `sign_constraint` to enforce Ink/Polar phase separation.
- **3-Stage Training**: Data Fitting $\rightarrow$ Physics with Contact $\rightarrow$ Refinement.

**Configuration**: `fix_v1.json`

---

## Baseline: Initial Framework
**Status:** Proof of Concept

**Characteristics:**
- **Pure Fluid Dynamics**: No electrostatic forces (weight = 0.0).
- **Basic Stages**: 3 stages focusing on Data, Continuity, and Transport sequentially.
- **Goal**: Verify the Level Set formulation without complex electrowetting physics.

**Configuration**: `baseline.json`

---

## Summary of Evolution

| Version | Key Characteristic | 30V Aperture | Stability | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline**| Initial Framework | N/A | ✅ Stable | No Physics |
| **v1.0** | First Physics | Low | ⚠️ Unstable | Feature Test |
| **v2.0** | Stable Training | Low | ✅ Stable | Optimization |
| **v5.0** | Scientific Reboot | N/A (Crashed) | ❌ NaN | Failed |
| **v5.1** | Stability Fix | ~15% | ✅ Stable | Stable but Inaccurate |
| **v5.2** | Data Fix | N/A | ✅ Stable | Corrected Data Labels |
| **v5.3** | Progressive Physics | 68.58% | ✅ Stable | Good Baseline |
| **v5.4** | Added Constraints | N/A | ✅ Stable | Feature Complete |
| **v5.5** | **Scale & Precision** | **74.12%** | ✅ Stable | **Best Model** |
