# Supplementary Materials

## EFD3D: A Physics-Informed Neural Network for Electrowetting Display Pixel Simulation

**Manuscript ID**: EFD-2026-001

---

### Appendix A: Implementation Details

#### A.1. Network Architecture

| Network | Layers | Activation | Output |
|---------|--------|------------|--------|
| $\phi$-net | 6→128→128→64→32→1 | Tanh + Sigmoid | $\phi \in [0,1]$ |
| Velocity-net | 7→64→64→32→4 | Tanh (linear final) | $(u, v, w, p)$ |

Total parameters: 34,661 ($\phi$-net: 27,777 + vel-net: 6,884)

#### A.2. Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Total epochs | 60,000 | Three-stage curriculum |
| Stage 1 epochs | 1,500 | Geometry fitting |
| Stage 2 epochs | 2,500 | Kinematics (1,500--4,000) |
| Stage 3 epochs | 56,000 | Full dynamics (4,000--60,000) |
| L-BFGS iterations | 3,000 | Fine-tuning |
| Batch size | 4,096 | Collocation points |
| Learning rate (Adam) | $3\times10^{-4}$ | Initial |
| Min learning rate | $1\times10^{-6}$ | After scheduling |
| Optimizer | Adam + L-BFGS | Hybrid optimization |

### Appendix B: Code Availability

The source code and trained models are available at:

- Main framework: https://github.com/lsm-428/EFD-PINNs

The framework is implemented in PyTorch 2.0+ with Python 3.12 and requires CUDA 11.8+ for GPU acceleration. Installation instructions and examples are provided in the repository README.
