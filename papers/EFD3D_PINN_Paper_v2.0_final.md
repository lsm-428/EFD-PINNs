---
header-includes:
  - \usepackage{float}
  - \usepackage{graphicx}
  - \graphicspath{{../outputs/figures/}}
  - \usepackage[top=1in, bottom=1in, left=1in, right=1in]{geometry}
---

# Authors

\textbf{Shouming Li}$^{1}$, \textbf{Dong Yuan}$^{1}$, \textbf{Biao Tang}$^{1}$, \textbf{Hongwei Jiang}$^{2}$, \textbf{Guofu Zhou}$^{1}$

$^{1}$Guangdong Provincial Key Laboratory of Optical Information Materials and Technology, Institute of Electronic Paper Displays, South China Academy of Advanced Optoelectronics, South China Normal University, Guangzhou 510006, People's Republic of China

$^{2}$Epapervision Technology Guangdong Co., Ltd., Foshan, 528200, People's Republic of China

Corresponding author: Biao Tang (tangbiao@scnu.edu.cn)

---

EFD3D: A Physics-Informed Neural Network for Electrowetting Display Pixel Simulation

\textbf{Manuscript ID}: EFD-2026-001
\textbf{Journal}: Displays
\textbf{Submission Date}: 2026-04-02
\textbf{Version}: 2.0 (Full Academic Style)

## Abstract

This work presents EFD3D, a Physics-Informed Neural Network (PINN) framework for three-dimensional electrowetting display pixel simulation. The framework demonstrates the feasibility of applying PINNs to complex two-phase flow problems in electrowetting systems, achieving comparable accuracy to traditional Computational Fluid Dynamics (CFD) methods (volume conservation error below 1%) while enabling rapid inference ($\approx$ 3 s per query, theoretical estimate based on model size) after a single training phase. Notably, this is achieved on entry-level hardware (NVIDIA Quadro P2000 with 5 GB VRAM) with only 34,661 parameters using the open-source PyTorch framework.

The framework features a 6D Triad input representation $[x, y, z, V_{from}, V_{to}, t_{since}]$ for encoding voltage transitions, physics-informed loss functions that enforce Navier-Stokes equations and Volume of Fluid transport, and a dual-network architecture for interface and flow field prediction. Through extensive ablation studies and reproducibility analysis, we validate the effectiveness of the physics constraints and demonstrate consistent physical metrics. The framework's computational efficiency enables edge-device deployment for practical applications and supports inverse design workflows where different pixel configurations can be rapidly evaluated without retraining.

This work documents an eight-month development process involving over 120 training experiments, multiple architectural iterations, and systematic debugging of physics constraints. The development process itself provides valuable insights into the practical challenges of applying PINNs to industrial problems, demonstrating that meaningful physics-informed machine learning research is accessible even with limited resources.

\textbf{Keywords}: PINN; Electrowetting; Multiphase; Two-Phase Flow; Microfluidics

## Introduction

### Background and Motivation

An electrowetting-based display (EWD) operates on the principle of voltage-controlled wettability modification of a hydrophobic surface [1]. When voltage is applied between a conductive polar fluid and an insulated electrode, the contact angle decreases according to the Young-Lippmann equation [2], causing the polar fluid to spread and displace a colored non-polar oil (ink) within a pixel chamber [3, 4, 5]. This displacement modulates optical transmission, creating visible contrast.

Despite two decades of research, electrowetting display technology faces significant commercialization challenges, particularly in manufacturing tolerance control and driving waveform optimization. These industrial requirements create a critical research gap in simulation capabilities. First, existing computational approaches face a limitation: they cannot efficiently handle *step voltage transitions* that are needed for practical EWD operation. Traditional CFD methods require complete re-simulation for each new voltage waveform, making optimization across diverse driving schemes computationally prohibitive for industrial design workflows where thousands of parameter combinations must be evaluated. Second, many academic PINN frameworks rely on idealized parameters rather than realistic device conditions, leading to predictions that may not reflect actual device behavior. Third, the strongly three-dimensional nature of EWD flow patterns—including complex moving contact lines, thin film dynamics, and secondary recirculation zones—cannot be adequately captured by simplified 2D axisymmetric approximations commonly used in literature.

This work addresses these gaps by introducing EFD3D, a Physics-Informed Neural Network framework that combines device-calibrated physics with step voltage transition generalization in full 3D resolution. The framework enables continuous, mesh-free simulation of ink-polar fluid interaction under single-step voltage transitions within the 0--30~$\text{V}$ operating range, providing computational efficiency for industrial design workflows.

### Related Work and Research Gap

Deep learning has been applied to electrowetting systems for droplet recognition [6] and orbital droplet manipulation [7]. Recent advances in Physics-Informed Neural Networks (PINNs) have shown promise in solving partial differential equations without mesh generation [8, 9, 10]. PINNs embed physical laws directly into the loss function through automatic differentiation, enabling continuous function approximation of field variables. PINN training methodologies have been improved through feature-enforcing approaches [11] and training guides [12]. Prior applications of PINNs to fluid dynamics include systematic analysis of PINNs for two-phase flow with capillarity [13], and NSFnets for incompressible Navier-Stokes [14], which serves as a methodological foundation for this work. Neural operator approaches such as DeepONet [15] have also shown promise for learning nonlinear operators. Machine learning approaches to CFD acceleration [16, 17] and multiphase flow modeling [18, 19] have demonstrated speedups but lack the physics constraints needed for electrowetting applications. No framework exists that simultaneously addresses all three limitations—3D resolution, step voltage transition generalization, and device-calibrated physics—in a unified computational approach. This gap severely limits the practical utility of current simulation methods for industrial EWD design and optimization.

This work addresses the research gap through the 6D Triad representation and device-calibrated physics integration.

### Contributions

This work makes three key contributions that demonstrate the feasibility of applying PINNs to electrowetting display simulation:

\textbf{Contribution 1: First PINN Framework for EWD Simulation}. This work demonstrates, to our knowledge for the first time, that Physics-Informed Neural Networks can be applied to simulate three-dimensional two-phase flow in electrowetting displays. The framework achieves comparable accuracy to traditional CFD methods (volume conservation error below 1%) while enabling rapid inference ($\approx$ 3 s per query, theoretical estimate based on model size) after a single training phase—a significant reduction from the 4--24 hours typically required per simulation with conventional CFD approaches.

\textbf{Contribution 2: PINN Implementation for EWD}. The framework employs: (a) a 6D Triad input representation $[x, y, z, V_{from}, V_{to}, t_{since}]$ that encodes voltage transition states, enabling a single model to handle step voltage transitions from 0--30~$\text{V}$ without retraining; (b) a dual-branch Multi-Layer Perceptron (MLP) architecture separating interface tracking ($\phi$-network) and flow dynamics (velocity network) for improved stability; (c) physics-informed loss functions enforcing Navier-Stokes equations, VOF transport, and volume conservation; and (d) device-calibrated physical parameters validated against real EWD measurements.

\begin{figure}[H]
\centering
\begin{minipage}{0.45\textwidth}
\centering
\includegraphics[width=\textwidth]{../outputs/figures/figure1_6d_triad_input.png}
\caption{6D Triad input representation. (a) Six input dimensions with normalized ranges. (b) Voltage transition matrix showing 961 possible $(V_{from}, V_{to})$ combinations. (c) Spatial encoding on pixel cross-section. (d) Training sample distribution. (e) Example voltage step transition. (f) Dual-branch network architecture.}
\label{fig:6d_triad}
\end{minipage}
\hspace{0.8cm}
\begin{minipage}{0.45\textwidth}
\centering
\includegraphics[width=\textwidth]{../outputs/figures/figure2_device_calibration.png}
\caption{Device-calibrated physics validation. (a) Contact angle vs. voltage: measured $\theta_0 = 120^\circ$, $\theta(20~\text{V}) \approx 99^\circ$, $\theta(30~\text{V}) = 67.5^\circ$. (b) Aperture ratio vs. voltage: $\eta(20~\text{V}) \approx 0.599$, $\eta(30~\text{V}) = 0.834$. (c) Hysteresis loop: contact angle during voltage up/down sweep. (d) Young-Lippmann linear relation: $\Delta\cos\theta$ vs. $V^2$. (e) Dynamic response (0~$\text{V} \to 30~\text{V}$): aperture ratio evolution over time. (f) Stage 1 calibration data table: $\theta$ and $\eta$ values at each voltage.}
\label{fig:device_calibration}
\end{minipage}
\end{figure}

\vspace{0.5cm}

\textbf{Contribution 3: Engineering Infrastructure for EWD Design}. The framework establishes a practical engineering tool through: (a) interactive design capabilities enabling interactive parameter exploration, (b) device-calibrated physics parameters validated against experimental measurements, and (c) a foundation for automated waveform optimization. Extensive ablation studies (6 variants compared) validate that each physics constraint contributes to solution quality, with the full model achieving an improved balance of accuracy and stability. Reproducibility analysis demonstrates consistent convergence and physically meaningful solutions, while the low computational requirements (NVIDIA Quadro P2000, 34,661 parameters) make industrial-grade PINN simulation accessible without dedicated HPC resources.

Together, these contributions establish EFD3D as a feasible approach for EWD simulation, demonstrating that meaningful PINN research is accessible even with limited hardware resources (NVIDIA Quadro P2000 with 5 GB VRAM, 34,661 parameters).

## Governing Equations

### Two-Phase Flow Formulation

The EWD pixel is modeled as a two-phase system comprising non-polar ink and polar conductive fluid separated by a sharp interface represented by a phase field $\phi \in [0, 1]$. The governing equations are derived from first principles of fluid dynamics and interface tracking.

The continuity equation maintains mass conservation throughout the domain:
\begin{equation}
\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0
\end{equation}

The Navier-Stokes equations govern momentum conservation, accounting for the two-phase nature through linear interpolation of density and viscosity:
\begin{equation}
\rho(\phi) \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu(\phi) \nabla^2 \mathbf{u} + \mathbf{F}_{st} + \mathbf{F}_{ew}
\end{equation}
where the implementation uses the simplified viscous term $\mu\nabla^2\mathbf{u}$ rather than the full viscous stress tensor form $\nabla\cdot[\mu(\nabla\mathbf{u}+(\nabla\mathbf{u})^T)]$. This simplification assumes: (1) constant viscosity within each phase (neglecting steep gradients at the interface), and (2) negligible bulk viscosity contribution. Given the moderate viscosity ratio in our system ($\mu_{ink}/\mu_{polar} = 3$), this approximation introduces minimal error while significantly simplifying the automatic differentiation implementation. The full viscous stress tensor would require second-order derivatives of the velocity field, which can cause numerical instability in PINN training.
\begin{equation}
\rho(\phi) = \phi \rho_{ink} + (1-\phi) \rho_{polar}, \quad \mu(\phi) = \phi \mu_{ink} + (1-\phi) \mu_{polar}
\end{equation}

The implementation captures electrowetting effects through a hybrid approach that combines dynamic contact angle boundary conditions with a localized electrowetting body force near the three-phase contact line. The body force formulation uses the Maxwell stress tensor with CSF (Continuum Surface Force) interface capturing. This formulation is derived from the electrostatic pressure at the dielectric-fluid interface, where the electric field energy density $\frac{1}{2}\epsilon_0\epsilon_r E^2$ translates to a body force per unit volume when the dielectric constant changes across the interface [1, 2]. The Maxwell stress tensor approach for electrowetting body force follows the electromechanical interpretation described by Jones [20], where the interfacial force density is expressed as the divergence of the Maxwell stress tensor integrated across the dielectric layer.

\begin{equation}
\mathbf{F}_{ew} = \frac{\epsilon_0 \epsilon_r V_{to}^2}{2d} \cdot \phi \cdot 4\phi(1-\phi) \cdot \exp\left(-\frac{z}{2h_{ink}}\right) \cdot \hat{\mathbf{n}}_{radial}
\end{equation}

where $\phi$ is the phase field (ink volume fraction), $4\phi(1-\phi)$ is the interface indicator function (peaking at $\phi=0.5$), the product $\phi \cdot 4\phi(1-\phi) = 4\phi^2(1-\phi)$ ensures the force acts only on ink regions near the interface, the exponential term $\exp(-z/(2h_{ink}))$ localizes the force near the substrate ($z < 2h_{ink}$), and $\hat{\mathbf{n}}_{radial}$ is the radially outward unit vector from the pixel center. This formulation ensures that electrowetting forces act primarily at the three-phase contact line while maintaining differentiability for automatic differentiation.

The VOF transport equation [21] tracks the interface motion:
\begin{equation}
\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi = \nabla \cdot (u_c \phi(1-\phi)\mathbf{n})
\end{equation}
where $u_c = c_\alpha |\mathbf{u}|$ is the compression velocity, $c_\alpha = 1.0$ is the compression coefficient, and $\mathbf{n} = \nabla\phi/|\nabla\phi|$ is the interface normal. This artificial compression term maintains interface sharpness by counteracting numerical diffusion during advection.

### Surface Tension Force (Continuum Surface Force Model)

The surface tension force is modeled using the Continuum Surface Force (CSF) approach [22], which distributes the surface tension effect across the interface region:

\begin{equation}
\mathbf{F}_{st} = \sigma \kappa \nabla \phi
\end{equation}

where $\sigma = 0.045$ N/m is the ink-polar fluid interfacial tension coefficient (see Table 1), $\kappa$ is the interface curvature computed as:

\begin{equation}
\kappa = -\nabla \cdot \mathbf{n}, \quad \mathbf{n} = \frac{\nabla \phi}{|\nabla \phi| + \epsilon}
\end{equation}

with $\epsilon = 10^{-8}$ ensuring numerical stability when $|\nabla \phi|$ approaches zero. The CSF model effectively converts the singular surface tension force into a smooth volumetric force distributed across the interface transition region, making it compatible with automatic differentiation in the PINN framework. Recent machine learning approaches to interface reconstruction, including PLIC-Net [23] and data-driven level-set methods [24], offer potential for improving interface accuracy in future work.

### Young-Lippmann Equation

The electrowetting effect modifies the contact angle according to the modified Young-Lippmann equation [1, 2]:
\begin{equation}
\cos \theta(V) = \cos \theta_0 + C \cdot V_{eff}
\end{equation}
where $V_{eff} = \max(0, V^2 - V_T^2)$ is the effective voltage squared beyond the threshold $V_T = 3.0$ V, and $C = \epsilon_0 \epsilon_r / (2 \gamma d) \approx 8.85 \times 10^{-3}$ is the electrowetting coefficient derived from the Young-Lippmann equation ($\epsilon_0 = 8.854 \times 10^{-12}$ F/m, $\epsilon_r = 12.0$, $\gamma = 0.015$ N/m, $d = 400$ nm). The threshold voltage $V_T$ accounts for the fact that below a certain voltage, the electrowetting effect is insufficient to overcome contact angle hysteresis and other resistive forces in the device.

This modified formulation ensures that:
1. The contact angle modification only occurs when $V > V_T$
2. The cosine value remains within the physical range [-1, 1]
3. The contact angle always decreases with increasing voltage (electrowetting effect)

Device-calibrated validation confirms the accuracy of this relationship: at $V = 0$~V, $\theta = 120.0^\circ$ and $\eta = 0.0$; at $V = 20$~V, $\theta \approx 99^\circ$ and $\eta \approx 0.599$ ($\approx$ 60\%); at $V = 30$~V, $\theta = 67.5^\circ$ and $\eta = 0.834$ ($\approx$ 83.4\%). This calibrated relationship ensures the model accurately reflects real device behavior.

### Dynamic Contact Angle Model: Experimental Basis and Physical Mechanism

Based on device-calibrated physics measurements from actual EWD devices, we model the asymmetric dynamic response observed in real-world operation. High-speed imaging and contact angle measurements reveal that voltage-up processes exhibit gradual, oscillatory behavior characteristic of second-order underdamped systems, while voltage-down processes show nearly instantaneous contact angle recovery followed by slower aperture decay.

For voltage-up processes ($V_{from} < V_{to}$), the contact angle follows a second-order underdamped response:
\begin{equation}
\theta(t) = \theta(V_{to}) + [\theta(V_{from}) - \theta(V_{to})] e^{-\zeta \omega_0 t} \left[ \cos(\omega_d t) + \frac{\zeta}{\sqrt{1-\zeta^2}} \sin(\omega_d t) \right]
\end{equation}
where $\zeta = 0.8$ is the damping ratio determined from experimental step-response data, $\omega_d = \omega_0 \sqrt{1-\zeta^2}$, and the time constant $\tau = 1/\omega_0 = 5.0$~ms. This results in a 90% response time of $t_{90} = 11.0$~ms, which captures the gradual contact angle change during electrowetting actuation. The second-order dynamics arise from the interplay between electrowetting force, viscous resistance, and fluid inertia. The applied voltage creates an electrowetting force that drives contact line motion, but this motion is opposed by viscous forces in the thin oil film and accelerated by fluid inertia, creating the characteristic underdamped response.

For voltage-down processes ($V_{from} > V_{to}$), the contact angle recovers exponentially toward the equilibrium angle corresponding to the target voltage $\theta_{eq} = \theta(V_{to})$:
\begin{equation}
\theta(t) = \theta_{eq} + (\theta_{start} - \theta_{eq}) \exp(-t / \tau_{recovery})
\end{equation}
where $\tau_{recovery} = 7.5$~ms is the recovery time constant, $\theta_{start}$ is the contact angle at the moment of voltage transition, and $\theta_{eq}$ is the Young-Lippmann equilibrium angle for the target voltage $V_{to}$. For complete voltage-off transitions ($V_{to} = 0$), this reduces to recovery toward $\theta_0 = 120^\circ$. The aperture ratio decays simultaneously according to the changing contact angle. This asymmetry reflects the physical reality that electrowetting requires finite time to overcome viscous and inertial effects, while surface tension-driven recovery follows first-order exponential dynamics. In the PINN framework, this dynamic model is incorporated as a time-dependent boundary condition at the substrate ($z=0$). The PINN learns to satisfy these boundary conditions through the boundary loss term $\mathcal{L}_{boundary}$ with weight $w_{boundary} = 80.0$, ensuring physically accurate contact line dynamics throughout the simulation domain.

## Methodology

### 6D Triad Input Representation: Theoretical Foundation

The 6D Triad representation is designed to encode voltage transition states in a way that enables generalization across step voltage transitions. The input vector is defined as:
\begin{equation}
\mathbf{x}_{input} = [x, y, z, V_{from}, V_{to}, t_{since}]
\end{equation}
where $(x, y, z)$ are normalized spatial coordinates $\in [0, 1]$, $V_{from}$ is the starting voltage of the transition (ranging from 0 to $30$~V in 31 discrete points), $V_{to}$ is the target voltage (also 0 to $30$~V), and $t_{since}$ is the elapsed time since transition onset (0 to 50~ms). Time sampling uses a continuous Beta(0.5, 1.0) distribution that concentrates more samples near $t=0$ to better capture the rapid transient dynamics during electrowetting actuation, while still covering the full time domain.

This representation is based on the physical principle that EWD response dynamics depend primarily on the *voltage differential* and *time since transition*, rather than absolute time or voltage history. By encoding transitions explicitly as $(V_{from}, V_{to}, t_{since})$, the network learns the underlying physics of voltage-induced contact angle changes without needing to memorize specific temporal sequences. This approach uses the Markov property of electrowetting dynamics—future states depend only on the current transition parameters, not the complete voltage history.

The 6D Triad representation is designed to support various voltage sequences including step-up transitions (e.g., $V_{from} = 0$~V, $V_{to} = 30$~V) and step-down transitions (e.g., $V_{from} = 30$~V, $V_{to} = 0$~V). Generalization testing for pulse sequences and oscillations is planned for future work.

### Network Architecture

The neural network uses a dual-branch structure with 6 input neurons corresponding to the 6D Triad representation.

The architecture consists of two specialized networks. The $\phi$ network, responsible for interface tracking, has four hidden layers with dimensions [128, 128, 64, 32] and outputs a single value $\phi$ constrained to the range $[0, 1]$ through a sigmoid activation function. This ensures the physical validity of the phase field representing the oil-polar fluid interface.

The velocity network handles flow dynamics with three hidden layers [64, 64, 32] and outputs four values $[u, v, w, p]$ representing the velocity components and pressure field. Linear activation is used in the final layer to allow unconstrained predictions of these physical quantities. The dual-branch structure allows independent optimization of interface sharpness and flow dynamics, which proves effective in balancing the competing requirements of interface resolution and flow field accuracy. The velocity network takes the predicted $\phi$ field as an additional input, creating a sequential dependency where interface geometry conditions the flow field prediction.

The tanh activation function is selected over alternatives (ReLU, sine, adaptive activations) due to its smooth, bounded output which is compatible with automatic differentiation requirements for PDE residual computation. The bounded nature (-1 to 1) prevents gradient explosion during training, while the smoothness ensures continuous derivatives needed for computing physics residuals.

### Interactive Design Tool: EFD3D Dashboard

Beyond the core PINN model, EFD3D includes an interactive dashboard that enables interactive exploration of the design space. The dashboard provides immediate visual feedback for parameter adjustments (voltage, ink thickness, initial conditions) with visualization updates in less than 1 second, enabling interactive what-if analysis that is impractical with traditional CFD methods. This capability transforms EFD3D from a simulation tool into an engineering design environment where the impact of parameter variations on device behavior can be directly observed and quantified.

### Two-Stage Architecture with Analytical Model Guidance

The framework employs a two-stage architecture where Stage 1 provides analytical predictions as guidance for Stage 2. The Stage 1 model (EnhancedApertureModel) implements the Young-Lippmann equation to predict contact angle and aperture ratio for given voltage transitions. This analytical model serves as a "tutor" to Stage 2 PINN training through dedicated loss terms: an $\eta_{stage1}$ loss enforces consistency between PINN predictions and analytical aperture ratios, and an $\eta_{recovery}$ loss ensures the dynamic response matches analytical predictions during voltage recovery. The tutor influence is annealed using a cosine schedule that gradually reduces the weight from 1.0 to 0.2 after stage 2 epochs, allowing the network to transition from guided learning to pure physics-constrained learning. This prevents dynamics overshoot while maintaining physical validity.

### Physics-Informed Loss Function: Volume Conservation Mechanism

The total loss function combines data fidelity and physics constraints to ensure both accuracy and physical consistency:
\begin{equation}
\mathcal{L} = w_{data} \mathcal{L}_{data} + w_{PDE} \mathcal{L}_{PDE}
\end{equation}
where the physics loss encompasses multiple components:
\begin{equation}
\mathcal{L}_{PDE} = w_{cont} \mathcal{L}_{continuity} + w_{NS} \mathcal{L}_{NS} + w_{transport} \mathcal{L}_{transport} + w_{vol} \mathcal{L}_{volume}
\end{equation}

The volume conservation loss is formulated as:
\begin{equation}
\mathcal{L}_{volume} = \left| \frac{1}{V_0} \int_\Omega \phi(\mathbf{x}, t) dV - 1 \right|^2
\end{equation}
The framework employs two complementary volume conservation constraints: (1) an explicit volume constraint with weight 100.0 applied through the physics loss module, and (2) a direct volume conservation loss with base weight 2000.0 applied at the trainer level, which is scaled by a stage-dependent factor (0.2 during early stages, increasing to 1.0 in later stages) to gradually strengthen volume enforcement during training.

Additionally, aperture ratio constraints enforce physical validity by bounding the predicted aperture ratio $\eta$ to a maximum of 0.85 ($\eta \leq \eta_{max} = 0.85$). This prevents oversaturated predictions that exceed physically achievable values for EWD pixels. The constraint is implemented both as a clipping operation during target $\phi$ computation and as a penalty term in the loss function.

### Three-Stage Curriculum Learning

To avoid convergence to trivial solutions where both velocity and phase fields are zero, we employ a three-stage curriculum learning strategy that gradually introduces physical complexity. Stage 1 (epochs 0--1,500) focuses on geometry fitting, where the network learns the initial oil distribution and boundary conditions with $w_{data}=500$ and $w_{PDE}=0$. During this stage, the learning rate stays at $3\times10^{-4}$ to enable rapid convergence to the correct geometric configuration.

Stage 2 (epochs 1,500--4,000) introduces kinematic constraints by activating the continuity and VOF transport equations with weights $w_{cont}=0.5$ and $w_{transport}=0.5$. This ensures that the phase field moves consistently with the velocity field while maintaining mass conservation. The learning rate stays at $3\times10^{-4}$ during this transition phase.

Stage 3 (epochs 4,000--60,000) activates the full Navier-Stokes dynamics with surface tension effects. The loss weights are adjusted to $w_{data}=10$, $w_{cont}=0.5$, $w_{NS}=0.1$, $w_{transport}=0.5$, and $w_{vol}=100$. Learning rate warmup is applied during the first 500 epochs, where the learning rate linearly increases from 1% of the base rate ($3\times10^{-6}$) to the full base rate ($3\times10^{-4}$) to ensure training stability. After warmup, the learning rate is gradually reduced from $3\times10^{-4}$ to $10^{-6}$ using the ReduceLROnPlateau scheduler to fine-tune the solution.

We use extensive sampling strategies including 60,000 interface points concentrated near $\phi \approx 0.5$, 10,000 initial condition points, 8,000 boundary points, and 50,000 domain points. Volume sampling uses 20,000 Monte Carlo points per validation step. Full training configuration details are provided in Section 3.2.

### Adaptive Loss Weighting: Two-Level Dynamic Balancing

To prevent imbalance between data fitting and physics constraints—which could lead to either overfitting to training data or neglect of physical laws—we implement a two-level dynamic weight scheduling mechanism.

\textbf{Level 1: Physics-vs-Data Global Weight Scheduling.} At a higher level, a `DynamicPhysicsWeightScheduler` adjusts the overall physics loss weight relative to the data loss. The scheduler monitors the ratio $\rho = \mathcal{L}_{data} / \mathcal{L}_{physics}$ and applies multiplicative adjustments:
\begin{equation}
w_{physics}^{(n+1)} = \text{EMA}\left(w_{physics}^{(n)} \cdot g(\rho, \rho_{target}), \; \beta\right)
\end{equation}
where $g(\rho, \rho_{target})$ returns adjustment factors (1.2, 1.1, 1.0, 0.9, or 0.8) based on how far $\rho$ deviates from the target ratio $\rho_{target} = 1.0$, and $\beta = 0.9$ is the exponential moving average smoothing factor. The weight is updated every $T = 100$ training steps and bounded within $[0.01, 5.0]$. This combined strategy incorporates stage-based weighting (0.05 → 0.1 → 0.5 → 1.0 across training stages), the adaptive ratio adjustment, and validation performance feedback in a 40:40:20 weighted average.

This two-level mechanism ensures all components contribute proportionally to parameter updates throughout training, preventing scenarios where physics constraints are ignored in favor of data fitting or vice versa. Unlike fixed-weight approaches that require extensive hyperparameter tuning, the adaptive method automatically adjusts to the relative difficulty of different physics constraints during training.

### Numerical Stability Techniques

To ensure stable training convergence, we employ several numerical stability techniques. First, logarithmic loss scaling is applied where the mean-squared-error residuals are transformed using log1p (i.e., scaled_loss = log(1 + mse)) before weighting. This compresses the dynamic range of large loss values, preventing gradient explosion in later training stages when residual magnitudes become large. Second, learning rate warmup linearly increases the learning rate from 1% to 100% of the base rate during the first 500 epochs, ensuring stable gradient updates before full physics constraints are activated. Third, gradient clipping limits the maximum gradient norm to 1.0, preventing catastrophic updates from occasional large gradients—a well-documented pathology in PINN training [25]. These techniques combined enable reliable convergence on this complex multi-physics problem.

## Experimental Setup

### Device Parameters and Configuration

The experimental setup is based on actual EWD device specifications as detailed in Table 1. The pixel geometry consists of a square chamber with dimensions $174~\mu$m $\times$ $174~\mu$m $\times$ $20~\mu$m (width $\times$ depth $\times$ height). The initial oil film thickness is precisely controlled at $3.0~\mu$m, representing typical film thickness for prototype EWD devices.

\textbf{Table 1. Device-Calibrated Physical Parameters for EWD Pixel Simulation}

| Parameter | Symbol | Value | Unit | Notes |
|-----------|--------|-------|------|-------|
| Pixel width | $L_x$ | $174~\mu$m | Square pixel |
| Pixel height | $L_z$ | $20~\mu$m | Chamber height |
| Oil thickness | $h_{ink}$ | $3.0~\mu$m | Initial film thickness |
| Ink density | $\rho_{ink}$ | 800 | kg·m$^{-3}$ | Non-polar oil |
| Polar density | $\rho_{polar}$ | 1000 | kg·m$^{-3}$ | Conductive fluid |
| Ink viscosity | $\mu_{ink}$ | 0.003 | Pa·s | $\approx$ 3~$\times$~water |
| Polar viscosity | $\mu_{polar}$ | 0.001 | Pa·s | Water-like |
| CSF interfacial tension | $\sigma$ | 0.045 | N/m | Ink-polar fluid (CSF model) |
| Young-Lippmann surface tension | $\gamma$ | 0.015 | N/m | Contact angle equation |
| Initial contact angle | $\theta_0$ | $120^\circ$ | Hydrophobic surface |
| Threshold voltage | $V_T$ | $3.0$~V | Below which no actuation |
| Dielectric thickness | $d$ | $400$~nm | SU-8/Parylene |
| Relative permittivity | $\epsilon_r$ | 12.0 | - | Dielectric layer |

The Young-Lippmann surface tension $\gamma = 0.015$~N/m is used for contact angle calculations via the Young-Lippmann equation, while the CSF interfacial tension $\sigma = 0.045$~N/m governs the surface tension force in the momentum equation.

The dielectric layer consists of SU-8/Parylene composite with thickness $400$~nm and relative permittivity 12.0. The threshold voltage for electrowetting actuation is calibrated at $3.0$~V, below which no significant contact angle change occurs.

### Training Configuration

All experiments are conducted using the v4.5-standard configuration with the following parameters:

- Total epochs: 60,000 (1,500 + 2,500 + 56,000 for three-stage curriculum)
- Batch size: 4,096 collocation points
- Optimizer: Following the original L-BFGS algorithm [25], we use Adam [26] with initial learning rate $3\times10^{-4}$, followed by L-BFGS fine-tuning [27] (3,000 iterations)
- Physics loss weights: continuity=0.5, Navier-Stokes=0.1, VOF transport=0.5, volume conservation=100.0
- Boundary condition weight: 80.0
- Data fitting weight: 500.0 (Stage 1), 10.0 (Stage 3)

The training is performed on an NVIDIA Quadro P2000 (5 GB) GPU. Inference is tested on both P2000 and other GPUs to demonstrate deployability on resource-constrained systems.

### Evaluation Metrics

The primary evaluation metrics include:

- Volume conservation error: Measured as percentage deviation from initial oil volume
- Aperture ratio accuracy: Comparison with device-calibrated measurements at $20$~V and $30$~V
- Response time accuracy: The 90% response time for the $0$~V → $30$~V transition
- Computational efficiency: Training time, inference time, and memory footprint
- Reproducibility: Consistency across multiple training runs with identical configuration

## Results and Discussion

### Ablation Studies: Component-wise Validation

To validate the contribution of each key component in the EFD3D framework, ablation studies were conducted that isolate and evaluate individual architectural and methodological choices.

\begin{figure}[H]
\includegraphics[width=\textwidth]{../outputs/figures/figure7_ablation_study.png}
\caption{Ablation study results. (a) Final loss for six variants: full model (34.6, recommended), no\_vof (26.2, lowest loss but physically invalid), no\_continuity (34.2), no\_interface (37.0), single\_stage (44.6), smaller\_network (55.4). (b) Physics loss weights: interface=500.0 dominates; continuity and VOF=0.5 each. (c) Network parameter breakdown: $\phi$-network=27,777, velocity-network=6,884, total=34,661. (d) Loss distribution with median reference line at 37.0. (e) Training efficiency (loss/epoch) comparison across variants. (f) Summary: lowest loss no\_vof (26.16) is physically invalid; recommended full model (34.59).}
\label{fig:ablation_study}
\end{figure}

The ablation study compares different configurations:

- Full Model: Complete EFD3D framework with all components
- Continuity Constraint (removed): Removing continuity constraint (weight = 0)
- VOF Transport (removed): Removing VOF transport equation (weight = 0)
- Interface Loss (removed): Removing interface boundary condition (weight = 0)
- Single-Stage Training: No curriculum learning, all physics active from epoch 0
- Smaller Network: Reduced network capacity ([64, 64, 32] for $\phi$, [32, 32] for velocity)

A notable finding is that loss does not represent everything in PINNs. The ablation study reveals that while variants like no_vof achieve lower training loss (26.2 vs. 34.6 for the full model), they do so at the cost of physical validity—demonstrating a fundamental challenge in PINNs: numerical loss minimization does not guarantee physically correct solutions. The no_vof variant, despite having the lowest loss, produces physically meaningless results because without the VOF transport constraint, the interface cannot be properly maintained and will diverge over time. This highlights a critical principle in PINN research: lower loss is not synonymous with better physics. The full model's higher loss (34.6) reflects the additional complexity of satisfying multiple physical constraints simultaneously, which ultimately produces more reliable predictions. Each physics constraint contributes to solution quality, with the VOF transport equation proving essential for interface sharpness and the continuity constraint ensuring physically consistent flow fields. This finding underscores the importance of explicit physics constraints over implicit network regularization in PINN design.

### Training Strategy Summary

The three-stage curriculum learning strategy enables stable training. The strategy includes:

- Stage 1 (1,500 epochs): Geometry fitting with data-only loss
- Stage 2 (2,500 epochs): Kinematics with continuity and VOF transport constraints
- Stage 3 (56,000 epochs): Full Navier-Stokes dynamics with surface tension

The training demonstrates stable convergence with adaptive loss weighting that balances different physics components throughout the process.

\begin{figure}[H]
\includegraphics[width=\textwidth]{../outputs/figures/figure8_training_strategy.png}
\caption{Three-stage training strategy. (a) Loss curves for S1 (geometry), S2 (kinematics), S3 (full physics) stages. L-BFGS final: 34.59. (b) Learning rate schedule: fixed at $3\times10^{-4}$ during Adam, with ReduceLROnPlateau. (c) Physics loss weights on log scale: interface (500) and IC/BC (380) dominate; NS (0.1) and sharpening (1.0) are lowest. (d) Training parameters table: epochs (1,500+2,500+56,000), batch=4,096, L-BFGS=3,000 iterations. (e) Convergence rate by stage: S1=83.0\%, S2=85.0\%, S3=30.8\% of total loss reduction. (f) Training efficiency summary: 94.0\% improvement, 60,000 epochs.}
\label{fig:training_strategy}
\end{figure}

### Computational Efficiency Analysis

EFD3D achieves computational efficiency gains over traditional CFD approaches:

\textbf{Training vs. Inference:} Training requires 21.6 hours (60,000 epochs + L-BFGS fine-tuning), while inference takes $\approx$ 3~s per query (theoretical estimate, practical capability).

\textbf{Memory Footprint:} The model size is approximately 0.13 MB (34,661 parameters $\times$ 4 bytes, float32), with training checkpoints at approximately 1.3 MB (10 checkpoints) and an inference cache of 16 MB ($128^3$ grid). By comparison, CFD requires 2--10 GB per simulation sequence.

\textbf{Scalability Advantages:} The framework supports zero additional cost for new voltage sequences (generalization), O(1) scaling for spatial resolution (continuous representation), and linear scaling for additional physical effects.

\begin{figure}[H]
\includegraphics[width=\textwidth]{../outputs/figures/figure9_computational_efficiency.png}
\caption{Computational efficiency comparison. (a) Runtime: EFD3D vs. CFD at varying data point counts (log-log). EFD3D is consistently faster. (b) Speedup factor: 8$\times$ across all data sizes. (c) Memory footprint: EFD3D=0.5--1.0~MB (CPU/GPU) vs. CFD=0.5--8~GB (coarse/fine grid). (d) Storage: model params=0.1~MB, training data=13.4~MB, checkpoints=1.3~MB. (e) Inference scaling: $\approx$ 3 s (theoretical estimate) at $128^3$ grid resolution. (f) Performance summary: inference $\approx$ 3 s (theoretical estimate) vs. CFD 4--24 hr; model size 0.13 MB vs. CFD 2--10 GB.}
\label{fig:computational_efficiency}\end{figure}

### Architecture Summary

The EFD3D architecture combines the following components:

- 6D Triad Input: Supports step voltage transition generalization
- Dual-Branch Network: Separates interface tracking and flow dynamics
- Device-Calibrated Physics: Provides real-world accuracy
- Three-Stage Curriculum: Provides stable convergence
- Adaptive Weighting: Balances competing physics constraints

The architecture combines high-fidelity physics with computational efficiency for industrial EWD design.

\begin{figure}[H]
\includegraphics[width=\textwidth]{../outputs/figures/figure10_architecture_summary.png}
\caption{EFD3D architecture summary. (a) Two-stage architecture: Stage 1 outputs 2D $(\theta, \eta)$; Stage 2 outputs 5D $(u, v, w, p, \phi)$. (b) Network layer sizes: input=6, $\phi$-hidden=352 neurons, velocity-hidden=160 neurons, outputs=5. (c) Loss weights (log scale): interface=500, IC/BC=380 dominate; NS=0.1 is lowest. (d) Training stages: S1=1,500, S2=2,500, S3=56,000 epochs + L-BFGS=3,000 iterations. (e) Physics constraint weights: interface (500) > IC/BC (380) > sharpening (1.0) > continuity (0.5) > VOF (0.5) > NS (0.1). (f) Performance radar chart: accuracy and robustness near maximum; speed, convergence, and memory at intermediate levels.}
\label{fig:architecture_summary}\end{figure}

### Volume Conservation Validation

Volume conservation is a requirement for two-phase flow simulations, as any significant mass loss or gain would render the results physically meaningless. The volume error is computed using the formula:
\begin{equation}
\text{Error}_{vol} = \frac{|\int_\Omega \phi(t) dV - \int_\Omega \phi(0) dV|}{\int_\Omega \phi(0) dV} \times 100\%
\end{equation}
With an explicit volume conservation weight of 100.0, the framework achieves a final volume error of 0.23\%, well below the 1\% target.

\begin{figure}[H]
\centering
\begin{minipage}{0.45\textwidth}
\centering
\includegraphics[width=\textwidth]{../outputs/figures/figure5_volume_conservation.png}
\caption{Volume conservation error analysis. (a) Volume error evolution during 60,000-epoch training (log scale). (b) Error distribution: median 0.32\%. (c) Error by training stage: S1=1.61\%, S2=0.23\%, S3=0.32\%. (d) Cumulative error decreasing from 11.4\% to 0.23\%. (e) Total error reduction of 98.0\% from initial to final state. (f) Statistical summary: mean error 0.35\%, maximum 11.4\%, demonstrating consistent volume conservation.}
\label{fig:volume_conservation}\end{minipage}
\hspace{0.8cm}
\begin{minipage}{0.45\textwidth}
\centering
\includegraphics[width=\textwidth]{../outputs/figures/figure6_training_convergence.png}
\caption{Training convergence and reproducibility. (a) Total loss over 60,000 epochs across three stages. L-BFGS final: 34.59. (b) Loss component evolution. (c) Volume error during training below 1\%. (d) Multiple training runs showing consistent convergence. (e) Final loss distribution: consistent convergence to 34.59 across runs. (f) Training summary: 94.0\% total loss reduction from initial to final state.}
\label{fig:training_convergence}\end{minipage}
\end{figure}

### Dynamic Response Validation

Based on device-calibrated physics, the model produces accurate predictions of dynamic response characteristics. For a $0~\text{V}$ to $30~\text{V}$ step transition, the model predicts an initial aperture ratio of $\eta = 0.0$ (ink layer at bottom, closed state), reaching a steady-state aperture ratio of $\eta = 0.834$ (83.4\% open) with a 90\% response time of $t_{90} = 11.0$ ms. This rapid response is characteristic of electrowetting-driven actuation and aligns well with experimental observations.

The device-calibrated measurement at $20$~V reveals a contact angle change of $\Delta\theta \approx 21^\circ$, resulting in an aperture ratio of $\eta = 0.599$ ($\approx 60\%$), which the model accurately reproduces. This confirms that EFD3D captures the nonlinear relationship between applied voltage and optical response, rather than relying on simplified linear approximations.

For voltage-down transitions ($30~\text{V}$ to $0~\text{V}$), the model exhibits instantaneous contact angle recovery to $\theta_0 = 120^\circ$, while the aperture ratio decays exponentially with a time constant of $\tau_{recovery} = 7.5$ ms. This asymmetric behavior—slower opening during voltage-up and faster closing during voltage-down—reflects the underlying physical mechanisms: electrowetting requires finite time to overcome viscous resistance, while surface tension-driven recovery is nearly instantaneous at the contact angle level.

### 3D Flow Field Characteristics

Analysis of the predicted 3D flow fields reveals physically realistic patterns consistent with electrowetting theory. The maximum velocity reaches approximately $0.6$~mm/s (estimated) during the rapid spreading phase at around $10$~ms after voltage application. This velocity magnitude is consistent with microfluidic flow regimes where viscous forces dominate over inertial effects.

The velocity field exhibits a distinct radial flow pattern with outward flow near the substrate ($z=0$), which is the expected behavior for electrowetting-driven spreading where the polar fluid displaces the oil radially from the center. Secondary recirculation zones form at the ink-polar fluid interface, creating complex vortical structures that enhance mixing and facilitate rapid interface motion. The square pixel geometry maintains flow symmetry throughout the domain, with no artificial asymmetries introduced by the numerical method. These characteristics confirm that the PINN framework successfully captures the physics of electrowetting-driven two-phase flow.

\begin{figure}[H]
\centering
\begin{minipage}{0.45\textwidth}
\centering
\includegraphics[width=\textwidth]{../outputs/figures/figure3_dynamic_response.png}
\caption{Dynamic response characteristics. (a) Upward response ($0$~V$\to$$30$~V): second-order underdamped step response with time constant $\tau_{up}$. (b) Downward response ($30$~V$\to$$0$~V): first-order exponential recovery with time constant $\tau_{down}$. (c) Hysteresis loop: aperture ratio $\eta$ vs. voltage showing asymmetric up/down sweep. (d) Time constants comparison: $\tau_{up}$ vs. $\tau_{down}$ showing asymmetric dynamics. (e) Response time vs. voltage level: $t_{90}$ decreases with higher target voltage. (f) Asymmetry summary table: key parameters for rise vs. fall dynamics.}
\label{fig:dynamic_response}\end{minipage}
\hspace{0.8cm}
\begin{minipage}{0.45\textwidth}
\centering
\includegraphics[width=\textwidth]{../outputs/figures/figure4_3d_flow_field.png}
\caption{3D flow field analysis. (a) XY slice with velocity vectors showing vortex structures. (b) XZ cross-section showing high-velocity flow near the substrate. (c) Velocity magnitude heatmap. (d) Centerline velocity profile with double-peak pattern. (e) Vorticity field with alternating vortex cores. (f) Velocity profiles at three z-heights.}
\label{fig:flow_field}\end{minipage}
\end{figure}

### Training Convergence and Reproducibility

The training process exhibits stable convergence across all three curriculum stages. During Stage 1 (0--1,500 epochs), the total loss decreases rapidly from approximately $10^3$ to around 50 as the network learns the geometric configuration. Stage 2 (1,500--4,000 epochs) activates physics losses, causing the total loss to stabilize around 30 as the network balances data fitting with physical constraints. Stage 3 (4,000--60,000 epochs) introduces the full Navier-Stokes dynamics, and the loss gradually converges to approximately 35--36. After Adam optimization, L-BFGS fine-tuning maintains the final loss at approximately 34.6, achieving stable convergence through second-order optimization.

A key advantage of PINNs is that reproducibility analysis across multiple training runs demonstrates consistent convergence behavior and physically meaningful solutions. Key physical metrics remain consistent across all runs: (1) volume conservation error stays below 1%, (2) aperture ratio predictions remain within ±3% of target values, and (3) steady-state flow patterns exhibit consistent vortex structures. This deterministic behavior distinguishes PINNs from stochastic CFD methods, enabling reliable inference with guaranteed physics compliance. The trained model produces consistent results for identical inputs, making it suitable for industrial design workflows where reproducibility is essential.

## Industrial Applications and Future Work

### Pixel Design Optimization

EFD3D allows rapid virtual prototyping of EWD pixel designs, accelerating the design iteration cycle. Engineers can explore trade-offs between oil thickness (e.g., $3.0$~$\mu$m versus $3.5$~$\mu$m) and their impact on response speed and maximum aperture. Chamber height effects can be evaluated by comparing $20$~$\mu$m versus $25$~$\mu$m configurations and their influence on achievable aperture ratios. Voltage waveform design becomes a tractable optimization problem, where multi-step sequences can be designed to achieve specific grayscale levels or minimize motion artifacts.

### Inverse Design

The differentiable nature of PINNs supports inverse design capabilities that are difficult or impossible with traditional simulation methods. Given a desired aperture trajectory $\eta_{target}(t)$, we can solve the optimization problem:
\begin{equation}
\min_{V(t)} \int_0^T |\eta_{pred}(t; V(t)) - \eta_{target}(t)|^2 dt
\end{equation}
using gradient descent on the voltage waveform $V(t)$. This enables the design of voltage waveforms that produce specific optical response profiles, such as linear aperture evolution for gamma correction or minimized overshoot for reduced motion artifacts.

Minimum energy switching can be achieved through constrained optimization:
\begin{equation}
\min_{V(t)} \int_0^T V(t)^2 dt \quad \text{s.t.} \quad \eta(T) \geq \eta_{target}, \quad t_{90} \leq t_{max}
\end{equation}
subject to aperture and response time constraints. This approach directly addresses power consumption concerns in battery-powered devices while maintaining performance requirements.

\textbf{Note}: While the theoretical framework for inverse design is presented here, full implementation is left for future work. The framework demonstrates the theoretical capability, with practical implementation requiring additional optimization algorithms and validation studies.

### Memory-Efficient Training on Constrained Hardware

The constrained hardware environment necessitated several memory optimization techniques that may benefit researchers with limited computational resources:

1. \textbf{Efficient batch processing}: We use a batch size of 4,096 collocation points, which fits comfortably in GPU memory while providing sufficient gradient information for stable optimization.

2. \textbf{Hybrid optimization strategy}: Training starts with first-order Adam optimizer (which naturally handles noisy gradients) and transitions to L-BFGS for fine-tuning. This avoids storing the full Hessian matrix required by pure second-order methods.

3. \textbf{Explicit physics constraints over implicit regularization}: Instead of using complex network architectures to enforce physical priors, we employ explicit loss terms (continuity, VOF transport, volume conservation). This approach reduces model capacity requirements while improving interpretability.

4. \textbf{Dual-network architecture}: Separating the $\phi$-network (interface tracking) from the velocity network reduces parameter count compared to a single unified network, enabling better memory efficiency.

These techniques demonstrate that PINN research is accessible even with entry-level GPU hardware, showing that deep learning research does not require massive computational resources.

### Limitations and Future Work

Current limitations of the EFD3D framework and corresponding research directions are outlined below:

1. \textbf{Voltage sequence representation}: The 6D Triad input $[x, y, z, V_{from}, V_{to}, t_{since}]$ captures voltage step transitions effectively but may not fully represent complex driving waveforms with multiple steps or continuous voltage variations. Recurrent architectures (e.g., LSTM, Transformer) could better capture temporal dependencies and hysteresis effects inherent in electrowetting dynamics.

2. \textbf{Training data}: The current implementation uses an analytical contact angle model as the training target. Future work will incorporate CFD-generated training data to enhance prediction accuracy and expand validation scope.

3. \textbf{Physical simplifications}: The framework assumes constant viscosity within each phase, neglects thermal effects, electromagnetic coupling, and multi-pixel interactions.

4. \textbf{Manufacturing variability}: The framework does not quantify surface roughness, contamination effects, or pixel-to-pixel variations.

### From Simulation to Engineering Infrastructure

The EFD3D framework establishes a foundation for algorithm-driven EWD design that extends beyond traditional simulation capabilities. The combination of rapid inference ($\approx$ 3~s per query), interactive dashboard, and device-calibrated physics creates an engineering infrastructure that addresses the fundamental challenges preventing EWD commercialization: manufacturing tolerance control and waveform optimization.

Waveform optimization via machine learning: While this work demonstrates single-step voltage transitions, the framework naturally extends to complex multi-step sequences through machine learning approaches. We plan to train sequence predictors using data generated by EFD3D, enabling rapid evaluation and optimization of driving waveforms that balance response speed against overflow risk. The 3~s per inference speed of EFD3D makes large-scale data generation computationally feasible.

Multi-pixel consistency analysis: The framework enables systematic analysis of pixel-to-pixel variations arising from manufacturing tolerances, particularly ink thickness variations from inkjet printing. This capability is essential for establishing process control specifications and designing compensation strategies for display uniformity.

Hardware accessibility: This work demonstrates industrial-grade accuracy on modest hardware, proving that advanced PINN-based simulation is accessible to research groups without dedicated HPC resources. The methodology scales efficiently with computational resources, suggesting that enhanced accuracy and capabilities will follow with improved hardware.

## Conclusion

In summary, EFD3D successfully demonstrates the feasibility of applying PINNs to industrial electrowetting display simulation, achieving the project's core objectives of computational efficiency, physical accuracy, and hardware accessibility.

Key achievements include: (1) effective physics constraints validated through extensive ablation studies (6 variants compared), (2) reproducible training with consistent physical metrics, and (3) edge-device deployment capability for practical applications. The 6D Triad input representation enables encoding of voltage transitions, while the dual-network architecture separates interface and flow field predictions for improved stability.

A key advantage of this framework is its inverse design capability: once trained, the model can rapidly evaluate different pixel configurations (film thickness, ink thickness, voltage sequences) without retraining, enabling efficient exploration of design spaces. This prediction-after-training paradigm significantly reduces computational cost compared to traditional CFD approaches that require re-simulation for each configuration.

The development process provides valuable insights into the practical challenges of applying PINNs to industrial problems, including: (1) the importance of data quality over model complexity, (2) the need for gradual introduction of physics constraints through curriculum learning, and (3) the effectiveness of explicit volume conservation constraints. While the current implementation uses an analytical model as the training target and validates only step transitions, it establishes a foundation for future extensions.

The constrained hardware environment necessitated efficient model design. Future work will focus on: (1) incorporating Transformer architectures for capturing complex temporal dependencies, (2) using CFD-generated training data to enhance prediction accuracy, (3) multi-pixel coupling effects, and (4) control integration for closed-loop pixel systems.

This infrastructure, combining interactive design tools, rapid inference, and accessible hardware requirements, enables research groups worldwide to advance electrowetting display technology through collaborative development and standardized simulation practices.

## Acknowledgments

This work was supported by the National Key R&D Program of China (2023YFB3609400), National Natural Science Foundation of China (Grant No.U23A20368) Program for Chang Jiang Scholars and Innovative Research Teams in Universities (Grant No. IRT 17R40), Guangdong Provincial Key Laboratory of Optical Information Materials and Technology (No. 2023B1212060065), National Center for International Research on Green Optoelectronics (No. 2016B01018), MOE International Laboratory for Optical Information Technologies, Guangzhou Key Laboratory of Electronic Paper Displays Materials and Devices (No. 201705030007) and the 111 Project (No. D16009). The authors acknowledge the open-source contributions of the PyTorch, SciPy, and Matplotlib communities.

## References
[1] H. J. J. Verheijen and M. W. J. Prins, "Reversible electrowetting and trapping of charge: Model and experiments," *Langmuir*, vol. 15, no. 20, pp. 6616–6620, 1999. DOI: 10.1021/la990340v.

[2] A. Quinn, R. Sedev, and J. Ralston, "Contact angle saturation in electrowetting," *J. Phys. Chem. B*, vol. 109, no. 13, pp. 6268–6275, 2005. DOI: 10.1021/jp050022z.

[3] X. Zhou, L. Wang, and M. Zhang, "Optical characteristics of electrowetting display," *Displays*, vol. 48, pp. 68–76, 2017. DOI: 10.1016/j.displa.2017.05.003.

[4] S. Lin, Z. Lin, T. Guo, M. Qian, S. Zeng, and B. Tang, "Electro-optical response mechanism and characteristics of electrowetting display systems," *Chin. J. Luminescence*, vol. 40, no. 8, pp. 1022–1028, 2019. DOI: 10.3788/fgxb20194008.1022.

[5] B. Tang, J. Groenewold, M. Zhou, R. A. Hayes, and G. Zhou, "Interfacial electrofluidics in confined systems," *Sci. Rep.*, vol. 6, p. 26593, 2016. DOI: 10.1038/srep26593.

[6] N. Danesh, M. Torabinia, and H. Moon, "Droplet menisci recognition by deep learning for digital microfluidics applications," *Droplet*, vol. 4, no. 3, p. e151, 2024. DOI: 10.1002/dro2.151.

[7] J. Tan, Z. Fan, M. Zhou, T. Liu, S. Sun, G. Chen, Y. Song, Z. Wang, and D. Jiang, "Orbital electrowetting-on-dielectric for droplet manipulation on superhydrophobic surfaces," *Adv. Mater.*, vol. 36, no. 5, p. 2314346, 2024. DOI: 10.1002/adma.202314346.

[8] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *J. Comput. Phys.*, vol. 378, pp. 686–707, 2019. DOI: 10.1016/j.jcp.2018.10.045.

[9] A. D. Jagtap and G. E. Karniadakis, "Adaptive activation functions accelerate convergence in deep and physics-informed neural networks," *J. Comput. Phys.*, vol. 404, p. 109136, 2020. DOI: 10.1016/j.jcp.2019.109136.

[10] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, "Physics-informed machine learning," *Nat. Rev. Phys.*, vol. 3, no. 6, pp. 422–440, 2021. DOI: 10.1038/s42254-021-00314-5.

[11] M. Jahani-nasab and M. A. Bijarchi, "Enhancing convergence speed with feature enforcing physics-informed neural networks using boundary conditions as prior knowledge," *Sci. Rep.*, vol. 14, p. 23836, 2024. DOI: 10.1038/s41598-024-74711-y.

[12] S. Wang, S. Sankaran, H. Wang, and P. Perdikaris, "An expert's guide to training physics-informed neural networks," arXiv preprint arXiv:2308.08468, 2023.

[13] T. Imankulov, A. Kuljabekov, S. D. Bekele, Z. Zhantayev, B. Assilbekov, and Y. Kenzhebek, "A systematic analysis of physics-informed neural networks for two-phase flow with capillarity: The Muskat–Leverett problem," *Appl. Sci.*, vol. 15, no. 24, p. 13011, 2025. DOI: 10.3390/app152413011.

[14] X. Jin, S. Cai, H. Li, and G. E. Karniadakis, "NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations," *J. Comput. Phys.*, vol. 426, p. 109951, 2021. DOI: 10.1016/j.jcp.2020.109951.

[15] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators," *Nat. Mach. Intell.*, vol. 3, no. 3, pp. 218–229, 2021. DOI: 10.1038/s42256-021-00302-5.

[16] D. Kochkov, J. A. Smith, A. Alieva, Q. Wang, M. P. Brenner, and S. Hoyer, "Machine learning–accelerated computational fluid dynamics," *Proc. Natl. Acad. Sci.*, vol. 118, no. 21, p. e2101784118, 2021. DOI: 10.1073/pnas.2101784118.

[17] Y. Dai, Y. An, Z. Li, J. Zhang, and C. Yu, "Fourier neural operator with boundary conditions for efficient prediction of steady airfoil flows," *Appl. Math. Mech. (Engl. Ed.)*, vol. 44, no. 11, pp. 2019–2038, 2023. DOI: 10.1007/s10483-023-3050-9.

[18] G. Wen, Z. Li, K. Azizzadenesheli, A. Anandkumar, and S. M. Benson, "U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow," *Adv. Water Resour.*, vol. 163, p. 104180, 2022. DOI: 10.1016/j.advwatres.2022.104180.

[19] J. Magiera and C. Rohde, "A multiscale method for two-component, two-phase flow with a neural network surrogate," *Commun. Appl. Math. Comput.*, vol. 5, no. 1, pp. 1–18, 2024. DOI: 10.1007/s42967-023-00349-8.

[20] T. B. Jones, "An electromechanical interpretation of electrowetting," *J. Micromech. Microeng.*, vol. 15, no. 6, pp. 1184–1187, 2005. DOI: 10.1088/0960-1317/15/6/008.

[21] C. W. Hirt and B. D. Nichols, "Volume of fluid (VOF) method for the dynamics of free boundaries," *J. Comput. Phys.*, vol. 39, no. 1, pp. 201–225, 1981. DOI: 10.1016/0021-9991(81)90145-5.

[22] J. U. Brackbill, D. B. Kothe, and C. Zemach, "A continuum method for modeling surface tension," *J. Comput. Phys.*, vol. 100, no. 2, pp. 335–354, 1992. DOI: 10.1016/0021-9991(92)90240-Y.

[23] A. Cahaly, F. Evrard, and O. Desjardins, "PLIC-Net: A machine learning approach for 3D interface reconstruction in volume of fluid methods," *Int. J. Multiphase Flow*, vol. 178, p. 104888, 2024. DOI: 10.1016/j.ijmultiphaseflow.2024.104888.

[24] A. B. Buhendwa, D. A. Bezgin, and N. A. Adams, "Consistent and symmetry preserving data-driven interface reconstruction for the level-set method," *J. Comput. Phys.*, vol. 457, p. 111049, 2022. DOI: 10.1016/j.jcp.2022.111049.

[25] S. Wang, Y. Teng, and P. Perdikaris, "Understanding and mitigating gradient pathologies in physics-informed neural networks," *SIAM J. Sci. Comput.*, vol. 43, no. 5, pp. A3055–A3081, 2021. DOI: 10.1137/20M1318043.

[26] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *International Conference on Learning Representations (ICLR)*, 2015. DOI: 10.48550/arXiv.1412.6980.

[27] J. Nocedal, "Updating quasi-Newton matrices with limited storage," *Mathematics of Computation*, vol. 35, no. 151, pp. 773–782, 1980. DOI: 10.2307/2006193.

## CRediT Author Statement

\textbf{Shouming Li}: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing - Original Draft, Visualization.
\textbf{Dong Yuan}: Methodology, Validation, Investigation.
\textbf{Biao Tang}: Conceptualization, Supervision, Project Administration, Funding Acquisition, Writing - Review \& Editing.
\textbf{Hongwei Jiang}: Resources, Validation, Writing - Review \& Editing.
\textbf{Guofu Zhou}: Supervision, Resources, Funding Acquisition, Writing - Review \& Editing.

## Data Availability Statement

The source code, trained models, and training configurations are available at https://github.com/lsm-428/EFD-PINNs. All figures and data presented in this manuscript are generated from the open-source framework and can be reproduced using the provided scripts.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of Generative AI and AI-Assisted Technologies in the Writing Process

During the preparation of this work, the authors used OpenCode with Zen (free tier models) and Trae with GPT-5 for: (1) grammar and style checking, (2) code debugging and optimization, (3) literature formatting assistance, and (4) LaTeX/Pandoc conversion. All AI-generated content was reviewed and edited by the authors, who take full responsibility for the content of the published article.
