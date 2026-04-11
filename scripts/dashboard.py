import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.inference import PINNInferenceEngine
from src.dashboard.plotter import FlowFieldPlotter
from src.dashboard.training_output_analyzer import TrainingOutputAnalyzer
from src.dashboard.benchmark_panel import render_benchmark_tab
from src.dashboard.compare_panel import render_compare_tab
from src.dashboard.stage1_panel import render_stage1_tab

# --- Page Configuration ---
st.set_page_config(
    page_title="EFD3D Physics Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styles ---
st.markdown(
    """
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #1f77b4;
    }
    h2, h3 {
        color: #333333;
    }
    .stButton>button {
        color: white;
        background-color: #1f77b4;
        border-radius: 5px;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar: Configuration & Model Loading ---
st.sidebar.title("⚙️ System Config")

# 1. Model Selection
st.sidebar.subheader("1. Model Checkpoint")
model_dir = PROJECT_ROOT / "outputs"
# Recursive search for .pth files
model_files = sorted(
    list(model_dir.rglob("*.pth")) + list(model_dir.rglob("*.pt")),
    key=lambda x: x.stat().st_mtime,
    reverse=True,
)

if not model_files:
    st.error("No model checkpoints found in `outputs/`. Please run training first.")
    st.stop()

model_options = {
    f"{m.parent.name}/{m.name} ({time.ctime(m.stat().st_mtime)})": str(m)
    for m in model_files
}
selected_model_key = st.sidebar.selectbox(
    "Select Checkpoint", options=list(model_options.keys())
)
model_path = model_options[selected_model_key]

# 2. Compute Device
st.sidebar.subheader("2. Compute Resource")
if "cuda_available" not in st.session_state:
    import torch

    st.session_state.cuda_available = torch.cuda.is_available()

device_options = ["auto", "cpu"]
if st.session_state.cuda_available:
    device_options.insert(1, "cuda")

device = st.sidebar.selectbox(
    "Inference Device",
    options=device_options,
    index=1 if st.session_state.cuda_available else 0,
    help="CUDA is recommended for 3D volume rendering and high-res slices.",
)


# 3. Model Loading (Cached)
@st.cache_resource(show_spinner="Loading PINN Model & Physics Engine...")
def load_engine(path, dev):
    try:
        return PINNInferenceEngine(path, device=dev)
    except Exception as e:
        return None, str(e)


engine = load_engine(model_path, device)

# Handle loading errors
if isinstance(engine, tuple):  # Error case
    st.error(f"Failed to load model: {engine[1]}")
    st.stop()

# Display Model Metadata
with st.sidebar.expander("ℹ️ Model Architecture Info", expanded=False):
    st.markdown(f"**Loaded Path:** `{Path(model_path).name}`")
    st.markdown(f"**Training V_max:** `{engine.V_max_train} V`")
    st.markdown(
        f"**Domain:** {engine.Lx * 1e6:.0f} x {engine.Ly * 1e6:.0f} x {engine.Lz * 1e6:.0f} μm"
    )
    if hasattr(engine.model, "config"):
        st.json(engine.model.config)

plotter = FlowFieldPlotter()

# --- Main Interface ---
st.title("🌊 EFD3D Physics Dashboard")
st.markdown("Physics-Informed Neural Network analysis for Electro-Fluid Dynamics.")

# Tabs for different analysis modes
tab_names = [
    "📊 2D Field Analysis",
    "🧊 3D Volumetric View",
    "📈 Transient Response",
    "🩺 Physics Diagnostics",
    "📊 训练输出分析",
    "⏱️ Benchmark",
    "🔄 Compare",
    "📐 Stage 1",
]
tabs = st.tabs(tab_names)

# ==============================================================================
# TAB 1: 2D Field Analysis (Slices)
# ==============================================================================
with tabs[0]:
    st.markdown("### Cross-Sectional Field Analysis")

    col_ctrl, col_view = st.columns([1, 3])

    with col_ctrl:
        st.markdown("#### Driving Signal")

        # Mode Selection
        drive_mode = st.radio(
            "Drive Mode", ["Step Response", "Custom Waveform"], index=0
        )

        v_limit = float(engine.V_max_train)

        if drive_mode == "Step Response":
            v_from = st.number_input("V_prev (V)", -v_limit, v_limit, 0.0, step=1.0)
            v_to = st.number_input("V_curr (V)", -v_limit, v_limit, 30.0, step=1.0)
            t_ms = st.slider("Time (ms)", 0.0, 100.0, 0.5, step=0.1)

            # Map to engine inputs
            e_v_from, e_v_to, e_t = v_from, v_to, t_ms / 1000.0

        else:  # Custom Waveform
            wf_type_2d = st.selectbox(
                "Waveform", ["Square Pulse", "Sine Wave"], key="wf_2d"
            )

            if wf_type_2d == "Square Pulse":
                amp = st.number_input(
                    "Amplitude (V)", -v_limit, v_limit, 30.0, key="amp_2d"
                )
                width = st.number_input(
                    "Pulse Width (ms)", 0.1, 100.0, 10.0, key="wd_2d"
                )
                t_ms = st.slider(
                    "Global Time (ms)", 0.0, width * 3, width / 2, step=0.1
                )

                # Logic to map Global Time -> Local Step
                # If t < width: Step 0->Amp, local_t = t
                # If t > width: Step Amp->0, local_t = t - width
                if t_ms <= width:
                    e_v_from, e_v_to = 0.0, amp
                    e_t = t_ms / 1000.0
                    st.info(f"Phase: Rising Edge (t={e_t * 1000:.1f}ms)")
                else:
                    e_v_from, e_v_to = amp, 0.0
                    e_t = (t_ms - width) / 1000.0
                    st.info(f"Phase: Falling Edge (t={e_t * 1000:.1f}ms)")

            elif wf_type_2d == "Sine Wave":
                freq = st.number_input("Freq (Hz)", 1, 200, 10, key="fr_2d")
                amp = st.number_input(
                    "Amplitude (V)", 0.0, v_limit, 30.0, key="sin_amp_2d"
                )
                t_ms = st.slider(
                    "Global Time (ms)", 0.0, 1000.0 / freq * 2, 0.0, step=0.1
                )

                # Quasi-static approx
                # V_inst = amp * sin(2*pi*f*t)
                v_inst = amp * np.sin(2 * np.pi * freq * (t_ms / 1000.0))
                e_v_from, e_v_to = 0.0, v_inst
                e_t = 1.0  # Assumed equilibrium
                st.caption(f"Quasi-static: V_inst = {v_inst:.1f}V")

        st.markdown("#### Slice Config")
        plane = st.selectbox(
            "Cut Plane", ["xy (Top View)", "xz (Side View)", "yz (Front View)"]
        ).split()[0]

        # Dynamic Slider for Slice Position
        if plane == "xy":
            max_pos = engine.Lz * 1e6
            def_pos = engine.h_ink * 1e6 / 2
            label = "Z-height (μm)"
        elif plane == "xz":
            max_pos = engine.Ly * 1e6
            def_pos = engine.Ly * 1e6 / 2
            label = "Y-position (μm)"
        else:  # yz
            max_pos = engine.Lx * 1e6
            def_pos = engine.Lx * 1e6 / 2
            label = "X-position (μm)"

        slice_pos = st.slider(label, 0.0, max_pos, def_pos)

        st.markdown("#### Options")
        res_2d = st.select_slider("Resolution", [64, 128, 256], value=128)
        show_vec = st.checkbox("Velocity Vectors", True)
        show_pot = st.checkbox("Electric Potential", False)

        run_2d = st.button("Render Slice", type="primary", use_container_width=True)

    with col_view:
        if run_2d:
            with st.spinner("Computing fields..."):
                start_time = time.time()
                data = engine.predict_field(
                    t=e_t,
                    voltage_from=e_v_from,
                    voltage_to=e_v_to,
                    plane=plane,
                    slice_val=slice_pos * 1e-6,
                    resolution=res_2d,
                )
                dt = time.time() - start_time

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Inference Time", f"{dt * 1000:.1f} ms")
                m2.metric("Max Velocity", f"{np.max(data['vel_mag']):.4f} m/s")
                m3.metric("Avg Phase", f"{np.mean(data['phi']):.3f}")
                m4.metric(
                    "Reynolds No.",
                    f"{np.max(data['vel_mag']) * engine.h_ink / 1.0e-6:.2f}",
                )  # Approx

                # Plotting
                fig = plotter.create_dashboard_figure(
                    data, show_electric_field=show_pot
                )
                st.pyplot(fig)

                # Export Data
                # df = pd.DataFrame({'x': data['X_grid'].flatten(), 'y': data['Y_grid'].flatten(), 'phi': data['phi'].flatten()})
                # st.download_button("Download CSV", df.to_csv(), "slice_data.csv")
        else:
            st.info("👈 Adjust parameters and click 'Render Slice' to visualize.")

# ==============================================================================
# TAB 2: 3D Volumetric View
# ==============================================================================
with tabs[1]:
    st.markdown("### 3D Ink Interface Reconstruction")
    st.caption(
        "Visualizes the iso-surface of the phase field (φ=0.5) representing the ink-air interface."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        v3_f = st.number_input(
            "Initial Voltage (V)",
            -v_limit,
            v_limit,
            0.0,
            help="Voltage before switching",
        )
    with c2:
        v3_t = st.number_input(
            "Target Voltage (V)",
            -v_limit,
            v_limit,
            30.0,
            help="Voltage after switching",
        )
    with c3:
        t3_ms = st.number_input("Time since switch (ms)", 0.0, 100.0, 1.0, step=0.1)

    c4, c5 = st.columns(2)
    with c4:
        res_3d = st.slider(
            "Grid Resolution (Voxel Density)",
            30,
            80,
            40,
            help="Higher = Smoother but slower",
        )
    with c5:
        iso_val = st.slider("Iso-value (Phase Threshold)", 0.1, 0.9, 0.5)

    if st.button("Generate 3D Model", type="primary"):
        if not st.session_state.cuda_available and res_3d > 50:
            st.warning("⚠️ CPU inference with high resolution may be slow.")

        with st.spinner("Raymarching 3D Volume (this may take a moment)..."):
            vol_data = engine.predict_3d_volume(
                t=t3_ms / 1000.0,
                voltage_from=v3_f,
                voltage_to=v3_t,
                resolution_xy=res_3d,
                resolution_z=int(
                    res_3d * (engine.Lz / engine.Lx)
                ),  # Scale Z res proportionally
            )

            fig_3d = plotter.create_3d_isosurface_figure(vol_data, isovalue=iso_val)
            st.plotly_chart(fig_3d, use_container_width=True)

# ==============================================================================
# TAB 3: Transient Response (Waveforms)
# ==============================================================================
with tabs[2]:
    st.markdown("### Dynamic Aperture Response Simulation")

    row_cfg, row_plot = st.columns([1, 2])

    with row_cfg:
        st.markdown("#### Waveform Generator")
        wf_type = st.selectbox(
            "Type",
            [
                "Step Function",
                "Square Pulse",
                "Linear Ramp",
                "Sine Wave",
                "Custom Expression",
            ],
        )

        duration = st.number_input("Total Duration (ms)", 0.5, 200.0, 50.0)
        points = st.slider("Sampling Points", 20, 500, 100)

        # Waveform Logic
        if wf_type == "Step Function":
            step_v = st.number_input("Step Amplitude (V)", -v_limit, v_limit, 30.0)
            delay = st.number_input("Delay (ms)", 0.0, duration, 0.1)
            func = (
                lambda t: (0.0, step_v, t - delay / 1000.0)
                if t > delay / 1000.0
                else (0.0, 0.0, 0.0)
            )
            preview_y = lambda t_arr: np.where(t_arr > delay / 1000.0, step_v, 0.0)

        elif wf_type == "Square Pulse":
            amp = st.number_input("Pulse Amplitude (V)", -v_limit, v_limit, 30.0)
            start = st.number_input("Start Time (ms)", 0.0, duration, 0.2)
            width = st.number_input("Pulse Width (ms)", 0.1, duration, 0.5)

            def pulse_func(t):
                t_s = t * 1000
                if start <= t_s <= start + width:
                    # In pulse: Transition from 0 to Amp
                    return (0.0, amp, t - start / 1000.0)
                elif t_s > start + width:
                    # After pulse: Transition from Amp to 0
                    return (amp, 0.0, t - (start + width) / 1000.0)
                else:
                    return (0.0, 0.0, 0.0)

            func = pulse_func
            preview_y = lambda t_arr: np.where(
                (t_arr * 1000 >= start) & (t_arr * 1000 <= start + width), amp, 0.0
            )

        elif wf_type == "Sine Wave":
            freq = st.number_input("Frequency (Hz)", 10, 1000, 100)
            amp = st.number_input("Amplitude (V)", 0.0, v_limit, 30.0)
            offset = st.number_input("DC Offset (V)", -v_limit, v_limit, 0.0)
            st.info(
                "Uses quasi-static approximation (Instantaneous Voltage -> Equilibrium)"
            )
            func = lambda t: (
                0.0,
                offset + amp * np.sin(2 * np.pi * freq * t),
                1.0,
            )  # t=1.0 assumes equilibrium
            preview_y = lambda t_arr: offset + amp * np.sin(2 * np.pi * freq * t_arr)

        else:  # Ramp or Custom
            st.warning(
                "Custom expressions not yet fully implemented, using linear ramp default."
            )
            slope = st.number_input("Slope (V/ms)", 0.0, 100.0, 10.0)
            func = lambda t: (0.0, min(v_limit, slope * t * 1000), t)
            preview_y = lambda t_arr: np.minimum(v_limit, slope * t_arr * 1000)

        run_traj = st.button("Simulate Trajectory", type="primary")

    with row_plot:
        # Preview Waveform
        t_preview = np.linspace(0, duration / 1000.0, 200)
        v_preview = preview_y(t_preview)

        fig_wv, ax_wv = plt.subplots(figsize=(8, 2))
        ax_wv.plot(t_preview * 1000, v_preview, "k--", alpha=0.6, label="Input Voltage")
        ax_wv.set_ylabel("Voltage (V)")
        ax_wv.set_xlabel("Time (ms)")
        ax_wv.legend(loc="upper right")
        ax_wv.grid(True, alpha=0.3)
        st.pyplot(fig_wv)

        if run_traj:
            with st.spinner("Calculating Aperture Dynamics..."):
                t_sim = np.linspace(0, duration / 1000.0, points)
                traj_data = engine.predict_trajectory(func, t_sim)

                # Plot Result
                fig_res, ax1 = plt.subplots(figsize=(8, 4))

                color = "tab:red"
                ax1.set_xlabel("Time (ms)")
                ax1.set_ylabel("Voltage (V)", color=color)
                ax1.plot(
                    traj_data["t"] * 1000,
                    traj_data["voltage"],
                    color=color,
                    linestyle="--",
                    alpha=0.5,
                )
                ax1.tick_params(axis="y", labelcolor=color)

                ax2 = ax1.twinx()
                color = "tab:blue"
                ax2.set_ylabel("Aperture Ratio (η)", color=color)
                ax2.plot(
                    traj_data["t"] * 1000, traj_data["eta"], color=color, linewidth=2.5
                )
                ax2.tick_params(axis="y", labelcolor=color)
                ax2.set_ylim(-0.1, 1.1)

                plt.title("System Response")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig_res)

# ==============================================================================
# TAB 4: Physics Diagnostics
# ==============================================================================
with tabs[3]:
    st.markdown("### 🩺 Physics Constraints Verification")
    st.markdown(
        "Check if the model predictions satisfy physical laws (Mass Conservation & Navier-Stokes)."
    )

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Conservation of Mass")
        st.write("Calculates total liquid volume. Should remain constant over time.")

        check_t = st.slider("Check Time (ms)", 0.0, 100.0, 0.5, key="diag_t")
        check_v = st.number_input("Voltage", 0.0, v_limit, 30.0, key="diag_v")

        if st.button("Check Volume"):
            vol = engine.check_mass_conservation(check_t / 1000.0, 0.0, check_v)
            st.metric("Total Volume Integral", f"{vol:.4e} m³")

            # Theoretical volume (approx)
            theor_vol = engine.Lx * engine.Ly * engine.h_ink * 0.5  # Assuming 50% fill
            st.metric("Theoretical Volume (50%)", f"{theor_vol:.4e} m³")

    with d2:
        st.markdown("#### PDE Residuals (Loss Landscape)")
        st.write(
            "Visualizes where the model struggles to satisfy Navier-Stokes equations."
        )

        if st.button("Compute Residual Map"):
            with st.spinner("Evaluating derivatives..."):
                # Compute residuals on central slice
                res_map = engine.compute_residuals(
                    t=check_t / 1000.0,
                    voltage_from=0.0,
                    voltage_to=check_v,
                    plane="xz",  # Side view is usually most interesting for physics
                    resolution=64,
                )

                # Plot
                fig_res, axes = plt.subplots(1, 2, figsize=(10, 4))

                # Continuity Residual
                if "continuity" in res_map:
                    im1 = axes[0].imshow(
                        res_map["continuity"].T,
                        origin="lower",
                        cmap="coolwarm",
                        aspect="auto",
                    )
                    axes[0].set_title("Mass Continuity Residual")
                    fig_res.colorbar(im1, ax=axes[0])

                # Momentum Residual (X)
                if "momentum_x" in res_map:
                    im2 = axes[1].imshow(
                        res_map["momentum_x"].T,
                        origin="lower",
                        cmap="coolwarm",
                        aspect="auto",
                    )
                    axes[1].set_title("Momentum X Residual")
                    fig_res.colorbar(im2, ax=axes[1])

                st.pyplot(fig_res)
                st.caption(
                    "Red/Blue indicates high error in satisfying physics equations."
                )

# ==============================================================================
# TAB 5: Training Output Analyzer (训练输出分析)
# ==============================================================================
with tabs[4]:
    st.markdown("### 📊 训练输出分析")
    st.caption("分析和可视化训练输出目录中的训练运行。")

    analyzer = TrainingOutputAnalyzer(train_outputs_dir="outputs/train")
    analyzer.render()

# ==============================================================================
# TAB 6: Benchmark Performance (⏱️ Benchmark)
# ==============================================================================
with tabs[5]:
    render_benchmark_tab(engine)

# ==============================================================================
# TAB 7: Compare (🔄 Compare)
# ==============================================================================
with tabs[6]:
    render_compare_tab()

# ==============================================================================
# TAB 8: Stage 1 (📐 Stage 1)
# ==============================================================================
with tabs[7]:
    render_stage1_tab()

st.markdown("---")
st.caption(f"EFD3D Dashboard v2.0 | Project Root: `{PROJECT_ROOT}`")
