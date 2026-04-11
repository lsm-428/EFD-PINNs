"""
Stage 1 分析模型演示面板组件
============================

用于在 Streamlit 仪表板中演示 Stage 1 分析模型（接触角/开口率预测）。
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.aperture_model import EnhancedApertureModel
from src.config import PHYSICS


# ============================================================================
# Core Model Functions
# ============================================================================


@st.cache_resource
def get_stage1_model() -> EnhancedApertureModel:
    """
    Get or create Stage 1 analytical model instance.

    Returns:
        EnhancedApertureModel instance
    """
    return EnhancedApertureModel()


def compute_steady_state(
    model: EnhancedApertureModel, voltage: float
) -> Dict[str, float]:
    """
    Compute steady state contact angle and aperture ratio.

    Uses Young-Lippmann equation to calculate steady-state contact angle
    and converts it to aperture ratio.

    Args:
        model: EnhancedApertureModel instance
        voltage: Voltage (V)

    Returns:
        Dictionary with:
        - theta: Contact angle (degrees)
        - aperture_ratio: Aperture ratio (0-1)
        - r_open: Transparent region radius (m)
    """
    # Get steady-state contact angle using Young-Lippmann
    theta = model._get_predictor().young_lippmann(voltage)

    # Convert to aperture ratio
    aperture_ratio = model.contact_angle_to_aperture_ratio(theta)

    # Calculate transparent region radius
    r_open = model.aperture_ratio_to_open_radius(aperture_ratio)

    return {"theta": theta, "aperture_ratio": aperture_ratio, "r_open": r_open}


def compute_dynamic_response(
    model: EnhancedApertureModel,
    V_from: float,
    V_to: float,
    duration: float = 0.05,
    num_points: int = 500,
) -> Dict[str, np.ndarray]:
    """
    Compute dynamic response for voltage jump.

    Simulates the transient response from V_from to V_to over specified duration.
    Uses the triad interface (V_from, V_to, t_since) for prediction.

    Args:
        model: EnhancedApertureModel instance
        V_from: Initial voltage (V)
        V_to: Target voltage (V)
        duration: Total duration (s)
        num_points: Number of time points

    Returns:
        Dictionary with:
        - t: Time array (s)
        - theta: Contact angle array (degrees)
        - aperture_ratio: Aperture ratio array (0-1)
        - voltage: Voltage array (V)
    """
    # Generate time series
    t = np.linspace(0, duration, num_points)

    # Initialize arrays
    theta = np.zeros(num_points)
    aperture_ratio = np.zeros(num_points)
    voltage = np.zeros(num_points)

    # Compute response at each time point
    for i, ti in enumerate(t):
        # Use triad interface for prediction
        theta_i, eta_i = model.theta_eta_from_triad(V_from, V_to, ti)

        theta[i] = theta_i
        aperture_ratio[i] = eta_i
        voltage[i] = V_to  # After jump, voltage is V_to

    return {
        "t": t,
        "theta": theta,
        "aperture_ratio": aperture_ratio,
        "voltage": voltage,
    }


def create_stage1_plots(
    results: Dict[str, Any], plot_type: str = "dynamic"
) -> go.Figure:
    """
    Create Plotly figure for Stage 1 results.

    Args:
        results: Dictionary containing simulation results
        plot_type: Type of plot ('dynamic', 'steady', 'both')

    Returns:
        Plotly Figure object
    """
    if plot_type == "dynamic":
        # Dynamic response plot
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Contact Angle Response", "Aperture Ratio Response"),
            vertical_spacing=0.12,
        )

        # Plot contact angle
        fig.add_trace(
            go.Scatter(
                x=results["t"] * 1000,  # Convert to ms
                y=results["theta"],
                mode="lines",
                name="Contact Angle",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Plot aperture ratio
        fig.add_trace(
            go.Scatter(
                x=results["t"] * 1000,  # Convert to ms
                y=results["aperture_ratio"] * 100,  # Convert to %
                mode="lines",
                name="Aperture Ratio",
                line=dict(color="red", width=2),
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Contact Angle (°)", row=1, col=1)
        fig.update_yaxes(title_text="Aperture Ratio (%)", row=2, col=1)

        fig.update_layout(
            height=600, title_text="Stage 1 Dynamic Response", showlegend=True
        )

    elif plot_type == "steady":
        # Steady state plot
        fig = go.Figure()

        voltages = np.linspace(0, 30, 31)
        theta_list = []
        aperture_list = []

        for V in voltages:
            result = compute_steady_state(results["model"], V)
            theta_list.append(result["theta"])
            aperture_list.append(result["aperture_ratio"] * 100)

        # Plot contact angle
        fig.add_trace(
            go.Scatter(
                x=voltages,
                y=theta_list,
                mode="lines+markers",
                name="Contact Angle",
                yaxis="y",
                line=dict(color="blue", width=2),
            )
        )

        # Plot aperture ratio
        fig.add_trace(
            go.Scatter(
                x=voltages,
                y=aperture_list,
                mode="lines+markers",
                name="Aperture Ratio",
                yaxis="y2",
                line=dict(color="red", width=2),
            )
        )

        # Create dual y-axis
        fig.update_layout(
            yaxis=dict(
                title="Contact Angle (°)",
                title_font=dict(color="blue"),
                tickfont=dict(color="blue"),
            ),
            yaxis2=dict(
                title="Aperture Ratio (%)",
                title_font=dict(color="red"),
                tickfont=dict(color="red"),
                overlaying="y",
                side="right",
            ),
            xaxis=dict(title="Voltage (V)"),
            height=500,
            title_text="Stage 1 Steady-State Characteristics",
            showlegend=True,
        )

    elif plot_type == "both":
        # Combined plot with steady state and dynamic response
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Steady-State Contact Angle",
                "Steady-State Aperture Ratio",
                "Dynamic Contact Angle Response",
                "Dynamic Aperture Ratio Response",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        # Steady state contact angle
        voltages = np.linspace(0, 30, 31)
        theta_list = [
            compute_steady_state(results["model"], V)["theta"] for V in voltages
        ]
        fig.add_trace(
            go.Scatter(
                x=voltages,
                y=theta_list,
                mode="lines+markers",
                name="Contact Angle",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Steady state aperture ratio
        aperture_list = [
            compute_steady_state(results["model"], V)["aperture_ratio"] * 100
            for V in voltages
        ]
        fig.add_trace(
            go.Scatter(
                x=voltages,
                y=aperture_list,
                mode="lines+markers",
                name="Aperture Ratio",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=2,
        )

        # Dynamic contact angle
        if "t" in results and "theta" in results:
            fig.add_trace(
                go.Scatter(
                    x=results["t"] * 1000,
                    y=results["theta"],
                    mode="lines",
                    name="Dynamic θ",
                    line=dict(color="darkblue", width=2),
                ),
                row=2,
                col=1,
            )

        # Dynamic aperture ratio
        if "t" in results and "aperture_ratio" in results:
            fig.add_trace(
                go.Scatter(
                    x=results["t"] * 1000,
                    y=results["aperture_ratio"] * 100,
                    mode="lines",
                    name="Dynamic η",
                    line=dict(color="darkred", width=2),
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_xaxes(title_text="Voltage (V)", row=1, col=1)
        fig.update_xaxes(title_text="Voltage (V)", row=1, col=2)
        fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=2, col=2)
        fig.update_yaxes(title_text="Contact Angle (°)", row=1, col=1)
        fig.update_yaxes(title_text="Aperture Ratio (%)", row=1, col=2)
        fig.update_yaxes(title_text="Contact Angle (°)", row=2, col=1)
        fig.update_yaxes(title_text="Aperture Ratio (%)", row=2, col=2)

        fig.update_layout(
            height=800,
            title_text="Stage 1: Steady-State and Dynamic Response",
            showlegend=True,
        )

    else:
        raise ValueError(
            f"Unknown plot_type: {plot_type}. Must be 'dynamic', 'steady', or 'both'."
        )

    return fig


# ============================================================================
# UI Rendering Functions
# ============================================================================


def render_stage1_tab():
    """渲染 Stage 1 分析模型标签页

    展示 Young-Lippmann 方程、接触角预测、开口率预测等功能。
    """
    st.subheader("📐 Stage 1 Analytical Model")
    st.markdown("""
    Stage 1 模型基于 **Young-Lippmann 方程** 预测接触角和开口率。
    通过电润湿效应，电压改变界面张力，从而改变接触角。
    """)

    # 获取模型
    model = get_stage1_model()

    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["📊 Dynamic Response", "📈 Steady State", "📖 Theory"])

    # =========================================================================
    # Tab 1: Dynamic Response
    # =========================================================================
    with tab1:
        st.markdown("### 动态响应分析")
        st.markdown("模拟电压跳变下的接触角和开口率瞬态响应。")

        # 电压输入控制
        col1, col2 = st.columns(2)
        with col1:
            V_from = st.slider(
                "起始电压 V_from (V)",
                min_value=0.0,
                max_value=30.0,
                value=0.0,
                step=0.5,
                key="v_from_dynamic",
                help="电压跳变前的初始电压",
            )
        with col2:
            V_to = st.slider(
                "目标电压 V_to (V)",
                min_value=0.0,
                max_value=30.0,
                value=30.0,
                step=0.5,
                key="v_to_dynamic",
                help="电压跳变后的目标电压",
            )

        # 时间参数
        col3, col4 = st.columns(2)
        with col3:
            duration = st.number_input(
                "仿真时长 (ms)",
                min_value=1.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                key="duration_dynamic",
                help="动态响应仿真时长",
            )
        with col4:
            num_points = st.number_input(
                "采样点数",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
                key="num_points_dynamic",
                help="时间序列采样点数",
            )

        # 计算按钮
        if st.button("🔄 计算动态响应", key="compute_dynamic"):
            with st.spinner("正在计算动态响应..."):
                # 计算动态响应
                dynamic_results = compute_dynamic_response(
                    model, V_from, V_to, duration / 1000.0, int(num_points)
                )

                # 显示稳态结果
                st.markdown("#### 稳态结果")
                steady_from = compute_steady_state(model, V_from)
                steady_to = compute_steady_state(model, V_to)

                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric(
                        "初始接触角",
                        f"{steady_from['theta']:.1f}°",
                        f"V={V_from}V",
                    )
                with mcol2:
                    st.metric(
                        "目标接触角",
                        f"{steady_to['theta']:.1f}°",
                        f"{steady_to['theta'] - steady_from['theta']:+.1f}°",
                    )
                with mcol3:
                    st.metric(
                        "开口率变化",
                        f"{steady_to['aperture_ratio'] * 100:.1f}%",
                        f"{(steady_to['aperture_ratio'] - steady_from['aperture_ratio']) * 100:+.1f}%",
                    )

                # 绘制动态响应图
                st.markdown("#### 动态响应曲线")
                fig = create_stage1_plots(dynamic_results, plot_type="dynamic")
                st.plotly_chart(fig, use_container_width=True)

                # 显示数值结果
                with st.expander("📋 查看数值数据"):
                    # 创建 DataFrame
                    df = pd.DataFrame(
                        {
                            "时间 (ms)": dynamic_results["t"] * 1000,
                            "接触角 (°)": dynamic_results["theta"],
                            "开口率 (%)": dynamic_results["aperture_ratio"] * 100,
                        }
                    )
                    st.dataframe(df, use_container_width=True)

                    # CSV 下载
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 下载 CSV",
                        data=csv,
                        file_name=f"stage1_dynamic_V{V_from}_to_V{V_to}.csv",
                        mime="text/csv",
                    )

    # =========================================================================
    # Tab 2: Steady State
    # =========================================================================
    with tab2:
        st.markdown("### 稳态特性分析")
        st.markdown("展示不同电压下的稳态接触角和开口率。")

        # 显示稳态特性曲线
        with st.spinner("正在计算稳态特性..."):
            steady_results = {"model": model}
            fig_steady = create_stage1_plots(steady_results, plot_type="steady")
            st.plotly_chart(fig_steady, use_container_width=True)

        # 特定点查询
        st.markdown("#### 特定电压查询")
        query_voltage = st.slider(
            "查询电压 (V)",
            min_value=0.0,
            max_value=30.0,
            value=15.0,
            step=0.5,
            key="query_voltage",
        )

        if st.button("📊 查询该电压", key="query_steady"):
            result = compute_steady_state(model, query_voltage)

            qcol1, qcol2, qcol3 = st.columns(3)
            with qcol1:
                st.metric("接触角", f"{result['theta']:.1f}°")
            with qcol2:
                st.metric("开口率", f"{result['aperture_ratio'] * 100:.1f}%")
            with qcol3:
                st.metric("透明区半径", f"{result['r_open'] * 1e6:.2f} μm")

    # =========================================================================
    # Tab 3: Theory
    # =========================================================================
    with tab3:
        render_young_lippmann_theory()


def render_young_lippmann_theory():
    """渲染 Young-Lippmann 理论说明"""
    st.markdown("### Young-Lippmann 方程")

    st.markdown("""
    #### 电润湿原理

    电润湿 (Electrowetting) 是一种通过施加电压改变固体表面润湿性的现象。
    当在绝缘层两侧施加电压时，液滴的接触角会发生变化。

    #### Young-Lippmann 方程

    电润湿效应由 Young-Lippmann 方程描述：
    """)

    # 使用 LaTeX 渲染方程
    st.latex(
        r"\cos\theta(V) = \cos\theta_0 + \frac{\varepsilon_r \varepsilon_0}{2\gamma d} V^2"
    )

    st.markdown("""
    其中：
    - $\\theta(V)$: 电压 $V$ 下的接触角
    - $\\theta_0$: 初始接触角（零电压时）
    - $\\varepsilon_r$: 相对介电常数
    - $\\varepsilon_0$: 真空介电常数 ($8.854 \\times 10^{-12}$ F/m)
    - $\\gamma$: 表面张力 (N/m)
    - $d$: 介电层厚度 (m)
    - $V$: 施加电压 (V)
    """)

    # 显示物理参数
    st.markdown("#### 本模型物理参数")

    params = {
        "初始接触角 θ₀": f"{PHYSICS['theta0']}°",
        "像素壁接触角": f"{PHYSICS['theta_wall']}°",
        "有效介电常数 εᵣ": f"{PHYSICS['epsilon_r']}",
        "表面张力 γ": f"{PHYSICS['gamma']} N/m",
        "介电层厚度 d": f"{PHYSICS['d_dielectric'] * 1e9:.0f} nm",
        "疏水层厚度": f"{PHYSICS['d_hydrophobic'] * 1e9:.0f} nm",
        "阈值电压": f"{PHYSICS['V_threshold']} V",
        "像素宽度 Lx": f"{PHYSICS['Lx'] * 1e6:.0f} μm",
    }

    param_df = pd.DataFrame(
        {
            "参数": list(params.keys()),
            "值": list(params.values()),
        }
    )
    st.table(param_df)

    # 动态参数
    st.markdown("#### 动态响应参数")

    dynamic_params = {
        "时间常数 τ": f"{PHYSICS['tau'] * 1000:.1f} ms",
        "恢复时间常数 τ_recovery": f"{PHYSICS['tau_recovery'] * 1000:.1f} ms",
        "阻尼比 ζ": f"{PHYSICS['zeta']}",
    }

    dynamic_df = pd.DataFrame(
        {
            "参数": list(dynamic_params.keys()),
            "值": list(dynamic_params.values()),
        }
    )
    st.table(dynamic_df)

    # 开口率计算
    st.markdown("#### 开口率计算")

    st.markdown("""
    开口率 (Aperture Ratio) 表示像素的透明区域比例，由接触角推导：

    $$\\eta = \\sin^2\\left(\\frac{\\theta}{2}\\right)$$

    开口率越高，像素越亮（透光越多）。
    """)

    # 示例计算
    st.markdown("#### 示例计算")

    example_volts = [0, 10, 20, 30]
    example_results = []

    for V in example_volts:
        result = compute_steady_state(EnhancedApertureModel(), V)
        example_results.append(
            {
                "电压 (V)": V,
                "接触角 (°)": f"{result['theta']:.1f}",
                "开口率 (%)": f"{result['aperture_ratio'] * 100:.1f}",
                "透明区半径 (μm)": f"{result['r_open'] * 1e6:.2f}",
            }
        )

    example_df = pd.DataFrame(example_results)
    st.table(example_df)


def render_voltage_sequence_input() -> Tuple[float, float, float]:
    """
    Render voltage sequence input controls in Streamlit.

    Returns:
        (V_from, V_to, t_since) Voltage jump sequence parameters
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        V_from = st.number_input(
            "起始电压 V_from (V)",
            min_value=0.0,
            max_value=30.0,
            value=0.0,
            step=1.0,
            key="v_from",
        )

    with col2:
        V_to = st.number_input(
            "目标电压 V_to (V)",
            min_value=0.0,
            max_value=30.0,
            value=30.0,
            step=1.0,
            key="v_to",
        )

    with col3:
        t_since = st.number_input(
            "时间 t_since (s)",
            min_value=0.0,
            max_value=0.10,
            value=0.005,
            step=0.001,
            format="%.4f",
            key="t_since",
        )

    return V_from, V_to, t_since


def render_contact_angle_prediction(V_from: float, V_to: float, t_since: float) -> None:
    """渲染接触角预测结果

    Args:
        V_from: 起始电压 (V)
        V_to: 目标电压 (V)
        t_since: 时间间隔 (s)
    """
    pass


def render_aperture_prediction(V_from: float, V_to: float, t_since: float) -> None:
    """渲染开口率预测结果

    Args:
        V_from: 起始电压 (V)
        V_to: 目标电压 (V)
        t_since: 时间间隔 (s)
    """
    pass


def render_young_lippmann_equation() -> None:
    """渲染 Young-Lippmann 方程可视化

    展示电润湿接触角方程的物理原理。
    """
    pass


def render_stage1_calibration_info() -> None:
    """渲染 Stage 1 模型校准信息

    显示校准参数和验证结果。
    """
    pass
