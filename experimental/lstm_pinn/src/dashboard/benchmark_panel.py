"""
性能基准测试面板组件
===================

用于在 Streamlit 仪表板中展示性能基准测试结果。
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.dashboard.inference import PINNInferenceEngine


def run_single_benchmark(
    model: torch.nn.Module,
    batch_sizes: Optional[List[int]] = None,
    device: str = "cuda",
) -> Dict[int, Dict[str, float]]:
    """Run inference latency benchmark.

    Args:
        model: PyTorch model to benchmark
        batch_sizes: List of batch sizes to test (default: [1, 8, 64, 512, 4096])
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dictionary mapping batch_size to performance metrics:
        {batch_size: {avg_ms, min_ms, max_ms, std_ms, samples_per_sec}}
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 64, 512, 4096]

    model.eval()
    results = {}

    for bs in batch_sizes:
        x = torch.randn(bs, 6, device=device)

        with torch.no_grad():
            for _ in range(5):
                _ = model(x)

        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            times.append((time.perf_counter() - start) * 1000)

        times = sorted(times)[5:-5]

        avg = sum(times) / len(times)
        results[bs] = {
            "avg_ms": avg,
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": np.std(times),
            "samples_per_sec": 1000 / avg if avg > 0 else 0,
        }

    return results


def run_throughput_test(
    model: torch.nn.Module, max_batch: int = 10000, device: str = "cuda"
) -> Dict[str, Any]:
    """Run throughput test across different batch sizes.

    Args:
        model: PyTorch model to test
        max_batch: Maximum batch size to test
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dictionary containing throughput results:
        {batch_sizes: [], throughputs: [], max_throughput: float, max_batch_size: int}
    """
    model.eval()

    batch_sizes = [1, 10, 100, 1000, 5000, 10000]
    batch_sizes = [bs for bs in batch_sizes if bs <= max_batch]

    throughputs = []

    for bs in batch_sizes:
        x = torch.randn(bs, 6, device=device)

        with torch.no_grad():
            for _ in range(3):
                _ = model(x)

        start = time.perf_counter()
        iterations = 100
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / (iterations * bs)) * 1000
        throughput = 1000 / latency_ms
        throughputs.append(throughput)

    max_tp = max(throughputs)
    max_bs = batch_sizes[throughputs.index(max_tp)]

    return {
        "batch_sizes": batch_sizes,
        "throughputs": throughputs,
        "max_throughput": max_tp,
        "max_batch_size": max_bs,
    }


def get_gpu_info(model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """Get GPU information and model statistics.

    Args:
        model: Optional PyTorch model to analyze

    Returns:
        Dictionary containing GPU info and model stats:
        {device, gpu_name, memory_allocated_mb, memory_reserved_mb,
         total_parameters, trainable_parameters, device_type}
    """
    info = {
        "device": None,
        "gpu_name": None,
        "memory_allocated_mb": 0,
        "memory_reserved_mb": 0,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "device_type": "cpu",
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
        info["device"] = str(device)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["device_type"] = "cuda"

        torch.cuda.synchronize()
        info["memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024**2
        info["memory_reserved_mb"] = torch.cuda.memory_reserved(0) / 1024**2
    else:
        device = torch.device("cpu")
        info["device"] = str(device)
        info["device_type"] = "cpu"

    if model is not None:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info["total_parameters"] = total
        info["trainable_parameters"] = trainable

    return info


def render_benchmark_tab(engine: Optional["PINNInferenceEngine"]) -> None:
    """Render the benchmark tab in Streamlit dashboard.

    Args:
        engine: PINNInferenceEngine instance for model access
    """
    import plotly.graph_objects as go

    st.subheader("⚡ Performance Benchmark")

    # Handle case when engine is None
    if engine is None:
        st.warning("⚠️ No model loaded. Please load a model first.")
        st.info("Use the 'Load Model' button in the sidebar to load a trained model.")
        return

    # Get the model from engine
    model = getattr(engine, "model", None)
    if model is None:
        st.warning("⚠️ Model not available in engine.")
        return

    # GPU info section
    st.markdown("### 🖥️ Device Information")
    gpu_info = get_gpu_info(model)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Device", gpu_info.get("device_type", "cpu").upper())
    with col2:
        if gpu_info.get("gpu_name"):
            st.metric("GPU", gpu_info.get("gpu_name"))
        else:
            st.metric("GPU", "N/A")
    with col3:
        st.metric("Total Parameters", f"{gpu_info.get('total_parameters', 0):,}")
    with col4:
        if gpu_info.get("device_type") == "cuda":
            st.metric("Memory (MB)", f"{gpu_info.get('memory_allocated_mb', 0):.1f}")
        else:
            st.metric("Memory (MB)", "N/A")

    st.divider()

    # Benchmark controls
    st.markdown("### ⚙️ Benchmark Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Device selector
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.append("cuda")
        device = st.selectbox(
            "Select Device",
            available_devices,
            index=len(available_devices) - 1 if torch.cuda.is_available() else 0,
            help="Choose the device to run benchmark on",
        )

    with col2:
        # Batch size selector
        default_batch_sizes = [1, 8, 64, 512, 4096]
        batch_sizes = st.multiselect(
            "Select Batch Sizes",
            options=[1, 8, 64, 256, 512, 1024, 2048, 4096, 8192],
            default=default_batch_sizes,
            help="Select batch sizes to benchmark",
        )

    # Run benchmark button
    run_benchmark = st.button(
        "🚀 Run Benchmark", type="primary", use_container_width=True
    )

    # Initialize session state for results
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None
    if "throughput_results" not in st.session_state:
        st.session_state.throughput_results = None

    # Run benchmarks when button is clicked
    if run_benchmark:
        if not batch_sizes:
            st.error("Please select at least one batch size.")
        else:
            # Move model to selected device
            model_to_device = model.to(device)
            model_to_device.eval()

            # Run single benchmark
            with st.spinner(f"Running latency benchmark on {device.upper()}..."):
                benchmark_results = run_single_benchmark(
                    model_to_device, batch_sizes=batch_sizes, device=device
                )
                st.session_state.benchmark_results = benchmark_results

            # Run throughput test
            max_batch = max(batch_sizes) if batch_sizes else 4096
            with st.spinner("Running throughput test..."):
                throughput_results = run_throughput_test(
                    model_to_device, max_batch=max_batch, device=device
                )
                st.session_state.throughput_results = throughput_results

            st.success("✅ Benchmark completed!")

    # Display results
    benchmark_results = st.session_state.benchmark_results
    throughput_results = st.session_state.throughput_results

    if benchmark_results:
        st.divider()
        st.markdown("### 📊 Benchmark Results")

        # Create results table
        results_data = []
        for bs, metrics in benchmark_results.items():
            results_data.append(
                {
                    "Batch Size": bs,
                    "Avg Latency (ms)": f"{metrics['avg_ms']:.3f}",
                    "Min Latency (ms)": f"{metrics['min_ms']:.3f}",
                    "Max Latency (ms)": f"{metrics['max_ms']:.3f}",
                    "Std (ms)": f"{metrics['std_ms']:.3f}",
                    "Throughput (samples/s)": f"{metrics['samples_per_sec']:,.0f}",
                }
            )

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Latency chart
        st.markdown("#### ⏱️ Latency vs Batch Size")
        fig_latency = go.Figure()

        batch_size_list = list(benchmark_results.keys())
        avg_latencies = [benchmark_results[bs]["avg_ms"] for bs in batch_size_list]

        fig_latency.add_trace(
            go.Bar(
                x=batch_size_list,
                y=avg_latencies,
                name="Avg Latency",
                marker_color="#1f77b4",
                text=[f"{lat:.2f} ms" for lat in avg_latencies],
                textposition="outside",
            )
        )

        fig_latency.update_layout(
            title="Average Latency by Batch Size",
            xaxis_title="Batch Size",
            yaxis_title="Latency (ms)",
            template="plotly_white",
            showlegend=False,
        )

        st.plotly_chart(fig_latency, use_container_width=True)

        # Throughput chart
        st.markdown("#### 🚀 Throughput vs Batch Size")
        fig_throughput = go.Figure()

        throughputs = [
            benchmark_results[bs]["samples_per_sec"] for bs in batch_size_list
        ]

        fig_throughput.add_trace(
            go.Scatter(
                x=batch_size_list,
                y=throughputs,
                mode="lines+markers",
                name="Throughput",
                line=dict(color="#2ca02c", width=3),
                marker=dict(size=10),
                text=[f"{tp:,.0f} samples/s" for tp in throughputs],
            )
        )

        # Find optimal batch size
        max_tp_idx = throughputs.index(max(throughputs))
        fig_throughput.add_annotation(
            x=batch_size_list[max_tp_idx],
            y=throughputs[max_tp_idx],
            text=f"Peak: {throughputs[max_tp_idx]:,.0f} samples/s",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ff7f0e",
            font=dict(color="#ff7f0e", size=12),
        )

        fig_throughput.update_layout(
            title="Throughput by Batch Size",
            xaxis_title="Batch Size",
            yaxis_title="Throughput (samples/s)",
            template="plotly_white",
            showlegend=False,
        )

        st.plotly_chart(fig_throughput, use_container_width=True)

    if throughput_results:
        st.markdown("#### 📈 Throughput Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Max Throughput",
                f"{throughput_results['max_throughput']:,.0f} samples/s",
            )
        with col2:
            st.metric(
                "Optimal Batch Size",
                f"{throughput_results['max_batch_size']:,}",
            )


def load_benchmark_data(benchmark_file: str) -> Optional[Dict[str, Any]]:
    """加载基准测试数据

    Args:
        benchmark_file: 基准测试数据文件路径

    Returns:
        基准测试数据字典，如果加载失败则返回 None
    """
    pass


def render_benchmark_charts(benchmark_data: Dict[str, Any]) -> None:
    """渲染基准测试图表

    Args:
        benchmark_data: 基准测试数据
    """
    pass


def render_benchmark_summary(benchmark_data: Dict[str, Any]) -> None:
    """渲染基准测试摘要信息

    Args:
        benchmark_data: 基准测试数据
    """
    pass
