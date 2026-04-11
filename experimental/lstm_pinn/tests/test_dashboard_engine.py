import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add project root
sys.path.append(os.getcwd())

from src.dashboard.inference import PINNInferenceEngine


def test_engine():
    checkpoint_path = "outputs/train/lstm_hybrid_20260122_234758/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Initializing engine with {checkpoint_path}...")
    engine = PINNInferenceEngine(checkpoint_path, device="cpu")  # Force CPU for test

    print("Testing predict_field...")
    field = engine.predict_field(
        t=0.01, voltage_from=0, voltage_to=30, plane="xz", resolution=32
    )
    print(f"Field keys: {field.keys()}")
    print(f"Phi shape: {field['phi'].shape}")

    print("Testing predict_3d_volume...")
    vol = engine.predict_3d_volume(
        t=0.01, voltage_from=0, voltage_to=30, resolution_xy=16, resolution_z=8
    )
    print(f"Volume keys: {vol.keys()}")
    print(f"Phi volume shape: {vol['phi'].shape}")

    print("Testing predict_trajectory...")
    t_sim = np.linspace(0, 0.05, 10)
    func = lambda t: (0.0, 30.0, t)
    traj = engine.predict_trajectory(func, t_sim)
    print(f"Trajectory keys: {traj.keys()}")
    print(f"Eta shape: {traj['eta'].shape}")

    print("Testing check_mass_conservation...")
    mass = engine.check_mass_conservation(t=0.01, voltage_from=0, voltage_to=30)
    print(f"Mass (Volume): {mass}")

    print("Testing compute_residuals...")
    res = engine.compute_residuals(t=0.01, voltage_from=0, voltage_to=30, resolution=32)
    print(f"Residuals keys: {res.keys()}")

    print("All tests passed!")


def test_predict_point():
    """Test predict_point() method for single-point inference."""
    # Use a known existing checkpoint
    checkpoint_paths = [
        "outputs/train/pinn_20260205_174333/best_model.pth",
        "outputs/train/lstm_hybrid_20260122_234758/best_model.pth",
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("No checkpoint found, skipping test_predict_point")
        return

    engine = PINNInferenceEngine(checkpoint_path, device="cpu")

    # Test single point inference with typical 6D Triad input
    result = engine.predict_point(
        x=87e-6, y=87e-6, z=10e-6, V_from=0.0, V_to=20.0, t_since=0.01
    )

    # Verify return type and keys
    assert isinstance(result, dict), "Result should be a dict"
    assert "u" in result, "Result should have 'u' key"
    assert "v" in result, "Result should have 'v' key"
    assert "w" in result, "Result should have 'w' key"
    assert "p" in result, "Result should have 'p' key"
    assert "phi" in result, "Result should have 'phi' key"

    # Verify types
    assert isinstance(result["u"], float), "u should be float"
    assert isinstance(result["v"], float), "v should be float"
    assert isinstance(result["w"], float), "w should be float"
    assert isinstance(result["p"], float), "p should be float"
    assert isinstance(result["phi"], float), "phi should be float"

    # Verify phi is in valid range
    assert 0.0 <= result["phi"] <= 1.0, f"phi should be in [0,1], got {result['phi']}"

    print(f"test_predict_point passed: {result}")


def test_predict_batch():
    """Test predict_batch() method for batch inference."""
    checkpoint_paths = [
        "outputs/train/pinn_20260205_174333/best_model.pth",
        "outputs/train/lstm_hybrid_20260122_234758/best_model.pth",
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("No checkpoint found, skipping test_predict_batch")
        return

    engine = PINNInferenceEngine(checkpoint_path, device="cpu")

    # Create batch of 5 points
    points = np.array(
        [
            [87e-6, 87e-6, 10e-6, 0.0, 20.0, 0.01],
            [87e-6, 87e-6, 10e-6, 0.0, 10.0, 0.01],
            [87e-6, 87e-6, 10e-6, 10.0, 20.0, 0.01],
            [50e-6, 50e-6, 5e-6, 0.0, 20.0, 0.005],
            [100e-6, 100e-6, 15e-6, 0.0, 30.0, 0.02],
        ]
    )

    result = engine.predict_batch(points)

    # Verify return type and keys
    assert isinstance(result, dict), "Result should be a dict"
    assert "u" in result, "Result should have 'u' key"
    assert "v" in result, "Result should have 'v' key"
    assert "w" in result, "Result should have 'w' key"
    assert "p" in result, "Result should have 'p' key"
    assert "phi" in result, "Result should have 'phi' key"

    # Verify shapes
    n_points = len(points)
    assert result["u"].shape == (n_points,), (
        f"u shape should be ({n_points},), got {result['u'].shape}"
    )
    assert result["v"].shape == (n_points,), (
        f"v shape should be ({n_points},), got {result['v'].shape}"
    )
    assert result["w"].shape == (n_points,), (
        f"w shape should be ({n_points},), got {result['w'].shape}"
    )
    assert result["p"].shape == (n_points,), (
        f"p shape should be ({n_points},), got {result['p'].shape}"
    )
    assert result["phi"].shape == (n_points,), (
        f"phi shape should be ({n_points},), got {result['phi'].shape}"
    )

    print(f"test_predict_batch passed: shapes={result['u'].shape}")


def test_check_point_physics():
    """Test check_point_physics() method for physics validation."""
    checkpoint_paths = [
        "outputs/train/pinn_20260205_174333/best_model.pth",
        "outputs/train/lstm_hybrid_20260122_234758/best_model.pth",
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("No checkpoint found, skipping test_check_point_physics")
        return

    engine = PINNInferenceEngine(checkpoint_path, device="cpu")

    # Test physics validation
    result = engine.check_point_physics(
        x=87e-6, y=87e-6, z=10e-6, V_from=0.0, V_to=20.0, t_since=0.01
    )

    # Verify return type and keys
    assert isinstance(result, dict), "Result should be a dict"
    assert "continuity_residual" in result, (
        "Result should have 'continuity_residual' key"
    )
    assert "momentum_residual" in result, "Result should have 'momentum_residual' key"
    assert "mass_conservation_error" in result, (
        "Result should have 'mass_conservation_error' key"
    )

    # Verify types
    assert isinstance(result["continuity_residual"], float), (
        "continuity_residual should be float"
    )
    assert isinstance(result["momentum_residual"], float), (
        "momentum_residual should be float"
    )
    assert isinstance(result["mass_conservation_error"], float), (
        "mass_conservation_error should be float"
    )

    print(f"test_check_point_physics passed: {result}")


if __name__ == "__main__":
    test_engine()
    test_predict_point()
    test_predict_batch()
    test_check_point_physics()
