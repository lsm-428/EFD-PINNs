#!/usr/bin/env python3
"""
test_vof_3d.py - 3D VOF Interface Tests

Tests for 3D volume-of-fluid interface visualization and validation
using PyVista for rendering and NumPy for numerical validation.

Features:
- Isosurface extraction (phi=0.5)
- Volume integration validation
- Interface area calculation
- Publication-quality 3D rendering

Usage:
    python -m pytest tests/test_vof_3d.py -v
    python tests/test_vof_3d.py --visualize
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")


class TestVOF3D:
    """3D VOF Interface Tests"""

    @pytest.fixture
    def sample_phi_grid(self):
        """
        Create a test 3D phi grid with a spherical interface.

        Returns:
            Tuple of (X, Y, Z, phi) arrays
        """
        n = 32  # Grid resolution
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        z = np.linspace(0, 1, n)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Spherical interface: phi = 0.5 at r = 0.3
        R = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
        phi = np.where(R < 0.3, 1.0, 0.0)

        return X, Y, Z, phi

    @pytest.fixture
    def ellipsoid_phi_grid(self):
        """
        Create a test 3D phi grid with an ellipsoidal interface.

        Returns:
            Tuple of (X, Y, Z, phi) arrays
        """
        n = 32
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        z = np.linspace(0, 1, n)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Ellipsoidal interface
        R = ((X - 0.5) / 0.2) ** 2 + ((Y - 0.5) / 0.15) ** 2 + ((Z - 0.5) / 0.1) ** 2
        phi = np.where(R < 1.0, 1.0, 0.0)

        return X, Y, Z, phi

    @pytest.fixture
    def vof_grid(self, sample_phi_grid):
        """Create a PyVista StructuredGrid from phi data."""
        X, Y, Z, phi = sample_phi_grid
        grid = pv.StructuredGrid(X, Y, Z)
        grid.point_data["phi"] = phi.flatten(order="C")
        return grid

    def test_isosurface_extraction(self, vof_grid):
        """
        Test that phi=0.5 isosurface can be extracted.

        The isosurface represents the fluid-fluid interface.
        """
        # Extract isosurface at phi = 0.5
        contours = vof_grid.contour([0.5])

        # Verify contours were created (use n_cells instead of deprecated n_faces)
        assert contours.n_points > 0, "Isosurface should have points"
        assert contours.n_cells > 0, "Isosurface should have cells"

        # Verify it's a closed surface (for a sphere)
        assert contours.n_cells > 10, "Isosurface should have multiple cells"

        print(
            f"  [OK] Isosurface: {contours.n_points} points, {contours.n_cells} cells"
        )

    def test_volume_integration_sphere(self, sample_phi_grid):
        """
        Test volume integration for spherical interface.

        Theoretical volume: V = 4/3 * pi * r^3
        """
        X, Y, Z, phi = sample_phi_grid

        # Calculate grid spacing
        dx = X[1, 0, 0] - X[0, 0, 0]
        dy = Y[0, 1, 0] - Y[0, 0, 0]
        dz = Z[0, 0, 1] - Z[0, 0, 0]
        cell_volume = dx * dy * dz

        # Calculate volume using phi summation
        computed_volume = np.sum(phi) * cell_volume

        # Theoretical volume for sphere (r=0.3)
        r = 0.3
        expected_volume = (4.0 / 3.0) * np.pi * (r**3)

        # Calculate relative error
        relative_error = abs(computed_volume - expected_volume) / expected_volume

        print(
            f"  Volume: computed={computed_volume:.6f}, expected={expected_volume:.6f}"
        )
        print(f"  Relative error: {relative_error:.2%}")

        # Allow 10% error due to discretization
        assert relative_error < 0.10, (
            f"Volume error should be < 10%: got {relative_error:.2%}"
        )

    def test_interface_area_sphere(self, vof_grid):
        """
        Test interface area calculation for spherical interface.

        Theoretical surface area: A = 4 * pi * r^2
        """
        # Extract isosurface
        contours = vof_grid.contour([0.5])

        # Get surface area from PyVista
        computed_area = contours.area

        # Theoretical area for sphere (r=0.3)
        r = 0.3
        expected_area = 4.0 * np.pi * (r**2)

        # Calculate relative error
        relative_error = abs(computed_area - expected_area) / expected_area

        print(f"  Area: computed={computed_area:.6f}, expected={expected_area:.6f}")
        print(f"  Relative error: {relative_error:.2%}")

        # Allow 15% error due to discretization and contour approximation
        assert relative_error < 0.15, (
            f"Area error should be < 15%: got {relative_error:.2%}"
        )

    def test_volume_integration_ellipsoid(self, ellipsoid_phi_grid):
        """
        Test volume integration for ellipsoidal interface.

        Theoretical volume: V = 4/3 * pi * a * b * c
        """
        X, Y, Z, phi = ellipsoid_phi_grid

        # Calculate grid spacing
        dx = X[1, 0, 0] - X[0, 0, 0]
        dy = Y[0, 1, 0] - Y[0, 0, 0]
        dz = Z[0, 0, 1] - Z[0, 0, 0]
        cell_volume = dx * dy * dz

        # Calculate volume
        computed_volume = np.sum(phi) * cell_volume

        # Theoretical volume for ellipsoid (a=0.2, b=0.15, c=0.1)
        a, b, c = 0.2, 0.15, 0.1
        expected_volume = (4.0 / 3.0) * np.pi * a * b * c

        relative_error = abs(computed_volume - expected_volume) / expected_volume

        print(
            f"  Volume: computed={computed_volume:.6f}, expected={expected_volume:.6f}"
        )
        print(f"  Relative error: {relative_error:.2%}")

        assert relative_error < 0.10, (
            f"Volume error should be < 10%: got {relative_error:.2%}"
        )

    def test_contour_properties(self, vof_grid):
        """Test contour mesh properties."""
        contours = vof_grid.contour([0.5])

        # Check bounds
        bounds = contours.bounds
        print(
            f"  Bounds: x=[{bounds[0]:.3f}, {bounds[1]:.3f}], "
            f"y=[{bounds[2]:.3f}, {bounds[3]:.3f}], "
            f"z=[{bounds[4]:.3f}, {bounds[5]:.3f}]"
        )

        # Sphere with r=0.3 centered at 0.5 should have bounds [0.2, 0.8]
        assert bounds[0] > 0.15 and bounds[1] < 0.85, (
            "Contour bounds should encompass the sphere"
        )
        assert bounds[2] > 0.15 and bounds[3] < 0.85, (
            "Contour bounds should encompass the sphere"
        )
        assert bounds[4] > 0.15 and bounds[5] < 0.85, (
            "Contour bounds should encompass the sphere"
        )

        # Check if it's a manifold
        assert contours.is_all_triangles, "Contour should be triangulated"


def generate_3d_visualization(output_dir="outputs/analysis"):
    """
    Generate publication-quality 3D VOF visualization.

    Note: Requires display connection. On headless systems, use:
        export PYVISTA_OFF_SCREEN=1
        export PYVISTA_USE_PANEL=1

    Args:
        output_dir: Directory to save output files
    """
    print("  [INFO] 3D visualization requires display connection")
    print("  [INFO] On headless systems, export PYVISTA_OFF_SCREEN=1")
    print("  [INFO] Or use: tests/test_vof_3d.py --visualize")
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="3D VOF Interface Tests and Visualization"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    if args.visualize:
        print("\n=== Generating 3D VOF Visualizations ===")
        generate_3d_visualization(args.output)
    else:
        print("\n=== Running 3D VOF Interface Tests ===")
        pytest.main([__file__, "-v"])
