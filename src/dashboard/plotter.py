"""
Flow Field Plotter
==================

Standardized plotting utilities for EFD3D project.
Ensures consistent styling (fonts, labels, colormaps) across Web App, CLI, and Notebooks.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
from typing import Optional, Dict

try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: PyVista not found. 3D high-quality rendering disabled.")


class FlowFieldPlotter:
    def __init__(self):
        # Set global style
        self._set_style()

    def _set_style(self):
        """Configure matplotlib for publication-quality figures"""
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 12

    def _get_grid(self, data, key):
        """
        Helper to get grid coordinates, handles both 'X_grid' and 'X' key formats.
        """
        # Try 'X_grid' first (expected by plotter), fall back to 'X' (returned by inference)
        grid_key = f"{key}_grid"
        if grid_key in data:
            return data[grid_key]
        elif key in data:
            return data[key]
        else:
            raise KeyError(
                f"Neither '{grid_key}' nor '{key}' found in data. Available keys: {list(data.keys())}"
            )

    def plot_phase_field(
        self,
        ax: matplotlib.axes.Axes,
        data: Dict[str, np.ndarray],
        title: Optional[str] = None,
    ):
        """
        Plot Phase Field (Phi) Contour.

        Args:
            ax: Matplotlib axes
            data: Dictionary from PINNInferenceEngine.predict_field
            title: Custom title
        """
        X = self._get_grid(data, "X") * 1e6  # Convert to um
        Y = self._get_grid(data, "Y") * 1e6
        phi = data["phi"]
        labels = data.get("labels", ("x", "y"))

        # Contour Plot
        levels = np.linspace(0, 1, 21)
        im = ax.contourf(X, Y, phi, levels=levels, cmap="Blues", vmin=0, vmax=1)

        # Interface Line (phi=0.5)
        ax.contour(X, Y, phi, levels=[0.5], colors="red", linewidths=2)

        ax.set_xlabel(f"{labels[0]} (μm)")
        ax.set_ylabel(f"{labels[1]} (μm)")

        # Auto aspect ratio is usually better for XZ/YZ which are thin
        # But 'equal' is physically correct.
        # Let's use 'equal' but ensure limits are correct
        ax.set_aspect("equal")

        if title:
            ax.set_title(title)
        else:
            t_ms = data["t"] * 1000
            plane = data.get("plane", "xy").upper()
            ax.set_title(f"{plane} Phase Field φ (t={t_ms:.1f}ms)")

        return im

    def plot_velocity_field(
        self,
        ax: matplotlib.axes.Axes,
        data: Dict[str, np.ndarray],
        title: Optional[str] = None,
    ):
        """
        Plot Velocity Magnitude and Streamlines.
        """
        X = self._get_grid(data, "X") * 1e6
        Y = self._get_grid(data, "Y") * 1e6

        # Use in-plane velocity components if available, else fallback
        u = data.get("vel_u", data["u"])
        v = data.get("vel_v", data["v"])
        vel_mag = data["vel_mag"]
        labels = data.get("labels", ("x", "y"))

        # Magnitude Contour
        im = ax.contourf(X, Y, vel_mag, levels=20, cmap="viridis")

        # Streamlines
        # Adjust density based on aspect ratio?
        # For XZ, Y is small (20um), X is large (174um). Streamplot might need help.
        density = 1.0
        if data.get("plane") in ["xz", "yz"]:
            density = [1.0, 2.0]  # Higher density in Z direction maybe?

        try:
            ax.streamplot(
                X, Y, u, v, color="white", linewidth=0.5, density=density, arrowsize=1.0
            )
        except Exception as e:
            print(f"Streamplot failed: {e}")

        ax.set_xlabel(f"{labels[0]} (μm)")
        ax.set_ylabel(f"{labels[1]} (μm)")
        ax.set_aspect("equal")

        if title:
            ax.set_title(title)
        else:
            plane = data.get("plane", "xy").upper()
            ax.set_title(f"{plane} Velocity Field")

        return im

    def plot_dynamic_response(
        self,
        ax: matplotlib.axes.Axes,
        t: np.ndarray,
        eta: np.ndarray,
        highlight_t: Optional[float] = None,
    ):
        """
        Plot Aperture Ratio vs Time.
        """
        t_ms = t * 1000

        ax.plot(t_ms, eta, "b-", linewidth=2, label="PINN Prediction")

        # Highlight specific time point if provided
        if highlight_t is not None:
            highlight_ms = highlight_t * 1000
            # Find closest eta value
            idx = (np.abs(t - highlight_t)).argmin()
            current_eta = eta[idx]

            ax.plot(highlight_ms, current_eta, "ro", markersize=8)
            ax.axvline(highlight_ms, color="r", linestyle="--", alpha=0.5)
            ax.axhline(current_eta, color="r", linestyle="--", alpha=0.5)
            ax.text(
                highlight_ms,
                current_eta + 0.05,
                f"t={highlight_ms:.1f}ms\nη={current_eta:.2f}",
                color="red",
                fontsize=10,
            )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Aperture Ratio (η)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title("Dynamic Response (0V -> 30V -> 0V)")
        ax.legend()

    def create_dashboard_figure(
        self, data: Dict[str, np.ndarray], show_electric_field: bool = False
    ):
        """
        Create a combined figure for dashboard view.
        Returns matplotlib Figure object.
        """
        cols = 3 if show_electric_field else 2
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))

        # 1. Phase Field
        im1 = self.plot_phase_field(axes[0], data)
        plt.colorbar(im1, ax=axes[0], label="Phase (φ)", fraction=0.046, pad=0.04)

        # 2. Velocity Field
        im2 = self.plot_velocity_field(axes[1], data)
        plt.colorbar(im2, ax=axes[1], label="Velocity (m/s)", fraction=0.046, pad=0.04)

        # 3. Electric Field (Optional)
        if show_electric_field:
            im3 = self.plot_electric_potential(axes[2], data)
            plt.colorbar(
                im3, ax=axes[2], label="Potential (V)", fraction=0.046, pad=0.04
            )

        plt.tight_layout()
        return fig

    def plot_electric_potential(
        self,
        ax: matplotlib.axes.Axes,
        data: Dict[str, np.ndarray],
        title: Optional[str] = None,
    ):
        """
        Plot Electric Potential (Psi).
        """
        X = self._get_grid(data, "X") * 1e6
        Y = self._get_grid(data, "Y") * 1e6
        psi = data["psi"]
        labels = data.get("labels", ("x", "y"))

        # Contour
        im = ax.contourf(X, Y, psi, levels=20, cmap="plasma")

        ax.set_xlabel(f"{labels[0]} (μm)")
        ax.set_ylabel(f"{labels[1]} (μm)")
        ax.set_aspect("equal")

        if title:
            ax.set_title(title)
        else:
            plane = data.get("plane", "xy").upper()
            ax.set_title(f"{plane} Electric Potential (V)")

        return im

    def plot_residual_heatmap(self, data: Dict[str, np.ndarray], residual_key: str):
        """
        Plot heatmap for a specific residual field.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        X = self._get_grid(data, "X") * 1e6
        Y = self._get_grid(data, "Y") * 1e6
        labels = data.get("labels", ("x", "y"))

        if residual_key not in data:
            ax.text(0.5, 0.5, f"Key '{residual_key}' not found", ha="center")
            return fig

        val = data[residual_key]

        # Log scale for better visualization of errors
        # Use abs value
        abs_val = np.abs(val)
        log_val = np.log10(abs_val + 1e-16)

        im = ax.contourf(X, Y, log_val, levels=20, cmap="magma")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"Log10 |Residual|")

        ax.set_xlabel(f"{labels[0]} (μm)")
        ax.set_ylabel(f"{labels[1]} (μm)")
        ax.set_title(f"Residual: {residual_key}")
        ax.set_aspect("equal")

        return fig

    def generate_3d_html(
        self, data: Dict[str, np.ndarray], filename: str = "temp_viz.html"
    ) -> str:
        """
        Generate high-quality 3D visualization using PyVista (VTK).
        Returns the path to the generated HTML file.
        """
        # 1. Create Grid
        # Data comes in (nx, ny, nz) order from inference.py
        # PyVista StructuredGrid expects (nx, ny, nz) but flattened in specific order
        # We can just use the meshgrid arrays directly

        X = data["X"] * 1e6  # um
        Y = data["Y"] * 1e6
        Z = data["Z"] * 1e6
        phi = data["phi"]

        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        grid["phi"] = phi.flatten(
            order="F"
        )  # F-order to match VTK's internal x-fastest memory layout

        # 2. Setup Plotter
        pl = pv.Plotter(off_screen=True)
        pl.set_background("white")

        # 3. Add Ink Interface (Isosurface)
        try:
            contours = grid.contour(isosurfaces=[0.5], scalars="phi")

            # Liquid Ink Style
            pl.add_mesh(
                contours,
                color="#1f77b4",
                smooth_shading=True,
                specular=0.5,  # Shininess
                specular_power=15,  # Tight highlights
                opacity=0.9,
                label="Ink Interface",
            )

        except Exception as e:
            print(f"Isosurface generation failed: {e}")
            # Fallback: Volume rendering if contour fails (e.g., if field is constant)
            # pl.add_volume(grid, cmap="viridis", opacity="linear")
            pass

        # 4. Add Domain Box
        pl.add_bounding_box(color="black", line_width=2)
        pl.show_bounds(
            grid="back", location="outer", ticks="both", font_size=10, color="black"
        )

        # 5. Add Electric Potential (Optional slice?)
        # For now, keep it simple: just the ink.

        # 6. Lighting
        pl.enable_lightkit()

        # 7. Export
        pl.export_html(filename)
        pl.close()

        return filename

    def create_3d_isosurface_figure(
        self, data: Dict[str, np.ndarray], isovalue: float = 0.5
    ):
        """
        Create a 3D interactive plot using Plotly.
        Shows the ink interface (phi=0.5).
        """
        X = data["X"] * 1e6
        Y = data["Y"] * 1e6
        Z = data["Z"] * 1e6
        phi = data["phi"]

        fig = go.Figure(
            data=go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=phi.flatten(),
                isomin=isovalue,
                isomax=isovalue,
                surface_count=1,
                colorscale="Blues",
                caps=dict(x_show=False, y_show=False),
            )
        )

        # Add wireframe box for domain
        t_value = data.get("t", 0)  # Default to 0 if 't' key is missing
        fig.update_layout(
            title=f"3D Ink Interface (t={t_value * 1000:.1f}ms)",
            scene=dict(
                xaxis_title="X (μm)",
                yaxis_title="Y (μm)",
                zaxis_title="Z (μm)",
                aspectmode="data",  # Maintain physical aspect ratio
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )

        return fig
