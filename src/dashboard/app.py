#!/usr/bin/env python3
"""
Panel main application for EFD3D dashboard.

Integrates all 4 functional modules as tabs with shared DataStore and Material theme.
"""

try:
    import panel as pn

    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False
    pn = None

from src.dashboard.datastore import DataStore


# Configure Panel with Material theme if available
if PANEL_AVAILABLE:
    pn.extension(design="material")


# Shared DataStore singleton
DATA_STORE = DataStore()


def create_parameter_sweep_tab():
    """Create Parameter Sweep tab."""
    try:
        from src.dashboard.components.parameter_sweep import ParameterSweep

        # Create parameter sweep instance with shared datastore access
        param_sweep = ParameterSweep()

        if PANEL_AVAILABLE:
            # Create basic controls for parameter sweep
            voltage_min = pn.widgets.FloatSlider(
                name="Min Voltage (V)", start=0, end=30, value=0
            )
            voltage_max = pn.widgets.FloatSlider(
                name="Max Voltage (V)", start=0, end=30, value=30
            )
            time_step = pn.widgets.FloatSlider(
                name="Time Step (s)", start=0.001, end=0.01, value=0.002
            )

            def run_sweep(*events):
                results = param_sweep.perform_grid_sweep(
                    voltage_ranges=[(voltage_min.value, voltage_max.value)],
                    time_steps=[time_step.value],
                )
                # Store results in shared datastore
                DATA_STORE.cache_results("parameter_sweep", results)
                fig = param_sweep.generate_response_surface(results)
                return pn.pane.Matplotlib(fig, dpi=144)

            button = pn.widgets.Button(
                name="Run Parameter Sweep", button_type="primary"
            )
            plot_pane = pn.panel(pn.bind(run_sweep, button))

            return pn.Column(
                pn.Row(voltage_min, voltage_max), pn.Row(time_step, button), plot_pane
            )
        else:
            return "Parameter Sweep Component (requires Panel)"

    except Exception as e:
        return f"Parameter Sweep Error: {e}"


def create_model_comparison_tab():
    """Create Model Comparison tab."""
    try:
        from src.dashboard.components.model_comparison import ModelComparison

        if PANEL_AVAILABLE:
            model_comp = ModelComparison()

            # File selector for checkpoints
            file_input = pn.widgets.FileInput(
                name="Select Checkpoint Files", multiple=True
            )
            compare_button = pn.widgets.Button(
                name="Compare Models", button_type="success"
            )

            def compare_models(*events):
                if file_input.value:
                    # Store selected models in shared datastore
                    DATA_STORE.update_model_paths(file_input.filename)
                    model_comp.select_checkpoints(file_input.filename)
                    model_comp.compare_loss_curves()
                    return pn.pane.Markdown("Model comparison completed")
                return pn.pane.Markdown("Please select checkpoint files first")

            results_pane = pn.panel(pn.bind(compare_models, compare_button))

            return pn.Column(file_input, compare_button, results_pane)
        else:
            return "Model Comparison Component (requires Panel)"

    except Exception as e:
        return f"Model Comparison Error: {e}"


def create_experiment_simulator_tab():
    """Create Experiment Simulator tab."""
    try:
        from src.dashboard.components.experiment_sim import (
            ExperimentSimulator,
            WaveformType,
        )

        if PANEL_AVAILABLE:
            exp_sim = ExperimentSimulator()

            # Waveform type selector
            waveform_type = pn.widgets.Select(
                name="Waveform Type",
                options=[wt.value for wt in WaveformType],
                value=WaveformType.STEP.value,
            )

            # Parameters for different waveforms
            step_params = pn.Column(
                pn.widgets.FloatInput(name="Initial Voltage (V)", value=0),
                pn.widgets.FloatInput(name="Final Voltage (V)", value=30),
                pn.widgets.FloatInput(name="Rise Time (s)", value=0.001),
                pn.widgets.FloatInput(name="Total Time (s)", value=0.01),
            )

            square_params = pn.Column(
                pn.widgets.FloatInput(name="Low Voltage (V)", value=0),
                pn.widgets.FloatInput(name="High Voltage (V)", value=30),
                pn.widgets.FloatInput(name="On Time (s)", value=0.005),
                pn.widgets.FloatInput(name="Off Time (s)", value=0.005),
                pn.widgets.FloatInput(name="Total Time (s)", value=0.02),
            )

            sine_params = pn.Column(
                pn.widgets.FloatInput(name="Offset Voltage (V)", value=15),
                pn.widgets.FloatInput(name="Amplitude (V)", value=15),
                pn.widgets.FloatInput(name="Frequency (Hz)", value=50),
                pn.widgets.FloatInput(name="Total Time (s)", value=0.02),
            )

            def update_params(event):
                # This would dynamically show/hide parameter panels
                pass

            def run_simulation(*events):
                # Get parameters and run simulation
                # Store simulation parameters in shared datastore
                params = {
                    "waveform_type": waveform_type.value,
                    "initial_voltage": step_params[0].value
                    if waveform_type.value == "step"
                    else 0,
                    "final_voltage": step_params[1].value
                    if waveform_type.value == "step"
                    else 0,
                }
                DATA_STORE.update_parameters(**params)

                results = exp_sim.run_experiment(
                    waveform_type=WaveformType(waveform_type.value),
                    params={
                        "V_initial": params["initial_voltage"],
                        "V_final": params["final_voltage"],
                    },
                )
                DATA_STORE.cache_results("experiment_simulation", results)
                return pn.pane.Markdown("Simulation completed")

            run_button = pn.widgets.Button(name="Run Simulation", button_type="primary")
            results_pane = pn.panel(pn.bind(run_simulation, run_button))

            return pn.Column(
                waveform_type,
                pn.Tabs(
                    ("Step", step_params),
                    ("Square", square_params),
                    ("Sine", sine_params),
                ),
                run_button,
                results_pane,
            )
        else:
            return "Experiment Simulator Component (requires Panel)"

    except Exception as e:
        return f"Experiment Simulator Error: {e}"


def create_training_monitor_tab():
    """Create Training Monitor tab."""
    try:
        from src.dashboard.components.training_monitor import TrainingMonitor

        if PANEL_AVAILABLE:
            monitor = TrainingMonitor()

            # Log file input
            log_file = pn.widgets.TextInput(
                name="Training Log Path", placeholder="/path/to/training.log"
            )
            start_button = pn.widgets.Button(
                name="Start Monitoring", button_type="primary"
            )
            stop_button = pn.widgets.Button(
                name="Stop Monitoring", button_type="danger"
            )

            def start_monitoring(*events):
                if log_file.value:
                    monitor.start_watching(log_file.value)
                    DATA_STORE.update_parameters(log_path=log_file.value)
                    return "Monitoring started"
                return "Please specify log file path"

            def stop_monitoring(*events):
                monitor.stop_watching()
                return "Monitoring stopped"

            def update_metrics(*events):
                if monitor.is_watching():
                    metrics = monitor.get_latest_metrics()
                    DATA_STORE.cache_results("training_metrics", metrics)
                    return f"Epoch: {metrics.get('epoch', 'N/A')}, Loss: {metrics.get('loss_total', 'N/A'):.4f}"
                return "Not monitoring"

            start_status = pn.panel(pn.bind(start_monitoring, start_button))
            stop_status = pn.panel(pn.bind(stop_monitoring, stop_button))
            metrics_display = pn.panel(
                pn.bind(update_metrics, start_button, stop_button)
            )

            loss_plot = pn.pane.Markdown("Loss curve will appear here")

            return pn.Column(
                log_file,
                pn.Row(start_button, stop_button),
                start_status,
                stop_status,
                metrics_display,
                loss_plot,
            )
        else:
            return "Training Monitor Component (requires Panel)"

    except Exception as e:
        return f"Training Monitor Error: {e}"


def create_app():
    """
    Create the main Panel application with 4 tabs.

    Returns:
        pn.Tabs: Panel application with integrated components or error messages
    """
    if not PANEL_AVAILABLE:
        raise ImportError("Panel is required to run the dashboard application")

    tabs = pn.Tabs(
        ("Parameter Sweep", create_parameter_sweep_tab()),
        ("Model Comparison", create_model_comparison_tab()),
        ("Experiment Simulator", create_experiment_simulator_tab()),
        ("Training Monitor", create_training_monitor_tab()),
        dynamic=True,
        width=1200,
        height=800,
    )

    return tabs


# Main application entry point
if PANEL_AVAILABLE:
    app = create_app()
else:
    app = None

if __name__ == "__main__":
    if app:
        app.show()
    else:
        print("Panel is not installed. Please install panel to run the dashboard:")
        print("pip install panel")
