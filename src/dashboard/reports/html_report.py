#!/usr/bin/env python3
"""
HTML Report Generator for Interactive Dashboard

Provides HTMLReport class for generating interactive, responsive HTML reports
with embedded Plotly visualizations for training analysis.

Usage:
    from src.dashboard.reports.html_report import HTMLReport

    report = HTMLReport(title="Training Report")
    report.add_loss_curves(records)
    report.add_metrics(metrics)
    report.generate("output.html")
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HTMLReport:
    """
    Interactive HTML report generator with embedded Plotly charts.

    Features:
        - Embedded Plotly charts (zoomable, filterable)
        - Responsive design for desktop browsers
        - Multiple visualization types supported
        - Automatic styling and layout
    """

    def __init__(self, title: str = "Training Report", subtitle: str = ""):
        """
        Initialize HTML report generator.

        Args:
            title: Main report title
            subtitle: Optional subtitle
        """
        self.title = title
        self.subtitle = subtitle
        self.charts: List[Dict[str, Any]] = []
        self.sections: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}

    def add_section(self, title: str, content: str, collapsible: bool = False) -> None:
        """
        Add a text section to the report.

        Args:
            title: Section title
            content: HTML content
            collapsible: Whether section is collapsible
        """
        self.sections.append(
            {"title": title, "content": content, "collapsible": collapsible}
        )

    def add_metric(self, name: str, value: Any, unit: str = "") -> None:
        """
        Add a single metric to the report.

        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit
        """
        self.metrics[name] = {"value": value, "unit": unit}

    def add_metrics_grid(self, metrics: Dict[str, Dict[str, Any]]) -> None:
        """
        Add a grid of metrics.

        Args:
            metrics: Dict with format {name: {"value": val, "unit": unit}}
        """
        self.metrics.update(metrics)

    def add_loss_curves(
        self, records: Dict[str, List[Any]], log_scale: bool = True
    ) -> None:
        """
        Add loss curves visualization.

        Args:
            records: Training records dict
            log_scale: Whether to use log scale for y-axis
        """
        epochs = records.get("epoch", list(range(len(list(records.values())[0]))))

        # Identify loss components
        exclude = {"epoch", "stage", "loss_total", "lr"}
        component_names = [k for k in records.keys() if k not in exclude]

        fig = make_subplots(rows=1, cols=1)

        # Total loss
        if "loss_total" in records:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=records["loss_total"],
                    name="Total Loss",
                    mode="lines",
                    line=dict(color="black", width=2),
                )
            )

        # Component losses
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for i, name in enumerate(component_names):
            if name in records:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=records[name],
                        name=name,
                        mode="lines",
                        line=dict(color=colors[i % len(colors)], width=1.5),
                        opacity=0.85,
                    )
                )

        title = "Training Loss Curves" + (" (Log Scale)" if log_scale else "")
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(traceorder="normal"),
        )

        if log_scale:
            fig.update_yaxes(type="log")

        self.charts.append(
            {"title": "Loss Components", "figure": fig, "type": "loss_curves"}
        )

    def add_learning_rate(self, records: Dict[str, List[Any]]) -> None:
        """
        Add learning rate visualization.

        Args:
            records: Training records dict
        """
        epochs = records.get("epoch", list(range(len(list(records.values())[0]))))

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Training Loss", "Learning Rate"),
        )

        # Loss curve
        if "loss_total" in records:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=records["loss_total"],
                    name="Loss",
                    mode="lines",
                    line=dict(color="#1f77b4", width=1.5),
                ),
                row=1,
                col=1,
            )
            fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)

        # Learning rate
        if "lr" in records:
            lr = records["lr"]
            valid = [i for i, v in enumerate(lr) if v is not None and str(v) != "nan"]
            if valid:
                fig.add_trace(
                    go.Scatter(
                        x=[epochs[i] for i in valid],
                        y=[lr[i] for i in valid],
                        name="LR",
                        mode="lines",
                        line=dict(color="#ff7f0e", width=1.5),
                    ),
                    row=2,
                    col=1,
                )
                fig.update_yaxes(title_text="Learning Rate", type="log", row=2, col=1)

        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_layout(
            title="Training Dynamics", template="plotly_white", showlegend=False
        )

        self.charts.append(
            {"title": "Learning Rate", "figure": fig, "type": "learning_curve"}
        )

    def add_loss_fraction(self, records: Dict[str, List[Any]]) -> None:
        """
        Add loss fraction stacked area plot.

        Args:
            records: Training records dict
        """
        epochs = records.get("epoch", list(range(len(list(records.values())[0]))))

        # Identify loss components
        exclude = {"epoch", "stage", "loss_total", "lr"}
        component_names = [k for k in records.keys() if k not in exclude]

        # Prepare fraction data
        comp_values = []
        comp_labels = []

        for name in component_names:
            if name in records:
                values = records[name]
                # Filter valid values
                valid_values = [
                    v for v in values if v is not None and str(v) != "nan" and v > 0
                ]
                if valid_values:
                    comp_values.append(valid_values)
                    comp_labels.append(name)

        if not comp_values:
            return

        # Calculate fractions
        min_len = min(len(v) for v in comp_values)
        frac_array = []
        for i in range(len(comp_values)):
            vals = comp_values[i][:min_len]
            vals = [max(v, 0) for v in vals]
            frac_array.append(vals)

        # Normalize
        total = [
            sum(frac_array[i][j] for i in range(len(frac_array)))
            for j in range(min_len)
        ]
        total = [max(t, 1e-10) for t in total]
        frac_array = [
            [frac_array[i][j] / total[j] for j in range(min_len)]
            for i in range(len(frac_array))
        ]

        fig = go.Figure()

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for i, label in enumerate(comp_labels):
            fig.add_trace(
                go.Scatter(
                    x=epochs[:min_len],
                    y=frac_array[i],
                    name=label,
                    stackgroup="one",
                    mode="none",
                    fillcolor=colors[i % len(colors)],
                    opacity=0.85,
                )
            )

        fig.update_layout(
            title="Relative Loss Contributions",
            xaxis_title="Epoch",
            yaxis_title="Fraction",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            legend=dict(traceorder="normal"),
        )

        self.charts.append(
            {"title": "Loss Fraction", "figure": fig, "type": "loss_fraction"}
        )

    def add_metrics_table(self) -> None:
        """
        Add metrics table visualization.
        """
        if not self.metrics:
            return

        names = []
        values = []
        units = []

        for name, data in self.metrics.items():
            names.append(name)
            values.append(data["value"])
            units.append(data.get("unit", ""))

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["Metric", "Value", "Unit"],
                        fill_color="#667eea",
                        font=dict(color="white", family="Arial", size=12),
                        align="left",
                    ),
                    cells=dict(
                        values=[names, values, units],
                        fill_color="white",
                        font=dict(color="black", family="Arial", size=11),
                        align="left",
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Training Metrics",
            template="plotly_white",
            margin=dict(l=0, r=0, t=30, b=0),
        )

        self.charts.append({"title": "Metrics", "figure": fig, "type": "metrics_table"})

    def add_training_summary(self, records: Dict[str, List[Any]]) -> None:
        """
        Add training summary statistics.

        Args:
            records: Training records dict
        """
        epochs = records.get("epoch", list(range(len(list(records.values())[0]))))
        loss_total = records.get("loss_total", [0])

        epochs = [e for e in epochs if e is not None and str(e) != "nan"]
        loss_total = [l for l in loss_total if l is not None and str(l) != "nan"]

        if not epochs or not loss_total:
            return

        best_idx = loss_total.index(min(loss_total))
        final_loss = loss_total[-1]
        best_loss = loss_total[best_idx]
        best_epoch = epochs[best_idx]

        summary_html = f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0;">
            <div style="background: linear-gradient(135deg, #667eea20, #764ba220); 
                        padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: #667eea;">
                    {len(epochs):,}
                </div>
                <div style="color: #666; font-size: 0.9em; margin-top: 5px;">
                    Total Epochs
                </div>
            </div>
            <div style="background: linear-gradient(135deg, #667eea20, #764ba220); 
                        padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: #667eea;">
                    {best_epoch:,}
                </div>
                <div style="color: #666; font-size: 0.9em; margin-top: 5px;">
                    Best Epoch
                </div>
            </div>
            <div style="background: linear-gradient(135deg, #667eea20, #764ba220); 
                        padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: #667eea;">
                    {final_loss:.2f}
                </div>
                <div style="color: #666; font-size: 0.9em; margin-top: 5px;">
                    Final Loss
                </div>
            </div>
            <div style="background: linear-gradient(135deg, #667eea20, #764ba220); 
                        padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: #667eea;">
                    {best_loss:.2f}
                </div>
                <div style="color: #666; font-size: 0.9em; margin-top: 5px;">
                    Best Loss
                </div>
            </div>
        </div>
        """

        self.sections.append(
            {"title": "Training Summary", "content": summary_html, "collapsible": False}
        )

    def generate(self, output_path: str) -> str:
        """
        Generate HTML report file.

        Args:
            output_path: Path to save HTML file

        Returns:
            Absolute path to generated file
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Generate chart HTML
        chart_html = ""
        for chart in self.charts:
            fig = chart["figure"]
            fig_html = fig.to_html(include_plotlyjs=False, full_html=False)
            chart_html += f"""
            <div class="chart-section">
                <h3>{chart["title"]}</h3>
                <div class="chart-container">{fig_html}</div>
            </div>
            """

        # Generate sections HTML
        sections_html = ""
        for section in self.sections:
            if section["collapsible"]:
                sections_html += f"""
                <details class="section collapsible">
                    <summary>{section["title"]}</summary>
                    <div class="section-content">{section["content"]}</div>
                </details>
                """
            else:
                sections_html += f"""
                <div class="section">
                    <h3>{section["title"]}</h3>
                    <div class="section-content">{section["content"]}</div>
                </div>
                """

        # Generate metrics table HTML if we have metrics
        metrics_html = ""
        if self.metrics:
            names = []
            values = []
            units = []
            for name, data in self.metrics.items():
                names.append(name)
                values.append(data["value"])
                units.append(data.get("unit", ""))

            metrics_html = f"""
            <h2>Training Metrics</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Unit</th>
                    </tr>
                </thead>
                <tbody>
                    {
                "".join(
                    f"<tr><td>{n}</td><td>{v}</td><td>{u}</td></tr>"
                    for n, v, u in zip(names, values, units)
                )
            }
                </tbody>
            </table>
            """

        # Build complete HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        
        h1 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}
        
        h2 {{
            color: #444;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
            margin: 30px 0 20px 0;
            font-size: 1.5em;
        }}
        
        h3 {{
            color: #555;
            margin: 20px 0 10px 0;
            font-size: 1.2em;
        }}
        
        .subtitle {{
            color: #777;
            font-size: 0.95em;
            margin-bottom: 20px;
        }}
        
        /* Chart styling */
        .chart-section {{
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
            border: 1px solid #eee;
        }}
        
        .chart-container {{
            width: 100%;
            min-height: 400px;
        }}
        
        .chart-container .plotly {{
            width: 100% !important;
            height: 100% !important;
        }}
        
        /* Responsive chart sizing */
        @media (min-width: 768px) {{
            .chart-container {{
                min-height: 500px;
            }}
        }}
        
        /* Section styling */
        .section {{
            margin: 25px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
            border: 1px solid #eee;
        }}
        
        .section-content {{
            padding: 15px 0;
        }}
        
        /* Collapsible sections */
        details.section {{
            cursor: pointer;
        }}
        
        details.section[open] {{
            background: #f0f0f0;
        }}
        
        summary {{
            cursor: pointer;
            padding: 10px 0;
            font-weight: 600;
        }}
        
        summary::before {{
            content: '▼ ';
            font-size: 0.8em;
        }}
        
        details.section[open] summary::before {{
            content: '▲ ';
        }}
        
        /* Metrics table */
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
        }}
        
        .metrics-table thead {{
            background: #667eea;
            color: white;
        }}
        
        .metrics-table th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        .metrics-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        
        .metrics-table tr:hover {{
            background: #f5f5f5;
        }}
        
        .metrics-table tbody tr:last-child td {{
            border-bottom: none;
        }}
        
        /* Footer */
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #999;
            font-size: 0.85em;
            text-align: center;
        }}
        
        /* Blink animation for loading */
        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.3; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        {f'<p class="subtitle">{self.subtitle}</p>' if self.subtitle else ""}
        
        {metrics_html}
        
        {sections_html}
        
        {chart_html}
        
        <div class="footer">
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return os.path.abspath(output_path)
