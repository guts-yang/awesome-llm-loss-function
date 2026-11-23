"""3D surfaces, 2D downhill fields, and 1D logit thrust."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT = "PingFang SC, Noto Sans CJK SC, Arial Unicode MS, sans-serif"
COLORS = ["#0F6E6E", "#C4491D", "#3D4A8F", "#8A6A12"]


def _base(fig, title: str, height: int = 480):
    fig.update_layout(
        title=dict(text=title, x=0.0),
        font=dict(family=FONT, size=14, color="#1C2430"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F7F4EE",
        margin=dict(l=8, r=8, t=52, b=8),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def surface_figure(x_grid, y_grid, values, x_label: str, y_label: str, y_slice: float | None = None, title: str = "三维势能"):
    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=values,
            colorscale="Teal",
            opacity=0.96,
            showscale=True,
            colorbar=dict(title="L", len=0.7),
            name="损失",
        )
    )
    if y_slice is not None:
        x_line = x_grid[0]
        z_low = float(np.nanmin(values))
        z_high = float(np.nanmax(values))
        plane_x = np.vstack([x_line, x_line])
        plane_y = np.full_like(plane_x, y_slice)
        plane_z = np.vstack([np.full_like(x_line, z_low), np.full_like(x_line, z_high)])
        figure.add_trace(
            go.Surface(
                x=plane_x,
                y=plane_y,
                z=plane_z,
                opacity=0.35,
                showscale=False,
                colorscale=[[0, "#C4491D"], [1, "#C4491D"]],
                name="切面",
            )
        )
    figure.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title="损失 L",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.9)),
        )
    )
    return _base(figure, title, height=480)


def _downhill(x_grid, y_grid, values, stride: int = 4):
    y_step = float(y_grid[1, 0] - y_grid[0, 0])
    x_step = float(x_grid[0, 1] - x_grid[0, 0])
    d_loss_dy, d_loss_dx = np.gradient(values, y_step, x_step)
    x_points = x_grid[::stride, ::stride]
    y_points = y_grid[::stride, ::stride]
    downhill_x = -d_loss_dx[::stride, ::stride]
    downhill_y = -d_loss_dy[::stride, ::stride]
    norm = np.sqrt(downhill_x**2 + downhill_y**2) + 1e-12
    return x_points, y_points, downhill_x / norm, downhill_y / norm


def contour_figure(x_grid, y_grid, values, x_label: str, y_label: str, y_slice: float, paths=None, title: str = "二维下降"):
    figure = go.Figure()
    figure.add_trace(
        go.Contour(
            x=x_grid[0],
            y=y_grid[:, 0],
            z=values,
            colorscale="Teal",
            contours=dict(showlabels=False),
            colorbar=dict(title="L", len=0.7),
            name="等高线",
        )
    )
    x_points, y_points, arrow_x, arrow_y = _downhill(x_grid, y_grid, values)
    span = float(x_grid[0, -1] - x_grid[0, 0])
    length = 0.045 * span
    arrow_line_x = []
    arrow_line_y = []
    for row in range(x_points.shape[0]):
        for col in range(x_points.shape[1]):
            start_x = float(x_points[row, col])
            start_y = float(y_points[row, col])
            arrow_line_x.extend([start_x, start_x + length * float(arrow_x[row, col]), None])
            arrow_line_y.extend([start_y, start_y + length * float(arrow_y[row, col]), None])
    figure.add_trace(
        go.Scatter(x=arrow_line_x, y=arrow_line_y, mode="lines", line=dict(color="#1C2430", width=1.2), name="下降方向", hoverinfo="skip")
    )
    figure.add_trace(
        go.Scatter(
            x=x_grid[0],
            y=np.full(x_grid.shape[1], y_slice),
            mode="lines",
            line=dict(color="#C4491D", width=3, dash="dash"),
            name="一维切面",
        )
    )
    if paths:
        for index, path in enumerate(paths):
            figure.add_trace(
                go.Scatter(
                    x=path["x"],
                    y=path["y"],
                    mode="lines+markers",
                    name=path["name"],
                    line=dict(color=COLORS[index % len(COLORS)], width=3),
                )
            )
    figure.update_xaxes(title=x_label, range=[float(x_grid.min()), float(x_grid.max())])
    figure.update_yaxes(title=y_label, range=[float(y_grid.min()), float(y_grid.max())], scaleanchor="x", scaleratio=1)
    return _base(figure, title, height=480)


def slice_figure(curves, x_label: str, title: str = "一维切面"):
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("损失势能", "logit 梯度推力 |dL/dz|"))
    for index, curve in enumerate(curves):
        color = COLORS[index % len(COLORS)]
        figure.add_trace(
            go.Scatter(x=curve["probability"], y=curve["loss"], mode="lines", name=curve["name"], line=dict(color=color, width=2.5), legendgroup=curve["name"]),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=curve["probability"],
                y=np.abs(curve["thrust"]),
                mode="lines",
                name=curve["name"],
                line=dict(color=color, width=2.5),
                legendgroup=curve["name"],
                showlegend=False,
                customdata=curve["thrust"],
                hovertemplate="p=%{x:.3f}<br>|dL/dz|=%{y:.4f}<br>dL/dz=%{customdata:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.update_xaxes(title_text=x_label, row=2, col=1)
    figure.update_yaxes(title_text="L", row=1, col=1)
    figure.update_yaxes(title_text="|dL/dz|", row=2, col=1)
    return _base(figure, title, height=480)


def trajectory_figure(runs, title: str = "优化轨迹"):
    figure = go.Figure()
    for index, run in enumerate(runs):
        path = run["result"]["path"]
        color = COLORS[index % len(COLORS)]
        figure.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines+markers",
                name=run["name"],
                line=dict(color=color, width=3),
                marker=dict(size=6),
            )
        )
    figure.update_xaxes(title="参数 θ1（遗忘特征方向）")
    figure.update_yaxes(title="参数 θ2", scaleanchor="x", scaleratio=1)
    return _base(figure, title, height=480)


def arc_figure(runs, title: str = "弧长投影"):
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("损失", "保留通道概率"))
    for index, run in enumerate(runs):
        result = run["result"]
        color = COLORS[index % len(COLORS)]
        figure.add_trace(
            go.Scatter(x=result["arc"], y=result["loss"], mode="lines", name=run["name"], line=dict(color=color, width=2.5), legendgroup=run["name"]),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=result["arc"],
                y=result["retain_probability"],
                mode="lines",
                name=run["name"],
                line=dict(color=color, width=2.5),
                legendgroup=run["name"],
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    figure.update_xaxes(title_text="弧长 s", row=2, col=1)
    figure.update_yaxes(title_text="L", row=1, col=1)
    figure.update_yaxes(title_text="p_r", row=2, col=1)
    return _base(figure, title, height=480)
