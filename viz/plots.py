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


FRAME_COUNT = 36
FRAME_MS = 30
_PULSE_WIDTH = (2.5, 3.3, 2.5)
_PULSE_OPACITY = (1.0, 0.82, 1.0)
_PULSE_HEAD = (7, 11, 7)


def _frame_cuts(length: int, count: int = FRAME_COUNT) -> list[int]:
    if length <= 2:
        return [length]
    cuts = []
    for value in np.rint(np.linspace(2, length, count)).astype(int):
        point = int(max(2, min(length, value)))
        if not cuts or point != cuts[-1]:
            cuts.append(point)
    if cuts[-1] != length:
        cuts.append(length)
    return cuts


def _axis_span(values) -> tuple[float, float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    pad = max((high - low) * 0.08, 1e-3)
    return low - pad, high + pad


def _pulse(index: int, last: bool) -> tuple[float, float, int]:
    if last:
        return 2.5, 1.0, 0
    slot = index % 3
    return _PULSE_WIDTH[slot], _PULSE_OPACITY[slot], _PULSE_HEAD[slot]


def _slice_trace_group(curve, end: int, width: float, opacity: float, head: int):
    probability = curve["probability"][:end]
    loss = curve["loss"][:end]
    thrust = curve["thrust"][:end]
    color = curve["color"]
    name = curve["name"]
    loss_line = go.Scatter(
        x=probability,
        y=loss,
        mode="lines",
        name=name,
        legendgroup=name,
        line=dict(color=color, width=width),
        opacity=opacity,
    )
    thrust_line = go.Scatter(
        x=probability,
        y=np.abs(thrust),
        mode="lines",
        name=name,
        legendgroup=name,
        showlegend=False,
        line=dict(color=color, width=width),
        opacity=opacity,
        customdata=thrust,
        hovertemplate="p=%{x:.3f}<br>|dL/dz|=%{y:.4f}<br>dL/dz=%{customdata:.4f}<extra></extra>",
    )
    loss_head = go.Scatter(
        x=[probability[-1]],
        y=[loss[-1]],
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker=dict(size=head, color=color),
    )
    thrust_head = go.Scatter(
        x=[probability[-1]],
        y=[abs(float(thrust[-1]))],
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker=dict(size=head, color=color),
    )
    return [loss_line, thrust_line, loss_head, thrust_head]


def _place_traces(figure, traces):
    for index, trace in enumerate(traces):
        figure.add_trace(trace, row=1 if index % 2 == 0 else 2, col=1)


def _bind_frame_axes(frame_traces, base_traces):
    for frame_trace, base_trace in zip(frame_traces, base_traces):
        frame_trace.xaxis = base_trace.xaxis
        frame_trace.yaxis = base_trace.yaxis
    return frame_traces


def _add_margin_marker(figure, curve, marker):
    if not marker:
        return
    probability = curve["probability"]
    loss = curve["loss"]
    thrust = np.abs(curve["thrust"])
    x_value = float(marker["x"])
    loss_y = float(np.interp(x_value, probability, loss))
    thrust_y = float(np.interp(x_value, probability, thrust))
    for row, y_value in ((1, loss_y), (2, thrust_y)):
        figure.add_vline(x=x_value, row=row, col=1, line_width=1.5, line_dash="dot", line_color="#C4491D", opacity=0)
        figure.add_annotation(
            x=x_value,
            y=y_value,
            text=marker["label"],
            showarrow=False,
            yshift=14,
            font=dict(color="#C4491D", size=14),
            opacity=0,
            row=row,
            col=1,
        )


def slice_figure(curves, x_label: str, title: str = "一维切面", marker=None):
    prepared = []
    for index, curve in enumerate(curves):
        prepared.append(
            {
                "name": curve["name"],
                "color": COLORS[index % len(COLORS)],
                "probability": np.asarray(curve["probability"], dtype=float),
                "loss": np.asarray(curve["loss"], dtype=float),
                "thrust": np.asarray(curve["thrust"], dtype=float),
            }
        )
    cuts = _frame_cuts(len(prepared[0]["probability"]))
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("损失势能", "logit 梯度推力 |dL/dz|"))
    width, opacity, head = _pulse(0, last=False)
    grouped = []
    for curve in prepared:
        grouped.extend(_slice_trace_group(curve, cuts[0], width, opacity, head))
    _place_traces(figure, grouped)
    _add_margin_marker(figure, prepared[0], marker)
    probability = np.concatenate([item["probability"] for item in prepared])
    loss = np.concatenate([item["loss"] for item in prepared])
    thrust = np.concatenate([np.abs(item["thrust"]) for item in prepared])
    x_low, x_high = float(np.min(probability)), float(np.max(probability))
    loss_low, loss_high = _axis_span(loss)
    thrust_low, thrust_high = _axis_span(thrust)
    figure.update_xaxes(title_text=x_label, range=[x_low, x_high], row=2, col=1)
    figure.update_xaxes(range=[x_low, x_high], row=1, col=1)
    figure.update_yaxes(title_text="L", range=[loss_low, loss_high], row=1, col=1)
    figure.update_yaxes(title_text="|dL/dz|", range=[thrust_low, thrust_high], row=2, col=1)
    frames = []
    for index, cut in enumerate(cuts):
        frame_width, frame_opacity, frame_head = _pulse(index, last=index == len(cuts) - 1)
        frame_traces = []
        for curve in prepared:
            frame_traces.extend(_slice_trace_group(curve, cut, frame_width, frame_opacity, frame_head))
        frames.append(go.Frame(name=str(index), data=_bind_frame_axes(frame_traces, figure.data)))
    figure.frames = frames
    figure.layout.meta = {"frame_count": len(frames)}
    return _base(figure, title, height=480)


def slice_html(figure) -> str:
    frame_count = int(figure.layout.meta["frame_count"])
    names = "[" + ",".join(f"'{index}'" for index in range(frame_count)) + "]"
    last_name = str(frame_count - 1)
    post_script = f"""
var gd = document.getElementById('{{plot_id}}');
var names = {names};
var play = {{frame: {{duration: {FRAME_MS}, redraw: false}}, transition: {{duration: 0}}, mode: 'immediate'}};
var jump = {{frame: {{duration: 0, redraw: false}}, transition: {{duration: 0}}, mode: 'immediate'}};
if (!document.getElementById('loss-motion-style')) {{
  var style = document.createElement('style');
  style.id = 'loss-motion-style';
  style.textContent = '@keyframes lossBreathe {{0%,100%{{opacity:1}}50%{{opacity:0.78}}}} @keyframes lossFlash {{0%,100%{{opacity:1}}50%{{opacity:0.2}}}} .loss-flash {{animation: lossFlash 0.45s ease-in-out 3}}';
  document.head.appendChild(style);
}}
var settled = false;
function applyBreathe() {{
  var lines = gd.querySelectorAll('.js-line');
  for (var index = 0; index < lines.length; index++) {{
    lines[index].style.animation = 'lossBreathe 2.6s ease-in-out ' + (index * 0.15) + 's infinite';
  }}
}}
function revealMarker(flash) {{
  var annotations = gd.layout.annotations || [];
  var shapes = gd.layout.shapes || [];
  var update = {{}};
  for (var i = 0; i < annotations.length; i++) update['annotations[' + i + '].opacity'] = 1;
  for (var j = 0; j < shapes.length; j++) update['shapes[' + j + '].opacity'] = 1;
  var done = function () {{
    if (!flash) return;
    var nodes = gd.querySelectorAll('.annotation-text');
    for (var k = 0; k < nodes.length; k++) {{
      var content = nodes[k].textContent || '';
      if (content.indexOf('γ') === 0 || content.indexOf('δ') === 0) nodes[k].classList.add('loss-flash');
    }}
  }};
  if (!annotations.length && !shapes.length) return;
  Plotly.relayout(gd, update).then(done);
}}
gd.on('plotly_afterplot', function () {{ if (settled) applyBreathe(); }});
function settle(flash) {{
  settled = true;
  applyBreathe();
  revealMarker(flash);
}}
var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (reduce) {{
  Plotly.animate(gd, ['{last_name}'], jump).then(function () {{ settle(false); }});
}} else {{
  Plotly.animate(gd, names, play).then(function () {{ settle(true); }});
}}
"""
    return figure.to_html(
        include_plotlyjs=True,
        full_html=False,
        auto_play=False,
        post_script=post_script,
        config={"displayModeBar": True, "responsive": True},
    )


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
