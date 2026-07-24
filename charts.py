"""
charts.py — Visualisations for the Placement Intelligence Platform.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from theme import COLORS
from utils import FRIENDLY_LABELS, RADAR_FIELDS

PAPER = "rgba(0,0,0,0)"
FONT = dict(family="Inter, sans-serif", color=COLORS["text"], size=12)


def _layout(fig, height=320, **kw):
    fig.update_layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=FONT,
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        **kw,
    )
    return fig


def gauge_chart(proba_pct: float, placed: bool):
    color = COLORS["gold"] if placed else COLORS["danger"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba_pct,
        number={"suffix": "%", "font": {"size": 40, "color": COLORS["text"], "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text2"], "tickfont": {"color": COLORS["text2"]}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(240,68,56,0.15)"},
                {"range": [40, 70], "color": "rgba(155,123,212,0.15)"},
                {"range": [70, 100], "color": "rgba(215,177,90,0.15)"},
            ],
        },
    ))
    return _layout(fig, height=260)


def home_gauge(accuracy_pct, students_processed=None, predictions_made=None, confidence_pct=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy_pct,
        number={"suffix": "%", "font": {"size": 34, "color": COLORS["text"], "family": "Space Grotesk"}},
        title={"text": "Model Accuracy", "font": {"size": 13, "color": COLORS["text2"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text2"], "tickfont": {"color": COLORS["text2"]}},
            "bar": {"color": COLORS["gold"], "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
        },
    ))
    return _layout(fig, height=280)


def metrics_bar(metric_defs: list):
    labels = [d[0] for d in metric_defs]
    values = [d[1] * 100 for d in metric_defs]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=COLORS["purple2"],
        text=[f"{v:.1f}%" for v in values], textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig.update_yaxes(range=[0, 108], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, tickfont=dict(color=COLORS["text2"]))
    return _layout(fig, height=320)


def feature_importance_bar(importances: pd.Series, top_n=15):
    top = importances.head(top_n).sort_values(ascending=True)
    labels = [FRIENDLY_LABELS.get(f, f) for f in top.index]
    fig = go.Figure(go.Bar(
        x=top.values, y=labels, orientation="h",
        marker=dict(color=top.values, colorscale=[[0, COLORS["purple"]], [1, COLORS["gold"]]]),
        text=[f"{v:.3f}" for v in top.values], textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig.update_xaxes(title="Relative importance", showgrid=False, tickfont=dict(color=COLORS["text2"]))
    fig.update_yaxes(tickfont=dict(color=COLORS["text"]), showgrid=False)
    return _layout(fig, height=460)


def range_chart(range_feats: list, numeric_ranges: dict):
    fig = go.Figure()
    labels = [FRIENDLY_LABELS.get(f, f) for f in reversed(range_feats)]
    for i, feat in enumerate(reversed(range_feats)):
        lo, hi = numeric_ranges.get(feat, (None, None))
        if lo is None or hi is None:
            continue
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[labels[i], labels[i]], mode="lines+markers",
            line=dict(color=COLORS["purple2"], width=6),
            marker=dict(size=9, color=COLORS["gold"]),
            showlegend=False, hoverinfo="x+y",
        ))
    fig.update_xaxes(title="Value range", showgrid=False, tickfont=dict(color=COLORS["text2"]))
    fig.update_yaxes(tickfont=dict(color=COLORS["text"]), showgrid=False)
    return _layout(fig, height=max(280, 34 * len(range_feats) + 60))


def distribution_histogram(values: pd.Series, threshold: float):
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=20, marker_color=COLORS["purple2"],
        marker_line_color="rgba(255,255,255,0.25)", marker_line_width=1,
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["danger"], line_width=2)
    fig.update_xaxes(title="Placement probability (%)", showgrid=False, tickfont=dict(color=COLORS["text2"]))
    fig.update_yaxes(title="Students", showgrid=False, tickfont=dict(color=COLORS["text2"]))
    return _layout(fig, height=300)


def donut_split(placed_count: int, at_risk_count: int):
    fig = go.Figure(go.Pie(
        labels=["Placement-ready", "At risk"],
        values=[placed_count, at_risk_count],
        hole=0.62,
        marker=dict(colors=[COLORS["gold"], COLORS["danger"]]),
        textfont=dict(color=COLORS["text"]),
    ))
    return _layout(fig, height=300, showlegend=True, legend=dict(font=dict(color=COLORS["text2"])))


def radar_chart(values_dict: dict, title="Skill Profile"):
    labels = [FRIENDLY_LABELS.get(k, k).replace(" Score", "") for k in values_dict]
    values = list(values_dict.values())
    fig = go.Figure(go.Scatterpolar(
        r=values + values[:1], theta=labels + labels[:1],
        fill="toself", line=dict(color=COLORS["purple2"]),
        fillcolor="rgba(155,123,212,0.28)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(color=COLORS["text2"], size=8), gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(tickfont=dict(color=COLORS["text"], size=10), gridcolor="rgba(255,255,255,0.08)"),
        ),
        showlegend=False,
        title=dict(text=title, font=dict(color=COLORS["text"], size=13)),
    )
    return _layout(fig, height=340)


# ---------------------------------------------------------------------------
# Matplotlib — retained ONLY for the downloadable full-report PNG.
# ---------------------------------------------------------------------------
def report_figure(name, proba_pct, user_input, top_drivers=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GT_PURPLE, GT_PURPLE_LIGHT = "#4F2D7F", "#AA8CE0"
    GT_GOLD, GT_RED = "#C9A227", "#E24C63"
    GT_INK, GT_MUTED, GT_PANEL = "#241B36", "#8A7FA0", "#FFFFFF"
    plt.rcParams["font.family"]
