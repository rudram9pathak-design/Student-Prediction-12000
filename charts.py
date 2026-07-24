"""
charts.py — Visualisations for the Placement Intelligence Platform.

All INTERACTIVE, on-screen charts use Plotly (per the redesign brief) and
are themed to match the dark-glass palette in theme.py.

The one exception is `report_figure()`, which still renders with matplotlib.
That function only powers the downloadable PNG "full report" (a static,
printable artifact, not an on-screen chart) — keeping it on matplotlib avoids
a hard runtime dependency on a headless-browser PNG exporter (kaleido) just
for one offline export button, and changes nothing about what that button
has always produced.
"""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from theme import COLORS
from utils import FRIENDLY_LABELS, RADAR_FIELDS

PAPER = "rgba(0,0,0,0)"
FONT = dict(family="Inter, sans-serif", color=COLORS["text"], size=12)


def _layout(fig, height=320, **kw):
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PAPER, font=FONT,
        margin=dict(l=10, r=10, t=40, b=10), height=height,
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


def home_gauge(accuracy_pct, students_processed, predictions_made, confidence_pct):
    """The hero-section gauge: shows model accuracy with supporting stats as
    an annotation ring — the 'large animated Plotly Gauge' from the brief."""
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
    """metric_defs: list of (label, value_0_to_1)."""
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
    """Min–max spread chart (honest stand-in for a quartile box plot, same
    caveat as the original — there's no saved test-set sample for real quartiles)."""
    fig = go.Figure()
    labels = [FRIENDLY_LABELS.get(f, f) for f in reversed(range_feats)]
    for i, feat in enumerate(reversed(range_feats)):
        lo, hi = numeric_ranges[feat]
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
    """Builds the single downloadable PNG report: donut gauge + skill radar +
    academic bars + top model drivers. Logic/output unchanged from the
    original app; this is the same figure, just relocated here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GT_PURPLE, GT_PURPLE_LIGHT = "#4F2D7F", "#AA8CE0"
    GT_GOLD, GT_RED = "#C9A227", "#E24C63"
    GT_INK, GT_MUTED, GT_PANEL = "#241B36", "#8A7FA0", "#FFFFFF"
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(9, 6.2))
    fig.patch.set_facecolor(GT_PANEL)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(GT_PANEL)
    color = GT_GOLD if proba_pct >= 50 else GT_RED
    ax0.pie([proba_pct, 100 - proba_pct], colors=[color, "#EFEAF7"],
            startangle=90, counterclock=False, wedgeprops=dict(width=0.32))
    ax0.text(0, 0.05, f"{proba_pct:.1f}%", ha="center", va="center",
              fontsize=22, fontweight="bold", color=GT_INK)
    ax0.text(0, -0.28, "Placement Probability", ha="center", va="center",
              fontsize=8.5, color=GT_MUTED)
    ax0.set_title(f"{name}", fontsize=13, fontweight="bold", color=GT_INK, pad=6)

    radar_vals = {k: float(user_input.get(k, np.nan)) for k in RADAR_FIELDS if not pd.isna(user_input.get(k, np.nan))}
    if radar_vals:
        ax1 = fig.add_subplot(gs[0, 1], polar=True)
        labels = [FRIENDLY_LABELS.get(k, k).replace(" Score", "") for k in radar_vals]
        values = list(radar_vals.values())
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        ax1.plot(angles, values, color=GT_PURPLE, linewidth=2)
        ax1.fill(angles, values, color=GT_PURPLE_LIGHT, alpha=0.35)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(labels, fontsize=7, color=GT_INK)
        ax1.set_ylim(0, 100)
        ax1.set_yticklabels([])
        ax1.set_facecolor(GT_PANEL)
        ax1.spines["polar"].set_color(GT_MUTED)
        ax1.set_title("Skill Profile", fontsize=11, fontweight="bold", color=GT_INK, pad=12)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(GT_PANEL)
    academic_keys = ["CGPA", "10th_Percentage", "12th_Percentage", "Attendance_Percentage"]
    academic_vals, academic_labels = [], []
    for k in academic_keys:
        v = user_input.get(k, np.nan)
        if pd.isna(v):
            continue
        v = float(v) * 10 if k == "CGPA" else float(v)
        academic_vals.append(v)
        academic_labels.append(FRIENDLY_LABELS.get(k, k).replace("_Percentage", " %"))
    if academic_vals:
        bars = ax2.barh(academic_labels, academic_vals, color=GT_PURPLE_LIGHT)
        ax2.bar_label(bars, fmt="%.0f", fontsize=8, color=GT_INK, padding=3)
        ax2.set_xlim(0, 110)
        ax2.set_title("Academic Snapshot", fontsize=11, fontweight="bold", color=GT_INK)
        ax2.tick_params(labelsize=8, colors=GT_INK)
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.set_facecolor(GT_PANEL)
    lines = ["What the model weighs most:"]
    if top_drivers is not None:
        for feat, _ in top_drivers.head(6).items():
            lines.append(f"  •  {FRIENDLY_LABELS.get(feat, feat)}")
    ax3.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9.5, color=GT_INK)

    fig.suptitle("Placement Readiness — Full Report", fontsize=14, fontweight="bold",
                 color=GT_PURPLE, y=1.02)
    fig.tight_layout()
    return fig
