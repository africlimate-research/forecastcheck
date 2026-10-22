"""Plotly figures for the threshold-diagnostics tab (reliability / ROC / cost-loss).

Each figure draws one curve per model at a single threshold + altitude band — so users can
compare models head-to-head. ``series`` is a list of ``(label, dataset)`` pairs (one per
model). The figures consume the pure plot-data helpers in
:mod:`ui.components.threshold_charts` (which reuse :mod:`verification.diagnostics`). Plotly is
imported here, not in the data layer, so the unit tests stay free of a Plotly dependency.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ui.app_config.settings import MODEL_PALETTE
from ui.components.threshold_charts import cost_loss_xy, reliability_xy, roc_curve_xy

_GRID = "#9CA3AF"


def _color(i):
    """A stable curve color for the i-th series (model)."""
    return MODEL_PALETTE[i % len(MODEL_PALETTE)]


def _layout(fig, title, x_title, y_title):
    """Apply the shared plotly_white styling used across the threshold figures."""
    fig.update_layout(
        template="plotly_white", height=540,
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=14)),
        xaxis_title=x_title, yaxis_title=y_title,
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=11)),
        margin=dict(l=40, r=20, t=44, b=40),
    )
    return fig


def _title(metric, threshold, band, units=""):
    """Figure title with the fixed threshold and altitude band.

    ``units`` comes from the curve file's own threshold coord — mm/day for
    precipitation, degC for daily-max temperature. It used to be a hardcoded "mm/day",
    which silently mislabels every temperature figure.
    """
    thr = f"{int(threshold)} {units}".strip()
    from ui.app_config.settings import band_label

    return f"{metric} — {thr} — altitude: {band_label(band)}"


def roc_figure(series, threshold, band, units=""):
    """ROC curves (one per model, AUC in the legend) at a single threshold + altitude band."""
    fig = go.Figure()
    for i, (label, ds) in enumerate(series):
        fpr, tpr, auc = roc_curve_xy(ds, threshold, band)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label} (AUC={auc:.3f})",
                                 line=dict(color=_color(i), width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="no skill",
                             line=dict(color=_GRID, dash="dash", width=1)))
    _layout(fig, _title("ROC", threshold, band, units), "False alarm rate", "Hit rate")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def reliability_figure(series, threshold, band, units=""):
    """Reliability curves (one per model, with bootstrap CI) at a single threshold, auto-zoomed."""
    fig = go.Figure()
    top = 0.0
    for i, (label, ds) in enumerate(series):
        fp, of, lo, hi = reliability_xy(ds, threshold, band)
        top = float(np.nanmax([top, np.nanmax(fp), np.nanmax(of), np.nanmax(hi)]))
        fig.add_trace(go.Scatter(
            x=fp, y=of, mode="lines+markers", name=label,
            line=dict(color=_color(i), width=2),
            error_y=dict(type="data", symmetric=False, array=hi - of, arrayminus=of - lo, thickness=1)))
    lim = 1.0 if not np.isfinite(top) or top <= 0 else min(1.0, top * 1.08)
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="perfect",
                             line=dict(color=_GRID, dash="dash", width=1)))
    _layout(fig, _title("Reliability", threshold, band, units), "Forecast probability", "Observed frequency")
    fig.update_xaxes(range=[0, lim])
    fig.update_yaxes(range=[0, lim])
    return fig


def cost_loss_figure(series, threshold, band, logx=False, units=""):
    """Cost-loss value curves (one per model) at a single threshold + altitude band.

    The y-axis is strictly positive, floored at 0, and capped at the largest plotted value
    + 0.2 (so high thresholds, whose values are small, still fill the panel). ``logx`` puts
    the cost-loss ratio on a log scale to spread out the low-alpha region.
    """
    fig = go.Figure()
    top = 0.0
    for i, (label, ds) in enumerate(series):
        alphas, value, _ = cost_loss_xy(ds, threshold, band)
        top = float(np.nanmax([top, np.nanmax(value)]))
        fig.add_trace(go.Scatter(x=alphas, y=value, mode="lines", name=label,
                                 line=dict(color=_color(i), width=2.5)))
    upper = min(1.0, max(0.2, top + 0.2))
    _layout(fig, _title("Cost-loss value", threshold, band, units), "Cost-loss ratio  α = C/L",
            "Relative economic value V")
    if logx:
        fig.update_xaxes(type="log", range=[np.log10(0.01), 0.0])
    else:
        fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, upper])
    return fig
