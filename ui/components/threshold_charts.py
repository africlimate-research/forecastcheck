"""Plot-data helpers for the threshold-diagnostics tab.

The math lives in :mod:`verification.diagnostics` (single source of truth); these helpers
only slice a stored curve Dataset into plot-ready arrays. The Plotly drawing that consumes
them is added in the page layer.
"""
from __future__ import annotations

import numpy as np

from verification import diagnostics

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def roc_curve_xy(ds, threshold, band):
    """(fpr, tpr, auc) for one ROC curve, reusing diagnostics.roc_rates."""
    fpr, tpr = diagnostics.roc_rates(ds, threshold, band)
    return fpr, tpr, float(_trapezoid(tpr, fpr))


def reliability_xy(ds, threshold, band):
    """(forecast_probability, observed_frequency, lower, upper) arrays for one reliability curve."""
    sub = ds.sel(threshold=threshold, altitude=band)
    return (sub["forecast_probability"].values, sub["observed_frequency"].values,
            sub["obs_freq_lower"].values, sub["obs_freq_upper"].values)


def cost_loss_xy(ds, threshold, band):
    """(cost_loss_ratio, value, base_rate) for one cost-loss curve."""
    sub = ds.sel(threshold=threshold, altitude=band)
    return (ds["cost_loss_ratio"].values, sub["value"].values,
            float(sub["base_rate"].values))
