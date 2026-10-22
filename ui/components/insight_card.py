# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import streamlit as st
import xarray as xr
from ui.app_config.settings import MODEL_PALETTE
from ui.services.loader import LeadTimeData
from ui.utils.geo import regional_annual_mean


def render(
    model_results:     dict[str, dict[int, LeadTimeData]],
    metric_key:        str,
    climatology_value: float,
    region:            str,
    clim:              "object | None" = None,
) -> None:
    if climatology_value == 0:
        return

    active = [(m, lt_map) for m, lt_map in model_results.items() if lt_map]
    if not active:
        return

    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
        "color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;"
        "margin:24px 0 12px;'>Performance Insights</p>",
        unsafe_allow_html=True,
    )

    # Lazy import to share the per-model clim cache with time_serie_chart;
    # both renderers now use the same shared-mask diff so the headline
    # numbers agree with the Spatial Distribution panel.
    from ui.components.time_serie_chart import (
        _diff_mean, _matched_clim_for_model)

    cards_html = ""
    for idx, (model_name, lt_map) in enumerate(active):
        color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]

        clim_interp = (
            _matched_clim_for_model(model_name, lt_map, region, clim)
            if clim is not None else None
        )
        if clim_interp is not None:
            diffs = [
                d for d in (_diff_mean(ltd, clim_interp, region)
                            for ltd in lt_map.values() if ltd is not None)
                if d is not None
            ]
            avg_diff = float(np.mean(diffs)) if diffs else 0.0
        else:
            values    = [regional_annual_mean(ltd, region) for ltd in lt_map.values()]
            avg_score = float(np.mean(values))
            avg_diff  = avg_score - climatology_value
        is_better = avg_diff < 0
        pct       = (abs(avg_diff) / abs(climatology_value)) * 100

        badge_bg    = "rgba(10,153,56,0.09)"  if is_better else "rgba(242,176,41,0.18)"
        badge_text  = "#2e7d33"               if is_better else "#92580a"
        badge_icon  = "▼"                     if is_better else "▲"
        badge_label = "Better"                if is_better else "Worse"

        cards_html += (
            "<div style='flex:1;border:1px solid #E2E8F0;"
            f"border-top:3px solid {color};"
            "border-radius:8px;padding:14px;"
            "background:#FFFFFF;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"

            f"<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
            f"color:{color};margin-bottom:6px;text-transform:uppercase;"
            "letter-spacing:0.08em;'>"
            f"{model_name}</div>"

            f"<div style='font-family:Inter,sans-serif;font-size:20px;font-weight:800;"
            "color:#171d1a;line-height:1.2;margin-bottom:6px;'>"
            f"{abs(avg_diff):.4f}</div>"

            f"<span style='display:inline-block;background:{badge_bg};color:{badge_text};"
            "font-family:Inter,sans-serif;font-size:10px;font-weight:700;"
            "padding:2px 8px;border-radius:12px;'>"
            f"{badge_icon} {pct:.1f}% {badge_label}</span>"

            "<div style='font-family:Inter,sans-serif;font-size:10px;color:#9CA3AF;"
            "margin-top:6px;'>avg vs climatology baseline</div>"

            "</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:16px;align-items:stretch;'>{cards_html}</div>",
        unsafe_allow_html=True,
    )
