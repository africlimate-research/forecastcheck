# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Placeholder pages for the horizons that have no diagnostics yet.

Every entry in the header's horizon dropdowns navigates somewhere, including
the sub-seasonal and seasonal ones — landing on a dead menu item reads as a
bug. Each of those entries lands here instead, naming the horizon and the lens
it will eventually show. One render function, six thin wrappers, because
``st.Page`` wants a distinct callable and url_path per page.

When a horizon's data arrives, point its ``st.Page`` at the real page module in
``ui/app.py`` and delete the matching wrapper — nothing else references them.
"""
from __future__ import annotations

import streamlit as st

from ui.components.chrome import (HORIZONS, MUTED, NAV_PAGES, icon_img,
                                  render_footer, render_header)

# lens slug → what that lens will offer once the horizon has data. Kept short:
# the copy that describes the lens itself already lives in chrome._CARD_META.
_LENS_BLURB = {
    "national_overview":
        "Calibrated skill by season, lead time and region, alongside the raw "
        "models scored by MAE, bias and correlation.",
    "score_decomposition":
        "Miscalibration and discrimination against a reference forecast.",
    "detailed_metrics":
        "Reliability, ROC and cost-loss curves over thresholds, altitude "
        "bands, regions and seasons.",
}


def _render(horizon: str, lens: str) -> None:
    render_header(active=lens, horizon=horizon)
    h_label = HORIZONS[horizon][0].title()          # SUB-SEASONAL -> Sub-Seasonal
    lens_label = NAV_PAGES[lens][0].title().replace("Decomp.", "Decomposition")
    st.markdown(
        f"""
        <div style="padding:52px 6px 6px;max-width:680px;">
          <span class="fc-status fc-status-soon" style="margin-left:0;">
            COMING SOON</span>
          <h1 style="font-size:1.7rem;margin:14px 0 10px;">
            {h_label} · {lens_label}</h1>
          <p style="color:{MUTED};font-size:0.9rem;line-height:1.7;margin:0 0 22px;">
            {_LENS_BLURB.get(lens, "")} This horizon has no verification data
            on the platform yet — the page will light up here once its
            forecasts and observations are onboarded.
          </p>
          <a class="fc-btn-amber" href="national-overview" target="_self">
            Explore the short &amp; medium-range results ›</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='padding:34px 6px 0;'>{icon_img('seasonal_forecast', 64)}</div>",
        unsafe_allow_html=True,
    )
    render_footer()


def subseasonal_national_overview() -> None:
    _render("subseasonal", "national_overview")


def subseasonal_score_decomposition() -> None:
    _render("subseasonal", "score_decomposition")


def subseasonal_detailed_metrics() -> None:
    _render("subseasonal", "detailed_metrics")


def seasonal_national_overview() -> None:
    _render("seasonal", "national_overview")


def seasonal_score_decomposition() -> None:
    _render("seasonal", "score_decomposition")


def seasonal_detailed_metrics() -> None:
    _render("seasonal", "detailed_metrics")
