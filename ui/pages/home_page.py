# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""ForecastCheck home page — hero, then one section per verification
product (horizon), each followed by its own row of parameter cards."""
from __future__ import annotations

import streamlit as st

from ui.components.chrome import (AMBER, GREEN, INK, MUTED, _b64_svg, icon_img,
                                  render_explore_cards, render_footer,
                                  render_header)

# The three analysis lenses, in nav order. Every horizon offers the same three;
# only the medium-range one has data behind it today, so the others render the
# identical cards greyed out rather than a different, invented set.
_PARAMS = ("national_overview", "score_decomposition", "detailed_metrics")


def _product_header(icon: str, title: str, desc: str, *, live: bool) -> None:
    """Banded heading for one verification product, with its status."""
    cls, label = (("fc-status-live", "AVAILABLE") if live
                  else ("fc-status-soon", "COMING SOON"))
    st.markdown(
        f"""
        <div class="fc-product-head">
          <span class="fc-ph-icon">{icon_img(icon, 36)}</span>
          <div>
            <h2>{title}<span class="fc-status {cls}">{label}</span></h2>
            <p>{desc}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _explore_html(url_path: str) -> str:
    return f"<a class='fc-btn-amber' href='{url_path}' target='_self'>Explore ›</a>"


def render() -> None:
    render_header(active="home")

    # ── Hero ────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="padding:34px 6px 6px;max-width:640px;">
          <h1 style="font-size:1.9rem;margin:0 0 10px;">AI Forecast Verification Platform</h1>
          <p style="color:{MUTED};font-size:0.9rem;line-height:1.7;margin:0 0 18px;">
            A professional tool for evaluating the performance of AI weather models
            across Africa — comparing forecast skill against observational truth at
            multiple lead times, regions, and variables.
          </p>
          {_explore_html('national-overview')}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

    # ── Verification products, each followed by its own parameters ──────
    # Previously the two products sat side by side and ONE "Platform
    # Parameters" row served both, which read as if the parameters belonged to
    # the platform rather than to a horizon. Stacking each product above its
    # own row makes the ownership explicit and leaves room for the horizons
    # whose parameters will diverge later.
    st.markdown(
        f"""
        <div style="padding:26px 6px 0;">
          <h2 style="font-size:1.45rem;margin:0 0 6px;">Verification Products</h2>
          <p style="color:{MUTED};font-size:0.85rem;margin:0;">
            Each forecast horizon is explored through the same three
            parameters — pick a horizon, then a parameter.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _product_header(
        "smr_verification", "Short &amp; Medium-Range Verification",
        "Evaluate AI weather model skill across forecast horizons from 24&nbsp;h "
        "to 240&nbsp;h — lead-time error curves, pixel-level spatial error across "
        "the country and its sub-regions, and monthly seasonality, with several "
        "models side by side and benchmarked against climatology.",
        live=True,
    )
    render_explore_cards(*_PARAMS, spacer=16)

    # No dedicated sub-seasonal glyph exists under assets/icons yet, and icons
    # are the designer's SVGs rather than something to draw in code — so this
    # borrows the seasonal one. Sharing with the other forthcoming product is
    # less confusing than reusing the live product's glyph on a greyed section.
    _product_header(
        "seasonal_forecast", "Sub-Seasonal Forecast Verification",
        "Verification of week-2 to week-6 outlooks, bridging the gap between "
        "medium-range weather and the seasonal outlooks below.",
        live=False,
    )
    render_explore_cards(*_PARAMS, enabled=False, spacer=16)

    _product_header(
        "seasonal_forecast", "Seasonal Forecast Verification",
        "Verification of monthly-to-seasonal lead times covering 1 to 6 month "
        "horizons — how well extended-range forecasts capture rainfall and "
        "temperature anomalies across the continent and key sub-regions.",
        live=False,
    )
    render_explore_cards(*_PARAMS, enabled=False, spacer=16)

    # ── CTA pill ────────────────────────────────────────────────────────
    # Per the design: the AfriClimate leaf mark, white on a solid amber
    # circle, overlapping an amber-bordered message box. Our own message.
    symbol = _b64_svg("AfriClimate Symbol.svg")
    logo = (
        f"<img src='data:image/svg+xml;base64,{symbol}' alt='AfriClimate AI' "
        "style='width:56px;height:56px;object-fit:contain;"
        "filter:brightness(0) invert(1);'>"
        if symbol else "<span style='font-size:2rem;color:#fff;'>🌍</span>"
    )
    st.markdown(
        f"""
        <div style="display:flex;justify-content:center;padding:46px 0 8px;">
          <div style="display:inline-flex;align-items:center;">
            <span style="display:inline-flex;width:96px;height:96px;flex-shrink:0;
                border-radius:50%;background:{AMBER};align-items:center;
                justify-content:center;box-shadow:0 3px 12px rgba(242,176,41,0.35);
                position:relative;z-index:2;">{logo}</span>
            <div style="border:5px solid {AMBER};border-radius:22px 40px 40px 22px;
                background:#ffffff;padding:20px 40px 20px 64px;margin-left:-48px;max-width:560px;
                font-size:0.95rem;color:{INK};line-height:1.6;">
              Questions, collaborations, or data contributions?<br>
              <a href="mailto:research@africlimate.ai" style="color:{GREEN};
                  font-weight:600;text-decoration:none;">research@africlimate.ai</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()
