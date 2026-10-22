# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

import streamlit as st


def render() -> None:
    st.markdown("""
        <div style='max-width:720px;margin:40px auto 0;padding:0 8px;'>
          <p style='font-family:Inter,sans-serif;font-size:0.6rem;font-weight:700;
              letter-spacing:0.2em;text-transform:uppercase;color:#0a9938;
              margin:0 0 10px;'>AfriClimate AI</p>
          <h1 style='font-family:Inter,sans-serif;font-size:2rem;font-weight:800;
              color:#171d1a;line-height:1.25;margin:0 0 16px;'>
            AI Forecast <span style='color:#213b7d;'>Verification</span> Platform
          </h1>
          <p style='font-family:Inter,sans-serif;font-size:1rem;color:#4B5563;
              line-height:1.7;margin:0 0 40px;'>
            A professional tool for evaluating the performance of AI weather models
            across Africa — comparing forecast skill against observational truth
            at multiple lead times, regions, and variables.
          </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Two tab cards ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="large")

    def _tab_card(col, icon, label, title, body, accent, badge=None):
        badge_html = ""
        if badge:
            badge_html = f"""
              <div style='display:inline-flex;align-items:center;gap:6px;
                  background:rgba(10,153,56,0.08);border:1px solid rgba(10,153,56,0.25);
                  border-radius:99px;padding:4px 12px;margin-top:16px;'>
                <span style='width:7px;height:7px;border-radius:50%;background:#0a9938;
                    display:inline-block;animation:pulse 1.8s ease-in-out infinite;'></span>
                <span style='font-family:Inter,sans-serif;font-size:0.65rem;font-weight:700;
                    color:#0a9938;letter-spacing:0.08em;text-transform:uppercase;'>{badge}</span>
              </div>"""
        col.markdown(f"""
            <div style='border:1px solid #e5e7eb;border-top:4px solid {accent};
                border-radius:10px;padding:28px 24px;background:#ffffff;
                box-shadow:0 2px 10px rgba(0,0,0,0.06);height:100%;'>
              <div style='font-size:2rem;margin-bottom:14px;'>{icon}</div>
              <div style='font-family:Inter,sans-serif;font-size:0.58rem;font-weight:700;
                  letter-spacing:0.18em;text-transform:uppercase;color:{accent};
                  margin-bottom:6px;'>{label}</div>
              <div style='font-family:Inter,sans-serif;font-size:1.05rem;font-weight:800;
                  color:#171d1a;margin-bottom:12px;line-height:1.3;'>{title}</div>
              <div style='font-family:Inter,sans-serif;font-size:0.82rem;
                  color:#6B7280;line-height:1.7;'>{body}</div>
              {badge_html}
            </div>
            <style>
              @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.3; }} }}
            </style>
        """, unsafe_allow_html=True)

    _tab_card(
        col_a,
        icon="📈",
        label="24 h — 240 h",
        title="Short & Medium-Range Verification",
        body=(
            "Evaluate AI weather model skill across forecast horizons from "
            "<b>24 h to 240 h</b>. Analyse lead-time error curves, explore "
            "spatial error patterns at pixel level across Africa and sub-regions, "
            "and drill into monthly seasonality. Compare multiple models "
            "side-by-side and benchmark against climatology."
        ),
        accent="#213b7d",
    )

    _tab_card(
        col_b,
        icon="🗓️",
        label="1 month — 6 months",
        title="Seasonal Forecast Verification",
        body=(
            "Verification of monthly-to-seasonal lead-time forecasts covering "
            "<b>1 to 6 month horizons</b>. Assess how well extended-range "
            "forecasts capture rainfall and temperature anomalies across the "
            "African continent and key sub-regions."
        ),
        accent="#0a9938",
        badge="Coming soon",
    )

    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

    # ── Three capability cards ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="medium")

    def _card(col: Any, icon: str, title: str, body: str, accent: str) -> None:
        col.markdown(f"""
            <div style='border:1px solid #e5e7eb;border-top:3px solid {accent};
                border-radius:8px;padding:20px;background:#ffffff;height:100%;
                box-shadow:0 1px 6px rgba(0,0,0,0.05);'>
              <div style='font-size:1.6rem;margin-bottom:10px;'>{icon}</div>
              <div style='font-family:Inter,sans-serif;font-size:0.9rem;
                  font-weight:700;color:#171d1a;margin-bottom:8px;'>{title}</div>
              <div style='font-family:Inter,sans-serif;font-size:0.8rem;
                  color:#6B7280;line-height:1.6;'>{body}</div>
            </div>
        """, unsafe_allow_html=True)

    _card(c1, "📈", "Lead-Time Performance",
          "Analyse how model skill degrades across forecast horizons from 24h to 240h.",
          "#0a9938")
    _card(c2, "🗺️", "Spatial Distribution",
          "Explore the geographic pattern of forecast errors across Africa and sub-regions.",
          "#213b7d")
    _card(c3, "⚖️", "Climatology & Models Benchmark",
          "Measure genuine model skill by comparing against a climatology baseline and benchmark models against each other.",
          "#f2b029")
