# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Contact & FAQs page — ForecastCheck design.

Dark-green FAQ accordions (chevron rows, lighter-green expanded body),
followed by the get-in-touch statement and the amber mail button.
"""
from __future__ import annotations

import streamlit as st

from ui.components.chrome import GREEN_DARK, HEADER_GREEN, render_footer, render_header

_FAQS = [
    (
        "What is ForecastCheck?",
        "ForecastCheck is AfriClimate AI's forecast verification platform, "
        "built as part of the Forecast4Africa initiative. It evaluates the "
        "performance of AI weather models — comparing forecast skill against "
        "observational truth at multiple lead times, regions, and variables.",
    ),
    (
        "Which models and metrics are covered?",
        "AI weather models (GraphCast and others) and operational baselines "
        "(IFS), verified against gridded CHIRTS temperature observations, with "
        "the ERA5 reanalysis as a second reference. Metrics include MAE, CRPS, "
        "bias and correlation, plus threshold diagnostics (reliability, ROC and "
        "cost-loss curves).",
    ),
    (
        "How is calibration done?",
        "Model outputs are calibrated with Isotonic Distributional Regression "
        "(IDR) so that models of different kinds — deterministic and ensemble — "
        "can be compared fairly. The Score Decomposition page separates each "
        "model's error into miscalibration (what calibration removes) and "
        "discrimination (the information a calibrated model adds over a "
        "reference).",
    ),
    (
        "Why are some options missing?",
        "The platform is discovery-driven: the menus only ever offer the "
        "model / metric / region / season / lead-time combinations that the "
        "verification pipeline has actually produced on disk. Generating a new "
        "combination means running the corresponding `verification.run` command.",
    ),
    (
        "Can I use the data and results?",
        "Yes — data and results are released under the Creative Commons "
        "Attribution 4.0 International (CC BY 4.0) licence. Please cite the "
        "companion manuscript when using this resource.",
    ),
]


def render() -> None:
    render_header(active="contact")
    _inject_page_css()

    st.markdown(
        "<h1 style='margin:30px 0 18px;'>Contact &amp; FAQs</h1>",
        unsafe_allow_html=True,
    )

    with st.container(key="fc_faq"):
        for question, answer in _FAQS:
            with st.expander(question):
                st.markdown(answer)

    st.markdown(
        f"""
        <div style="margin:44px 0 8px;max-width:620px;">
          <div style="font-family:'Poppins',sans-serif;font-size:1.35rem;
              font-weight:700;color:{GREEN_DARK};line-height:1.4;">
            If you have information you think is relevant
            to this website, please do get in touch.
          </div>
          <div style="border-left:2px solid {GREEN_DARK};height:64px;
              margin:18px 0 18px 4px;"></div>
          <a class="fc-btn-amber" style="font-size:0.85rem;padding:9px 20px;"
              href="mailto:research@africlimate.ai">research@africlimate.ai</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()


def _inject_page_css() -> None:
    st.html(f"""
<style>
/* FAQ accordions: dark-green summary rows, lighter-green open body */
.st-key-fc_faq [data-testid="stExpander"] {{
  margin-bottom: 10px;
}}
.st-key-fc_faq [data-testid="stExpander"] details {{
  border: none !important;
  border-radius: 10px !important;
  overflow: hidden;
  background: transparent !important;
}}
.st-key-fc_faq [data-testid="stExpander"] summary {{
  background: {GREEN_DARK} !important;
  padding: 12px 18px !important;
  border-radius: 10px !important;
}}
.st-key-fc_faq [data-testid="stExpander"] details[open] summary {{
  border-radius: 10px 10px 0 0 !important;
}}
.st-key-fc_faq [data-testid="stExpander"] summary p,
.st-key-fc_faq [data-testid="stExpander"] summary span {{
  color: #ffffff !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
}}
.st-key-fc_faq [data-testid="stExpander"] summary svg,
.st-key-fc_faq [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{
  color: #ffffff !important;
  fill: #ffffff !important;
}}
.st-key-fc_faq [data-testid="stExpanderDetails"] {{
  background: {HEADER_GREEN} !important;
  border-radius: 0 0 10px 10px;
  padding: 14px 18px !important;
}}
.st-key-fc_faq [data-testid="stExpanderDetails"] p {{
  color: #ffffff !important;
  font-size: 0.84rem !important;
  line-height: 1.65 !important;
}}
</style>
    """)
