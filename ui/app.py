# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""ForecastCheck — local, discovery-driven multipage dashboard.

Green ForecastCheck chrome (ported from the AfriClimate `verification-ui`
redesign) over the LOCAL NetCDF verification engine. Pages are wired to the
on-disk discovery (`ui.services.*`), so the menus only ever offer what the
`verification.run` pipeline has actually produced under the data root.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Seed VERIFICATION_COUNTRY / VERIFICATION_TRUTH_PRODUCT from this machine's country
# file (deployments/{name}.toml, named by .deployment) before any module reads them. A
# real environment variable still overrides. This is what makes a machine set up once
# (echo senegal > .deployment) launch pre-configured — and what lets a second country
# be launched from the same tree with VERIFICATION_DEPLOYMENT=<name>.
from verification import config as _config

_config.apply_deployment_defaults()

import streamlit as st

st.set_page_config(
    page_title="ForecastCheck by AfriClimate AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui.components import chrome
from ui.components.page_background import set_background
from ui.pages import (coming_soon, contact, detailed_metrics, home_page,
                      national_overview, score_decomposition)


def _inject_heartbeat() -> None:
    """Auto-reload the page when the Streamlit WebSocket goes silent.

    On Cloud Run with the default scale-to-zero policy, a container can sleep
    after a few minutes of inactivity. The browser keeps the last-rendered
    page visible, but the WebSocket dies. This poller hits Streamlit's
    `/_stcore/health` every 10 s and reloads the parent page if the endpoint
    has been unreachable for ~45 s.
    """
    import streamlit.components.v1 as components

    components.html(
        """
<script>
(function () {
  let last_ok = Date.now();
  const origin = (function () {
    try { return window.parent.location.origin; }
    catch (_) { return window.location.origin; }
  })();
  setInterval(async () => {
    try {
      const r = await fetch(origin + "/_stcore/health", { cache: "no-store" });
      if (r.ok) { last_ok = Date.now(); return; }
    } catch (_) { /* network error — treat as silent */ }
    if (Date.now() - last_ok > 45000) {
      console.warn("[heartbeat] backend unreachable for 45s, reloading");
      try { window.parent.location.reload(); }
      catch (_) { window.location.reload(); }
    }
  }, 10000);
})();
</script>
        """,
        height=0,
    )


_PAGES = {
    "home": st.Page(
        home_page.render, title="ForecastCheck — Home",
        url_path="home", default=True,
    ),
    "national_overview": st.Page(
        national_overview.render, title="National Overview",
        url_path="national-overview",
    ),
    "score_decomposition": st.Page(
        score_decomposition.render, title="Score Decomposition",
        url_path="score-decomposition",
    ),
    "detailed_metrics": st.Page(
        detailed_metrics.render, title="Threshold Diagnostics",
        url_path="threshold-diagnostics",
    ),
    "contact": st.Page(
        contact.render, title="Contact & FAQs",
        url_path="contact",
    ),
    # The forthcoming horizons: every entry in their nav dropdowns navigates to
    # a placeholder rather than dead-ending. Keys mirror
    # chrome._page_key(horizon, lens) -- when a horizon's data lands, point
    # these at the real page modules and drop the coming_soon wrappers.
    "subseasonal_national_overview": st.Page(
        coming_soon.subseasonal_national_overview,
        title="Sub-Seasonal — National Overview",
        url_path="sub-seasonal-national-overview",
    ),
    "subseasonal_score_decomposition": st.Page(
        coming_soon.subseasonal_score_decomposition,
        title="Sub-Seasonal — Score Decomposition",
        url_path="sub-seasonal-score-decomposition",
    ),
    "subseasonal_detailed_metrics": st.Page(
        coming_soon.subseasonal_detailed_metrics,
        title="Sub-Seasonal — Threshold Diagnostics",
        url_path="sub-seasonal-threshold-diagnostics",
    ),
    "seasonal_national_overview": st.Page(
        coming_soon.seasonal_national_overview,
        title="Seasonal — National Overview",
        url_path="seasonal-national-overview",
    ),
    "seasonal_score_decomposition": st.Page(
        coming_soon.seasonal_score_decomposition,
        title="Seasonal — Score Decomposition",
        url_path="seasonal-score-decomposition",
    ),
    "seasonal_detailed_metrics": st.Page(
        coming_soon.seasonal_detailed_metrics,
        title="Seasonal — Threshold Diagnostics",
        url_path="seasonal-threshold-diagnostics",
    ),
}

chrome.register_pages(_PAGES)
chrome.inject_css()
set_background()  # isobar pattern from ui/static/isobar_white.png
_inject_heartbeat()

nav = st.navigation(list(_PAGES.values()), position="hidden")
nav.run()
