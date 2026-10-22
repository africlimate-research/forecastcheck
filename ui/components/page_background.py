# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import streamlit as st


def set_background(image_url: str = "app/static/isobar_white.png") -> None:
    """Inject isobar pattern as page background via Streamlit's static file server."""
    st.markdown(
        f"""<style>
        /* Base layer — the actual image. Applied to the app view container
           as well: when `html` has its own background (Streamlit sets one),
           the body background no longer propagates to the page canvas, so
           painting body alone renders nothing visible. */
        body, html {{
            /* The pattern's lines are WHITE on a transparent background —
               they only show against this faint sage tint. On pure white
               the pattern was invisible. */
            background-color: #eef3ee !important;
            background-image: url("{image_url}") !important;
            background-size: cover !important;
            background-position: center center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}
        /* The app view container is a composited scroll area — Chrome
           refuses to paint fixed-attachment backgrounds inside those, so
           this layer uses scroll attachment instead. */
        [data-testid="stAppViewContainer"] {{
            background-image: url("{image_url}") !important;
            background-size: cover !important;
            background-position: center center !important;
            background-repeat: no-repeat !important;
            background-attachment: scroll !important;
        }}
        /* Make every Streamlit wrapper transparent so the image shows
           through. background-color ONLY — the `background:` shorthand
           would reset the background-image set above. */
        #root,
        .stApp,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"],
        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"] {{
            background-color: transparent !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def set_sidebar_background(image_url: str) -> None:
    st.markdown(
        f"""<style>
        [data-testid="stSidebar"] > div:first-child {{
            background-image: url("{image_url}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: local !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )
