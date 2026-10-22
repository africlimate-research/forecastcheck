# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""ForecastCheck page chrome — header nav, footer, and global CSS.

Every page renders:  render_header(active=...) → page content → render_footer().
The design language: green rounded header bar, light-green bordered cards,
amber CTA chips, navy accents, isobar-white page background.

Ported from the AfriClimate `verification-ui` redesign so the local,
discovery-driven ForecastCheck build shares its look. Self-contained — no
cloud/data dependencies — so pages can wire it to the local NetCDF engine.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

# ── Palette — exact values sampled from the designer's July Layout PDF ─────
GREEN        = "#2f7e34"   # headings (H1/H2/H3), section titles
GREEN_BRIGHT = "#0a9a39"   # AfriClimate symbol green
HEADER_GREEN = "#6fba6d"   # header nav bar background
GREEN_DARK   = "#2b522c"   # card borders, dark buttons/config headers
GREEN_BG     = "#edf9ed"   # light green — card fills, icon boxes
GREEN_BORDER = GREEN_DARK  # design cards use the dark green as border
AMBER        = "#f5b029"   # Explore buttons
NAVY         = "#213b7c"   # AfriClimate wordmark / footer accents
CREAM        = "#fdfbf4"   # wordmark text on the green header
INK          = "#162116"   # body text
INK_SOFT     = "#313e2a"   # dark olive — nav labels, button text
MUTED        = "#706f6f"   # secondary text
GRAY_BG      = "#f1f1f1"   # neutral card fill

_ASSETS = Path(__file__).resolve().parents[1] / "assets"

# Pages of the app: slug → (nav label, url_path). Order = nav order.
NAV_PAGES = {
    "score_decomposition": ("SCORE DECOMP.",     "score-decomposition"),
    "national_overview":   ("NATIONAL OVERVIEW", "national-overview"),
    "detailed_metrics":    ("THRESHOLD DIAGNOSTICS", "threshold-diagnostics"),
}
# The forecast horizons that now head the nav. Each opens a dropdown of the
# SAME three analysis pages (NAV_PAGES) — the horizon picks the data, the
# dropdown entry picks the lens. Only the medium-range horizon has diagnostics
# on disk, so the other two list the lenses without links.
# slug → (nav label, has data yet)
HORIZONS = {
    "medium_range": ("SHORT & MEDIUM-RANGE", True),
    "subseasonal":  ("SUB-SEASONAL", False),
    "seasonal":     ("SEASONAL", False),
}
CONTACT_MAILTO = "mailto:research@africlimate.ai"

# st.Page objects, filled by app.py at startup via register_pages() so the
# header can st.page_link() to them (client-side switch, session preserved).
_PAGES: dict[str, "st.Page"] = {}


def register_pages(pages: dict[str, "st.Page"]) -> None:
    _PAGES.update(pages)


@st.cache_resource
def _b64_svg(name: str) -> str:
    path = _ASSETS / name
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


def _symbol_img(size: int = 30) -> str:
    """The Africa-leaf mark as an inline <img>."""
    data = _b64_svg("AfriClimate Symbol.svg")
    if not data:
        return ""
    return (
        f"<img src='data:image/svg+xml;base64,{data}' alt='' "
        f"style='width:{size}px;height:{size}px;object-fit:contain;'>"
    )


def icon_img(name: str, size: int = 44) -> str:
    """A ForecastCheck section icon (the designer's SVGs under assets/icons/)
    as an inline <img>. ``name`` is the file stem, e.g. ``nat_overview``.
    Falls back to an empty string if the asset is missing."""
    data = _b64_svg(f"icons/{name}.svg")
    if not data:
        return ""
    return (
        f"<img src='data:image/svg+xml;base64,{data}' alt='' "
        f"style='width:{size}px;height:{size}px;object-fit:contain;'>"
    )


# ═════════════════════════════════════════════════════════════════════════
#  Config-group header bars — three fixed groups per CONFIGURATION panel,
#  each an amber line-art glyph + white label (ported from the redesign).
# ═════════════════════════════════════════════════════════════════════════

_SVG_ATTRS = (f"xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
              f"fill='none' stroke='{AMBER}' stroke-width='2' "
              f"stroke-linecap='round' stroke-linejoin='round'")
_GROUP_SVGS = {
    "Forecast & Truth": (
        f"<svg {_SVG_ATTRS}><circle cx='12' cy='12' r='9'/>"
        "<circle cx='12' cy='12' r='4.5'/>"
        f"<circle cx='12' cy='12' r='1.2' fill='{AMBER}' stroke='none'/></svg>"),
    "Score": (
        f"<svg {_SVG_ATTRS}><line x1='4' y1='20' x2='20' y2='20'/>"
        "<rect x='6' y='10' width='3.5' height='8' rx='1'/>"
        "<rect x='11' y='6' width='3.5' height='12' rx='1'/>"
        "<rect x='16' y='13' width='3.5' height='5' rx='1'/></svg>"),
    "Conditioning": (
        f"<svg {_SVG_ATTRS}><line x1='4' y1='8' x2='20' y2='8'/>"
        "<line x1='4' y1='16' x2='20' y2='16'/>"
        "<circle cx='9' cy='8' r='2.6' fill='#ffffff'/>"
        "<circle cx='15' cy='16' r='2.6' fill='#ffffff'/></svg>"),
}


def config_head_html(label: str) -> str:
    """HTML for a CONFIGURATION group header bar: amber glyph + white label."""
    svg = _GROUP_SVGS.get(label.replace("&amp;", "&"), "")
    icon = ""
    if svg:
        data = base64.b64encode(svg.encode()).decode()
        icon = (f"<img src='data:image/svg+xml;base64,{data}' width='16' "
                "height='16' style='vertical-align:-3px;margin-right:8px;'>")
    return f"<div class='fc-config-head'>{icon}{label}</div>"


@contextmanager
def config_group(prefix: str, name: str, label: str):
    """A CONFIGURATION group as its own bordered sub-card: amber-glyph header
    bar + that group's inputs. The container key carries the ``_cfggrp_``
    marker the global CSS styles."""
    with st.container(key=f"{prefix}_cfggrp_{name}"):
        st.markdown(config_head_html(label), unsafe_allow_html=True)
        yield


# ═════════════════════════════════════════════════════════════════════════
#  Global CSS
# ═════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    # st.html — NOT st.markdown: markdown terminates raw-HTML blocks at the
    # first blank line, which would dump the tail of this stylesheet onto
    # the page as visible text.
    st.html(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600;700;800&display=swap');
:root {{
  --fc-green: {GREEN};
  --fc-green-bright: {GREEN_BRIGHT};
  --fc-header-green: {HEADER_GREEN};
  --fc-green-dark: {GREEN_DARK};
  --fc-green-bg: {GREEN_BG};
  --fc-green-border: {GREEN_BORDER};
  --fc-amber: {AMBER};
  --fc-navy: {NAVY};
  --fc-cream: {CREAM};
  --fc-ink: {INK};
  --fc-ink-soft: {INK_SOFT};
  --fc-muted: {MUTED};
  --fc-font: 'Poppins', 'Inter', sans-serif;
}}
/* Hide Streamlit default chrome (page background = isobar via set_background) */
header[data-testid="stHeader"] {{ display: none; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}
/* Content column. The design artboard's 1240px stranded a lot of a wide
   monitor as empty margin, so this is wider — but still capped: full-bleed
   leaves the config panel and prose absurdly stretched on an ultrawide.
   min() keeps it near-full-width on a laptop and caps the column on a big
   screen. Prose blocks keep their own inline max-width regardless. */
.block-container {{
  max-width: min(1600px, 94vw);
  padding-top: 1.2rem !important;
  padding-bottom: 2rem !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}}
/* Typography — sizes measured off the July Layout artboard.
   Material icon spans are excluded: overriding their font-family would
   render icon ligatures as literal text like "arrow_down". */
html, body,
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{
  font-family: var(--fc-font);
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important;
}}
h1, .fc-h1 {{
  font-family: var(--fc-font) !important;
  color: var(--fc-green) !important;
  font-weight: 700 !important;
  font-size: 1.55rem !important;
}}
h2, .fc-h2 {{
  font-family: var(--fc-font) !important;
  color: var(--fc-green) !important;
  font-weight: 700 !important;
  font-size: 1.3rem !important;
}}
p {{ color: var(--fc-ink); }}
/* ── Header bar ──────────────────────────────────────────────────────── */
.st-key-fc_header {{
  background: var(--fc-header-green);
  border-radius: 16px;
  padding: 10px 22px;
  box-shadow: 0 2px 10px rgba(47,126,52,0.20);
}}
.st-key-fc_header [data-testid="stHorizontalBlock"] {{
  align-items: center;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
}}
/* nav page-links: dark olive labels on the leaf-green bar (per design) */
.st-key-fc_header a[data-testid="stPageLink-NavLink"] {{
  border-radius: 8px;
  padding: 6px 10px;
  justify-content: center;
}}
.st-key-fc_header a[data-testid="stPageLink-NavLink"] p {{
  color: var(--fc-ink-soft) !important;
  font-family: var(--fc-font) !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em;
  text-align: center;
  line-height: 1.2 !important;
  /* One line per tab: wrapping made "THRESHOLD DIAGNOSTICS" two rows and left
     the bar taller than the logo beside it. The header column ratios are sized
     off the label lengths so nowrap has the room it needs. */
  white-space: nowrap;
}}
.st-key-fc_header a[data-testid="stPageLink-NavLink"]:hover {{
  background: rgba(255,255,255,0.25);
}}
/* ── Horizon dropdown triggers ────────────────────────────────────────
   Each horizon tab is an st.popover so the dropdown opens on click and its
   entries can be real page_links (client-side switch, session preserved).
   The trigger has to read as a nav tab, so its button chrome is stripped. */
[class*="st-key-fc_nav_"] button {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 6px 6px !important;
  border-radius: 8px !important;
  width: 100%;
  min-width: 0;
  justify-content: center !important;
}}
/* Belt and braces after the overflow above: let a header column shrink below
   its content instead of pushing its neighbours out of the bar, whatever a
   tab's contents turn out to measure. Deliberately NOT overflow:hidden — that
   would clip the dropdown panel if it renders inside the column. */
.st-key-fc_header [data-testid="stColumn"] {{
  min-width: 0;
}}
[class*="st-key-fc_nav_"] button:hover {{
  background: rgba(255,255,255,0.25) !important;
}}
/* ONLY the label paragraph. Overriding font-family on every descendant span
   also hit the trigger's Material chevron, whose glyph IS a ligature of its
   font — so it fell back to rendering the literal text "expand_more", which
   under white-space:nowrap overflowed the column and flex-shrank the logo and
   CONTACT beside it to nothing. Never restyle the icon's font. */
[class*="st-key-fc_nav_"] button [data-testid="stMarkdownContainer"] p {{
  color: var(--fc-ink-soft) !important;
  font-family: var(--fc-font) !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em;
  white-space: nowrap;
  line-height: 1.2 !important;
}}
/* The chevron keeps Material Symbols Rounded; only its colour and size are
   ours. It is the sole affordance that the tab opens a menu. */
[class*="st-key-fc_nav_"] button [data-testid="stIconMaterial"] {{
  color: var(--fc-ink-soft) !important;
  font-size: 1.1rem !important;
  flex-shrink: 0;
  margin-left: 2px;
}}
/* The horizon whose page is open keeps a standing highlight, so the bar still
   says where you are now that the tab itself is no longer the link. */
[class*="st-key-fc_navwrap_cur_"] button {{
  background: rgba(255,255,255,0.30) !important;
}}
/* Dropdown body: a lens with no data yet is listed but not linked, so the
   three horizons read as the same menu at different stages of readiness. */
.fc-nav-item-soon {{
  font-family: var(--fc-font);
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--fc-muted);
  padding: 6px 2px;
}}
.fc-nav-soon-note {{
  font-family: var(--fc-font);
  font-size: 0.66rem;
  letter-spacing: 0.06em;
  color: var(--fc-muted);
  border-bottom: 1px solid var(--fc-cream);
  padding-bottom: 6px;
  margin-bottom: 4px;
}}
/* ── Buttons ─────────────────────────────────────────────────────────── */
.fc-btn-amber {{
  display: inline-block;
  background: var(--fc-amber);
  color: var(--fc-ink-soft) !important;
  font-family: var(--fc-font);
  font-size: 0.72rem;
  font-weight: 700;
  padding: 5px 14px;
  border-radius: 6px;
  text-decoration: none !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.15);
}}
.fc-btn-amber:hover {{ filter: brightness(1.05); }}
.fc-btn-white {{
  display: inline-block;
  background: #ffffff;
  color: var(--fc-ink-soft) !important;
  font-family: var(--fc-font);
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  padding: 10px 26px;
  border-radius: 10px;
  text-decoration: none !important;
}}
.fc-btn-white:hover {{ background: var(--fc-cream); }}
/* ── Cards ───────────────────────────────────────────────────────────── */
.fc-card {{
  background: #ffffff;
  border: 1px solid var(--fc-green-dark);
  border-radius: 14px;
  padding: 26px 26px 24px;
  height: 100%;
}}
.fc-card.fc-card-tint {{ background: var(--fc-green-bg); }}
.fc-card h3 {{
  font-family: var(--fc-font);
  color: var(--fc-green);
  font-size: 1.05rem;
  font-weight: 700;
  margin: 14px 0 8px;
}}
.fc-card p {{
  color: var(--fc-ink);
  font-size: 0.84rem;
  line-height: 1.65;
  margin: 0 0 6px;
}}
.fc-icon-box {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 92px; height: 92px;
  background: var(--fc-green-bg);
  border: 1px solid var(--fc-green-dark);
  border-radius: 12px;
}}
.fc-coming-soon {{
  font-size: 0.72rem;
  color: var(--fc-muted);
  letter-spacing: 0.04em;
  margin-top: 10px;
  display: inline-block;
}}
/* ── Product sections (home) ──────────────────────────────────────────
   Each verification product on the home page is a banded header followed by
   its own row of parameter cards, so a parameter reads as belonging to the
   horizon above it rather than to the platform at large. */
.fc-product-head {{
  display: flex;
  align-items: center;
  gap: 16px;
  border-top: 2px solid var(--fc-green-dark);
  padding: 22px 2px 0;
  margin-top: 30px;
}}
.fc-product-head .fc-ph-icon {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 62px; height: 62px;
  flex-shrink: 0;
  background: var(--fc-green-bg);
  border: 1px solid var(--fc-green-dark);
  border-radius: 12px;
}}
.fc-product-head h2 {{
  font-family: var(--fc-font);
  color: var(--fc-green);
  font-size: 1.3rem;
  font-weight: 700;
  margin: 0 0 4px;
}}
.fc-product-head p {{
  color: var(--fc-muted);
  font-size: 0.84rem;
  line-height: 1.6;
  margin: 0;
  max-width: 760px;
}}
.fc-status {{
  font-family: var(--fc-font);
  font-size: 0.64rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 3px 9px;
  border-radius: 5px;
  margin-left: 10px;
  vertical-align: 2px;
  white-space: nowrap;
}}
.fc-status-live {{ background: var(--fc-green); color: #ffffff; }}
.fc-status-soon {{ background: var(--fc-cream); color: var(--fc-muted);
                   border: 1px solid var(--fc-muted); }}
/* A parameter card for a horizon that has no data yet: same shape, drained of
   colour, and no Explore link — nothing to click through to. */
.fc-card.fc-card-soon {{
  background: #fbfbfa;
  border: 1px dashed var(--fc-muted);
}}
.fc-card.fc-card-soon h3 {{ color: var(--fc-muted); }}
.fc-card.fc-card-soon p {{ color: var(--fc-muted); }}
.fc-card.fc-card-soon .fc-icon-box,
.fc-card.fc-card-soon svg {{ opacity: 0.45; }}
/* ── CONFIGURATION group sub-cards (Forecast & Truth / Score / Conditioning) ──
   Each config_group() renders a white bordered sub-card with a dark-green
   header bar (amber glyph + white label) holding that group's inputs. */
.fc-config-head {{
  background: var(--fc-green-dark);
  color: #ffffff;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  border-radius: 6px;
  padding: 5px 10px;
  margin: 2px 0 8px;
  display: flex;
  align-items: center;
}}
[class*="st-key-"][class*="_cfggrp_"] {{
  background: #ffffff;
  border: 1px solid var(--fc-green-dark);
  border-radius: 12px;
  padding: 8px 12px 4px !important;
  margin-bottom: 12px;
}}
[class*="st-key-"][class*="_cfggrp_"] .fc-config-head {{ margin: 2px 0 8px !important; }}
[class*="st-key-"][class*="_cfggrp_"] [data-testid="stWidgetLabel"] p {{
  font-size: 0.62rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em;
  color: var(--fc-green-dark) !important;
}}
/* The CONFIGURATION panel wrapper (light-green frame around the groups) */
[class*="st-key-"][class*="_config"] {{
  background: var(--fc-green-bg);
  border: 1px solid var(--fc-green-dark);
  border-radius: 14px;
  padding: 14px 14px 4px;
}}
/* ── Responsive ──────────────────────────────────────────────────────── */
@media (max-width: 900px) {{
  /* Reflowed onto several rows — wrapping inside a tab is fine again. */
  .st-key-fc_header a[data-testid="stPageLink-NavLink"] p {{
    white-space: normal;
  }}
  .block-container {{
    padding-left: 0.9rem !important;
    padding-right: 0.9rem !important;
  }}
  h1, .fc-h1 {{ font-size: 1.3rem !important; }}
  h2, .fc-h2 {{ font-size: 1.12rem !important; }}
  .fc-card {{ padding: 18px 18px 16px; }}
  .fc-icon-box {{ width: 68px; height: 68px; }}
  .fc-icon-box svg {{ width: 38px; height: 38px; }}
}}
@media (max-width: 640px) {{
  .st-key-fc_header {{ padding: 10px 14px; }}
  .st-key-fc_header [data-testid="stHorizontalBlock"] {{
    flex-wrap: wrap;
    row-gap: 4px;
  }}
  .st-key-fc_header [data-testid="stColumn"] {{
    width: auto !important;
    flex: 0 1 auto !important;
    min-width: 0 !important;
  }}
}}
</style>
        """,
    )


# ═════════════════════════════════════════════════════════════════════════
#  Header
# ═════════════════════════════════════════════════════════════════════════

def _page_key(horizon: str, lens: str) -> str:
    """Registry key for one horizon's take on one lens.

    The medium-range horizon owns the original pages, so its lenses keep their
    bare slugs; the forthcoming horizons prefix theirs. Mirrored by the
    st.Page registrations in ui/app.py.
    """
    return lens if horizon == "medium_range" else f"{horizon}_{lens}"


def _horizon_menu(horizon: str, label: str, live: bool) -> None:
    """One horizon tab: a click-to-open dropdown of the three analysis lenses.

    st.popover rather than a CSS menu so each entry can be a real st.page_link
    — a client-side page switch that preserves session state, where an <a href>
    would reload the app and drop every selection.
    """
    with st.popover(label, type="tertiary", use_container_width=True,
                    key=f"fc_nav_{horizon}"):
        if not live:
            st.markdown(
                "<div class='fc-nav-soon-note'>NO DATA ON THE PLATFORM YET</div>",
                unsafe_allow_html=True,
            )
        for lens, (lens_label, url_path) in NAV_PAGES.items():
            page_obj = _PAGES.get(_page_key(horizon, lens))
            if page_obj is not None:
                st.page_link(page_obj, label=lens_label)
            elif live:
                # Registry not populated (direct import, tests): fall back to a
                # plain link so the menu still navigates.
                st.markdown(
                    f"<a href='{url_path}' target='_self'>{lens_label}</a>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='fc-nav-item-soon'>{lens_label}</div>",
                    unsafe_allow_html=True,
                )


def render_header(active: str = "", horizon: str = "medium_range") -> None:
    """Green nav bar: logo · one dropdown per forecast horizon · CONTACT.

    The horizons are the top-level choice now; the three analysis pages hang
    off each as a dropdown, so a page is reached as (horizon, lens) rather than
    from a flat list of three. ``horizon`` says which tab to highlight — the
    pages themselves pass it, defaulting to the only horizon with data.
    ``active`` is accepted for call-site compatibility and no longer needed:
    st.page_link marks the current lens inside the open dropdown itself.
    """
    with st.container(key="fc_header"):
        # Ratios track the label lengths (20 / 12 / 8 chars) PLUS the dropdown
        # chevron, which the old plain-link tabs did not have to fit; the
        # spacer gives up the room. Every tab stays on ONE line under
        # white-space:nowrap.
        col_logo, col_sp, c1, c2, c3, col_contact = st.columns(
            [2.9, 0.3, 2.6, 1.6, 1.3, 1.3], gap="small",
        )
        with col_logo:
            logo_data = _b64_svg("forecastcheck_wordmark_dark.svg")
            st.markdown(
                f"""
                <a href="./" target="_self" style="text-decoration:none;
                    display:inline-flex;align-items:center;">
                  <img src="data:image/svg+xml;base64,{logo_data}" alt="ForecastCheck"
                      style="height:48px;width:auto;display:block;">
                </a>
                """,
                unsafe_allow_html=True,
            )
        for col, (hkey, (label, live)) in zip((c1, c2, c3), HORIZONS.items()):
            with col:
                # Highlight rides on a keyed wrapper: st.container(key=...)
                # emits a real class, which is the only handle CSS has on a
                # widget it cannot be given a class directly.
                state = "cur" if hkey == horizon else "off"
                with st.container(key=f"fc_navwrap_{state}_{hkey}"):
                    _horizon_menu(hkey, label, live)
        with col_contact:
            st.markdown(
                "<div style='text-align:right;'>"
                "<a class='fc-btn-white' href='contact' target='_self'>CONTACT</a></div>",
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════
#  Footer
# ═════════════════════════════════════════════════════════════════════════

_SOCIALS = [
    ("LinkedIn", "https://www.linkedin.com/company/africlimate-ai",
     "M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.36V9h3.41v1.56h.05c.47-.9 1.63-1.85 3.36-1.85 3.6 0 4.27 2.37 4.27 5.45v6.29zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.55V9h3.57v11.45z"),
    ("YouTube", "https://www.youtube.com/@africlimateai",
     "M23.5 6.19a3.02 3.02 0 0 0-2.12-2.14C19.5 3.55 12 3.55 12 3.55s-7.5 0-9.38.5A3.02 3.02 0 0 0 .5 6.19C0 8.07 0 12 0 12s0 3.93.5 5.81a3.02 3.02 0 0 0 2.12 2.14c1.88.5 9.38.5 9.38.5s7.5 0 9.38-.5a3.02 3.02 0 0 0 2.12-2.14C24 15.93 24 12 24 12s0-3.93-.5-5.81zM9.55 15.57V8.43L15.82 12l-6.27 3.57z"),
    ("Instagram", "https://www.instagram.com/africlimateai",
     "M12 2.16c3.2 0 3.58.01 4.85.07 1.17.05 1.8.25 2.23.41.56.22.96.48 1.38.9.42.42.68.82.9 1.38.16.42.36 1.06.41 2.23.06 1.27.07 1.65.07 4.85s-.01 3.58-.07 4.85c-.05 1.17-.25 1.8-.41 2.23-.22.56-.48.96-.9 1.38-.42.42-.82.68-1.38.9-.42.16-1.06.36-2.23.41-1.27.06-1.65.07-4.85.07s-3.58-.01-4.85-.07c-1.17-.05-1.8-.25-2.23-.41a3.72 3.72 0 0 1-1.38-.9 3.72 3.72 0 0 1-.9-1.38c-.16-.42-.36-1.06-.41-2.23-.06-1.27-.07-1.65-.07-4.85s.01-3.58.07-4.85c.05-1.17.25-1.8.41-2.23.22-.56.48-.96.9-1.38.42-.42.82-.68 1.38-.9.42-.16 1.06-.36 2.23-.41 1.27-.06 1.65-.07 4.85-.07zM12 0C8.74 0 8.33.01 7.05.07 5.78.13 4.9.33 4.14.63a5.88 5.88 0 0 0-2.13 1.38A5.88 5.88 0 0 0 .63 4.14C.33 4.9.13 5.78.07 7.05.01 8.33 0 8.74 0 12s.01 3.67.07 4.95c.06 1.27.26 2.15.56 2.91.31.8.72 1.48 1.38 2.13a5.88 5.88 0 0 0 2.13 1.38c.76.3 1.64.5 2.91.56C8.33 23.99 8.74 24 12 24s3.67-.01 4.95-.07c1.27-.06 2.15-.26 2.91-.56a5.88 5.88 0 0 0 2.13-1.38 5.88 5.88 0 0 0 1.38-2.13c.3-.76.5-1.64.56-2.91.06-1.28.07-1.69.07-4.95s-.01-3.67-.07-4.95c-.06-1.27-.26-2.15-.56-2.91a5.88 5.88 0 0 0-1.38-2.13A5.88 5.88 0 0 0 19.86.63c-.76-.3-1.64-.5-2.91-.56C15.67.01 15.26 0 12 0zm0 5.84A6.16 6.16 0 1 0 18.16 12 6.16 6.16 0 0 0 12 5.84zm0 10.15A4 4 0 1 1 16 12a4 4 0 0 1-4 3.99zm7.85-10.4a1.44 1.44 0 1 1-2.88 0 1.44 1.44 0 0 1 2.88 0z"),
    ("X", "https://x.com/africlimateai",
     "M18.24 2.25h3.31l-7.23 8.26 8.5 11.24h-6.66l-5.21-6.82-5.97 6.82H1.67l7.73-8.84L1.25 2.25h6.83l4.71 6.23 5.45-6.23zm-1.16 17.52h1.83L7.08 4.13H5.12l11.96 15.64z"),
]


def _social_icons_html() -> str:
    icons = ""
    for name, url, path_d in _SOCIALS:
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
            f"fill='#ffffff'><path d='{path_d}'/></svg>"
        )
        data = base64.b64encode(svg.encode()).decode()
        icons += (
            f"<a href='{url}' target='_blank' title='{name}' "
            f"style='display:inline-flex;align-items:center;justify-content:center;"
            f"width:48px;height:48px;background:{INK_SOFT};border-radius:11px;"
            f"border:1.5px solid {AMBER};margin-left:14px;text-decoration:none;'>"
            f"<img src='data:image/svg+xml;base64,{data}' width='22' height='22' "
            f"alt='{name}' style='display:block;'></a>"
        )
    return icons


@st.cache_resource
def _footer_html() -> str:
    symbol = _b64_svg("AfriClimate Symbol.svg")
    logo = (
        f"<img src='data:image/svg+xml;base64,{symbol}' alt='' "
        "style='width:58px;height:58px;object-fit:contain;'>"
        if symbol else ""
    )
    link = "text-decoration:none;color:#20261f;font-size:0.8rem;line-height:2;display:block;"
    edge = f"1.5px solid {INK_SOFT}"   # design: dark ink card border + dividers
    return f"""
<div style="border:{edge};border-radius:16px;margin-top:56px;
    font-family:'Poppins',sans-serif;overflow:hidden;background:#f4f5f2;">

  <div style="display:flex;flex-wrap:wrap;align-items:center;gap:28px;
      padding:30px 34px 26px;">
    <div style="display:flex;align-items:center;gap:14px;flex:2;min-width:260px;">
      {logo}
      <div style="line-height:1.1;">
        <div style="font-family:'Poppins',sans-serif;font-size:1.55rem;font-weight:700;
            color:{NAVY};">AfriClimate <span style="font-weight:800;">AI</span></div>
        <div style="font-size:0.62rem;color:{NAVY};opacity:0.75;letter-spacing:0.02em;">
            AI for a Climate-Resilient Africa</div>
      </div>
    </div>
    <div style="flex:1;min-width:170px;">
      <a style="{link}" href="score-decomposition" target="_self">Score Decomposition</a>
      <a style="{link}" href="national-overview" target="_self">National Overview</a>
      <a style="{link}" href="threshold-diagnostics"
         target="_self">Threshold Diagnostics</a>
    </div>
    <div style="flex:1;min-width:130px;border-left:1px solid {INK_SOFT};padding-left:24px;">
      <a style="{link}" href="https://africlimate.ai" target="_blank">About Us</a>
      <a style="{link}" href="https://africlimate.ai" target="_blank">The Team</a>
      <a style="{link}" href="./" target="_self">ForecastCheck</a>
    </div>
    <div style="flex:1;min-width:110px;border-left:1px solid {INK_SOFT};padding-left:24px;">
      <a style="{link}" href="contact" target="_self">Contact</a>
      <a style="{link}" href="https://africlimate.ai" target="_blank">Hiring</a>
    </div>
  </div>

  <div style="display:flex;flex-wrap:wrap;align-items:center;gap:16px;
      border-top:{edge};padding:18px 34px;">
    <div style="flex:1;min-width:280px;font-size:0.78rem;color:#20261f;line-height:1.6;
        border-right:1px solid {INK_SOFT};padding-right:24px;">
      <b>AfriClimate AI Verification Platform</b> is developed and maintained by<br>
      <b>AfriClimate AI</b> as part of the <b>Forecast4Africa</b> initiative.
    </div>
    <div style="display:flex;">{_social_icons_html()}</div>
  </div>

  <div style="border-top:{edge};padding:16px 34px;font-size:0.72rem;
      color:#454d46;line-height:1.7;">
    Data and results are released under the <b>Creative Commons Attribution 4.0
    International (CC BY 4.0) licence</b>.<br>
    Please cite the companion manuscript when using this resource.<br>
    Queries, collaborations, or data contributions:
    <a href="{CONTACT_MAILTO}" style="color:{GREEN};font-weight:600;
        text-decoration:none;">research@africlimate.ai</a>
  </div>
</div>
"""


def render_footer() -> None:
    # st.html, not st.markdown — the footer HTML contains blank lines,
    # which markdown would treat as the end of the raw-HTML block.
    st.html(_footer_html())


# ═════════════════════════════════════════════════════════════════════════
#  "Explore more of the platform" cards — two tinted cards above the footer
#  linking to the OTHER analysis pages (per the design). Kept here (not in a
#  page module) so every page can call it without a circular import.
# ═════════════════════════════════════════════════════════════════════════

# slug → (icon file stem, title, url_path, description). Icons are the
# designer's SVGs under assets/icons/ (never hand-drawn in code).
_CARD_META = {
    "national_overview": (
        "nat_overview", "National Overview", "national-overview",
        "Calibrated forecast skill by season, lead time and region, alongside "
        "the raw models scored by MAE, bias and correlation."),
    "score_decomposition": (
        "score_decomp", "Score Decomposition", "score-decomposition",
        "Investigate the miscalibration error of your model and its discrimination "
        "improvement in comparison to a reference forecast — either climatology or "
        "another calibrated forecast product."),
    "detailed_metrics": (
        "detailed_metrics", "Threshold Diagnostics", "threshold-diagnostics",
        "Reliability, ROC and cost-loss curves over the thresholds, altitude bands, "
        "regions and seasons the pipeline has pooled."),
}


def param_card_html(slug: str, *, enabled: bool = True) -> str:
    """One parameter card. ``enabled=False`` renders the same card for a
    horizon with no data yet: drained of colour and with no Explore link, since
    there is nothing to click through to. Copy comes from _CARD_META either
    way, so the live and forthcoming products describe a lens identically."""
    icon, title, url, desc = _CARD_META[slug]
    if not enabled:
        return (
            f"<div class='fc-card fc-card-soon'>{icon_img(icon, 48)}"
            f"<h3>{title}</h3><p>{desc}</p>"
            f"<span class='fc-coming-soon'>Coming Soon</span></div>"
        )
    return (
        f"<div class='fc-card fc-card-tint'>{icon_img(icon, 48)}"
        f"<h3>{title}</h3><p>{desc}</p>"
        f"<a class='fc-btn-amber' href='{url}' target='_self'>Explore ›</a></div>"
    )


def _card_html(slug: str) -> str:
    return param_card_html(slug)


def render_explore_cards(*slugs: str, enabled: bool = True,
                         spacer: int = 34) -> None:
    """Render one card per slug in a row (design: two cards above the footer
    that link to the other platform pages; the home page uses the same row
    under each product heading). ``enabled=False`` greys the row out."""
    if spacer:
        st.markdown(f"<div style='height:{spacer}px;'></div>",
                    unsafe_allow_html=True)
    cols = st.columns(len(slugs), gap="medium")
    for col, slug in zip(cols, slugs):
        with col:
            st.markdown(param_card_html(slug, enabled=enabled),
                        unsafe_allow_html=True)
