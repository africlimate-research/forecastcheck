# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""PDF report generator — Plotly → PNG via kaleido, assembled with reportlab."""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import xarray as xr
from ui.app_config.settings import (LEAD_TIMES, VARIABLE_UNITS, direction_note,
                                    higher_is_better, zero_is_best)
from ui.backend.data_loader.climatology_loader import ClimData
from ui.components.filters import (FilterState,
                                                variant_display_name)
from ui.services.loader import LeadTimeData
from ui.utils.geo import (annual_spatial_grid, band_annual_mean,
                                       regional_annual_mean,
                                       regional_monthly_means, subset_region)

# ── Stat helpers (call private numeric functions from stat_blocks) ─────────

def _hypothesis_cells(label_text: str, value_text: str, colors_mod):
    """Render the H0 / H1 row as Paragraphs.

    The raw Unicode subscripts H₀/H₁ aren't in Helvetica, so reportlab
    renders them as boxes. Paragraph supports inline <sub> markup, so we
    rewrite the label to use that, and run the value through Paragraph too
    so its `<` is escaped instead of being parsed as XML.
    """
    import xml.sax.saxutils as _sax

    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    label_style = ParagraphStyle(
        "HypLabel", fontName="Helvetica-Bold", fontSize=8, leading=10,
        textColor=colors_mod.HexColor("#9CA3AF"),
    )
    value_style = ParagraphStyle(
        "HypValue", fontName="Helvetica", fontSize=8, leading=10,
        textColor=colors_mod.HexColor("#374151"),
    )
    label_html = (label_text
                  .replace("H₀", "H<sub>0</sub>")
                  .replace("H₁", "H<sub>1</sub>"))
    value_html = _sax.escape(value_text)
    return Paragraph(label_html, label_style), Paragraph(value_html, value_style)


def _paired_ttest_table(vals_a: np.ndarray, vals_b: np.ndarray,
                         name_a: str, name_b: str, unit: str,
                         content_w: float, colors_mod,
                         higher_better: bool = False):
    """Return a reportlab Table for a paired t-test result, or None."""
    from ui.components.stat_blocks import _paired_t
    r = _paired_t(vals_a, vals_b, higher_better)
    if r is None:
        return None

    u    = f" ({unit})" if unit else ""
    sig  = r["p_better"] < 0.05
    pct_a = r["n_a_better"] / r["n"] * 100
    pct_b = r["n_b_better"] / r["n"] * 100

    from reportlab.platypus import Table, TableStyle
    GREEN  = colors_mod.HexColor("#0a9938")
    BLUE   = colors_mod.HexColor("#213b7d")
    DARK   = colors_mod.HexColor("#171d1a")
    LGRAY  = colors_mod.HexColor("#E5E7EB")
    OFFWHITE = colors_mod.HexColor("#F9FAFB")
    RESULT_COLOR = GREEN if sig else colors_mod.HexColor("#6B7280")

    h0_rel, h1_rel = ("≤", ">") if higher_better else ("≥", "<")
    title = f"Paired T-Test (one-sided)  -  {name_a} vs {name_b}"
    hyp_label, hyp_value = _hypothesis_cells(
        "H₀ / H₁",
        f"mean({name_a} − {name_b}) {h0_rel} 0   /   "
        f"{name_a} {h1_rel} {name_b}, i.e. {name_a} better   (α = 0.05)",
        colors_mod,
    )
    rows = [
        [title, ""],
        ["Statistic", "Value"],
        [hyp_label, hyp_value],
        ["n pixels", f"{r['n']:,}"],
        ["mean diff" + u, f"{r['mean_d']:+.4f}"],
        ["t-statistic", f"{r['t']:.3f}"],
        ["df", f"{r['df']:,}"],
        [f"95% CI{u}", f"[{r['ci_lo']:+.4f},  {r['ci_hi']:+.4f}]"],
        ["p-value (one-sided)", f"{r['p_better']:.4f}"],
        [f"{name_a} better", f"{pct_a:.1f}%  ({r['n_a_better']:,} pixels)"],
        [f"{name_b} better", f"{pct_b:.1f}%  ({r['n_b_better']:,} pixels)"],
        ["Result", "✓ Significant (p < 0.05)" if sig else "✗ Not significant (p ≥ 0.05)"],
    ]

    cw = content_w
    tbl = Table(rows, colWidths=[cw * 0.35, cw * 0.65], hAlign="LEFT")
    result_row_idx = len(rows) - 1
    tbl.setStyle(TableStyle([
        # Title row (spans both columns)
        ("SPAN",          (0, 0), (-1, 0)),
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors_mod.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  10),
        ("ALIGN",         (0, 0), (-1, 0),  "LEFT"),
        ("LEFTPADDING",   (0, 0), (-1, 0),  10),
        ("TOPPADDING",    (0, 0), (-1, 0),  7),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  7),
        # Column-header row
        ("BACKGROUND",    (0, 1), (-1, 1),  BLUE),
        ("TEXTCOLOR",     (0, 1), (-1, 1),  colors_mod.white),
        ("FONTNAME",      (0, 1), (-1, 1),  "Helvetica-Bold"),
        # Data rows
        ("BACKGROUND",    (0, 2), (0, -1),  OFFWHITE),
        ("FONTNAME",      (0, 2), (0, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 2), (0, -1),  colors_mod.HexColor("#9CA3AF")),
        ("FONTNAME",      (1, 2), (1, -1),  "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 2), (-1, -2), [colors_mod.white, OFFWHITE]),
        ("GRID",          (0, 1), (-1, -1), 0.4, LGRAY),
        ("PADDING",       (0, 1), (-1, -1), 5),
        # Verdict row
        ("TEXTCOLOR",     (1, result_row_idx), (1, result_row_idx), RESULT_COLOR),
        ("FONTNAME",      (1, result_row_idx), (1, result_row_idx), "Helvetica-Bold"),
    ]))
    return tbl


def _desc_stats_table(vals: np.ndarray, name: str, unit: str,
                       content_w: float, colors_mod):
    """Return a reportlab Table for descriptive stats, or None."""
    from ui.components.stat_blocks import _describe
    s = _describe(vals)
    if s is None:
        return None

    u = f" ({unit})" if unit else ""
    from reportlab.platypus import Table, TableStyle
    BLUE     = colors_mod.HexColor("#213b7d")
    DARK     = colors_mod.HexColor("#171d1a")
    LGRAY    = colors_mod.HexColor("#E5E7EB")
    OFFWHITE = colors_mod.HexColor("#F9FAFB")

    title = f"Descriptive Statistics  -  {name}"
    rows = [
        [title, ""],
        ["Statistic", "Value"],
        ["n pixels",         f"{s['n']:,}"],
        [f"mean{u}",         f"{s['mean']:+.4f}"],
        [f"std{u}",          f"{s['std']:.4f}"],
        ["median",           f"{s['median']:+.4f}"],
        ["min / max",        f"{s['min']:.4f}  /  {s['max']:.4f}"],
        ["IQR",              f"{s['iqr']:.4f}"],
        ["skewness",         f"{s['skew']:.3f}"],
    ]

    cw = content_w
    tbl = Table(rows, colWidths=[cw * 0.35, cw * 0.65], hAlign="LEFT")
    tbl.setStyle(TableStyle([
        # Title row (spans both columns)
        ("SPAN",          (0, 0), (-1, 0)),
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors_mod.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  10),
        ("ALIGN",         (0, 0), (-1, 0),  "LEFT"),
        ("LEFTPADDING",   (0, 0), (-1, 0),  10),
        ("TOPPADDING",    (0, 0), (-1, 0),  7),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  7),
        # Column-header row
        ("BACKGROUND",    (0, 1), (-1, 1),  BLUE),
        ("TEXTCOLOR",     (0, 1), (-1, 1),  colors_mod.white),
        ("FONTNAME",      (0, 1), (-1, 1),  "Helvetica-Bold"),
        # Data rows
        ("BACKGROUND",    (0, 2), (0, -1),  OFFWHITE),
        ("FONTNAME",      (0, 2), (0, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 2), (0, -1),  colors_mod.HexColor("#9CA3AF")),
        ("FONTNAME",      (1, 2), (1, -1),  "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 2), (-1, -1), [colors_mod.white, OFFWHITE]),
        ("GRID",          (0, 1), (-1, -1), 0.4, LGRAY),
        ("PADDING",       (0, 1), (-1, -1), 5),
    ]))
    return tbl


def _wilcoxon_table(vals_a: np.ndarray, vals_b: np.ndarray,
                     name_a: str, name_b: str, unit: str,
                     content_w: float, colors_mod,
                     higher_better: bool = False):
    """Return a reportlab Table for Wilcoxon + sign test results, or None."""
    from ui.components.stat_blocks import _wilcoxon
    r = _wilcoxon(vals_a, vals_b, higher_better)
    if r is None:
        return None

    u    = f" ({unit})" if unit else ""
    w_sig = (r["p_wilcoxon"] is not None and r["p_wilcoxon"] < 0.05)
    s_sig = r["p_sign"] < 0.05

    from reportlab.platypus import Table, TableStyle
    GREEN    = colors_mod.HexColor("#0a9938")
    BLUE     = colors_mod.HexColor("#213b7d")
    DARK     = colors_mod.HexColor("#171d1a")
    LGRAY    = colors_mod.HexColor("#E5E7EB")
    OFFWHITE = colors_mod.HexColor("#F9FAFB")

    pct_a = r["n_a_better"] / r["n"] * 100
    pct_b = r["n_b_better"] / r["n"] * 100

    verdict = ("Both significant" if w_sig and s_sig
               else "Partially significant" if w_sig or s_sig
               else "Not significant")
    result_color = GREEN if (w_sig and s_sig) else (
        colors_mod.HexColor("#f2b029") if (w_sig or s_sig)
        else colors_mod.HexColor("#6B7280")
    )

    sample_note = f" (sample n={r['n_wilcoxon']:,})" if r["sampled"] else ""
    w_pval = f"{r['p_wilcoxon']:.4f}{sample_note}" if r["p_wilcoxon"] is not None else "n/a"

    h0_rel, h1_rel = ("≤", ">") if higher_better else ("≥", "<")
    title = f"Wilcoxon Signed-Rank + Sign Test  -  {name_a} vs {name_b}"
    hyp_label, hyp_value = _hypothesis_cells(
        "H₀ / H₁",
        f"median({name_a} − {name_b}) {h0_rel} 0   /   "
        f"{name_a} {h1_rel} {name_b}, i.e. {name_a} better   (α = 0.05, no normality assumed)",
        colors_mod,
    )
    rows = [
        [title, ""],
        ["Statistic", "Value"],
        [hyp_label, hyp_value],
        ["n pixels (total)", f"{r['n']:,}"],
        ["Wilcoxon W" + sample_note, f"{r['w_stat']:.0f}" if r["w_stat"] is not None else "n/a"],
        ["Wilcoxon p (one-sided)", w_pval],
        [f"{name_a} wins (sign)", f"{r['n_a_better']:,}  ({pct_a:.1f}%)"],
        [f"{name_b} wins (sign)", f"{r['n_b_better']:,}  ({pct_b:.1f}%)"],
        ["tied pixels", f"{r['n_tied']:,}"],
        ["Sign test p (one-sided)", f"{r['p_sign']:.4f}"],
        ["Result", verdict],
    ]

    cw = content_w
    tbl = Table(rows, colWidths=[cw * 0.35, cw * 0.65], hAlign="LEFT")
    result_row_idx = len(rows) - 1
    tbl.setStyle(TableStyle([
        # Title row (spans both columns)
        ("SPAN",          (0, 0), (-1, 0)),
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors_mod.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  10),
        ("ALIGN",         (0, 0), (-1, 0),  "LEFT"),
        ("LEFTPADDING",   (0, 0), (-1, 0),  10),
        ("TOPPADDING",    (0, 0), (-1, 0),  7),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  7),
        # Column-header row
        ("BACKGROUND",    (0, 1), (-1, 1),  BLUE),
        ("TEXTCOLOR",     (0, 1), (-1, 1),  colors_mod.white),
        ("FONTNAME",      (0, 1), (-1, 1),  "Helvetica-Bold"),
        # Data rows
        ("BACKGROUND",    (0, 2), (0, -1),  OFFWHITE),
        ("FONTNAME",      (0, 2), (0, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 2), (0, -1),  colors_mod.HexColor("#9CA3AF")),
        ("FONTNAME",      (1, 2), (1, -1),  "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 2), (-1, -2), [colors_mod.white, OFFWHITE]),
        ("GRID",          (0, 1), (-1, -1), 0.4, LGRAY),
        ("PADDING",       (0, 1), (-1, -1), 5),
        # Verdict row
        ("TEXTCOLOR",     (1, result_row_idx), (1, result_row_idx), result_color),
        ("FONTNAME",      (1, result_row_idx), (1, result_row_idx), "Helvetica-Bold"),
    ]))
    return tbl


def _test_tables(vals_a, vals_b, name_a, name_b, unit, content_w, colors_mod, test: str,
                 higher_better: bool = False):
    """Return (ttest_tbl, wilcoxon_tbl) based on the selected test string."""
    t_choice = {"T-Test": "ttest", "Wilcoxon + Sign": "wilcoxon", "Both": "both"}.get(test, "ttest")
    tbl_t = _paired_ttest_table(vals_a, vals_b, name_a, name_b, unit, content_w, colors_mod,
                                higher_better) \
        if t_choice in ("ttest", "both") else None
    tbl_w = _wilcoxon_table(vals_a, vals_b, name_a, name_b, unit, content_w, colors_mod,
                            higher_better) \
        if t_choice in ("wilcoxon", "both") else None
    return tbl_t, tbl_w


# ── Kaleido helper ─────────────────────────────────────────────────────────

def _fig_to_png(fig, width: int = 870, height: int = 400) -> bytes | None:
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception as exc:
        print(f"[pdf_report] kaleido error: {exc}")
        return None


def _png_image(png: bytes, content_w: float, aspect: float):
    """Return a reportlab Image flowable from raw PNG bytes."""
    from reportlab.platypus import Image
    return Image(io.BytesIO(png), width=content_w, height=content_w * aspect)


# ── Main entry point ───────────────────────────────────────────────────────

def generate_report(
    f:               FilterState,
    all_data:        dict[str, dict[int, LeadTimeData]],
    clim:            ClimData | None,
    subtract_clim:   bool,
    spatial_lt:      int,
    seasonal_lt:     int | None      = None,
    comparison_lt:   int | None      = None,
    spatial_test:    str             = "T-Test",
    comparison_test: str             = "T-Test",
    altitude_band:   str             = "all",
) -> bytes:
    """Build a PDF verification report and return its bytes.

    Each tab in the app has its own lead-time slider, so the report takes
    one lead time per section and renders each exactly at that value. The
    Comparison section iterates every pairwise (A, B) combination so the
    PDF is a complete record — the user doesn't have to re-export once per
    pair.

    ``altitude_band`` mirrors the dashboard's altitude selector: scores and
    maps are restricted to that band and every caption says so (the caller
    disables ``subtract_clim`` when a band is selected, since the climatology
    baseline is all-altitudes only).
    """
    if seasonal_lt is None:
        seasonal_lt = spatial_lt
    if comparison_lt is None:
        comparison_lt = spatial_lt

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    buf = io.BytesIO()
    # A4 landscape gives ~27cm of horizontal room — enough for long
    # hypothesis text and three-column comparison tables without wrapping
    # or overlap.
    page_size = landscape(A4)
    W, H = page_size
    margin = 1.5 * cm
    doc = SimpleDocTemplate(
        buf, pagesize=page_size,
        rightMargin=margin, leftMargin=margin,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    content_w = W - 2 * margin

    styles  = getSampleStyleSheet()
    GREEN   = colors.HexColor("#0a9938")
    BLUE    = colors.HexColor("#213b7d")
    DARK    = colors.HexColor("#171d1a")
    GRAY    = colors.HexColor("#6B7280")
    LGRAY   = colors.HexColor("#E5E7EB")
    OFFWHITE = colors.HexColor("#F9FAFB")
    HEADER_BG = colors.HexColor("#F3F4F6")

    def _style(name, **kw):
        p = ParagraphStyle(name, parent=styles["Normal"], **kw)
        return p

    title_s    = _style("T", fontSize=20, textColor=GREEN,   fontName="Helvetica-Bold",  spaceAfter=10)
    sub_s      = _style("S", fontSize=9,  textColor=GRAY,    fontName="Helvetica",        spaceAfter=8)
    section_s  = _style("H", fontSize=12, textColor=DARK,    fontName="Helvetica-Bold",   spaceBefore=12, spaceAfter=5)
    body_s     = _style("B", fontSize=9,  textColor=colors.HexColor("#374151"), fontName="Helvetica", spaceAfter=3)
    caption_s  = _style("C", fontSize=8,  textColor=GRAY,    fontName="Helvetica-Oblique", spaceAfter=4)

    story = []

    # ── Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Forecast Verification Report", title_s))
    story.append(Paragraph(
        f"AfriClimate AI · Forecast4Africa · "
        f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        sub_s,
    ))

    # ── Configuration table ────────────────────────────────────────────────
    story.append(Paragraph("Configuration", section_s))

    model_labels = ", ".join(variant_display_name(fc) for fc in f.selected_variants) or "—"
    unit = VARIABLE_UNITS.get(f.variable, "")
    # Direction of goodness — drives one-sided tests, rankings, and captions.
    # Bias (zero-is-best) is compared by magnitude |bias|, then lower-is-better.
    hb = higher_is_better(f.metric_key)
    zb = zero_is_best(f.metric_key)
    metric_disp = f"|{f.metric_key.upper()}|" if zb else f.metric_key.upper()

    band_disp = "All" if altitude_band == "all" else f"{altitude_band} m"
    cfg_rows = [
        ["Region",          f.region,                              "Variable",        f.variable],
        ["Truth Source",    f.truth,                               "Year",            f.year],
        ["Metric",          f.metric_key.upper(),                  "vs Climatology",  "Yes" if subtract_clim else "No"],
        ["Spatial Lead",    f"{spatial_lt}h",                      "Seasonal Lead",   f"{seasonal_lt}h"],
        ["Comparison Lead", f"{comparison_lt}h",                   "Models",          model_labels],
        ["Altitude Band",   band_disp,                             "",                ""],
    ]
    cw = content_w / 4
    cfg_table = Table(cfg_rows, colWidths=[cw * 0.85, cw * 1.15, cw * 0.85, cw * 1.15], hAlign="LEFT")
    cfg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), HEADER_BG),
        ("BACKGROUND", (2, 0), (2, -1), HEADER_BG),
        ("TEXTCOLOR",  (0, 0), (0, -1), GRAY),
        ("TEXTCOLOR",  (2, 0), (2, -1), GRAY),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, LGRAY),
        ("PADDING",    (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, OFFWHITE]),
    ]))
    story.append(cfg_table)
    story.append(Spacer(1, 10))

    # ── Precompute climatology values ───────────────────────────────────────
    climatology_value: float | None = None
    clim_spatial: xr.DataArray | None = None
    clim_monthly: np.ndarray | None = None

    if subtract_clim and clim is not None:
        clim_spatial = subset_region(clim.annual_spatial_average, f.region)
        try:
            climatology_value = float(clim_spatial.mean().values)
        except Exception:
            climatology_value = float(clim.annual_average)
        try:
            clim_monthly = regional_monthly_means(clim, f.region)
        except Exception:
            clim_monthly = np.asarray(clim.monthly_average)

    # ── 1. Lead-time performance ────────────────────────────────────────────
    _band_note = "" if altitude_band == "all" else f" · altitude: {altitude_band} m"
    story.append(Paragraph("1 · Lead-Time Performance", section_s))
    story.append(Paragraph(
        f"Mean {f.metric_key.upper()} score across all forecast horizons "
        f"({'vs climatology' if subtract_clim else 'absolute'}) · region: {f.region}"
        f"{_band_note}",
        caption_s,
    ))

    from ui.components import time_serie_chart as _tsc
    ts_fig = _tsc._build_figure(
        all_data, f.metric_key, f.region, climatology_value, unit,
        clim=clim, altitude_band=altitude_band,
    )
    if ts_fig is not None:
        ts_fig.update_layout(
            height=360, width=870,
            margin=dict(l=55, r=20, t=30, b=50),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        png = _fig_to_png(ts_fig, width=870, height=360)
        if png:
            story.append(_png_image(png, content_w, 360 / 870))

    story.append(Spacer(1, 8))

    # ── 2. Spatial distribution ─────────────────────────────────────────────
    story.append(Paragraph(f"2 · Spatial Distribution at {spatial_lt}h Lead Time", section_s))
    story.append(Paragraph(
        f"{f.metric_key.upper()} spatial map "
        f"({'Δ vs climatology' if subtract_clim else 'absolute score'}) "
        f"for each selected model · region: {f.region}{_band_note}",
        caption_s,
    ))

    from ui.components.spatial_map import (_build_heatmap_figure,
                                                        _interpolate_clim,
                                                        _matched_clim_spatial)

    any_map = False
    for m_name, lt_map in all_data.items():
        ltd = lt_map.get(spatial_lt)
        if ltd is None:
            continue
        try:
            grid = annual_spatial_grid(ltd, f.region, altitude_band)
        except Exception:
            continue

        # Use a matched-month climatology (same months ltd covers) so the
        # PDF's spatial diff is the fair comparison the app shows.
        clim_interp: np.ndarray | None = None
        if subtract_clim and clim is not None:
            matched = _matched_clim_spatial(ltd, clim, f.region)
            if matched is not None:
                clim_interp = _interpolate_clim(grid, matched)

        map_fig = _build_heatmap_figure(grid, m_name, f.metric_key, spatial_lt, clim_interp,
                                        unit, f.region)
        if map_fig is not None:
            map_fig.update_layout(
                height=420, width=870,
                margin=dict(l=10, r=10, t=46, b=10),
                paper_bgcolor="white",
            )
            png = _fig_to_png(map_fig, width=870, height=420)
            if png:
                story.append(Paragraph(m_name, body_s))
                story.append(_png_image(png, content_w, 420 / 870))
                any_map = True

        # Statistical tests: model vs climatology (only when subtract_clim).
        # Stack tables vertically at full content width so long hypothesis
        # text never gets clipped and the two cards never overlap.
        if subtract_clim and clim_interp is not None:
            story.append(Paragraph(
                f"Statistical Test · {m_name} vs Climatology · {spatial_lt}h",
                caption_s,
            ))
            g_vals, c_vals = grid.values.ravel(), clim_interp.ravel()
            if zb:   # bias: closer to 0 is better, so test the magnitudes
                g_vals, c_vals = np.abs(g_vals), np.abs(c_vals)
            tbl_t, tbl_w = _test_tables(
                g_vals, c_vals,
                m_name, "Climatology", unit, content_w, colors, spatial_test,
                higher_better=hb,
            )
            if tbl_t is not None:
                story.append(tbl_t)
            if tbl_w is not None:
                if tbl_t is not None:
                    story.append(Spacer(1, 6))
                story.append(tbl_w)
        story.append(Spacer(1, 10))

    if not any_map:
        story.append(Paragraph(f"No spatial data available at {spatial_lt}h.", body_s))

    # ── 3. Seasonal evaluation ──────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph(f"3 · Seasonal Evaluation at {seasonal_lt}h Lead Time", section_s))
    story.append(Paragraph(
        f"Monthly {f.metric_key.upper()} score breakdown "
        f"{'(including climatology baseline)' if clim_monthly is not None else ''} "
        f"· region: {f.region}{_band_note}",
        caption_s,
    ))

    from ui.components import monthly_chart as _mc
    m_fig = _mc._build_figure(
        all_data, f.metric_key, f.region, seasonal_lt,
        clim_monthly, unit=unit, altitude_band=altitude_band,
    )
    if m_fig is not None:
        m_fig.update_layout(
            height=360, width=870,
            margin=dict(l=55, r=20, t=30, b=50),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        png = _fig_to_png(m_fig, width=870, height=360)
        if png:
            story.append(_png_image(png, content_w, 360 / 870))
    else:
        story.append(Paragraph(f"No monthly data at {seasonal_lt}h.", body_s))

    story.append(Spacer(1, 12))

    # ── 4. Models comparison (pairwise diff maps) ───────────────────────────
    # Render every pairwise comparison at the user's chosen comparison
    # lead time. A single static report covers all pairs so the user
    # doesn't have to re-export once per (A, B) combination.
    models_list = list(all_data.keys())
    if len(models_list) >= 2:
        story.append(PageBreak())
        story.append(Paragraph(
            f"4 · Models Comparison at {comparison_lt}h Lead Time",
            section_s,
        ))
        story.append(Paragraph(
            f"Spatial difference maps ({metric_disp} A − B) · "
            f"{'compared by magnitude |bias| — closer to 0 is better' if zb else direction_note(f.metric_key)} · "
            f"green = A better · blue = B better · region: {f.region}",
            caption_s,
        ))

        import itertools

        from ui.components.comparison_tab import \
            _build_diff_figure

        for model_a, model_b in itertools.combinations(models_list, 2):
            ltd_a = all_data[model_a].get(comparison_lt)
            ltd_b = all_data[model_b].get(comparison_lt)
            if ltd_a is None or ltd_b is None:
                continue
            try:
                grid_a = annual_spatial_grid(ltd_a, f.region, altitude_band)
                grid_b = annual_spatial_grid(ltd_b, f.region, altitude_band)
                if zb:
                    grid_a, grid_b = abs(grid_a), abs(grid_b)
                lat_a = "latitude" if "latitude" in grid_a.dims else "lat"
                lon_a = "longitude" if "longitude" in grid_a.dims else "lon"
                lat_b = "latitude" if "latitude" in grid_b.dims else "lat"
                lon_b = "longitude" if "longitude" in grid_b.dims else "lon"
                grid_b_aligned = grid_b.interp(
                    {lat_b: grid_a[lat_a], lon_b: grid_a[lon_a]}, method="linear",
                )
                diff = grid_a.values - grid_b_aligned.values
            except Exception:
                continue

            cmp_fig = _build_diff_figure(
                grid_a, grid_b_aligned, diff,
                model_a, model_b, comparison_lt, metric_disp, unit,
                higher_better=hb, region=f.region,
            )
            if cmp_fig is not None:
                cmp_fig.update_layout(
                    height=420, width=870,
                    margin=dict(l=10, r=10, t=46, b=10),
                    paper_bgcolor="white",
                )
                png = _fig_to_png(cmp_fig, width=870, height=420)
                if png:
                    story.append(_png_image(png, content_w, 420 / 870))

            a_flat = grid_a.values.ravel()
            b_flat = grid_b_aligned.values.ravel()
            desc_tbl = _desc_stats_table(
                diff.ravel(), f"{model_a} − {model_b}", unit,
                content_w, colors,
            )
            tbl_t, tbl_w = _test_tables(
                a_flat, b_flat, model_a, model_b, unit,
                content_w, colors, comparison_test,
                higher_better=hb,
            )
            first = True
            for tbl in (desc_tbl, tbl_t, tbl_w):
                if tbl is None:
                    continue
                if not first:
                    story.append(Spacer(1, 6))
                story.append(tbl)
                first = False
            story.append(Spacer(1, 10))

    # ── 5. Model rankings ───────────────────────────────────────────────────
    story.append(Paragraph("5 · Model Rankings", section_s))

    rank_rows: list[list] = [[
        "Rank", "Model",
        f"Mean {f.metric_key.upper()} (all lead times{_band_note})",
        "vs Climatology",
    ]]
    model_scores: list[tuple] = []
    for m_name, lt_map in all_data.items():
        scores = [
            band_annual_mean(ltd, altitude_band) if altitude_band != "all"
            else regional_annual_mean(ltd, f.region)
            for ltd in lt_map.values() if ltd is not None
        ]
        scores = [s for s in scores if np.isfinite(s)]
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        if climatology_value is not None and climatology_value != 0:
            # Positive pct = model beats climatology, in the metric's own
            # direction (smaller error / larger correlation / smaller |bias|).
            if zb:
                pct = (abs(climatology_value) - abs(mean_score)) / abs(climatology_value) * 100
            elif hb:
                pct = (mean_score - climatology_value) / abs(climatology_value) * 100
            else:
                pct = (climatology_value - mean_score) / abs(climatology_value) * 100
            vs_clim = f"{pct:+.1f}% {'better' if pct > 0 else 'worse'}"
        else:
            vs_clim = "N/A"
        model_scores.append((m_name, mean_score, vs_clim))

    # Rank best-first in the metric's direction: ascending error, descending
    # correlation, ascending |bias|.
    if zb:
        model_scores.sort(key=lambda r: abs(r[1]))
    else:
        model_scores.sort(key=lambda r: r[1], reverse=hb)
    for rank, (m_name, score, vs_clim) in enumerate(model_scores, start=1):
        rank_rows.append([str(rank), m_name, f"{score:.4f}", vs_clim])

    if len(rank_rows) > 1:
        rw = content_w
        rank_table = Table(
            rank_rows,
            colWidths=[rw * 0.08, rw * 0.37, rw * 0.30, rw * 0.25],
            hAlign="LEFT",
        )
        rank_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),  GREEN),
            ("TEXTCOLOR",  (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, OFFWHITE]),
            ("GRID",       (0, 0), (-1, -1), 0.4, LGRAY),
            ("PADDING",    (0, 0), (-1, -1), 7),
            ("ALIGN",      (0, 0), (0, -1),  "CENTER"),
        ]))
        story.append(KeepTogether(rank_table))
    else:
        story.append(Paragraph("No model data available for ranking.", body_s))

    story.append(Spacer(1, 16))

    # ── Footer note ─────────────────────────────────────────────────────────
    _dir = direction_note(f.metric_key)
    _delta = ("positive Δ = model better than climatology" if hb
              else "negative Δ = model better than climatology")
    _raw_note = ""
    if f.metric_key == "crps" and any(not v.endswith("-idr") for v in f.selected_variants):
        _raw_note = (" Raw forecasts are deterministic, so their CRPS equals "
                     "their MAE — raw and calibrated scores are directly comparable.")
    story.append(Paragraph(
        "Data source: AfriClimate AI · Forecast4Africa verification platform. "
        f"All scores are {f.metric_key.upper()} "
        f"({f'{_dir}; {_delta}' if subtract_clim else _dir}). "
        f"Lead times: {LEAD_TIMES[0]}–{LEAD_TIMES[-1]}h.{_raw_note}",
        caption_s,
    ))

    doc.build(story)
    return buf.getvalue()
