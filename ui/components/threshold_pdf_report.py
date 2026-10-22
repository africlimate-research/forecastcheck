# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""PDF report for the Threshold Diagnostics tab (reliability / ROC / cost-loss curves).

Sibling of :mod:`ui.components.pdf_report` (which covers the Short & Medium-Range tab). The two
tabs have independent filter states, so each exports its own report. This one renders the exact
curve figures shown on screen — built by :mod:`ui.components.threshold_figures` — to PNG via the
shared kaleido helper and assembles them with reportlab. Callers pass the already-loaded
``series`` per metric (``[(model_label, dataset), …]``) so this module doesn't import the page
layer (avoids a circular import) and never re-reads NetCDF the tab already loaded.
"""
from __future__ import annotations

import io
from datetime import datetime

from ui.app_config.settings import band_label
from ui.components import threshold_figures
from ui.components.pdf_report import _fig_to_png, _png_image
from verification.seasons import season_label

# Section order + human titles for the three curve metrics.
_METRICS: tuple[tuple[str, str], ...] = (
    ("reliability", "Reliability Diagram"),
    ("roc",         "ROC Curve"),
    ("cost_loss",   "Cost-Loss Value"),
)


def _metric_figure(metric, series, threshold, band, logx):
    """Build the on-screen Plotly figure for one curve metric."""
    if metric == "reliability":
        return threshold_figures.reliability_figure(series, threshold, band)
    if metric == "roc":
        return threshold_figures.roc_figure(series, threshold, band)
    return threshold_figures.cost_loss_figure(series, threshold, band, logx=logx)


def generate_threshold_report(
    *,
    year:            str,
    variable:        str,
    truth:           str,
    region:          str,
    season:          str,
    models_label:    str,
    lead:            int,
    threshold:       int,
    band:            str,
    #: Units of ``threshold``, read off the curve file's own coord (mm/day for
    #: precipitation, degC for daily-max temperature). Defaulted rather than required
    #: so an older caller still builds a report, just without the unit.
    threshold_units: str = "",
    series_by_metric: dict[str, list],
    logx:            bool = False,
) -> bytes:
    """Build the Threshold Diagnostics PDF and return its bytes.

    ``series_by_metric`` maps each curve metric ("reliability"/"roc"/"cost_loss") to the
    ``[(model_label, dataset), …]`` list the tab already loaded for the current selection. Each
    metric's figure is rendered exactly as shown (same threshold + altitude band) and embedded;
    a metric with no series gets a "no data" caption instead of a blank page.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    page_size = landscape(A4)
    W, _H = page_size
    margin = 1.5 * cm
    doc = SimpleDocTemplate(
        buf, pagesize=page_size,
        rightMargin=margin, leftMargin=margin, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    content_w = W - 2 * margin

    styles = getSampleStyleSheet()
    GREEN = colors.HexColor("#0a9938")
    DARK  = colors.HexColor("#171d1a")
    GRAY  = colors.HexColor("#6B7280")
    LGRAY = colors.HexColor("#E5E7EB")
    OFFWHITE  = colors.HexColor("#F9FAFB")
    HEADER_BG = colors.HexColor("#F3F4F6")

    def _style(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_s   = _style("T", fontSize=20, textColor=GREEN, fontName="Helvetica-Bold", spaceAfter=10)
    sub_s     = _style("S", fontSize=9,  textColor=GRAY,  fontName="Helvetica", spaceAfter=8)
    section_s = _style("H", fontSize=12, textColor=DARK,  fontName="Helvetica-Bold",
                       spaceBefore=12, spaceAfter=5)
    caption_s = _style("C", fontSize=8,  textColor=GRAY,  fontName="Helvetica-Oblique", spaceAfter=4)

    story: list = []
    story.append(Paragraph("Threshold Diagnostics Report", title_s))
    story.append(Paragraph(
        f"AfriClimate AI · Forecast4Africa · "
        f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        sub_s,
    ))

    # ── Configuration table ─────────────────────────────────────────────────
    story.append(Paragraph("Configuration", section_s))
    cfg_rows = [
        ["Region",         region,                    "Variable",       variable],
        ["Truth Source",   truth,                     "Year",           str(year)],
        ["Season",         season_label(season),      "Lead Time",      f"{lead}h"],
        ["Threshold",      f"{int(threshold)} {threshold_units}".strip(),
         "Altitude band", band_label(band)],
        ["Models",         models_label or "—",       "",               ""],
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

    # ── One section per curve metric ────────────────────────────────────────
    for i, (metric, human) in enumerate(_METRICS, start=1):
        series = series_by_metric.get(metric) or []
        story.append(Paragraph(f"{i} · {human}", section_s))
        story.append(Paragraph(
            f"{human} at {int(threshold)} {threshold_units}".strip()
            + f" · altitude: {band_label(band)} · {lead}h lead · "
            f"region: {region} · season: {season_label(season)}",
            caption_s,
        ))
        if not series:
            story.append(Paragraph("No data for the selected models at this configuration.", caption_s))
            story.append(Spacer(1, 8))
            continue
        fig = _metric_figure(metric, series, threshold, band, logx)
        fig.update_layout(height=420, width=870, margin=dict(l=40, r=20, t=40, b=45),
                          paper_bgcolor="white", plot_bgcolor="white")
        png = _fig_to_png(fig, width=870, height=420)
        if png:
            story.append(_png_image(png, content_w, 420 / 870))
        else:
            story.append(Paragraph(
                "Figure could not be rendered (image export unavailable — is 'kaleido' installed?).",
                caption_s,
            ))
        story.append(Spacer(1, 8))

    doc.build(story)
    return buf.getvalue()
