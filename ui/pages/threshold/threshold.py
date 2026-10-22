"""The Threshold Diagnostics tab — reliability / ROC / cost-loss curves.

A self-contained page (its own cascading filters) over the curve diagnostics discovered by
:mod:`ui.services.threshold_data`. Kept separate from the scalar `scores_tab` because curve
metrics are a different data shape (threshold × altitude × prob/bin/α), not the scalar
`LeadTimeData`. Like the Short & Medium-Range tab, the user picks **several models** and
compares them at one threshold + altitude (one curve per model).
"""
from __future__ import annotations

import streamlit as st

from ui.app_config.settings import ALTITUDE_UNIT
from ui.app_config.settings import forecast_label, target_label, variable_label
from ui.components import threshold_figures
from ui.services.data_cache import ui_results_root
from ui.services.engine import kind_of
from ui.services.threshold_data import discover_curves, load_curve
from ui.utils.leads import common_leads
from ui.utils.regions import default_region
from verification.seasons import DEFAULT_SEASON_SLUG, season_label

_CURVE_METRICS = ("reliability", "roc", "cost_loss")


def render() -> None:
    """Render the threshold-diagnostics tab: filters on the left, model-comparison curves on the right."""
    root = ui_results_root()
    df = discover_curves(root)
    if df.empty:
        st.info("No threshold diagnostics found for this country. "
                "Run `python -m verification.run` to generate them, or check "
                "`VERIFICATION_COUNTRY` / `VERIFICATION_DATA_ROOT`.")
        return
    df = df.copy()
    df["variable"] = df["native_var"].map(kind_of)

    col_filters, col_charts = st.columns([1, 3.5], gap="large")

    with col_filters:
        year = st.selectbox("Year", sorted(df["year"].unique()), key="t_year")
        d = df[df["year"] == year]
        variable = st.selectbox("Variable", sorted(d["variable"].unique()),
                                format_func=variable_label, key="t_var")
        d = d[d["variable"] == variable]
        truth = st.selectbox("Truth source", sorted(d["target"].unique()),
                             format_func=target_label, key="t_truth")
        d = d[d["target"] == truth]

        # Discovery-driven: offer only regions that have curve files on disk (whole country
        # first, then any admin-1 subregions generated via `verification.run --region`).
        regions_available = sorted(d["region"].unique())
        whole = default_region()
        if whole in regions_available:
            regions_available = [whole] + [r for r in regions_available if r != whole]
        region = st.selectbox("Region", regions_available, key="t_region")
        d = d[d["region"] == region]

        # Discovery-driven: offer only seasons with curve files on disk (Yearly first, then any
        # seasons generated via `verification.run --season`). Slugs are shown via season_label.
        seasons_available = sorted(d["season"].unique())
        if DEFAULT_SEASON_SLUG in seasons_available:
            seasons_available = [DEFAULT_SEASON_SLUG] + [s for s in seasons_available
                                                         if s != DEFAULT_SEASON_SLUG]
        season = st.selectbox("Season", seasons_available, format_func=season_label,
                              key="t_season")
        d = d[d["season"] == season]

        models_available = sorted(d["forecast"].unique())
        models = st.multiselect("Models", models_available, default=models_available,
                                format_func=forecast_label, key="t_models")
        if not models:
            st.info("Select at least one model.")
            return

        sel = d[d["forecast"].isin(models)]
        native_var = sel.iloc[0]["native_var"]
        # Only offer leads every selected model has for every curve metric, so the axes
        # probe below and each sub-tab render can never hit a missing file.
        leads = common_leads(sel["leads"])
        if not leads:
            st.info("No lead time is available for every selected model. "
                    "Select fewer models, or regenerate the missing diagnostics.")
            return
        lead = st.select_slider("Lead time (h)", options=leads,
                                value=leads[len(leads) // 2], key="t_lead")

        axes = load_curve(models[0], native_var, truth, _axes_metric(sel, models[0]),
                          year, lead, root, region=region, season=season)
        thr_options = [int(t) for t in axes["threshold"].values]
        # Units come from the FILE, not a constant: thresholds are per variable
        # (mm/day for precipitation, degC for daily-max temperature) and the writer
        # stamps them on the coord. 20 is the precipitation default and simply is not
        # an option for temperature, so the first threshold is the fallback.
        thr_units = axes["threshold"].attrs.get("units", "")
        default_thr = 20 if 20 in thr_options else thr_options[0]
        threshold = st.selectbox(f"Threshold ({thr_units})" if thr_units else "Threshold",
                                 thr_options,
                                 index=thr_options.index(default_thr), key="t_thr")
        band_options = [str(b) for b in axes["altitude"].values]
        # Region/model switches change which altitude bands exist; a persisted band from a
        # previous selection may no longer be an option. Reset it (prefer 'all' when present)
        # so the widget can never return a band absent from the datasets being drawn.
        if st.session_state.get("t_band") not in band_options:
            st.session_state["t_band"] = "all" if "all" in band_options else band_options[0]
        band = st.selectbox(f"Altitude band ({ALTITUDE_UNIT})", band_options,
                            key="t_band")

    with col_charts:
        # The tabs span the FULL width; the Export-PDF button is floated onto the top-right of
        # the tab row via CSS (absolute-positioned) so it shares the tab line without stealing
        # width from the plot below. A keyed wrapper (`.st-key-…`) anchors the absolute button.
        st.markdown(
            "<style>"
            ".st-key-t_charts_wrap { position: relative; }"
            ".st-key-t_export_wrap { position: absolute; top: 0; right: 0;"
            " width: 165px; z-index: 5; }"
            "</style>",
            unsafe_allow_html=True,
        )
        with st.container(key="t_charts_wrap"):
            with st.container(key="t_export_wrap"):
                _render_pdf_btn(
                    models=models, variable=variable, truth=truth, native_var=native_var,
                    year=year, lead=lead, root=root, sel=sel, region=region, season=season,
                    threshold=threshold, band=band,
                    logx=st.session_state.get("t_logx") == "log",
                )
            tab_rel, tab_roc, tab_cl = st.tabs(["Reliability", "ROC", "Cost-loss"])
        with tab_rel:
            series = _load_series(models, native_var, truth, "reliability", year, lead, root, sel, region, season, band)
            _draw(series, lambda s: threshold_figures.reliability_figure(s, threshold, band, thr_units))
        with tab_roc:
            series = _load_series(models, native_var, truth, "roc", year, lead, root, sel, region, season, band)
            _draw(series, lambda s: threshold_figures.roc_figure(s, threshold, band, thr_units))
        with tab_cl:
            logx = st.radio("Cost-loss ratio axis", ["linear", "log"], horizontal=True,
                            key="t_logx") == "log"
            series = _load_series(models, native_var, truth, "cost_loss", year, lead, root, sel, region, season, band)
            _draw(series, lambda s: threshold_figures.cost_loss_figure(s, threshold, band, logx=logx, units=thr_units))


def _axes_metric(sel, model):
    """A curve metric that exists for ``model`` — used only to read the threshold/altitude axes."""
    present = set(sel[sel["forecast"] == model]["metric"])
    return next(m for m in _CURVE_METRICS if m in present)


def _load_series(models, native_var, truth, metric, year, lead, root, sel, region,
                 season, band, threshold=None):
    """Load ``(label, dataset)`` for each model that has this metric at this lead + region + season + band.

    Skips any model whose curve file lacks the selected altitude band OR the
    selected ``threshold``: models generated with different ``--altitude`` /
    ``--threshold`` sets carry different coordinates, and the drawing layer slices
    with ``ds.sel(threshold=…, altitude=…)``, which raises ``KeyError`` on a value
    the file does not have. Skipping here means a partial selection plots the
    models that do have the value and an empty one yields the "No data" notice,
    instead of a traceback on the page.

    ``threshold=None`` disables the threshold check, for callers that only need
    the axes.
    """
    series = []
    for model in models:
        rows = sel[(sel["forecast"] == model) & (sel["metric"] == metric)]
        if rows.empty or lead not in rows.iloc[0]["leads"]:
            continue
        ds = load_curve(model, native_var, truth, metric, year, lead, root,
                        region=region, season=season)
        if band not in {str(b) for b in ds["altitude"].values}:
            continue
        if (threshold is not None
                and int(threshold) not in {int(t) for t in ds["threshold"].values}):
            continue
        series.append((forecast_label(model), ds))
    return series


def _draw(series, figure_fn):
    """Render a metric's figure, or a notice if no selected model has it."""
    if not series:
        st.info("No data for the selected models at this lead time.")
        return
    st.plotly_chart(figure_fn(series), width="stretch")


@st.fragment
def _render_pdf_btn(*, models, variable, truth, native_var, year, lead, root, sel,
                    region, season, threshold, band, logx) -> None:
    """Export/Download button for a Threshold Diagnostics PDF (this tab's results only).

    Runs as a fragment so Export/Download clicks re-render only this widget, not the curves.
    The report is keyed to the current selection: changing any filter reverts Download → Export
    so the user regenerates against the new configuration. ``truth``/``variable`` are the raw
    tokens (used to load the curves); their display labels are derived for the config table.
    """
    fp = (tuple(models), native_var, truth, str(year), int(lead), region, season,
          int(threshold), band)

    if (st.session_state.get("t_pdf_bytes") is not None
            and st.session_state.get("t_pdf_fp") == fp):
        fname = (f"threshold_{region}_{season}_{native_var}_{year}_{lead}h.pdf"
                 .lower().replace(" ", "_").replace("(", "").replace(")", ""))

        def _reset_pdf():
            st.session_state.pop("t_pdf_bytes", None)
            st.session_state.pop("t_pdf_fp", None)

        st.download_button("⬇️ Download PDF", data=st.session_state["t_pdf_bytes"],
                           file_name=fname, mime="application/pdf", key="t_download_pdf",
                           use_container_width=True, on_click=_reset_pdf)
        return

    if not st.button("📥 Export PDF", key="t_export_pdf", use_container_width=True,
                     help="Export this tab's reliability / ROC / cost-loss curves as a PDF."):
        return

    with st.spinner("Generating PDF report…"):
        try:
            from ui.components.threshold_pdf_report import generate_threshold_report
            series_by_metric = {
                m: _load_series(models, native_var, truth, m, year, lead, root, sel,
                                region, season, band)
                for m in _CURVE_METRICS
            }
            pdf_bytes = generate_threshold_report(
                year=str(year), variable=variable_label(variable), truth=target_label(truth),
                region=region, season=season,
                models_label=", ".join(forecast_label(m) for m in models),
                lead=int(lead), threshold=int(threshold), band=band,
                threshold_units=thr_units,
                series_by_metric=series_by_metric, logx=logx,
            )
            st.session_state["t_pdf_bytes"] = pdf_bytes
            st.session_state["t_pdf_fp"] = fp
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF generation failed: {exc}")
            return
    st.rerun(scope="fragment")
