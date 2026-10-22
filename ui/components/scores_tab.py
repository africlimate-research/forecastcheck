# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import threading

import numpy as np
import streamlit as st
import xarray as xr
from ui.app_config.settings import (CORRELATION_METRICS, LEAD_TIMES, REGIONS,
                                                 VARIABLE_UNITS, metric_label,
                                                 season_months, target_label,
                                                 variable_label)
from ui.backend.data_loader.climatology_loader import (
    ClimatologyLoader, ClimData, climatology_available)
from ui.components import (comparison_tab, insight_card,
                                        monthly_chart, spatial_map,
                                        time_serie_chart)
from ui.components.filters import (FilterState,
                                                variant_display_name)
from ui.services.data_cache import get_menu_index, ui_results_root
from ui.services.loader import (LeadTimeData, load_daily_series,
                                                load_run, load_season_spatial)
from ui.utils.geo import (annual_spatial_grid, ensure_region_warm,
                                       is_region_warm, regional_annual_mean,
                                       regional_monthly_means, subset_region)
from ui.utils.regions import prewarm_masks

_clim_loader = ClimatologyLoader()
_DATA_CACHE: dict[tuple, dict[int, LeadTimeData]] = {}
_CLIM_CACHE: dict[tuple, ClimData | None]         = {}

_PRECACHE_STARTED = False
_PRECACHE_LOCK    = threading.Lock()


def _resolve_run(forecast: str, target: str, metric: str, year: str) -> tuple[str | None, str]:
    """``(native_var, storage_metric)`` for a run, from the discovery resolver.

    ``storage_metric`` is the metric segment actually read on disk — it can
    differ from the UI metric (raw model under CRPS reads its mae files; see
    :meth:`MenuIndex.resolve_storage` for the policy). ``native_var`` is None
    when the selection has no loadable run.
    """
    r = get_menu_index().resolve_storage(forecast=forecast, target=target,
                                         metric=metric, year=year)
    return (r["native_var"], r["storage_metric"]) if r else (None, metric)


def _load(
    forecast:   str,
    variable:   str,
    truth:      str,
    metric:     str,
    year:       str,
    region:     str,
    lead_times: tuple[int, ...],
    native_var: str | None = None,
) -> dict[int, LeadTimeData]:
    nv, storage_metric = ((native_var, metric) if native_var
                          else _resolve_run(forecast, truth, metric, year))
    # region excluded from key: results are full-domain, subsetting is display-time only.
    # native_var excluded too: it's a pure function of (forecast, truth, metric, year).
    # Keyed by the STORAGE metric, so a raw model viewed under CRPS reuses the
    # entry the precache worker filled from its mae files — same file, one copy.
    key = (forecast, variable, truth, storage_metric, year, lead_times)
    if key not in _DATA_CACHE:
        # No native_var means the selection has no run on disk (e.g. a metric
        # picked that this forecast lacks) — a normal empty result, not an error.
        _DATA_CACHE[key] = (
            load_run(forecast, nv, truth, storage_metric, year, list(lead_times),
                     root=ui_results_root())
            if nv else {}
        )
    return _DATA_CACHE[key]


def _maybe_free_monthly(ltd) -> None:
    """Free the big gridded monthly_spatial_average (24 MB) to save RAM after
    its regional means are cached — but keep it for stations, where it's tiny
    and the station-detail month curve reads it per station."""
    msa = ltd.monthly_spatial_average
    if msa is not None and "station" not in msa.dims:
        ltd.monthly_spatial_average = None


def _precache_worker(table) -> None:
    """Pre-warm _DATA_CACHE and _CLIM_CACHE for every discovered run.

    ``table`` is the menu index built once on the main thread and passed in,
    so the precache and the sidebar share exactly one scan of one root — no
    divergent discovery between what's offered and what's warmed.
    """
    from concurrent.futures import ThreadPoolExecutor

    lt_tuple = tuple(LEAD_TIMES)

    # ── 1. Climatology — one file per (target, year, metric) present ──────────
    clim_combos: list[tuple] = []
    seen_clim: set[tuple] = set()
    for _, row in table.iterrows():
        ck = (row["target"], row["year"], row["metric"])
        if ck not in _CLIM_CACHE and ck not in seen_clim:
            seen_clim.add(ck)
            clim_combos.append(ck)

    def _fetch_clim(ck: tuple) -> None:
        truth, year, metric_key = ck
        try:
            _CLIM_CACHE[ck] = _clim_loader.load(truth, year, metric_key, root=ui_results_root())
        except Exception as exc:
            print(f"[precache] clim {truth}/{year}/{metric_key}: {exc}")

    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=8, thread_name_prefix="clim-pw") as ex:
        list(ex.map(_fetch_clim, clim_combos))
    print(f"[precache] climatology done — {len(clim_combos)} files loaded")

    # ── 2. Model data — one combo per discovered run ──────────────────────────
    combos: list[tuple] = []
    for _, row in table.iterrows():
        model, variable, truth = row["forecast"], row["variable"], row["target"]
        metric_key, year, native = row["metric"], row["year"], row["native_var"]
        key = (model, variable, truth, metric_key, year, lt_tuple)
        if key not in _DATA_CACHE:
            combos.append((model, variable, truth, metric_key, year, native, key))

    def _fetch(args: tuple) -> None:
        """Load one model combo, compute all regional display values inline, then
        free the 24 MB monthly_spatial_average DataArray.

        Peak memory per worker: 19 lead-times × 26 MB ≈ 500 MB.
        Retained after this call: 19 × 2 MB (annual_spatial_average only) ≈ 38 MB.
        Total retained across all combos: ~8 GB — well within the instance limit.
        """
        model, variable, truth, metric_key, year, native, key = args
        try:
            if key not in _DATA_CACHE:
                ltd_map = load_run(model, native, truth, metric_key, year,
                                   list(lt_tuple), root=ui_results_root())
                # Compute regional means BEFORE freeing monthly_spatial_average.
                for ltd in ltd_map.values():
                    if ltd is None:
                        continue
                    for region in REGIONS:
                        regional_annual_mean(ltd, region)
                        regional_monthly_means(ltd, region)
                    _maybe_free_monthly(ltd)
                _DATA_CACHE[key] = ltd_map
            else:
                # Key already filled by on-demand _load() — monthly_spatial_average
                # is still in memory so regional means can be computed normally.
                for ltd in _DATA_CACHE[key].values():
                    if ltd is None:
                        continue
                    for region in REGIONS:
                        regional_annual_mean(ltd, region)
                        regional_monthly_means(ltd, region)
                    _maybe_free_monthly(ltd)
        except Exception as exc:
            print(f"[precache] {model}/{variable}/{truth}/{metric_key}/{year}: {exc}")

    # 4 outer workers × 8 inner lead-time threads = 32 concurrent GCS reads max.
    # Regional means are computed inline in each worker so no separate pass is needed.
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="precache") as ex:
        list(ex.map(_fetch, combos))

    # Prewarm regionmask boolean masks so on-demand spatial map rendering is fast.
    # (Spatial grids themselves are not pre-cached — they are built lazily per
    # lead-time when the user opens the spatial map tab.)
    sample = next(
        (
            ltd.annual_spatial_average
            for lt_map in _DATA_CACHE.values()
            for ltd in lt_map.values()
            if ltd is not None
        ),
        None,
    )
    if sample is not None:
        print("[precache] warming region masks…")
        prewarm_masks(sample)

    print(f"[precache] complete — {len(combos)} combinations, "
          f"{len(_DATA_CACHE)} cached, ~{len(_DATA_CACHE) * 2 // 1024} GB retained")


def _start_precaching() -> None:
    global _PRECACHE_STARTED
    if _PRECACHE_STARTED:
        return
    with _PRECACHE_LOCK:
        if _PRECACHE_STARTED:
            return
        _PRECACHE_STARTED = True
    # Build the index here (main thread, cache hit) and hand it to the worker
    # so discovery happens once, on the same root the sidebar/loader use.
    table = get_menu_index().table
    t = threading.Thread(target=_precache_worker, args=(table,),
                         daemon=True, name="precache-worker")
    t.start()
    print("[precache] background thread started")


def _load_clim(
    truth:  str,
    year:   str,
    metric: str,
) -> ClimData | None:
    key = (truth, year, metric)
    if key not in _CLIM_CACHE:
        _CLIM_CACHE[key] = _clim_loader.load(truth, year, metric, root=ui_results_root())
    return _CLIM_CACHE[key]


def _pad12(arr: np.ndarray) -> np.ndarray:
    """Pad a monthly array to exactly 12 entries with NaN."""
    n = min(int(arr.shape[0]), 12)
    out = np.full(12, np.nan)
    out[:n] = arr[:n]
    return out


def _multi_year_monthly(
    years:    list[str],
    f:        "FilterState",
    all_data: dict[str, dict[int, LeadTimeData]],
) -> tuple:
    """Return (result, no_data, incomplete).

    `result` is a dict {display_name: {lt: (12,) nanmean}}; it is always a
    dict (possibly empty) when `years` is provided, so callers can tell
    "no overrides requested" (None) apart from "overrides requested but
    no model has data for the selected years" ({}). The seasonal chart
    uses that distinction to skip year-mixing fallbacks.
    """
    if not years:
        return None, [], {}

    _MNAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    result:     dict[str, dict[int, np.ndarray]] = {}
    no_data:    list[str] = []
    incomplete: dict[str, dict[str, list[str]]] = {}  # model → {year → [missing month names]}

    for forecast in f.selected_variants:
        label = variant_display_name(forecast)
        if label not in all_data:
            no_data.append(label)
            continue
        lt_results: dict[int, np.ndarray] = {}
        # Track missing months per year — only from lead times that actually have data
        yr_missing: dict[str, list[str]] = {}
        for lt in LEAD_TIMES:
            stacked = []
            for yr in years:
                yr_data = _load(
                    forecast, f.variable, f.truth, f.metric_key,
                    yr, f.region, tuple(LEAD_TIMES),
                )
                if yr_data and lt in yr_data:
                    arr = _pad12(regional_monthly_means(yr_data[lt], f.region))
                    # Only record missing months from real data (NaN = beyond n_actual)
                    if yr not in yr_missing:
                        nan_months = [_MNAMES[i] for i in range(12) if np.isnan(arr[i])]
                        if nan_months:
                            yr_missing[yr] = nan_months
                else:
                    arr = np.full(12, np.nan)
                stacked.append(arr)
            arr_stack = np.array(stacked)
            mean_arr = (np.full(12, np.nan) if np.all(np.isnan(arr_stack))
                        else np.nanmean(arr_stack, axis=0))
            if not np.all(np.isnan(mean_arr)):
                lt_results[lt] = mean_arr
        if yr_missing:
            incomplete[label] = yr_missing
        if lt_results:
            result[label] = lt_results
        else:
            no_data.append(label)

    # Return the dict even when empty — see docstring; the chart needs
    # the empty case to distinguish "no override mode" from "override
    # mode with zero qualifying models".
    return result, no_data, incomplete


def _multi_year_clim_monthly(
    years:  list[str],
    truth:  str,
    metric: str,
    region: str,
) -> np.ndarray | None:
    """Return (12,) nanmean climatology across years, or None if data unavailable."""
    stacked = []
    for yr in years:
        clim = _load_clim(truth, yr, metric)
        if clim is not None:
            try:
                arr = regional_monthly_means(clim, region)
            except Exception:
                arr = np.asarray(clim.monthly_average)
            stacked.append(_pad12(arr))
    return np.nanmean(stacked, axis=0) if stacked else None


# ═══════════════════════════════════════════════════════════════════════════
#  Public — benchmark summary (home page leaderboard)
# ═══════════════════════════════════════════════════════════════════════════

def get_benchmark_summary(
    variable: str,
    truth:    str,
    metric:   str,
    year:     str,
) -> list[dict] | None:
    """Return model rankings vs climatology from in-memory cache (no GCS calls).

    Returns None if the cache is cold.  Each dict has:
      label, score, clim_score, pct (positive = better than clim), is_better.
    """
    lt_tuple = tuple(LEAD_TIMES)
    # Climatology for CRPS and MAE both use the CRPS file
    clim_metric = "crps" if metric in ("mae", "crps") else metric
    clim = _CLIM_CACHE.get((truth, year, clim_metric))
    if clim is None:
        return None
    clim_score = float(clim.annual_average)
    if clim_score == 0:
        return None

    rows: list[dict] = []
    for model in get_menu_index().values("forecast", variable=variable,
                                         target=truth, metric=metric, year=year):
        key = (model, variable, truth, metric, year, lt_tuple)
        lt_map = _DATA_CACHE.get(key)
        if not lt_map:
            continue
        scores = [float(ltd.annual_average) for ltd in lt_map.values() if ltd is not None]
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        pct = (clim_score - mean_score) / abs(clim_score) * 100
        rows.append({
            "label":      variant_display_name(model),
            "score":      mean_score,
            "clim_score": clim_score,
            "pct":        pct,
            "is_better":  pct > 0,
        })

    if not rows:
        return None
    rows.sort(key=lambda r: r["score"])  # lower score = better for CRPS/MAE
    return rows


# ═══════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def show(f: FilterState) -> None:
    _start_precaching()
    col_toolbar, col_pdf = st.columns([6.5, 1], vertical_alignment="center")
    with col_toolbar:
        subtract_clim = _render_toolbar(f)
    with col_pdf:
        _render_pdf_btn(f, subtract_clim, f.altitude_band)
    st.divider()
    _charts_fragment(f, subtract_clim, f.altitude_band)


# ═══════════════════════════════════════════════════════════════════════════
#  Charts fragment
# ═══════════════════════════════════════════════════════════════════════════

def _charts_fragment(f: FilterState, subtract_clim: bool, altitude_band: str = "all") -> None:
    all_data: dict[str, dict[int, LeadTimeData]] = {}

    failed: list[str] = []
    for forecast in f.selected_variants:
        label = variant_display_name(forecast)
        with st.spinner(f"Loading {label}…"):
            try:
                result = _load(
                    forecast, f.variable, f.truth, f.metric_key,
                    f.year, f.region, tuple(LEAD_TIMES),
                )
                if result:
                    all_data[label] = result
                else:
                    failed.append(label)
            except Exception as exc:
                failed.append(label)
                print(f"[scores_tab] {label}: {exc}")

    if failed:
        st.warning(f"No data found for: {', '.join(failed)}")

    if not all_data:
        st.info("No data available for the current configuration.")
        return

    # all_data is keyed by DISPLAY LABEL; on-demand loaders (daily series, monthly map)
    # need the real forecast TOKEN to resolve paths — map back here.
    token_of = {variant_display_name(t): t for t in f.selected_variants}

    # Load climatology once (cached); only used when toggle is on
    clim: ClimData | None = None
    clim_spatial: xr.DataArray | None = None
    climatology_value: float | None = None
    clim_monthly: np.ndarray | None = None

    if subtract_clim:
        # The toggle is only shown when a baseline exists (see _render_toolbar),
        # so clim is normally present here; guard defensively anyway.
        with st.spinner("Loading climatology baseline…"):
            clim = _load_clim(f.truth, f.year, f.metric_key)
        if clim is not None:
            clim_spatial = subset_region(clim.annual_spatial_average, f.region)
            try:
                climatology_value = float(clim_spatial.mean().values)
            except Exception:
                climatology_value = clim.annual_average
            try:
                clim_monthly = regional_monthly_means(clim, f.region)
            except Exception:
                clim_monthly = clim.monthly_average

    # Ensure regional grids/means are cached before rendering any chart.
    sample_ltd = next(
        (ltd for lt_map in all_data.values() for ltd in lt_map.values() if ltd is not None),
        None,
    )
    if sample_ltd is not None and not is_region_warm(sample_ltd, f.region):
        # Host the progress bar inside an st.empty() slot we control,
        # then wipe the slot from a finally block. `_prog.empty()` alone
        # can leave a hollow container behind in some Streamlit builds —
        # the empty slot itself never renders DOM, so this guarantees a
        # clean removal even on early exceptions.
        prog_slot = st.empty()
        try:
            _prog = prog_slot.progress(0.0, text=f"Preparing **{f.region}** data…")
            annual_spatial_grid(sample_ltd, f.region)  # warms _MASK_CACHE once

            def _on_progress(done: int, total: int) -> None:
                pct = done / total if total else 1.0
                _prog.progress(pct, text=f"Preparing **{f.region}** data — {int(pct * 100)} %")

            ensure_region_warm(all_data, f.region, on_progress=_on_progress)
        finally:
            prog_slot.empty()

    unit = VARIABLE_UNITS.get(f.variable, "")

    # Season aggregate window (None = Yearly) — drives the Lead-Time, Spatial,
    # and Comparison sub-tabs. Seasonal Evaluation keeps its own month range.
    months = season_months(f.season)
    # Correlation metrics show a filtered daily series on the Lead-Time tab (not a mean), so
    # the "mean of monthly means" wording only fits the scalar metrics.
    season_caption = None
    if months:
        season_caption = (
            f"Showing **{f.season}** only."
            if f.metric_key.lower() in CORRELATION_METRICS
            else f"Showing the **{f.season}** mean (unweighted mean of the season's monthly means)."
        )

    # Kick off background prewarm for every selected model's spatial
    # figures. This is fire-and-forget — a bounded ThreadPoolExecutor in
    # a daemon thread, no blocking. By the time the user opens the
    # Spatial tab (or clicks a different model / lead time inside it),
    # most figures are cache hits and render instantly. Internally
    # deduped via _PREWARM_STARTED so re-renders don't double-spawn.
    try:
        from ui.components.spatial_map import prewarm_all_figures
        _prewarm_entries = [
            (label, lt_map, f.metric_key, clim, unit)
            for label, lt_map in all_data.items() if lt_map
        ]
        prewarm_all_figures(_prewarm_entries, regions=[f.region])
    except Exception as _exc:
        print(f"[scores_tab] spatial prewarm dispatch failed: {_exc}")

    # Expose the loaded data to the PDF-button fragment so it can generate
    # the report without re-loading anything — keeps the click cheap and
    # fragment-scoped.
    st.session_state["_pdf_all_data"] = all_data
    st.session_state["_pdf_clim"] = clim

    tab_lt, tab_spatial, tab_seasonal, tab_compare = st.tabs([
        "📈 Lead-Time Performance",
        "🗺️ Spatial Distribution",
        "📅 Seasonal Evaluation",
        "⚖️ Models Comparison",
    ])

    band_note = f" · altitude {altitude_band} m" if altitude_band != "all" else ""

    with tab_lt:
        clim_badge = (
            "<span style='background:rgba(33,59,125,0.09);color:#213b7d;"
            "border:1px solid rgba(33,59,125,0.25);border-radius:99px;"
            "padding:2px 9px;font-family:Inter,sans-serif;font-size:0.62rem;"
            "font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'>"
            "vs Climatology</span>"
        ) if (subtract_clim and climatology_value is not None) else ""

        st.markdown(f"""
            <div style='border-left:4px solid #0a9938;padding:10px 16px 10px 14px;
                background:#ffffff;border-radius:0 6px 6px 0;margin-bottom:12px;
                box-shadow:0 1px 6px rgba(0,0,0,0.05);'>
              <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>
                <span style='font-size:19px;'>📈</span>
                <span style='font-family:Inter,sans-serif;font-size:15px;
                    font-weight:700;color:#171d1a;'>Lead-Time Performance</span>
                <span style='background:rgba(10,153,56,0.09);color:#2e7d33;
                    border:1px solid rgba(10,153,56,0.25);border-radius:99px;
                    padding:2px 9px;font-family:Inter,sans-serif;font-size:0.62rem;
                    font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'>
                    {metric_label(f.metric_key)}</span>
                {clim_badge}
              </div>
              <div style='font-family:Inter,sans-serif;font-size:11px;color:#6B7280;
                  margin-top:3px;padding-left:27px;'>
                  Metric error across all forecast horizons{band_note}</div>
            </div>
        """, unsafe_allow_html=True)
        if season_caption:
            st.caption(season_caption)

        def _load_daily(label, lead):
            """Reload the daily metric series (correlation view of the lead-time tab)."""
            token = token_of.get(label, label)
            nv, storage_metric = _resolve_run(token, f.truth, f.metric_key, f.year)
            if not nv:
                return None
            return load_daily_series(token, nv, f.truth, storage_metric, f.year,
                                     lead, root=ui_results_root())

        time_serie_chart.render(
            model_results=all_data,
            metric_key=f.metric_key,
            region=f.region,
            climatology_value=climatology_value,
            unit=unit,
            clim=clim,
            load_daily=_load_daily,
            altitude_band=altitude_band,
            season_months=months,
        )

        # Raw forecasts shown under CRPS are scored from their mae files —
        # legitimately: deterministic CRPS equals MAE. Say so, so nobody
        # suspects a labeling mix-up.
        if f.metric_key == "crps" and any(
            not token_of.get(lbl, lbl).endswith("-idr") for lbl in all_data
        ):
            st.caption(
                "ℹ️ Raw forecasts are deterministic, so their CRPS equals their "
                "MAE — raw and calibrated (IDR) scores are directly comparable."
            )

        if climatology_value is not None:
            insight_card.render(
                model_results=all_data,
                metric_key=f.metric_key,
                climatology_value=climatology_value,
                region=f.region,
                clim=clim,
            )

    def _load_season(label, lead):
        """Reload the season-mean spatial field for one model + lead (on demand)."""
        if not months:
            return None
        token = token_of.get(label, label)
        nv, storage_metric = _resolve_run(token, f.truth, f.metric_key, f.year)
        if not nv:
            return None
        return load_season_spatial(token, nv, f.truth, storage_metric, f.year,
                                   lead, months, root=ui_results_root())

    with tab_spatial:
        if season_caption:
            st.caption(season_caption)
        spatial_map.render(
            model_results=all_data,
            metric_key=f.metric_key,
            region=f.region,
            subtract_clim=subtract_clim and clim is not None,
            clim=clim,
            unit=unit,
            altitude_band=altitude_band,
            load_season=_load_season,
            season_months=months,
        )

    with tab_seasonal:
        # Header first so the "Average across years" filter visually sits
        # under it (matches the rest of the page where filters follow the
        # section header instead of preceding it).
        monthly_chart.render_header(f.metric_key, clim_monthly)

        # When the sidebar year changes, reset the multiselect to that
        # year. Otherwise stale selections from a previous year carry
        # over and the chart silently mixes calendar months across years
        # (e.g. Jan–Mar pulled from 2025 while Apr–Dec come from 2024,
        # because 2024's IFS+IDR file only covers Apr-Dec).
        if altitude_band != "all":
            # Band view: the multi-year blend aggregates regional means, not
            # band means — offer the sidebar year only rather than mixing.
            st.caption(f"Altitude {altitude_band} m · sidebar year ({f.year}) only")
            selected_years = [f.year]
            monthly_overrides, seasonal_no_data, seasonal_incomplete = None, [], None
        else:
            if st.session_state.get("_seasonal_prev_year") != f.year:
                st.session_state["seasonal_years"] = [f.year]
                st.session_state["_seasonal_prev_year"] = f.year
            if "seasonal_years" not in st.session_state:
                st.session_state["seasonal_years"] = [f.year]
            selected_years = st.multiselect(
                "Average across years",
                options=get_menu_index().values("year"),
                key="seasonal_years",
                help="Defaults to the sidebar year; add more to blend a multi-year mean.",
            )
            if not selected_years:
                selected_years = [f.year]

            monthly_overrides, seasonal_no_data, seasonal_incomplete = _multi_year_monthly(
                selected_years, f, all_data
            )
        if seasonal_no_data:
            st.warning(
                "No data available for: "
                + ", ".join(f"**{m}**" for m in seasonal_no_data)
                + f" — year(s): {', '.join(selected_years)}"
            )
        seasonal_clim = clim_monthly
        if subtract_clim and len(selected_years) > 1:
            seasonal_clim = _multi_year_clim_monthly(selected_years, f.truth, f.metric_key, f.region)

        monthly_chart.render(
            model_results=all_data,
            metric_key=f.metric_key,
            region=f.region,
            clim_monthly=seasonal_clim,
            model_monthly_overrides=monthly_overrides,
            unit=unit,
            incomplete_info=seasonal_incomplete if len(selected_years) > 1 else None,
            include_header=False,
            altitude_band=altitude_band,
        )

    with tab_compare:
        if season_caption:
            st.caption(season_caption)
        comparison_tab.render(
            model_results=all_data,
            metric_key=f.metric_key,
            region=f.region,
            unit=unit,
            altitude_band=altitude_band,
            load_season=_load_season,
            season_months=months,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Toolbar — config summary + climatology toggle only
# ═══════════════════════════════════════════════════════════════════════════

def _render_toolbar(f: FilterState) -> bool:
    """Config pills + the climatology toggle.

    The altitude band comes from the sidebar (``f.altitude_band``); a selected
    band forces the climatology comparison off — the climatology baseline
    exists only as an all-altitudes value, and mixing it with a band-filtered
    model score would be a dishonest comparison.
    """
    col_info, col_clim = st.columns([5, 1], vertical_alignment="center")

    with col_info:
        def _pill(text: str, bg: str, color: str, border: str) -> str:
            return (
                f"<span style='background:{bg};color:{color};"
                f"border:1px solid {border};border-radius:99px;"
                f"padding:3px 11px;font-family:Inter,sans-serif;"
                f"font-size:0.68rem;font-weight:700;white-space:nowrap;'>{text}</span>"
            )

        metric_pill = _pill(
            f"📊 {metric_label(f.metric_key)}",
            "rgba(10,153,56,0.09)", "#2e7d33", "rgba(10,153,56,0.3)",
        )
        var_pill = _pill(
            f"🌧️ {variable_label(f.variable)}",
            "#F3F4F6", "#374151", "#E5E7EB",
        )
        truth_pill = _pill(
            f"📡 {target_label(f.truth)}",
            "#F3F4F6", "#374151", "#E5E7EB",
        )
        region_pill = _pill(
            f"🌍 {f.region}",
            "#F3F4F6", "#374151", "#E5E7EB",
        )
        year_pill = _pill(
            f"🗓️ {f.year}",
            "#F3F4F6", "#374151", "#E5E7EB",
        )
        band_pill = _pill(
            f"🏔️ {f.altitude_band} m",
            "rgba(242,176,41,0.12)", "#92580a", "rgba(242,176,41,0.35)",
        ) if f.altitude_band != "all" else ""

        variants_html = "".join(
            _pill(variant_display_name(fc),
                  "rgba(10,153,56,0.09)", "#2e7d33", "rgba(10,153,56,0.25)")
            for fc in f.selected_variants
        ) if f.selected_variants else "<span style='color:#9CA3AF;font-size:0.72rem;'>—</span>"

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap;"
            f"padding:6px 0;'>"
            f"{metric_pill}{var_pill}{truth_pill}{region_pill}{year_pill}{band_pill}"
            f"<span style='color:#D1D5DB;padding:0 2px;'>|</span>"
            f"{variants_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_clim:
        # Offer the toggle only when a climatology baseline exists for this
        # selection — no toggle that can't do anything (it lights up
        # automatically once climatology is produced on disk). A selected
        # altitude band replaces it with a note: the baseline has no bands.
        if f.altitude_band != "all":
            subtract_clim = False
            st.caption("Climatology baseline is all-altitudes only")
        elif climatology_available(f.truth, f.year, f.metric_key, root=ui_results_root()):
            subtract_clim = st.toggle(
                "Compare vs Climatology",
                value=False,
                key="clim_toggle",
                help="Subtract a climatology baseline to show genuine model skill.",
            )
        else:
            subtract_clim = False

    return subtract_clim


def _pdf_lead_times(default_lt: int) -> tuple[int, int, int]:
    """Resolve the (spatial, seasonal, comparison) lead times.

    Each tab has its own slider; if a slider hasn't been touched yet, fall
    back to the spatial-tab value so the PDF still renders something
    sensible.
    """
    def _parse(s: str | None) -> int | None:
        if not s:
            return None
        try:
            return int(s.rstrip("h"))
        except ValueError:
            return None
    seasonal_lt   = _parse(st.session_state.get("monthly_lt_slider")) or default_lt
    comparison_lt = _parse(st.session_state.get("cmp_lt")) or default_lt
    return default_lt, seasonal_lt, comparison_lt


def _pdf_fingerprint(f: FilterState, subtract_clim: bool, altitude_band: str = "all") -> tuple:
    """Build a fingerprint of every config knob that affects PDF content.

    If any of these change between PDF generation and the next render, the
    button reverts from "Download" to "Export" so the user regenerates with
    the updated configuration.
    """
    spatial_lt_idx = st.session_state.get("sm_lt_idx", len(LEAD_TIMES) // 2)
    spatial_lt = LEAD_TIMES[min(int(spatial_lt_idx), len(LEAD_TIMES) - 1)]
    spatial_lt, seasonal_lt, comparison_lt = _pdf_lead_times(spatial_lt)
    # cmp_model_a / cmp_model_b are intentionally NOT in the fingerprint:
    # the PDF renders every pairwise (A, B) combination, so changing the
    # selected pair in the Comparison tab doesn't invalidate the report.
    return (
        f,
        subtract_clim,
        altitude_band,
        spatial_lt,
        seasonal_lt,
        comparison_lt,
        st.session_state.get("sm_test_choice", "T-Test"),
        st.session_state.get("cmp_test_choice", "T-Test"),
    )


@st.fragment
def _render_pdf_btn(f: FilterState, subtract_clim: bool, altitude_band: str = "all") -> None:
    """Self-contained PDF button — runs as a fragment so Export/Download
    clicks only re-render this widget, never the surrounding charts or tabs.
    """
    spatial_lt_idx = st.session_state.get("sm_lt_idx", len(LEAD_TIMES) // 2)
    spatial_lt = LEAD_TIMES[min(int(spatial_lt_idx), len(LEAD_TIMES) - 1)]
    spatial_lt, seasonal_lt, comparison_lt = _pdf_lead_times(spatial_lt)
    fp = _pdf_fingerprint(f, subtract_clim, altitude_band)

    if (
        st.session_state.get("pdf_report_bytes") is not None
        and st.session_state.get("pdf_report_fp") == fp
    ):
        fname = (
            f"verification_{f.region}_{f.variable}_{f.year}_{spatial_lt}h.pdf"
            .lower().replace(" ", "_").replace("(", "").replace(")", "")
        )

        def _reset_pdf():
            st.session_state.pop("pdf_report_bytes", None)
            st.session_state.pop("pdf_report_fp", None)

        st.download_button(
            "⬇️ Download PDF",
            data=st.session_state["pdf_report_bytes"],
            file_name=fname,
            mime="application/pdf",
            key="download_pdf_btn",
            use_container_width=True,
            on_click=_reset_pdf,
        )
        return

    if not st.button(
        "📥 Export PDF",
        key="export_pdf_btn",
        use_container_width=True,
        help=(
            f"Export PDF for {f.region} · {f.variable} · {f.year} · "
            f"spatial {spatial_lt}h, seasonal {seasonal_lt}h, "
            f"comparison {comparison_lt}h"
        ),
    ):
        return

    all_data = st.session_state.get("_pdf_all_data")
    clim = st.session_state.get("_pdf_clim")
    if not all_data:
        st.warning("Data not loaded yet — please wait for charts to load.")
        return

    with st.spinner("Generating PDF report…"):
        try:
            from ui.components import pdf_report as _pdf
            pdf_bytes = _pdf.generate_report(
                f, all_data, clim, subtract_clim,
                spatial_lt=spatial_lt,
                seasonal_lt=seasonal_lt,
                comparison_lt=comparison_lt,
                spatial_test=st.session_state.get("sm_test_choice", "T-Test"),
                comparison_test=st.session_state.get("cmp_test_choice", "T-Test"),
                altitude_band=altitude_band,
            )
            st.session_state["pdf_report_bytes"] = pdf_bytes
            st.session_state["pdf_report_fp"] = fp
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")
            return
    st.rerun(scope="fragment")
