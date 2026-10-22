# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""CLI to compute, store, and plot verification diagnostics.

Covers the reliability diagram, ROC contingency table, cost-loss value, the scalar CRPS
summary, and the deterministic metrics (mae / bias / correlation). Lead time is given in
days and converted to filename hours by x24. Cost-loss reuses — and always stores — the
ROC contingency table it is derived from.

The **model token drives the source**: a plain token (``graphcast``) scores the raw
forecast; a ``-idr`` token (``graphcast-idr``) scores the IDR-calibrated forecast. Each is
written under exactly the token passed, so ``--model graphcast graphcast-idr`` produces both
side by side. crps and the threshold curves exist only for ``-idr`` tokens (they need the
calibration's output); the deterministic metrics (mae/bias/correlation) are the mirror image —
raw tokens only, since the calibrated forecast is scored by CRPS, not q50 point/rank error.
"""
from __future__ import annotations

import argparse

import numpy as np

from verification import diagnostics as d
from verification.regions import default_region, regions_for_domain
from verification.store import BY_REGION_SUBDIR, PRODUCTS as _PRODUCTS
from verification.seasons import (DEFAULT_SEASON_SLUG, season_months,
                                  seasons_for_country)

_METRICS = ["roc", "cost_loss", "reliability", "crps", "mae",
             "spearmanr", "pearsonr", "bias"]
_THRESHOLD_METRICS = {"roc", "cost_loss", "reliability"}
_DETERMINISTIC_METRICS = ["mae", "spearmanr", "pearsonr", "bias"]
_CALIBRATED_ONLY = _THRESHOLD_METRICS | {"crps"}   # need the IDR output, so '-idr' tokens only

def select_deterministic_metrics(requested, *, calibrated, is_station):
    """Which of the requested deterministic metrics to compute for one model token.

    Deterministic metrics (mae/bias/correlation) score a single-valued forecast, so they
    apply to the **raw** forecast only — the calibrated ('-idr') forecast is scored by CRPS
    (whole distribution) plus the threshold curves, not q50 point/rank error — hence a
    calibrated token gets none. Stations further drop correlation, since the gridded corr
    reducer needs two spatial dims and a gauge series has one. Order follows ``requested``.
    """
    if calibrated:
        return []
    det = [m for m in requested if m in _DETERMINISTIC_METRICS]
    if is_station:
        det = [m for m in det if m in ("mae", "bias")]
    return det


def _available_leads(fc_root):
    """Lead times in days for which a forecast file exists under ``fc_root``.

    The ``{lead}h.nc`` filename scheme is parsed in exactly one place
    (:meth:`verification.store.CountryStore.leads_in`); this converts to days, which is
    the unit the CLI speaks.
    """
    return [h // 24 for h in d.store().leads_in(fc_root)]


def _emit_plots(args, ds, plot_fn, metric, lead_hours, model):
    """Save the requested plots for one diagnostic Dataset (filenames namespaced by model)."""
    prefix = f"{model}_{metric}"
    if args.per_altitude:
        for png in d.plot_by_altitude(ds, "plots", plot_fn, prefix):
            print(f"  plotted {png}")
    elif args.plot:
        band = str(ds["altitude"].values[0])
        print(f"  plotted {plot_fn(ds, f'plots/{prefix}_{lead_hours}h.png', band=band)}")


def _write_legacy_curves(args, samples, roc_store, rel_store, hours, day,
                         year, token, target, variable):
    """Write the per-region / per-season curve files in the pre-existing layout.

    Every one of these is **derived** from the by-region stores by summation rather than
    re-pooled, which is what makes the one-pass run possible while leaving
    ``tools/compare_runs`` able to diff this branch against the write-protected baseline
    with no changes. Since that baseline was produced by the old code path, agreement is
    independent evidence that the summation reproduces a direct pool — the point of
    keeping these files during the transition. They go away with the legacy writer.

    A country with no region partition has no store to sum, so it falls back to scoring
    the pooled samples directly, with the month filter applied as it always was.
    """
    from verification import curves

    whole = default_region()
    catalogue = _catalogue_or_none()
    for region in args.region:
        region_arg = None if region == whole else region
        leaves = None
        if region_arg is not None:
            if catalogue is None:
                print(f"  note: no region catalogue; skipping region {region!r}")
                continue
            region_id = catalogue.id_for(region_arg)
            if region_id is None:
                print(f"  note: {region!r} is not a region of this country; skipped")
                continue
            leaves = list(catalogue.leaves_under(region_id))

        for season in args.season:
            season_arg = None if season == DEFAULT_SEASON_SLUG else season
            months = season_months(season)
            print(f"  region: {region} | season: {season}")
            # PNG previews only for the whole country + yearly; other combinations would
            # collide on the flat plots/ filenames.
            emit = region_arg is None and season_arg is None
            # native_var is NOT optional here. write_dataset defaults it to
            # DEFAULT_VARIABLE, which was invisibly correct while threshold curves only
            # ever ran for precipitation — and silently filed every temperature_max
            # curve under .../precipitation/... the moment they did not.
            common = dict(year=year, model=token, target=target, native_var=variable,
                          region=region_arg, season=season_arg)

            roc = None
            if "roc" in args.metric or "cost_loss" in args.metric:
                roc = (curves.roc_for(roc_store, regions=leaves, months=months)
                       if roc_store is not None
                       else d.compute_roc(_pooled(samples, months)))
                print(f"    wrote {d.write_dataset(roc, 'roc', hours, args.outdir, **common)}")
                if "roc" in args.metric and emit:
                    _emit_plots(args, roc, d.plot_roc, "roc", hours, token)
            if "cost_loss" in args.metric:
                cl = d.compute_cost_loss(roc)
                print(f"    wrote {d.write_dataset(cl, 'cost_loss', hours, args.outdir, **common)}")
                if emit:
                    _emit_plots(args, cl, d.plot_cost_loss, "cost_loss", hours, token)
            if "reliability" in args.metric:
                rel = (curves.reliability_for(rel_store, regions=leaves, months=months,
                                              n_bootstraps=args.n_bootstraps)
                       if rel_store is not None
                       else d.compute_reliability(_pooled(samples, months),
                                                  n_bins=args.n_bins,
                                                  n_bootstraps=args.n_bootstraps))
                print(f"    wrote {d.write_dataset(rel, 'reliability', hours, args.outdir, **common)}")
                if emit:
                    _emit_plots(args, rel, d.plot_reliability, "reliability", hours, token)


def _catalogue_or_none():
    """This deployment's region catalogue, or None when no boundaries are onboarded."""
    from verification import admin

    catalogue = admin.catalogue_for()
    return catalogue if catalogue.pool_level > 0 else None


def _pooled(samples, months):
    """``samples`` restricted to a season's months — the fallback with no region store.

    Only reached for a country whose boundaries are not onboarded, where there is nothing
    to sum over and a season still has to be selected somehow.
    """
    if not months:
        return samples
    keep_month = set(months)
    day_month = samples.day_months
    if day_month is None:
        return samples
    pairs, blocks = {}, {}
    for key, (p, o) in samples.pairs.items():
        day = samples.blocks[key]
        take = np.isin(day_month[day], list(keep_month))
        pairs[key] = (p[take], o[take])
        blocks[key] = day[take]
    return d.Samples(lead_hours=samples.lead_hours, pairs=pairs, blocks=blocks,
                     thresholds=samples.thresholds, bands=samples.bands,
                     day_months=day_month, target=samples.target)


def _verify_year(args, year, variable, target, obs_dem, want_threshold):
    """Run every requested model/metric/lead/region/season for one test year."""
    for token in args.model:
        # The token a user types selects the source and is also the token written out.
        base, calibrated = d.parse_model_token(token)
        if calibrated:
            fc_root = args.fc_root or str(
                d.station_fc_root(base, year) if args.is_station
                else d.model_fc_root(base, year, variable))
        else:
            fc_root = args.fc_root or str(
                d.station_raw_fc_root(base, year) if args.is_station
                else d.model_fc_root(base, year, variable, raw=True))
            calib_only = sorted(_CALIBRATED_ONLY.intersection(args.metric))
            if calib_only:
                print(f"note: {token}: {calib_only} need the calibrated forecast; use {base}-idr")
        source = "calibrated" if calibrated else "raw"

        # Deterministic metrics apply to the raw forecast only; a calibrated token is scored
        # by CRPS + threshold curves. Stations further drop correlation (one spatial dim).
        det_metrics = select_deterministic_metrics(args.metric, calibrated=calibrated,
                                                   is_station=args.is_station)
        dropped_det = [m for m in args.metric if m in _DETERMINISTIC_METRICS
                       and m not in det_metrics]
        if dropped_det:
            reason = ("raw-only (the calibrated forecast is scored by CRPS)" if calibrated
                      else "not supported at stations; deterministic there is mae/bias only")
            print(f"note: {token}: {dropped_det} {reason}; skipped")

        lead_days = args.lead_day or _available_leads(fc_root)
        if not lead_days:
            # No forecast files for this (token, year) — most often a year that
            # hasn't been calibrated yet. Say so explicitly so a multi-year run
            # doesn't silently drop a requested year.
            print(f"{token} | {variable} | {target} {year}: no forecast files "
                  f"found ({fc_root}); calibrate this year first — skipping")
            continue
        for day in lead_days:
            hours = day * 24
            print(f"{token} | lead {day}d ({hours}h) | {variable} | {target} {year}")

            if calibrated and "crps" in args.metric:
                try:
                    crps = d.compute_crps(hours, model=base, year=year, variable=variable, fc_root=fc_root)
                except FileNotFoundError:
                    print(f"  skip crps {hours}h: no forecast file")
                else:
                    print(f"  wrote {d.write_dataset(crps, 'crps', hours, args.outdir, year=year, subdir='annual_monthly_averages', model=token, target=target, native_var=variable)}")
                    if args.plot:
                        print(f"  plotted {d.plot_crps(crps, f'plots/{token}_{variable}_crps_{hours}h.png')}")

            # Deterministic metrics: the raw forecast only (det_metrics is empty for a
            # calibrated token), written under the token as-is.
            for metric in det_metrics:
                try:
                    if args.is_station:
                        dm = d.compute_deterministic_station(hours, model=base, year=year,
                                                             metric=metric, source=source)
                    else:
                        dm = d.compute_deterministic(hours, model=base, year=year, variable=variable,
                                                     metric=metric, source=source)
                except (FileNotFoundError, KeyError):
                    print(f"  skip {metric} {hours}h: no forecast file")
                    continue
                print(f"  wrote {d.write_dataset(dm, metric, hours, args.outdir, year=year, subdir='annual_monthly_averages', model=token, target=target, native_var=variable)}")
                if args.plot:
                    print(f"  plotted {d.plot_spatial(dm, f'plots/{token}_{metric}_{hours}h.png', metric=metric.upper())}")

            if not (calibrated and want_threshold):
                continue

            # ONE pass over the data, whatever was asked for. Every sample is tagged with
            # the region it falls in and the day it happened, so the by-region stores
            # written below already contain every region and every season — they are
            # re-cut by summing at read time (verification.curves), not by pooling again.
            try:
                if args.is_station:
                    # --obs/--dem don't apply: the event truth is the gauge archive and
                    # each station carries its own elevation, so there's no DEM to interp.
                    samples = d.build_samples_station(
                        hours, model=base, year=year, thresholds=args.threshold,
                        bands=args.altitude, fc_root=args.fc_root)
                else:
                    samples = d.build_samples(
                        hours, model=base, year=year, thresholds=args.threshold,
                        bands=args.altitude, fc_root=args.fc_root,
                        variable=variable, **obs_dem)
            except FileNotFoundError:
                print(f"  skip lead {day}d ({hours}h): no forecast file")
                continue

            roc_store = d.compute_roc_by_region(samples)
            rel_store = (d.compute_reliability_counts(samples, n_bins=args.n_bins)
                         if "reliability" in args.metric else None)
            if roc_store is not None:
                if "roc" in args.metric or "cost_loss" in args.metric:
                    print(f"    wrote {d.write_dataset(roc_store, 'roc', hours, args.outdir, year=year, model=token, target=target, native_var=variable, subdir=BY_REGION_SUBDIR)}")
                if rel_store is not None:
                    print(f"    wrote {d.write_dataset(rel_store, 'reliability', hours, args.outdir, year=year, model=token, target=target, native_var=variable, subdir=BY_REGION_SUBDIR)}")

            # The legacy per-region / per-season files, DERIVED from the store above rather
            # than re-pooled. Keeping them means tools/compare_runs still diffs this branch
            # against the pre-change baseline unchanged — and because the baseline was
            # written by the old code, that comparison is an independent check that the
            # summation is right, not a circular one.
            _write_legacy_curves(args, samples, roc_store, rel_store, hours, day,
                                 year, token, target, variable)


def main():
    """Parse arguments, then for each model × lead compute, store, and plot the requested metrics."""
    from verification import config
    config.apply_deployment_defaults()   # seed this machine's country (.deployment); env var still wins
    ap = argparse.ArgumentParser(description="Forecast verification diagnostics.")
    _model_choices = list(d.MODELS) + [f"{m}-idr" for m in d.MODELS]
    ap.add_argument("--model", nargs="+", choices=_model_choices, default=["graphcast-idr"],
                    help="model token(s) to evaluate. A plain token (graphcast) scores the raw "
                         "forecast; a '-idr' token (graphcast-idr) scores the IDR-calibrated "
                         "forecast. Output is keyed by exactly the token passed.")
    ap.add_argument("--metric", nargs="+", choices=_METRICS, default=_METRICS)
    ap.add_argument("--lead-day", nargs="+", type=int, default=None,
                    help="lead times in days (x24 -> hours); default: all available")
    # No static `choices`: the valid set depends on --variable, which argparse has not
    # resolved yet at this point. Validated against the chosen variable below instead,
    # so the error names the variable and lists ITS thresholds.
    ap.add_argument("--threshold", nargs="+", type=int, default=None,
                    help="exceedance thresholds, in the variable's own units "
                         "(precipitation: mm/day; temperature: degC). Default: all of "
                         "the variable's thresholds in config/verification.toml.")
    # Per-country strata. apply_deployment_defaults() has already run, so this sees the
    # deployment country.
    _bands = list(d.altitude_bands())
    ap.add_argument("--altitude", nargs="+", choices=_bands,
                    default=_bands, help="altitude bands (metres, per country)")
    ap.add_argument("--region", nargs="+", choices=regions_for_domain(),
                    default=[default_region()],
                    help="region(s) for threshold curves. Default = the whole country "
                         "(written region-less, unchanged). An admin-1 subregion subsets the "
                         "IDR in memory before computing and writes under a region path "
                         "segment (.../{year}/{region}/{lead}h.nc). Regions run sequentially.")
    # Offer only the seasons this country actually has a calendar for; the slug
    # namespace stays global so discovery can still tell a season from a region.
    ap.add_argument("--season", nargs="+",
                    choices=list(seasons_for_country(config.country())),
                    default=[DEFAULT_SEASON_SLUG],
                    help="season(s) for threshold curves. Default = 'yearly' (whole year, "
                         "written season-less, unchanged). A season subsets the calibrated "
                         "forecast to its calendar months before pooling and writes under a "
                         "season path segment (.../{year}/[{region}/]{season}/{lead}h.nc). "
                         "Composes with --region; seasons run sequentially.")
    ap.add_argument("--n-bootstraps", type=int, default=d.RELIABILITY_N_BOOTSTRAPS)
    ap.add_argument("--n-bins", type=int, default=d.RELIABILITY_N_BINS)
    ap.add_argument("--is-station", action="store_true",
                    help="verify against gauge stations instead of the gridded product")
    ap.add_argument("--product", default=None, choices=sorted(_PRODUCTS),
                    help="observational product to verify against (default: the "
                         "deployment's truth_product). Selects which truth/ and idr/ "
                         "trees are read AND the product segment written into the "
                         "diagnostics path, so scoring a country against a second "
                         "product never overwrites the first.")
    ap.add_argument("--year", nargs="+", type=int, default=[d.DEFAULT_YEAR],
                    help="one or more test years to verify (each must be calibrated "
                         "first); runs sequentially; default 2022")
    ap.add_argument("--variable", default=d.DEFAULT_VARIABLE,
                    choices=list(d.VARIABLES),
                    help="calibrated variable to verify (default precipitation). "
                         "Threshold curves need the variable to have exceedance "
                         "thresholds (config/verification.toml); a variable with none "
                         "gets crps + the deterministic metrics only.")
    ap.add_argument("--outdir", default=None,
                    help="base directory for the diagnostics tree. Default: the "
                         "deployment's own diagnostics root, so the writer always "
                         "agrees with what the dashboard discovers.")
    ap.add_argument("--fc-root", default=None, help="override the forecast root (single model only)")
    ap.add_argument("--obs", default=None)
    ap.add_argument("--dem", default=None)
    ap.add_argument("--plot", action="store_true", help="save a preview PNG per metric")
    ap.add_argument("--per-altitude", action="store_true",
                    help="save one plot per altitude band instead of the first band")
    args = ap.parse_args()

    # --product is applied to the ENVIRONMENT rather than threaded through every call:
    # store() is constructed per use and reads the config at call time, so seeding the
    # variable here reaches the truth reader, the IDR reader and the diagnostics writer
    # at once. Threading it instead would mean a new argument on a dozen functions, any
    # one of which could be missed — and a missed one writes to the default product's
    # tree, which is the exact failure this axis exists to prevent.
    if args.product:
        import os

        os.environ["VERIFICATION_TRUTH_PRODUCT"] = args.product

    # Fail here, not minutes into a lead loop, when the country holds
    # several products and nothing says which.
    from verification.store import CountryStore, require_unambiguous_product

    problem = require_unambiguous_product(CountryStore.current())
    if problem:
        ap.error(problem)

    # The two verification targets differ only in their on-disk SHAPE (gridded field vs
    # gauge points); the diagnostics themselves are identical. Pick the front-end
    # (sample builder + forecast root) and the output segment from --is-station.
    #
    # Which observational PRODUCT these are scored against is a separate axis and is not
    # decided here: it comes from the deployment config (or --product) and the store
    # writes it as its own path segment.
    target = "station" if args.is_station else "gridded"
    variable = args.variable
    obs_dem = {k: v for k, v in (("obs_path", args.obs), ("dem_path", args.dem)) if v is not None}
    # Exceedance metrics (roc/cost_loss/reliability) need p_exceed, which the
    # calibration emits only for a variable with thresholds.
    # Whether the threshold metrics apply is a REGISTRY fact, not a name comparison:
    # a variable supports them exactly when it declares exceedance thresholds. This
    # used to read `variable == "precipitation"`, which is what kept temperature from
    # ever having curves even once it had thresholds to pool at.
    variable_thresholds = d.thresholds_for(variable)
    asked = _THRESHOLD_METRICS.intersection(args.metric)
    want_threshold = bool(asked) and bool(variable_thresholds)
    if asked and not variable_thresholds:
        print(f"note: {sorted(asked)} need exceedance thresholds and {variable} "
              f"defines none; running the other metrics only")

    if args.threshold is None:
        args.threshold = variable_thresholds
    else:
        unknown = [t for t in args.threshold if t not in variable_thresholds]
        if unknown:
            ap.error(f"--threshold {unknown} not defined for {variable}; "
                     f"it has {variable_thresholds} "
                     f"({d.threshold_units(variable) or 'no units'}). A threshold the "
                     f"calibration never emitted p_exceed for would pool an empty curve.")

    if args.fc_root and len(args.year) > 1:
        ap.error("--fc-root applies to a single forecast root; it cannot be "
                 "combined with multiple --year values")
    for year in args.year:
        _verify_year(args, year, variable, target, obs_dem, want_threshold)

if __name__ == "__main__":
    main()
