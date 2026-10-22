"""Produce calibrated IDR forecasts that the verification engine consumes.

Fits Isotonic Distributional Regression per grid cell on the training years and
applies it to the test year, writing one file per lead of calibrated CRPS +
quantiles (+ exceedance probabilities where the variable defines thresholds) into the slot
``verification.run`` reads.

Pipeline stage: harmonise → **calibrate (this)** → verify → dashboard.

One ``--variable`` drives precipitation, temperature_max or temperature_min from a
single spec table — the IDR engine is variable-agnostic; only the I/O, units,
wet-day filter, and exceedance thresholds differ:

Every location comes from the per-country store (:mod:`verification.store`), so this
runs for any onboarded agency without editing paths here. Per variable, what differs:

  precipitation (default):
    forecast : raw gridded `precipitation`   var 'precipitation' (mm/hr)
    obs      : gridded truth                 var 'precip' (mm/day → ÷24)
    out      : calibrated gridded `precipitation`
               crps,q5,q50,q95,q99,pit (mm/hr) + p_exceed over THRESHOLDS (13 values, mm/day)
  temperature_max:
    forecast : raw gridded `temperature_2m`  var 'temperature_2m' (°C)
    obs      : gridded truth                 var 'temp' (°C, no ÷24)
    out      : calibrated gridded `temperature_max`
               crps,q5,q50,q95,q99,pit (degC) + p_exceed over the hot-day
               thresholds in config/verification.toml (degC)
  temperature_min:
    forecast : raw gridded `temperature_2m_min`  var 'temperature_2m'
               (the daily-MIN reduction of 2m temperature — a separate tree from the
                daily-max one above, but the variable inside is still temperature_2m)
    obs      : gridded truth                 var 'temp' (°C)
    out      : calibrated gridded `temperature_min`
               crps,q5,q50,q95,q99,pit (degC); no exceedance (CRPS + quantiles only)

These resolve through the store to the uniform per-country tree —
``data/{country}/forecast/gridded/...`` in, ``data/{country}/truth/{product}/gridded/...``
for the observations, ``data/{country}/idr/{product}/gridded/...`` out. The IDR fit is
stamped with the product it was trained against because that is what it *is*: the same
forecast calibrated against two different products gives two different models, and
they occupy two different trees rather than overwriting each other.

Forecast and obs are already daily and on the same grid, so the only alignment needed
is a date intersection — no regridding here.

Usage:
  python -m calibration.run --model graphcast --leads 1,3,5,7 \
      --train-years 2016,2017,2018,2019,2020,2021 --test-year 2022
  python -m calibration.run --variable temperature_max --model graphcast --leads 1,3,5,7 \
      --train-years 2016,2017,2018,2019,2020,2021 --test-year 2022
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from verification import provenance as _provenance
from verification.store import VARIABLES as _STORE_VARIABLES
from verification.store import truth_field as _truth_field
from verification.store import CountryStore as _CountryStore
from verification.store import PRODUCTS as _PRODUCTS
from verification.store import exceedance_internal as _exceedance_internal
from verification.store import exceedance_thresholds as _exceedance_thresholds

from .idr import idr_on_grid


def _store():
    """The deployment's CountryStore, resolved at call time (never at import)."""
    return _CountryStore.current()


_MM_DAY_TO_MM_HR = 1.0 / 24.0
_AMOUNT_VARS = ("crps", "q5", "q50", "q95", "q99")      # carry the variable's units

# Calibration POLICY per variable. The IDR engine is variable-agnostic; these are the
# only differences. precipitation is the native case (wet-day filter on, ÷24 to mm/hr,
# mm/day exceedance); the temperatures disable the wet-day filter (min_wet_days=0) and
# emit no exceedance — CRPS + quantiles only. The in-file facts (fc_var, obs_var,
# to_rate) are NOT restated here — they come from the one registry in
# verification.store, which this module and the verification engine both read. What
# stays local is genuinely this stage's business:
#   units             — the units stamped on the IDR OUTPUT (note: not the raw file's
#                       units; temperature raw files say "°C", the output "degC")
#   min_wet_days      — the wet-day gate, off for temperature (not a flux)
#
# The exceedance thresholds are NOT here. They live in verification.store.VARIABLES
# because the verification engine pools its curves at exactly the values this stage
# emits p_exceed over, and selects them back BY LABEL — so two copies could disagree
# about which event a curve describes, and a mismatched label yields an empty
# selection rather than an error.
_CALIBRATION_POLICY = {
    "precipitation": {
        "units": "mm/hr", "min_wet_days": 50,
    },
    # Daily-max temperature. The harmoniser reduces 2m temperature over the daily
    # window with max, min and mean, and all three store the SAME variable name
    # (temperature_2m) -- so the VARIABLE SEGMENT of the path is the only thing that
    # says which reduction a file holds, which is why the tree is keyed by it.
    "temperature_max": {
        "units": "degC", "min_wet_days": 0,
    },
    "temperature_min": {
        "units": "degC", "min_wet_days": 0,
    },
}

#: Per-variable spec: shared path/obs facts + this stage's policy, in one mapping so
#: the existing ``spec[...]`` call sites are unchanged.
_VARIABLE_SPEC = {
    name: {
        "fc_var": spec.raw_fc_var,
        "obs_var": spec.obs_var, "to_rate": spec.to_rate,
        "exceedance_units": spec.exceedance_units,
        **_CALIBRATION_POLICY[name],
    }
    for name, spec in _STORE_VARIABLES.items() if name in _CALIBRATION_POLICY
}


def _load_forecast(year: int, model: str, lead_hours: int, variable: str) -> xr.DataArray:
    """Harmonised daily forecast for one (year, model, lead) → (time, lat, lon).

    The forecast ``time`` is relabelled from init date to **valid date** (init +
    prediction_timedelta, floored to the day): the day the daily 06–06 UTC window
    actually covers, which is the date the observations label the same window by. Without this
    the per-lead forecast aligns to obs one lead-day too early.
    """
    store = _store()
    f = store.lead_file(
        store.forecast_dir(target="gridded", variable=variable, year=year,
                           model=model), lead_hours)
    if not f.exists():
        raise FileNotFoundError(f)
    da = xr.open_dataset(f)[_VARIABLE_SPEC[variable]["fc_var"]]
    ptd = da["prediction_timedelta"].values.reshape(-1)
    if ptd.size == 0:                      # e.g. day-9: window runs past the forecast horizon
        raise FileNotFoundError(f)         # skipped by the caller, like a missing file
    if "prediction_timedelta" in da.dims:
        da = da.squeeze("prediction_timedelta", drop=True)
    return da.assign_coords(time=(da["time"] + ptd[0]).dt.floor("D"))


def _load_obs(year: int, variable: str) -> xr.DataArray:
    """Daily gridded obs for the variable, from the configured truth product.

    Lat/Lon are lowercased to match the forecast; precipitation is divided by 24
    (mm/day → mm/hr), temperature is left in °C (not a flux).
    """
    f = _store().truth(target="gridded", variable=variable, year=year)
    if not f.exists():
        raise FileNotFoundError(f)
    obs = xr.open_dataset(f)
    rename = {o: o.lower() for o in ("Lat", "Lon") if o in obs.dims or o in obs.coords}
    obs = obs.rename(rename)
    spec = _VARIABLE_SPEC[variable]
    da = obs[_truth_field(obs, variable)]
    return da * _MM_DAY_TO_MM_HR if spec["to_rate"] else da


def _snap_grid(da: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Snap ``da``'s lat/lon onto ``ref``'s identical grid.

    Forecast and obs are the same physical grid, but their stored lat/lon
    can differ in the last floating-point digits (3.0375 vs 3.0375000000000014)
    when they come from different files (harmoniser regrid vs concatenated obs).
    ``xr.align(join="inner")`` needs *exact* float equality, so such noise
    silently drops the mismatched cells. When the spatial shapes match, reassign
    ``da``'s coords to ``ref``'s so the join keeps the full grid. No-op for
    precipitation (its forecast + obs already bit-match).
    """
    if (da.sizes.get("lat"), da.sizes.get("lon")) == (ref.sizes.get("lat"), ref.sizes.get("lon")):
        return da.assign_coords(lat=ref["lat"], lon=ref["lon"])
    return da


def _stack_years(years, load_one) -> xr.DataArray:
    """Concatenate per-year arrays along time (sorted, de-duplicated)."""
    arrs = [load_one(y) for y in years]
    return xr.concat(arrs, dim="time").sortby("time").drop_duplicates("time")


def calibrate_lead(variable, model, lead_day, train_years, test_year, use_parallel,
                   n_jobs, thresholds=None):
    """Calibrate one lead of `variable`; write the per-lead output file. Returns (path, result).

    ``thresholds`` overrides the variable's exceedance list (mm/day). It matters
    for more than taste: ``p_exceed`` is (threshold, time, lat, lon) float64, so
    each extra threshold costs a full field on disk — 0.375 GB per threshold for
    a 364-day gridded year. The registry default of 13 makes a ~7 GB file per
    lead, which is why a run may need a narrower list. It is also the only way to
    REPRODUCE an existing file: without it the list was unparameterised, so the
    five-threshold files already on disk could not be regenerated as they were.
    """
    spec = _VARIABLE_SPEC[variable]
    lead_hours = lead_day * 24
    # Resolved HERE, not in _VARIABLE_SPEC: that dict is built at import, and thresholds
    # are a per-country fact read at call time, so freezing them there would give every
    # deployment whichever country's climate happened to be configured at import.
    default = list(_exceedance_thresholds(variable))
    # A variable with no exceedance (temperature_min) stays that way — an override must
    # not invent exceedance probabilities the variable does not define.
    exceedance = list(thresholds) if (thresholds and default) else default
    thr_internal = _exceedance_internal(variable, exceedance)

    # The test-year forecast defines the reference grid; every forecast/obs array
    # is snapped onto it so float-repr noise can't shrink the date/grid join.
    ref = _load_forecast(test_year, model, lead_hours, variable)

    # Train: stack years (snapped to ref), intersect forecast & obs on date.
    x_tr = _stack_years(train_years, lambda y: _snap_grid(_load_forecast(y, model, lead_hours, variable), ref))
    y_tr = _stack_years(train_years, lambda y: _snap_grid(_load_obs(y, variable), ref))
    x_tr, y_tr = xr.align(x_tr, y_tr, join="inner")

    # Test.
    x_te, y_te = xr.align(ref, _snap_grid(_load_obs(test_year, variable), ref), join="inner")

    result = idr_on_grid(
        x_tr, y_tr, x_te, y_te,
        spatial_dims=("lat", "lon"), time_dim="time",
        exceedance_thresholds=thr_internal,
        min_wet_days=spec["min_wet_days"],
        use_parallel=use_parallel, n_jobs=n_jobs,
    ).compute()

    # The engine labels rain-amount outputs mm/hr; stamp the variable's real units.
    for v in _AMOUNT_VARS:
        if v in result:
            result[v].attrs["units"] = spec["units"]

    # Relabel the exceedance threshold coord to its display values/units (the
    # verification engine indexes p_exceed positionally against these).
    if "p_exceed" in result:
        result = result.assign_coords(threshold=exceedance)
        result["threshold"].attrs["units"] = spec["exceedance_units"]

    store = _store()
    out = store.lead_file(
        store.calibrated_dir(target="gridded", variable=variable, year=test_year,
                             model=model), lead_hours)
    _provenance.stamp(
        result, stage="calibration", target="gridded", variable=variable,
        model=model, lead_hours=lead_hours, test_year=test_year,
        train_years=list(train_years),
        # the RESOLVED list, not the spec's: thresholds are per country and this run
        # may also have narrowed them with --threshold, and the stamp must record what
        # was actually written rather than what the registry would have defaulted to
        thresholds=list(exceedance) or None,
        threshold_units=spec["exceedance_units"],
        min_wet_days=spec["min_wet_days"],
        layout=type(store.layout).__name__, domain=store.domain,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(out)
    return out, result


def main(argv=None) -> int:
    from verification import config
    config.apply_deployment_defaults()   # seed this machine's country (.deployment); env var still wins
    p = argparse.ArgumentParser(
        description="Produce calibrated IDR forecasts (crps/quantiles[/p_exceed]) for verification.")
    p.add_argument("--variable", default="precipitation", choices=list(_VARIABLE_SPEC),
                   help="observation variable to calibrate (default precipitation)")
    p.add_argument("--model", default="graphcast")
    p.add_argument("--leads", default="1,3,5,7", help="lead days, comma-separated")
    p.add_argument("--train-years", default="2016,2017,2018,2019,2020,2021")
    p.add_argument("--test-year", type=int, default=2022)
    p.add_argument("--threshold", nargs="+", type=int, default=None,
                   help="exceedance thresholds in the variable's own units (mm/day, "
                        "degC), overriding the configured list. Each threshold is a full "
                        "float64 field in p_exceed (~0.4 GB per threshold for a gridded "
                        "year), so a narrower list is often the difference between "
                        "fitting on disk and not. Ignored for variables with no "
                        "exceedance. Pass the SAME list a file already has plus any "
                        "additions: the output replaces the file wholesale, so omitting "
                        "an existing threshold deletes it.")
    p.add_argument("--no-parallel", action="store_true",
                   help="disable joblib parallelism (sequential per-cell fit)")
    p.add_argument("--n-jobs", type=int, default=-1, help="joblib workers (-1 = all cores)")
    p.add_argument("--product", default=None, choices=sorted(_PRODUCTS),
                   help="observational product to calibrate against (default: the "
                        "deployment's truth_product). Selects the truth/ tree read AND "
                        "the idr/ tree written, so a second product's calibration is a "
                        "separate model rather than an overwrite of the first.")
    args = p.parse_args(argv)
    # Applied to the environment, not threaded: the store is built per call and reads
    # the config at call time, so this reaches the truth reader and the IDR writer at
    # once. See verification/run.py for the same note.
    if args.product:
        import os

        os.environ["VERIFICATION_TRUTH_PRODUCT"] = args.product

    # Fail here, not minutes into a lead loop, when the country holds
    # several products and nothing says which.
    from verification.store import CountryStore, require_unambiguous_product

    problem = require_unambiguous_product(CountryStore.current())
    if problem:
        p.error(problem)

    leads = [int(x) for x in args.leads.split(",") if x.strip()]
    train_years = [int(x) for x in args.train_years.split(",") if x.strip()]
    use_parallel = not args.no_parallel

    thr_note = (f" thresholds={args.threshold}" if args.threshold
                and _exceedance_thresholds(args.variable) else "")
    print(f"variable={args.variable} model={args.model} train={train_years} "
          f"test={args.test_year} leads={leads}{thr_note}")
    for day in leads:
        try:
            out, res = calibrate_lead(args.variable, args.model, day, train_years,
                                      args.test_year, use_parallel, args.n_jobs,
                                      thresholds=args.threshold)
        except FileNotFoundError as e:
            print(f"  lead {day}d ({day*24}h): SKIP — missing file: {e}")
            continue
        ncal = int(np.isfinite(res["crps"].mean("time")).sum())
        print(f"  lead {day}d ({day*24}h) -> {out}  (calibrated cells: {ncal})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
