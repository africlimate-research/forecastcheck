"""Produce calibrated IDR forecasts at station locations for the verification engine.

The station sibling of ``run.py``. Fits one Isotonic Distributional Regression
model per *station* on the stacked train years and applies it to the test year,
writing the same output contract as the gridded driver — crps,q5,q50,q95,q99,pit
(mm/hr) and p_exceed(threshold=[10,20,40,50,60] mm/day, time, station) — but on a
(time, station) grid.

The certified 2-D IDR engine (idr.idr_on_grid) is called unchanged: stations are
presented as the first spatial dim plus a synthetic length-1 second dim, squeezed
off the result. The engine never learns it is scoring stations, so the
byte-identical equivalence proof carries over untouched.

Every location comes from the per-country store (:mod:`verification.store`), so this
runs for any onboarded agency without editing paths here:

Inputs (per year / model / lead-hour):
  forecast : {raw}/station/precipitation/{year}/{model}/{H}h.nc  var 'precipitation' (mm/hr)
             read through the store, which returns the canonical
             (station, prediction_timedelta, time) dim order regardless of which
             harmoniser wrote the file
  obs      : engine-tier gauge truth for {year}  var 'total_precipitation_rate' (mm/day → ÷24)
Output:
  {idr}/station/precipitation/{test_year}/{model}/{H}h.nc
    crps,q5,q50,q95,q99,pit  (time,station)            in mm/hr
    p_exceed                 (threshold,time,station)  threshold = the deployment's
                             precipitation thresholds, mm/day (config/verification.toml)

Usage:
  python -m calibration.run_station --model graphcast --leads 1,3,5,7,9 \
      --train-years 2016,2017,2018,2019,2020,2021 --test-year 2022
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from verification import provenance as _provenance
from verification.store import CountryStore as _CountryStore
from verification.store import PRODUCTS as _PRODUCTS
from verification.store import exceedance_thresholds as _exceedance_thresholds

from .idr import idr_on_grid

_VARIABLE = "precipitation"          # stations are precipitation-only


def _store():
    """The deployment's CountryStore, resolved at call time (never at import)."""
    return _CountryStore.current()


#: Per-station fit gates, hoisted from the calibrate_stations signature so the values
#: recorded in provenance are the ones actually applied. A station failing either gate
#: is all-NaN for every output — silently, apart from the "calibrated stations" count —
#: so they belong in the audit trail.
_MIN_VALID = 10          # minimum finite train pairs
_MIN_WET_DAYS = 50       # minimum train days with obs > 0.1 mm/hr

_DUMMY = "_dummy"
_MM_DAY_TO_MM_HR = 1.0 / 24.0

# Decision thresholds. p_exceed is indexed positionally by the verification engine
# against this exact list, so order/units (mm/day) must match it.
#
# Resolved at CALL time, never frozen into a module constant: thresholds are a
# per-country fact that ``config/verification.toml`` can retune, and a literal here
# would quietly give the station pipeline different events from the gridded one —
# which shared `store.exceedance_thresholds` already. It WAS such a literal.
def default_thresholds_mm_day() -> list[float]:
    """This deployment's precipitation thresholds, mm/day."""
    return list(_exceedance_thresholds("precipitation"))


def _station_keys(da):
    """A unique per-station key: name plus the gauge lat/lon (rounded).

    Station names are not unique, so name alone cannot join the
    two files. The gauge coordinate disambiguates: the forecast stores it in
    ``station_lat``/``station_lon`` (``lat``/``lon`` there is the regridded cell),
    the obs in ``lat``/``lon``. Falls back to name-only when no coords are present
    (test fixtures).
    """
    names = [str(n) for n in da["name"].values]
    if "station_lat" in da.coords:
        lat, lon = da["station_lat"].values, da["station_lon"].values
    elif "lat" in da.coords and da["lat"].dims == ("station",):
        lat, lon = da["lat"].values, da["lon"].values
    else:
        return names
    return [f"{n}|{round(float(a), 4)}|{round(float(o), 4)}"
            for n, a, o in zip(names, lat, lon)]


def align_stations(forecast, obs):
    """Pair a station forecast and gauge obs by station identity and calendar date.

    Joins on a (name, gauge-lat, gauge-lon) key, never on axis position — the two
    files' station orders are not guaranteed equal and names are not unique. A
    station present in one file but not the other fails loud. Obs is then
    reindexed onto the forecast's station order, both time axes are floored to
    calendar date (obs is stamped at the 06:00-UTC window end, the forecast at
    00:00 — a raw-timestamp join would be empty), and the dates are intersected.
    Returns (forecast, obs) on the shared station × date grid.
    """
    fc_keys = _station_keys(forecast)
    obs_keys = _station_keys(obs)
    if len(set(fc_keys)) != len(fc_keys):
        raise ValueError("non-unique station keys in forecast (name+coords collision)")
    if len(set(obs_keys)) != len(obs_keys):
        raise ValueError("non-unique station keys in obs (name+coords collision)")
    missing = set(fc_keys) - set(obs_keys)
    extra = set(obs_keys) - set(fc_keys)
    if missing or extra:
        raise ValueError(
            f"station mismatch — in forecast only: {sorted(missing)}; "
            f"in obs only: {sorted(extra)}")

    st = np.arange(forecast.sizes["station"])
    obs_r = (obs.assign_coords(_key=("station", obs_keys))
                .swap_dims({"station": "_key"})
                .sel(_key=fc_keys)
                .swap_dims({"_key": "station"})
                .drop_vars("_key")
                .assign_coords(station=("station", st)))
    forecast = forecast.assign_coords(station=("station", st))

    fc_d = forecast.assign_coords(time=forecast["time"].dt.floor("D"))
    obs_d = obs_r.assign_coords(time=obs_r["time"].dt.floor("D"))
    return xr.align(fc_d, obs_d, join="inner")


def calibrate_stations(forecast_train, obs_train, forecast_test, obs_test,
                       thresholds_mm_day, *, use_parallel=False, n_jobs=1,
                       min_valid=_MIN_VALID, min_wet_days=_MIN_WET_DAYS):
    """Calibrate per station via IDR; return crps/quantiles/pit + p_exceed.

    All rain-amount inputs are mm/hr; ``thresholds_mm_day`` is in mm/day. Stations
    are bridged to the 2-D engine by a length-1 dummy spatial dim that is squeezed
    off the result.
    """
    thr_hr = [t * _MM_DAY_TO_MM_HR for t in thresholds_mm_day]

    def _grid(da):  # -> (time, station, _dummy=1), one canonical order for all four
        return da.transpose("time", "station").expand_dims({_DUMMY: 1}, axis=-1)

    result = idr_on_grid(
        _grid(forecast_train), _grid(obs_train), _grid(forecast_test), _grid(obs_test),
        spatial_dims=("station", _DUMMY), time_dim="time",
        exceedance_thresholds=thr_hr,
        use_parallel=use_parallel, n_jobs=n_jobs,
        min_valid=min_valid, min_wet_days=min_wet_days,
    ).squeeze(_DUMMY, drop=True).compute()

    # Relabel the threshold coord from internal mm/hr to the mm/day the
    # verification engine indexes against (probabilities are unchanged).
    if "p_exceed" in result:
        result = result.assign_coords(threshold=list(thresholds_mm_day))
        result["threshold"].attrs["units"] = "mm/day"

    # Carry station identity + the gauge's valid-day count onto the result, so the
    # verification consumer can join the gauge obs by name and stratify by altitude,
    # and the viz layer can map by lat/lon and station-day-weight by n_valid — all
    # without re-opening the obs file. n_valid is the gauge's annual report count,
    # distinct from the engine's min_valid fit gate.
    carry = ("name", "lat", "lon", "elevation", "n_valid")
    for c in carry:
        if c in obs_test.coords and obs_test[c].dims == ("station",):
            result = result.assign_coords({c: ("station", obs_test[c].values)})

    return result


# ---------------------------------------------------------------------------
# IO + CLI (mirrors run.py)
# ---------------------------------------------------------------------------

def _load_forecast(year: int, model: str, lead_hours: int):
    """Forecast-at-station (mm/hr) for one (year, model, lead) → (station, time).

    The forecast ``time`` is relabelled from init date to **valid date** (init +
    prediction_timedelta, floored to the day) so each lead pairs with the gauge day its
    06–06 UTC window covers; ``align_stations`` then floors both and intersects by date.
    """
    store = _store()
    f = store.lead_file(
        store.forecast_dir(target="station", variable=_VARIABLE, year=year,
                               model=model), lead_hours)
    if not f.exists():
        raise FileNotFoundError(f)
    # open_raw_forecast validates units and returns the canonical dim order, so a
    # (prediction_timedelta, time, station) file is transposed before anything
    # downstream sees it.
    da = store.open_raw_forecast(target="station", variable=_VARIABLE, year=year,
                                 model=model, lead=lead_hours)["precipitation"]
    ptd = da["prediction_timedelta"].values.reshape(-1)
    if ptd.size == 0:                      # e.g. day-9: window runs past the forecast horizon
        raise FileNotFoundError(f)         # skipped by the caller, like a missing file
    if "prediction_timedelta" in da.dims:
        da = da.squeeze("prediction_timedelta", drop=True)
    return da.assign_coords(time=(da["time"] + ptd[0]).dt.floor("D"))


def _load_obs(year: int, obs_dir=None):
    """Gauge obs, mm/day → mm/hr, keeping name / n_valid / lat / lon coords.

    ``obs_dir`` overrides the location for tests; None asks the store for this
    country's engine-tier truth. It is a keyword seam rather than a module global so a
    test cannot leak a temp path into the rest of the process.
    """
    f = (Path(obs_dir) / f"stations_{year}.nc" if obs_dir is not None
         else _store().truth(target="station", variable=_VARIABLE, year=year))
    if not f.exists():
        raise FileNotFoundError(f)
    obs = xr.open_dataset(f)
    return obs["total_precipitation_rate"] * _MM_DAY_TO_MM_HR


def _stack_years(years, load_one):
    """Concatenate per-year arrays along time (sorted, de-duplicated)."""
    arrs = [load_one(y) for y in years]
    return xr.concat(arrs, dim="time").sortby("time").drop_duplicates("time")


def calibrate_lead(model, lead_day, train_years, test_year, use_parallel, n_jobs,
                   thresholds=None):
    """Calibrate one lead at stations; write the per-lead file. Returns (path, result).

    ``thresholds`` (mm/day) defaults to :func:`default_thresholds_mm_day` — this
    deployment's configured list, read now rather than at import. Pass an explicit list
    to reproduce a run made when that list differed.
    """
    thresholds = list(default_thresholds_mm_day() if thresholds is None else thresholds)
    lead_hours = lead_day * 24

    x_tr = _stack_years(train_years, lambda y: _load_forecast(y, model, lead_hours))
    y_tr = _stack_years(train_years, _load_obs)
    x_tr, y_tr = align_stations(x_tr, y_tr)

    x_te, y_te = align_stations(_load_forecast(test_year, model, lead_hours),
                                _load_obs(test_year))

    result = calibrate_stations(
        x_tr, y_tr, x_te, y_te, thresholds,
        use_parallel=use_parallel, n_jobs=n_jobs,
    )

    store = _store()
    out = store.lead_file(
        store.calibrated_dir(target="station", variable=_VARIABLE, year=test_year,
                             model=model), lead_hours)
    # Train years are the fact the path cannot record and that most changes the
    # numbers; without this the output is unauditable after the fact.
    _provenance.stamp(
        result, stage="calibration", target="station", variable=_VARIABLE,
        model=model, lead_hours=lead_hours, test_year=test_year,
        train_years=list(train_years), thresholds_mm_day=list(thresholds),
        min_valid=_MIN_VALID, min_wet_days=_MIN_WET_DAYS,
        layout=type(store.layout).__name__, domain=store.domain,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(out)
    return out, result


def main(argv=None):
    from verification import config
    config.apply_deployment_defaults()   # seed this machine's country (.deployment); env var still wins
    p = argparse.ArgumentParser(
        description="Produce calibrated IDR forecasts (p_exceed/crps) at stations.")
    p.add_argument("--model", default="graphcast")
    p.add_argument("--leads", default="1,3,5,7,9", help="lead days, comma-separated")
    p.add_argument("--train-years", default="2016,2017,2018,2019,2020,2021")
    p.add_argument("--test-year", type=int, default=2022)
    p.add_argument("--no-parallel", action="store_true",
                   help="disable joblib parallelism (sequential per-station fit)")
    p.add_argument("--n-jobs", type=int, default=-1, help="joblib workers (-1 = all cores)")
    p.add_argument("--thresholds", default=None,
                   help="exceedance thresholds in mm/day, comma-separated. Defaults to "
                        "this country's configured list. The verification engine selects p_exceed by "
                        "LABEL, so whatever is written here must cover the thresholds "
                        "that stage asks for. Provided so a historical run can be "
                        "reproduced exactly: output written before this list grew to 13 "
                        "values carries only [10,20,40,50,60].")
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

    leads = [int(x) for x in args.leads.split(",") if x.strip()]
    thresholds = ([int(x) for x in args.thresholds.split(",") if x.strip()]
                  if args.thresholds else default_thresholds_mm_day())
    train_years = [int(x) for x in args.train_years.split(",") if x.strip()]
    use_parallel = not args.no_parallel

    print(f"model={args.model} train={train_years} test={args.test_year} leads={leads}")
    for day in leads:
        try:
            out, res = calibrate_lead(args.model, day, train_years, args.test_year,
                                      use_parallel, args.n_jobs,
                                      thresholds=thresholds)
        except FileNotFoundError as e:
            print(f"  lead {day}d ({day*24}h): SKIP — missing file: {e}")
            continue
        ncal = int(np.isfinite(res["crps"].mean("time")).sum())
        print(f"  lead {day}d ({day*24}h) -> {out}  (calibrated stations: {ncal})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
