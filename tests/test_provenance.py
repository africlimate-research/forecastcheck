"""Tests for run provenance (``verification.provenance``).

Provenance records what the path cannot: training years, fit gates, threshold subsets,
code version. It is written as global attributes precisely so it stays invisible to the
two systems that must not notice it:

  E1  discovery is unaffected — ``SummarySet.discover`` reads dims and path segments,
      never attributes, so a stamped file indexes exactly as before
  E2  the equivalence gate is unaffected — ``tools.compare_runs`` ignores global attrs,
      so a timestamp that changes every run does not fail every comparison
  E3  values are coerced to NetCDF-safe scalars and survive a round trip to disk
  E4  provenance never breaks a run, even with no git available
  E5  ``write_dataset`` actually stamps, and a real diagnostic file reads back

Runs standalone or under pytest.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compare_runs import compare_datasets
from verification import config, diagnostics as d, provenance


def _tiny_ds():
    return xr.Dataset(
        {"annual_average": (("year",), np.array([0.5]))},
        coords={"year": [2022]},
    )


def test_e1_stamp_adds_only_prefixed_global_attrs():
    """E1: nothing but prefixed global attrs change — no new vars, coords or dims."""
    ds = _tiny_ds()
    before_vars, before_coords = set(ds.data_vars), set(ds.coords)
    before_dims = dict(ds.sizes)
    provenance.stamp(ds, stage="verification", train_years=[2016, 2017])
    assert set(ds.data_vars) == before_vars
    assert set(ds.coords) == before_coords
    assert dict(ds.sizes) == before_dims
    assert all(k.startswith(provenance.PREFIX) for k in ds.attrs)
    got = provenance.read(ds)
    assert got["stage"] == "verification"
    assert got["train_years"] == "2016,2017"
    assert "created_utc" in got and "code_version" in got


def test_e2_provenance_does_not_fail_the_equivalence_gate():
    """E2: two runs differing only in provenance compare equal."""
    a, b = _tiny_ds(), _tiny_ds()
    provenance.stamp(a, stage="calibration", train_years=[2016])
    provenance.stamp(b, stage="calibration", train_years=[2016])
    a.attrs[f"{provenance.PREFIX}created_utc"] = "2020-01-01T00:00:00+00:00"
    b.attrs[f"{provenance.PREFIX}created_utc"] = "2026-09-07T12:00:00+00:00"
    a.attrs[f"{provenance.PREFIX}code_version"] = "aaaaaaa"
    b.attrs[f"{provenance.PREFIX}code_version"] = "bbbbbbb+dirty"
    assert compare_datasets(a, b) == []


def test_e3_values_are_netcdf_safe_and_survive_a_round_trip():
    """E3: lists, None and bools are coerced; the file reads back with them intact."""
    ds = _tiny_ds()
    provenance.stamp(ds, train_years=[2016, 2017], region=None, calibrated=True,
                     thresholds=(1, 5, 10), min_wet_days=50)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.nc"
        ds.to_netcdf(path)                     # would raise on an unsupported attr type
        got = provenance.read(xr.open_dataset(path))
    assert got["train_years"] == "2016,2017"
    assert got["thresholds"] == "1,5,10"
    assert got["region"] == "none"             # explicit, not dropped
    assert got["calibrated"] == "true"
    assert got["min_wet_days"] == 50


def test_e4_code_version_never_raises():
    """E4: provenance must not be able to take down a calibration run."""
    v = provenance.code_version()
    assert isinstance(v, str) and v
    assert provenance.code_version() is v or provenance.code_version() == v  # cached


def test_e5_write_dataset_stamps_and_discovery_still_works():
    """E5: a real diagnostic file carries provenance AND still indexes identically."""
    from ui.services.engine import SummarySet

    ds = xr.Dataset(
        {"annual_average": (("year",), np.array([0.5])),
         "annual_spatial_average": (("year", "lat", "lon"), np.ones((1, 2, 2))),
         "monthly_average": (("month",), np.arange(1, 13.0)),
         "monthly_spatial_average": (("month", "lat", "lon"), np.ones((12, 2, 2)))},
        coords={"year": [2022], "lat": [0.0, 1.0], "lon": [0.0, 1.0],
                "month": list(range(1, 13))},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = d.write_dataset(ds, "crps", 24, tmp, year=2022,
                               subdir="annual_monthly_averages",
                               model="graphcast-idr", target="gridded")
        got = provenance.read(xr.open_dataset(path))
        assert got["stage"] == "verification"
        assert got["metric"] == "crps" and got["model"] == "graphcast-idr"
        assert got["lead_hours"] == 24
        assert got["region"] == "none" and got["season"] == "none"

        # the stamped file must index exactly as an unstamped one would. An explicit
        # outdir IS the diagnostics root (the domain lives above diagnostics/ in the
        # layout, so it is not appended here), and discovery walks exactly that.
        root = Path(tmp)
        idx = SummarySet.discover(root).index
        assert len(idx) == 1, idx
        row = idx.iloc[0]
        assert row["forecast"] == "graphcast-idr" and row["metric"] == "crps"
        assert row["target"] == "chirts-gridded" and row["leads"] == [24]
        assert row["target_kind"] == "gridded"


def _write_station_tree(root: Path, product: str = "gauges") -> None:
    """A tiny synthetic gauge tree under ``root/senegal``: three stations, 2020-2022.

    Forecast (station, prediction_timedelta, time) in mm/hr, gauges in mm/day, laid
    out exactly where the store looks, so the real calibration driver can run on it.
    """
    import pandas as pd

    names = ["Podor", "Matam", "Kedougou"]
    lats, lons = [16.65, 15.66, 12.56], [-14.96, -13.26, -12.18]
    rng = np.random.default_rng(0)
    for year in (2020, 2021, 2022):
        time = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        obs = rng.gamma(0.6, 8.0, size=(time.size, 3))                    # mm/day
        fc = obs / 24.0 * rng.uniform(0.5, 1.5, size=obs.shape)            # mm/hr
        stn = {"station": np.arange(3), "name": ("station", names),
               "lat": ("station", lats), "lon": ("station", lons)}
        f = (root / "senegal/forecast/station/precipitation" / str(year)
             / "graphcast" / "24h.nc")
        f.parent.mkdir(parents=True, exist_ok=True)
        da = xr.DataArray(fc.T[:, None, :], dims=("station", "prediction_timedelta", "time"),
                          coords={**stn, "time": time,
                                  "prediction_timedelta": [np.timedelta64(30, "h")]},
                          attrs={"units": "mm/hr"})
        da.to_dataset(name="precipitation").to_netcdf(f)
        o = root / "senegal/truth" / product / "station/precipitation" / f"{year}.nc"
        o.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"total_precipitation_rate": (("time", "station"), obs,
                                                 {"units": "mm/day"})},
                   coords={**stn, "time": time + pd.Timedelta(hours=6)}).to_netcdf(o)


def test_e6_provenance_agrees_with_the_file_it_describes():
    """E6: a stamped setting must match the data in the same file.

    Regression guard for a real bug: the station stamp wrote THRESHOLDS_MM_DAY verbatim,
    so a run with --thresholds recorded 13 thresholds on a file that contained 5. A
    provenance record contradicting its own data is worse than none, because it gets
    trusted. Asserted behaviourally — run the real calibration on a synthetic gauge tree
    and read the file back — so any future way of desynchronising them fails too.
    """
    import os

    from calibration import run_station as rs

    override = [10, 20, 40, 50, 60]
    keys = ("VERIFICATION_DATA_ROOT", "VERIFICATION_COUNTRY", "VERIFICATION_TRUTH_PRODUCT")
    saved = {k: os.environ.get(k) for k in keys}
    with tempfile.TemporaryDirectory() as tmp:
        _write_station_tree(Path(tmp))
        try:
            os.environ["VERIFICATION_DATA_ROOT"] = tmp
            os.environ["VERIFICATION_COUNTRY"] = "Senegal"
            os.environ.pop("VERIFICATION_TRUTH_PRODUCT", None)
            out, result = rs.calibrate_lead(
                "graphcast", 1, [2020, 2021], 2022,
                use_parallel=False, n_jobs=1, thresholds=override)
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

        assert "/idr/gauges/station/" in str(out), out
        ds = xr.open_dataset(out)
        got = provenance.read(ds)
        # the stamp must describe THIS file, not the module defaults
        assert got["thresholds_mm_day"] == "10,20,40,50,60", got["thresholds_mm_day"]
        assert list(ds["threshold"].values) == override
        assert got["train_years"] == "2020,2021"
        assert got["test_year"] == 2022 and got["lead_hours"] == 24
        assert got["min_wet_days"] == rs._MIN_WET_DAYS
        ds.close()


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
