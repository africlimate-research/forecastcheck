"""Tests for the station IDR calibration driver (calibration/run_station.py).

Runs standalone (``python tests/test_run_station.py``) — no pytest required — and
also works under pytest if installed. Uses tiny synthetic station data so the
real IDR engine runs end-to-end in well under a second.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import calibration.run_station as rs
from calibration.run_station import align_stations, calibrate_stations


def _synth(nstation, ntime, seed, t0="2018-01-01", hour=0, names=None, n_valid=None,
           elevation=None):
    """A tiny (time, station) precip array (mm/hr) with the coords the driver needs."""
    rng = np.random.default_rng(seed)
    time = pd.date_range(t0, periods=ntime, freq="D") + pd.Timedelta(hours=hour)
    names = names if names is not None else [f"S{i}" for i in range(nstation)]
    data = rng.gamma(0.6, 2.0, size=(ntime, nstation))
    coords = {
        "time": time,
        "station": np.arange(nstation),
        "name": ("station", names),
        "lat": ("station", np.linspace(4, 14, nstation)),
        "lon": ("station", np.linspace(34, 47, nstation)),
    }
    if n_valid is not None:
        coords["n_valid"] = ("station", n_valid)
    if elevation is not None:
        coords["elevation"] = ("station", elevation)
    return xr.DataArray(data, dims=("time", "station"), coords=coords, name="precipitation")


def test_calibrate_stations_returns_crps_on_station_grid():
    """Tracer: the driver fits per station and returns crps on (time, station)."""
    x_tr = _synth(3, 150, seed=1)
    y_tr = _synth(3, 150, seed=2)
    x_te = _synth(3, 60, seed=3, t0="2022-01-01")
    y_te = _synth(3, 60, seed=4, t0="2022-01-01")

    out = calibrate_stations(x_tr, y_tr, x_te, y_te,
                             thresholds_mm_day=[10, 20, 40, 50, 60],
                             min_wet_days=20)

    assert "crps" in out.data_vars
    assert tuple(out["crps"].dims) == ("time", "station")
    assert out.sizes["station"] == 3
    assert rs._DUMMY not in out.dims  # the adapter axis must be squeezed away


def test_calibrate_handles_station_time_forecast_layout():
    """The real forecast is (station, time); obs is (time, station). Both work."""
    x_tr = _synth(3, 150, 1).transpose("station", "time")      # forecast layout
    y_tr = _synth(3, 150, 2)
    x_te = _synth(3, 60, 3, t0="2022-01-01").transpose("station", "time")
    y_te = _synth(3, 60, 4, t0="2022-01-01")

    out = calibrate_stations(x_tr, y_tr, x_te, y_te,
                             thresholds_mm_day=[10, 20, 40, 50, 60], min_wet_days=20)

    assert tuple(out["crps"].dims) == ("time", "station")
    assert out.sizes["station"] == 3


def test_station_identity_coords_carried_onto_output():
    """name/lat/lon/elevation (station identity) ride onto the output for the consumer."""
    nv = np.array([365, 120, 60]); elev = np.array([1500.0, 800.0, 2400.0])
    x_tr, y_tr = _synth(3, 150, 1), _synth(3, 150, 2)
    x_te = _synth(3, 60, 3, t0="2022-01-01")
    y_te = _synth(3, 60, 4, t0="2022-01-01", n_valid=nv, elevation=elev)

    out = calibrate_stations(x_tr, y_tr, x_te, y_te,
                             thresholds_mm_day=[10, 20, 40, 50, 60], min_wet_days=20)

    for c in ("name", "lat", "lon", "elevation"):
        assert c in out.coords, c
    assert list(out["elevation"].values) == list(elev)
    assert tuple(out["name"].dims) == ("station",)


def test_p_exceed_threshold_coord_is_mm_day():
    """p_exceed is present and its threshold coord is relabelled to mm/day."""
    x_tr, y_tr = _synth(3, 150, 1), _synth(3, 150, 2)
    x_te = _synth(3, 60, 3, t0="2022-01-01")
    y_te = _synth(3, 60, 4, t0="2022-01-01")

    out = calibrate_stations(x_tr, y_tr, x_te, y_te,
                             thresholds_mm_day=[10, 20, 40, 50, 60], min_wet_days=20)

    assert "p_exceed" in out.data_vars
    assert list(out["threshold"].values) == [10, 20, 40, 50, 60]
    assert out["threshold"].attrs.get("units") == "mm/day"
    # probabilities stay in [0, 1]
    p = out["p_exceed"].values
    assert np.nanmin(p) >= 0.0 and np.nanmax(p) <= 1.0


def test_n_valid_carried_onto_output():
    """The gauge's per-station n_valid is carried onto the result for weighting."""
    nv = np.array([365, 120, 60])
    x_tr, y_tr = _synth(3, 150, 1), _synth(3, 150, 2)
    x_te = _synth(3, 60, 3, t0="2022-01-01")
    y_te = _synth(3, 60, 4, t0="2022-01-01", n_valid=nv)

    out = calibrate_stations(x_tr, y_tr, x_te, y_te,
                             thresholds_mm_day=[10, 20, 40, 50, 60], min_wet_days=20)

    assert "n_valid" in out.coords
    assert list(out["n_valid"].values) == list(nv)


def _named(values_by_name, ntime, t0="2022-01-01", hour=0):
    """A (time, station) array whose per-station value encodes its name, for
    detecting mis-pairing. values_by_name: {name: constant}."""
    names = list(values_by_name)
    time = pd.date_range(t0, periods=ntime, freq="D") + pd.Timedelta(hours=hour)
    data = np.tile([values_by_name[n] for n in names], (ntime, 1))
    return xr.DataArray(data, dims=("time", "station"),
                        coords={"time": time, "station": np.arange(len(names)),
                                "name": ("station", names)}, name="precipitation")


def _geo(rows, ntime=5, t0="2022-01-01", hour=0, gauge="station"):
    """A (time, station) array carrying gauge coords, for the duplicate-name case.

    rows: list of (name, gauge_lat, gauge_lon, value). gauge="station" stores the
    gauge location in station_lat/station_lon (forecast layout, plus an offset
    lat/lon 'cell'); gauge="obs" stores it in lat/lon (obs layout).
    """
    names = [r[0] for r in rows]
    lat = np.array([r[1] for r in rows]); lon = np.array([r[2] for r in rows])
    time = pd.date_range(t0, periods=ntime, freq="D") + pd.Timedelta(hours=hour)
    data = np.tile([r[3] for r in rows], (ntime, 1))
    coords = {"time": time, "station": np.arange(len(rows)), "name": ("station", names)}
    if gauge == "station":
        coords["station_lat"] = ("station", lat); coords["station_lon"] = ("station", lon)
        coords["lat"] = ("station", lat + 0.04); coords["lon"] = ("station", lon + 0.04)
    else:
        coords["lat"] = ("station", lat); coords["lon"] = ("station", lon)
    return xr.DataArray(data, dims=("time", "station"), coords=coords, name="precipitation")


def test_align_disambiguates_duplicate_names_by_coords():
    """Two stations share a name; the gauge coords disambiguate the pairing."""
    fc = _geo([("Ndiaye", 7.88, 39.62, 0.0), ("Matam", 7.75, 38.5, 0.0),
               ("Ndiaye", 7.13, 40.05, 0.0)], gauge="station")
    obs = _geo([("Ndiaye", 7.13, 40.05, 30.0), ("Ndiaye", 7.88, 39.62, 10.0),
                ("Matam", 7.75, 38.5, 20.0)], gauge="obs")        # shuffled

    fc2, obs2 = align_stations(fc, obs)

    # obs reordered onto fc: Ndiaye@39.62→10, Matam→20, Ndiaye@40.05→30
    assert list(obs2.isel(time=0).values) == [10.0, 20.0, 30.0]


def test_align_raises_on_nonunique_obs_key():
    """A name+coord collision within obs fails loud, not a silent wrong pairing."""
    fc = _geo([("A", 1.0, 1.0, 0.0), ("B", 2.0, 2.0, 0.0)], gauge="station")
    obs = _geo([("A", 1.0, 1.0, 1.0), ("B", 2.0, 2.0, 2.0),
                ("B", 2.0, 2.0, 9.0)], gauge="obs")            # B collides
    try:
        align_stations(fc, obs)
    except ValueError as e:
        assert "obs" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on non-unique obs key")


def test_align_pairs_by_name_not_index():
    """obs in a different station order is reindexed to the forecast's name order."""
    fc = _named({"A": 0.0, "B": 0.0, "C": 0.0}, 5)              # names A,B,C
    obs = _named({"C": 3.0, "A": 1.0, "B": 2.0}, 5)             # shuffled order

    fc2, obs2 = align_stations(fc, obs)

    assert list(obs2["name"].values) == list(fc2["name"].values) == ["A", "B", "C"]
    # value encodes name → after reindex, column i must hold name i's value
    assert list(obs2.isel(time=0).values) == [1.0, 2.0, 3.0]


def test_align_raises_on_station_set_mismatch():
    """A station present in one file but not the other fails loud, naming it."""
    fc = _named({"A": 0.0, "B": 0.0, "C": 0.0}, 5)
    obs = _named({"A": 1.0, "B": 2.0}, 5)                       # missing C
    try:
        align_stations(fc, obs)
    except ValueError as e:
        assert "C" in str(e)
    else:
        raise AssertionError("expected ValueError on station set-mismatch")


def test_align_date_floors_then_intersects_time():
    """obs stamped 06:00 UTC and forecast at 00:00 still pair by calendar date.

    A naive inner-join on raw timestamps would give an empty intersection; the
    driver must floor to date first. Partial-overlap windows intersect by date.
    """
    fc = _named({"A": 0.0, "B": 0.0, "C": 0.0}, 5, t0="2022-01-01", hour=0)   # Jan 1-5
    obs = _named({"A": 1.0, "B": 2.0, "C": 3.0}, 5, t0="2022-01-03", hour=6)  # Jan 3-7

    fc2, obs2 = align_stations(fc, obs)

    assert fc2.sizes["time"] == obs2.sizes["time"] == 3                       # Jan 3,4,5
    assert list(fc2["time"].values) == list(obs2["time"].values)             # shared axis


def test_load_obs_converts_mm_day_to_mm_hr(tmp_path=None):
    """_load_obs returns mm/hr (÷24) and keeps the name + n_valid coords."""
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    time = pd.date_range("2022-01-01", periods=4, freq="D") + pd.Timedelta(hours=6)
    obs = xr.Dataset(
        {"total_precipitation_rate": (("time", "station"),
                                      np.array([[24.0, 48.0]] * 4))},
        coords={"time": time, "station": ["SNPOD001", "SNMAT001"],
                "name": ("station", ["Podor", "Matam"]),
                "n_valid": ("station", [365, 120])},
    )
    obs["total_precipitation_rate"].attrs["units"] = "mm/day"
    obs.to_netcdf(tmp / "stations_2022.nc")

    # Inject through the keyword seam. This used to assign rs._OBS_DIR, which was
    # never restored and so leaked a temp path into every later test in the process.
    da = rs._load_obs(2022, obs_dir=tmp)

    assert float(da.isel(time=0, station=0)) == 1.0     # 24 mm/day -> 1 mm/hr
    assert float(da.isel(time=0, station=1)) == 2.0     # 48 mm/day -> 2 mm/hr
    assert list(da["name"].values) == ["Podor", "Matam"]
    assert list(da["n_valid"].values) == [365, 120]


if __name__ == "__main__":
    test_calibrate_stations_returns_crps_on_station_grid()
    test_calibrate_handles_station_time_forecast_layout()
    test_station_identity_coords_carried_onto_output()
    test_p_exceed_threshold_coord_is_mm_day()
    test_n_valid_carried_onto_output()
    test_align_disambiguates_duplicate_names_by_coords()
    test_align_raises_on_nonunique_obs_key()
    test_align_pairs_by_name_not_index()
    test_align_raises_on_station_set_mismatch()
    test_align_date_floors_then_intersects_time()
    test_load_obs_converts_mm_day_to_mm_hr()
    print("ok")
