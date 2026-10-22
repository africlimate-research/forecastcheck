"""Tests for the station branch of the verification consumer.

Standalone (``python tests/test_diagnostics_station.py``) or under pytest. Tiny
synthetic station data with known answers; the threshold machinery (ROC,
reliability, cost-loss) is shape-agnostic and already covered for the gridded
case, so these tests target only the station-specific seams: pooling over a
single ``station`` spatial dim, and the station sample/obs front-end.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import diagnostics as d


def test_pool_pairs_over_single_station_dim():
    """pool_pairs handles (time, station) — one spatial dim — not just (lat, lon)."""
    precip = xr.DataArray(  # (time=3, station=2), mm/day
        np.array([[5.0, 50.0], [0.0, 30.0], [100.0, 1.0]]),
        dims=("time", "station"))
    nthr = len(d.THRESHOLDS_MM_DAY)
    p_exceed = xr.DataArray(  # (threshold, time, station) — only thr=20 is selected below
        np.broadcast_to(0.5, (nthr, 3, 2)).copy(),
        dims=("threshold", "time", "station"),
        coords={"threshold": d.THRESHOLDS_MM_DAY})
    masks = {"all": np.array([True, True])}

    pairs, blocks, _ = d.pool_pairs(p_exceed, precip, masks, thresholds=[20])
    p, o = pairs[(20, "all")]

    assert p.shape == o.shape == (6,)                 # 3 times × 2 stations
    assert int(o.sum()) == 3                           # 50, 30, 100 exceed 20 mm/day
    assert set(blocks[(20, "all")]) == {0, 1, 2}       # day index preserved


def _write_station_idr(path, names, lats, lons, elevs, ntime=4, t0="2022-01-01"):
    """A minimal station IDR output file: p_exceed + station identity coords."""
    M = len(names)
    time = pd.date_range(t0, periods=ntime, freq="D")     # midnight (producer floors)
    p = np.full((len(d.THRESHOLDS_MM_DAY), ntime, M), 0.3)
    ds = xr.Dataset(
        {"p_exceed": (("threshold", "time", "station"), p)},
        coords={"threshold": d.THRESHOLDS_MM_DAY, "time": time,
                "station": np.arange(M), "name": ("station", names),
                "lat": ("station", lats), "lon": ("station", lons),
                "elevation": ("station", elevs)},
    )
    ds.to_netcdf(path)


def _write_gauge(path, names, lats, lons, elevs, precip, t0="2022-01-01"):
    """A minimal gauge obs file (mm/day), stamped at 06:00 UTC like the real one."""
    ntime = precip.shape[0]
    time = pd.date_range(t0, periods=ntime, freq="D") + pd.Timedelta(hours=6)
    ds = xr.Dataset(
        {"total_precipitation_rate": (("time", "station"), precip)},
        coords={"time": time, "station": [f"code{i}" for i in range(len(names))],
                "name": ("station", names),
                "lat": ("station", lats), "lon": ("station", lons),
                "elevation": ("station", elevs)},
    )
    ds["total_precipitation_rate"].attrs["units"] = "mm/day"
    ds.to_netcdf(path)


def test_pool_pairs_gridded_flatten_is_order_preserving():
    """The (lat, lon) flatten must be C-order, so gridded pooling stays byte-identical."""
    precip = xr.DataArray(  # (time=1, lat=2, lon=2), mm/day
        np.array([[[5.0, 50.0], [30.0, 1.0]]]), dims=("time", "lat", "lon"))
    p_exceed = xr.DataArray(
        np.broadcast_to(0.4, (len(d.THRESHOLDS_MM_DAY), 1, 2, 2)).copy(),
        dims=("threshold", "time", "lat", "lon"),
        coords={"threshold": d.THRESHOLDS_MM_DAY})
    masks = {"all": np.array([[True, True], [True, True]])}

    pairs, _, _ = d.pool_pairs(p_exceed, precip, masks, thresholds=[20])
    _, o = pairs[(20, "all")]
    # C-order ravel of [[5,50],[30,1]] = [5,50,30,1]; events > 20 → [0,1,1,0]
    assert list(o) == [0, 1, 1, 0]


def test_build_samples_station_pools_and_stratifies_by_elevation():
    """End-to-end station front-end: align gauge by name+date, band by elevation, pool."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    fc_root = tmp / "idr"; fc_root.mkdir()
    # canonical per-station identity (IDR-file order)
    names = np.array(["Podor", "Matam", "Kedougou"])
    lats = np.array([16.6, 15.6, 12.5]); lons = np.array([-15.0, -13.3, -12.2])
    elev = np.array([500.0, 1500.0, 2500.0])               # bands 0-1000, 1000-2000, 2000-3000
    _write_station_idr(fc_root / "24h.nc", names, lats, lons, elev, ntime=4)
    precip = np.array([[50.0, 5.0, 0.0], [0.0, 30.0, 100.0],
                       [25.0, 0.0, 5.0], [5.0, 60.0, 0.0]])  # (time, station) in IDR order
    # gauge rows shuffled to exercise the name-join (identity reindexed consistently)
    o = [2, 0, 1]
    _write_gauge(tmp / "stations_2022.nc", names[o], lats[o], lons[o], elev[o], precip[:, o])

    samples = d.build_samples_station(24, fc_root=fc_root, obs_dir=tmp, year=2022)

    # one (threshold, band) per axis combination
    assert (20, "all") in samples.pairs
    # band "0-1000" must contain only station 0 (Podor, 500 m): 4 days × 1 station
    p, o = samples.pairs[(20, "0-1000")]
    assert p.shape == o.shape == (4,)
    # Podor daily mm/day = [50, 0, 25, 5]; events > 20 → [1,0,1,0]
    assert list(o) == [1, 0, 1, 0]


def test_year_routes_forecast_roots_obs_and_output():
    """--year threads through forecast roots, the gridded obs path, and output path."""
    assert str(d.model_fc_root("ifs", 2021)).endswith("idr/chirts/gridded/precipitation/2021/ifs")
    assert str(d.station_fc_root("graphcast", 2023)).endswith("idr/chirts/station/precipitation/2023/graphcast")
    # gridded obs routes by product, then target, then variable
    assert str(d.gridded_obs_path(2020)).endswith(
        "truth/chirts/gridded/precipitation/2020.nc")
    assert str(d.gridded_obs_path(2020, "temperature_max")).endswith(
        "truth/chirts/gridded/temperature_max/2020.nc")
    # IDR forecast root + output native-var segment route by variable
    assert str(d.model_fc_root("graphcast", 2022, "temperature_max")).endswith(
        "idr/chirts/gridded/temperature_max/2022/graphcast")
    assert "/temperature_max/" in str(
        d.output_path("crps", 24, "out", year=2022, model="graphcast", native_var="temperature_max"))
    # default stays 2022 + precipitation (backward compatible)
    assert str(d.model_fc_root("graphcast")).endswith("idr/chirts/gridded/precipitation/2022/graphcast")
    assert "/precipitation/" in str(d.output_path("crps", 24, "out", model="graphcast"))
    assert str(d.gridded_obs_path()).endswith(
        "truth/chirts/gridded/precipitation/2022.nc")
    # output path carries the requested year segment
    assert "/2021/" in str(d.output_path("roc", 24, "out", year=2021, model="ifs", target="station"))


def test_parse_model_token_is_inverse_of_ui_model_token():
    """parse_model_token splits a token into (base, calibrated); round-trips ui_model_token."""
    assert d.parse_model_token("graphcast") == ("graphcast", False)
    assert d.parse_model_token("graphcast-idr") == ("graphcast", True)
    assert d.parse_model_token("ifs-idr") == ("ifs", True)
    for base in ("graphcast", "ifs"):
        for cal in (True, False):
            tok = d.ui_model_token(base, cal)
            assert d.parse_model_token(tok) == (base, cal)


def test_idr_token_and_discovery_roundtrip():
    """Calibrated → '{model}-idr', raw → '{model}'; both at UI-discoverable paths (no 'raw/')."""
    assert d.ui_model_token("graphcast", calibrated=True) == "graphcast-idr"
    assert d.ui_model_token("graphcast", calibrated=False) == "graphcast"

    cal = d.output_path("crps", 24, "out", year=2022, subdir="annual_monthly_averages",
                        model="graphcast-idr", native_var="precipitation")
    raw = d.output_path("mae", 24, "out", year=2022, subdir="annual_monthly_averages",
                        model="graphcast", native_var="precipitation")
    assert "/graphcast-idr/" in str(cal)
    assert "/graphcast/" in str(raw) and "/raw/" not in str(raw)   # 'raw/' segment dropped

    # Both surface as distinct models in the UI discovery (5-segment parse: model above
    # native_var, year directly above annual_monthly_averages).
    import tempfile
    from ui.services.engine import SummarySet
    root = Path(tempfile.mkdtemp()) / "africa"
    for model, metric in [("graphcast-idr", "crps"), ("graphcast", "mae")]:
        p = (root / model / "precipitation" / "chirts" / "gridded" / metric / "2022"
             / "annual_monthly_averages" / "24h.nc")
        p.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"annual_average": (("year",), [0.1])}, coords={"year": [2022]}).to_netcdf(p)
    idx = SummarySet.discover(root).index
    got = set(zip(idx.forecast, idx.metric))
    assert ("graphcast-idr", "crps") in got
    assert ("graphcast", "mae") in got


if __name__ == "__main__":
    test_pool_pairs_over_single_station_dim()
    test_parse_model_token_is_inverse_of_ui_model_token()
    test_idr_token_and_discovery_roundtrip()
    test_pool_pairs_gridded_flatten_is_order_preserving()
    test_build_samples_station_pools_and_stratifies_by_elevation()
    test_year_routes_forecast_roots_obs_and_output()
    print("ok")
