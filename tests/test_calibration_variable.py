"""Tests for the --variable spec in the gridded IDR calibration driver.

Standalone or under pytest. Pure (no data): guards that precipitation keeps its
native IDR behaviour and temperature_max correctly disables the precip-specific
assumptions (wet-day filter, ÷24, mm/day exceedance).
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.run import _VARIABLE_SPEC, _snap_grid


def test_precip_spec_preserves_native_behaviour():
    s = _VARIABLE_SPEC["precipitation"]
    assert s["to_rate"] is True                     # ÷24 mm/day -> mm/hr
    assert s["units"] == "mm/hr"
    assert s["min_wet_days"] == 50                  # wet-day filter as before
    assert s["exceedance_units"] == "mm/day"
    # thresholds are NOT in the spec dict: it is built at import, and they are a
    # per-country fact resolved at call time
    assert "exceedance" not in s
    from verification.store import exceedance_thresholds
    assert list(exceedance_thresholds("precipitation")) == [
        1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    assert s["fc_var"] == "precipitation" and s["obs_var"] == "precip"


def test_temperature_spec_disables_precip_assumptions():
    s = _VARIABLE_SPEC["temperature_max"]
    assert s["to_rate"] is False                    # no ÷24 — temperature is not a flux
    assert s["units"] == "degC"
    assert s["min_wet_days"] == 0                   # wet-day filter OFF
    # Hot-day exceedance, in the variable's OWN units — not mm/day, and not converted:
    # only a flux divides by 24 (see store.exceedance_internal). Resolved per country,
    # so the spec dict deliberately does not carry it.
    assert "exceedance" not in s
    assert s["exceedance_units"] == "degC"
    from verification.store import exceedance_thresholds
    assert list(exceedance_thresholds("temperature_max", "Atlantis")) == [37, 38, 39, 40]
    assert list(exceedance_thresholds("temperature_max", "Senegal")) == [
        37, 38, 39, 40, 41, 42, 43, 44, 45]
    # All three temperature reductions store the SAME in-file name, so the variable
    # segment of the path is the only thing distinguishing max from min from mean.
    # The spec therefore carries the in-file name only; the tree is keyed by the
    # variable's own name, and the path fragments that used to live here are gone.
    assert s["fc_var"] == "temperature_2m" and s["obs_var"] == "temp"
    tmin = _VARIABLE_SPEC["temperature_min"]
    assert tmin["fc_var"] == "temperature_2m" and tmin["obs_var"] == "temp"
    assert "fc_subdir" not in s and "obs_subdir" not in s and "obs_prefix" not in s


def test_snap_grid_fixes_float_coord_mismatch():
    """Float-repr noise in obs coords must not shrink the inner-join grid."""
    lat_clean = np.array([3.0, 3.0375, 3.075])
    lat_noisy = lat_clean + np.array([0.0, 1e-15, 2e-15])      # harmoniser-style noise
    lon = np.array([33.0, 33.0375])
    fc = xr.DataArray(np.ones((3, 2)), dims=("lat", "lon"),
                      coords={"lat": lat_noisy, "lon": lon})
    obs = xr.DataArray(np.ones((3, 2)), dims=("lat", "lon"),
                       coords={"lat": lat_clean, "lon": lon})

    # Without snapping, the inner join drops the float-mismatched rows.
    bare, _ = xr.align(fc, obs, join="inner")
    assert bare.sizes["lat"] < 3

    # Snapping obs onto the forecast grid keeps the full grid.
    snapped, _ = xr.align(fc, _snap_grid(obs, fc), join="inner")
    assert snapped.sizes["lat"] == 3 and snapped.sizes["lon"] == 2


def test_cli_accepts_a_threshold_override(tmp_path, monkeypatch):
    """`--threshold` must not crash the CLI before any lead runs.

    It used to read `_VARIABLE_SPEC[...]["exceedance"]`, a key removed when thresholds
    became per-country — a KeyError on every run that passed `--threshold`.
    """
    from calibration import run

    # set through monkeypatch so main()'s deployment defaults cannot leak into later tests
    monkeypatch.setenv("VERIFICATION_COUNTRY", "Senegal")
    monkeypatch.setenv("VERIFICATION_DATA_ROOT", str(tmp_path))
    assert run.main(["--variable", "temperature_max", "--leads", "",
                     "--threshold", "37", "38"]) == 0


if __name__ == "__main__":
    test_precip_spec_preserves_native_behaviour()
    test_temperature_spec_disables_precip_assumptions()
    test_snap_grid_fixes_float_coord_mismatch()
    print("ok")
