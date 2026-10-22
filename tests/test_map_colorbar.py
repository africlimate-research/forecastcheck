# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Diverging map colorbars are symmetric about zero.

A diverging scale claims that colour encodes signed magnitude. That claim only holds
if the two halves carry the same units per colour step — otherwise full amber can mean
+0.3 while full green means -3.5 on the same bar, and a negligible warm bias is drawn
as vividly as a twelve-times-larger cool one. No reader can correct for that by eye,
and it is invisible unless you read the tick labels on both ends.

So the range is ``[-lim, +lim]`` whatever the data does, even when that leaves half the
ramp unused on a one-signed field. Wasting half the ramp is the cheaper error.

The Plotly station maps already did this (``zmin, zmax = -m, m`` in
``ui/components/spatial_map.py``); the matplotlib grid maps did not, so the same field
was coloured by two different rules depending on whether it was gridded or gauges.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402
import xarray as xr                                                   # noqa: E402

from ui.utils.map_plot import spatial_map_figure                      # noqa: E402


def _clim(vals: np.ndarray, scale: str):
    """Colour limits actually handed to the mesh, read back off the figure."""
    da = xr.DataArray(
        vals, dims=("lat", "lon"),
        coords={"lat": np.linspace(12.0, 16.0, vals.shape[0]),
                "lon": np.linspace(-17.0, -12.0, vals.shape[1])},
    )
    fig = spatial_map_figure(vals, da, title="", cbar_label="x",
                             diverging=True, scale=scale)
    assert fig is not None
    try:
        for ax in fig.axes:
            for coll in ax.collections:
                lo, hi = coll.get_clim()
                if lo is not None and hi is not None:
                    return float(lo), float(hi)
    finally:
        matplotlib.pyplot.close(fig)
    raise AssertionError("no coloured mesh on the figure")


_RNG = np.random.default_rng(0)

#: The shapes that matter. The one-signed rows are the ones that used to come out
#: lopsided, and they are not hypothetical: a miscalibration map is negative almost
#: everywhere, which is exactly why the old code anchored zero and let each side run
#: to its own extent.
FIELDS = {
    "mostly negative (miscalibration)": _RNG.uniform(-3.5, 0.3, (40, 40)),
    "mostly negative (bias)":           _RNG.uniform(-4.0, 0.5, (40, 40)),
    "mostly positive (crps delta)":     _RNG.uniform(-0.05, 0.20, (40, 40)),
    "wholly positive":                  _RNG.uniform(0.1, 2.0, (40, 40)),
    "wholly negative":                  _RNG.uniform(-2.0, -0.1, (40, 40)),
    "balanced":                         _RNG.uniform(-1.0, 1.0, (40, 40)),
}


# --------------------------------------------------------------------------- m1
@pytest.mark.parametrize("scale", ["delta", "signed"])
@pytest.mark.parametrize("name", sorted(FIELDS))
def test_m1_diverging_limits_are_symmetric(scale, name):
    """-lim to +lim, so a bar reading -4 also reads +4."""
    lo, hi = _clim(FIELDS[name], scale)
    assert lo < 0 < hi, (name, scale, lo, hi)
    assert lo == pytest.approx(-hi, rel=1e-12, abs=1e-15), (name, scale, lo, hi)


# --------------------------------------------------------------------------- m2
@pytest.mark.parametrize("scale", ["delta", "signed"])
def test_m2_sign_flip_flips_the_limits_and_nothing_else(scale):
    """Negating the field must mirror the bar, not rescale it.

    The sharpest statement of the property: under the old anchored-zero rule the same
    magnitudes drawn with the sign flipped produced a DIFFERENT span, so "how green"
    depended on which side the data happened to fall.
    """
    field = FIELDS["mostly negative (miscalibration)"]
    lo, hi = _clim(field, scale)
    lo_flipped, hi_flipped = _clim(-field, scale)
    assert (lo, hi) == pytest.approx((lo_flipped, hi_flipped)), \
        (lo, hi, lo_flipped, hi_flipped)


# --------------------------------------------------------------------------- m3
def test_m3_correlation_stays_within_its_natural_range():
    """Correlation is symmetric too, and never claims more than [-1, 1]."""
    lo, hi = _clim(_RNG.uniform(-0.2, 0.95, (40, 40)), "correlation")
    assert lo == pytest.approx(-hi)
    assert -1.0 <= lo < 0 < hi <= 1.0, (lo, hi)


# --------------------------------------------------------------------------- m4
@pytest.mark.parametrize("scale", ["delta", "signed"])
def test_m4_a_constant_field_still_yields_a_usable_bar(scale):
    """A flat field has no spread; the bar must not collapse to zero width.

    ``percentile(|v|, 98)`` is 0 for an all-zero field, which would make vmin == vmax
    and matplotlib draw an empty scale.
    """
    lo, hi = _clim(np.zeros((20, 20)), scale)
    assert lo < hi, (lo, hi)
    assert lo == pytest.approx(-hi)


# --------------------------------------------------------------------------- m5
@pytest.mark.parametrize("scale", ["delta", "signed"])
def test_m5_nans_are_ignored_rather_than_poisoning_the_range(scale):
    """Masked ocean/out-of-country cells are NaN and must not reach the limits."""
    field = FIELDS["balanced"].copy()
    field[:10, :] = np.nan
    lo, hi = _clim(field, scale)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo == pytest.approx(-hi)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
