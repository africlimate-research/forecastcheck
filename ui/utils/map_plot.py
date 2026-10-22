# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Publication-style spatial maps (cartopy) for the ForecastCheck UI.

Light-blue ocean, beige land context, solid coastlines, dotted country
borders, °N/°E gridline labels, and a labelled colorbar — the same look as
the AfriClimate `verification-ui` redesign, so the local (discovery-driven)
build renders identical maps from local NetCDF fields.

The ocean is drawn ABOVE the data layer as well — models that emit values
over the sea get those pixels masked, so the coastline always reads clearly.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless server — no display

# Cartopy fetches Natural Earth shapefiles on first draw. On macOS the stock
# Python often can't verify the download's TLS cert; point the default HTTPS
# context at certifi's CA bundle so those one-time downloads succeed anywhere
# with internet. No-op once the files are cached under cartopy's data dir.
import ssl

try:
    import certifi

    _CTX = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda *a, **k: _CTX  # type: ignore[assignment]
except Exception:  # pragma: no cover — certifi absent → rely on system certs
    pass

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from ui.app_config.settings import (
    MAP_CORRELATION_SCALE,
    MAP_DIVERGING_STOPS,
    MAP_LAND,
    MAP_OCEAN,
    MAP_SEQUENTIAL_SCALE,
    MAP_SIGNED_STOPS,
)

OCEAN_COLOR = MAP_OCEAN
LAND_COLOR  = MAP_LAND
LABEL_COLOR = "#706f6f"   # same muted grey as the section descriptor text
LABEL_SIZE  = 6           # very small map annotations

# Register the design font (Poppins) so map labels match the page text.
_FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
for _ttf in _FONT_DIR.glob("Poppins-*.ttf"):
    font_manager.fontManager.addfont(str(_ttf))
LABEL_FONT = "Poppins" if _FONT_DIR.exists() else "sans-serif"

# Diverging scale for Δ maps: green (better / below baseline) → white → amber
# (worse), centred at zero. The stops live in ui.app_config.settings so the
# Plotly station maps colour the same field identically — see MAP_DIVERGING_STOPS.
DESIGN_DIVERGING = LinearSegmentedColormap.from_list(
    "fc_green_amber", MAP_DIVERGING_STOPS,
)
# Value-neutral cool→warm for signed metrics whose ideal is 0 (bias): neither
# pole may read as "good", which rules out the green/amber ramp above.
DESIGN_SIGNED = LinearSegmentedColormap.from_list(
    "fc_cool_warm", MAP_SIGNED_STOPS,
)

_OCEAN_50M = cfeature.OCEAN.with_scale("50m")
_LAND_50M  = cfeature.LAND.with_scale("50m")
_COAST_50M = cfeature.COASTLINE.with_scale("50m")
_BORDER_50M = cfeature.BORDERS.with_scale("50m")


def spatial_map_figure(vals, ref, title: str, cbar_label: str,
                       diverging: bool = False, scale: str | None = None):
    """Render a 2-D field on a cartopy map.

    vals : 2-D np.ndarray aligned with ``ref``'s lat/lon coordinates
    ref  : xr.DataArray providing the lat/lon coordinates
    scale : which colour family to use — ``sequential`` (error metrics),
        ``signed`` (bias: 0 ideal, neither direction good), ``correlation``
        (higher better, may be negative) or ``delta`` (difference vs a baseline,
        where green means better). See ``settings.map_scale_for``.
    diverging : legacy shorthand for ``scale="delta"``; ignored when ``scale``
        is given. Kept so existing Δ-map callers need no change.
    """
    if scale is None:
        scale = "delta" if diverging else "sequential"
    lat_name = "latitude" if "latitude" in ref.dims else "lat"
    lon_name = "longitude" if "longitude" in ref.dims else "lon"
    lats = ref[lat_name].values
    lons = ref[lon_name].values

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return None
    # Limits come from the field being drawn, so an altitude band uses its own
    # full range rather than the whole-domain one — a band whose values are much
    # larger than average (miscalibration at >3000 m) would otherwise saturate.
    basis = finite

    norm = None
    if scale in ("delta", "signed", "correlation"):
        cmap = {"delta": DESIGN_DIVERGING, "signed": DESIGN_SIGNED,
                "correlation": MAP_CORRELATION_SCALE}[scale]
        if scale == "correlation":
            # Correlation lives in [-1, 1]; keep it symmetric so 0 is the middle.
            lim = min(1.0, max(0.1, float(np.nanpercentile(np.abs(basis), 98))))
            vmin, vmax = -lim, lim
        else:
            # SYMMETRIC about 0, always: -lim to +lim, so a bar reading -4 also
            # reads +4 and the two halves carry the same degrees per colour.
            #
            # This deliberately replaces a TwoSlopeNorm that anchored 0 and let
            # each side reach its own extent. That filled the ramp when a field
            # was one-signed, but it made the colour LIE about magnitude: on a
            # miscalibration map running -3.5 to +0.3, full amber meant +0.3
            # while full green meant -3.5, so a negligible warm bias was drawn
            # as vividly as a twelve-times-larger cool one. A reader comparing
            # two patches, or the same metric across regions, cannot correct for
            # that by eye. Wasting half the ramp is the cheaper error.
            #
            # percentile(|v|, 98) rather than the signed 1st/99th: it is the rule
            # the Plotly station maps already use (ui/components/spatial_map.py),
            # so the same field drawn as a grid and as gauges now shares a scale.
            lim = float(np.nanpercentile(np.abs(basis), 98))
            if not np.isfinite(lim) or lim <= 0:
                lim = max(float(np.nanmax(np.abs(basis))), 1e-9)
            vmin, vmax = -lim, lim
    else:
        vmin = float(np.nanpercentile(basis, 2))
        vmax = float(np.nanpercentile(basis, 98))
        cmap = MAP_SEQUENTIAL_SCALE

    fig, ax = plt.subplots(
        figsize=(5.9, 4.7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    pad = 1.0
    ax.set_extent(
        [float(lons.min()) - pad, float(lons.max()) + pad,
         float(lats.min()) - pad, float(lats.max()) + pad],
        crs=ccrs.PlateCarree(),
    )

    # Context layers under the data
    ax.add_feature(_OCEAN_50M, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(_LAND_50M,  facecolor=LAND_COLOR, alpha=0.4, zorder=0)

    mesh = ax.pcolormesh(
        lons, lats, vals, transform=ccrs.PlateCarree(),
        cmap=cmap, shading="auto", zorder=2,
        **({"norm": norm} if norm is not None
           else {"vmin": vmin, "vmax": vmax}),
    )

    # Ocean AGAIN above the data so any sea values are masked and the
    # coastline stays crisp.
    ax.add_feature(_OCEAN_50M, facecolor=OCEAN_COLOR, zorder=3)
    ax.add_feature(_COAST_50M, linewidth=0.8, zorder=4)
    ax.add_feature(_BORDER_50M, linewidth=0.6, linestyle=":", zorder=4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = True
    gl.right_labels = True
    label_style = {"size": LABEL_SIZE, "color": LABEL_COLOR, "family": LABEL_FONT}
    gl.xlabel_style = label_style
    gl.ylabel_style = label_style

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.10)
    cbar.ax.tick_params(labelsize=LABEL_SIZE, colors=LABEL_COLOR)
    plt.setp(cbar.ax.get_yticklabels(), family=LABEL_FONT)
    cbar.set_label(cbar_label, size=LABEL_SIZE, color=LABEL_COLOR, family=LABEL_FONT)
    if title:
        ax.set_title(title, fontsize=11)
    fig.patch.set_facecolor("white")
    return fig
