# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Presentation constants for the dashboard.

What is *available* (which forecasts / targets / metrics / variables /
years) is discovered from disk via
:class:`ui.services.menu.MenuIndex` — not listed here. This
module holds only cosmetics: the canonical lead-time axis, the brand
palette, units, and the token→label maps that turn engine tokens
(``graphcast``, ``rfe2``, ``mae``, ``precip``) into pretty display names.

Unknown tokens fall back to a derived label, so a newly-onboarded
forecast / target / metric still shows up (just less prettily) instead of
vanishing from the menu.
"""
from __future__ import annotations

from verification import config as _config
from ui.utils.regions import available_regions as _available_regions
from ui.utils.regions import default_region as _default_region
from verification.seasons import SEASONS as _CANON_SEASONS
from verification.seasons import seasons_for_country as _seasons_for_country

# ── Canonical lead-time axis (hours) ──────────────────────────────────────
# The day-N×24 axis shared by gridded and station results. The loader
# requests these leads; any a given run lacks are skipped.
LEAD_TIMES: list[int] = [24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144,
                         156, 168, 180, 192, 204, 216, 228, 240]

# ── Seasons — THIS DEPLOYMENT'S calendar, for the global Season selector ───
# Derived from the canonical registry in verification/seasons.py (shared with the
# diagnostics pipeline so definitions never drift), then re-keyed by display label —
# what the dropdown offers and what FilterState.season carries; the scalar pages
# aggregate the stored monthly means over these months.
#
# Narrowed to the deployment country's calendar. A cross-year season (NDJF) uses
# Nov–Dec + Jan–Feb of the *same* verification year.
SEASONS: dict[str, tuple[int, ...] | None] = {
    _CANON_SEASONS[slug][0]: _CANON_SEASONS[slug][1]
    for slug in _seasons_for_country(_config.country())
}
DEFAULT_SEASON: str = "Yearly"


#: Every canonical season by display label, regardless of country. Resolving and offering
#: are different jobs: the selector should only *offer* this deployment's calendar, but a
#: label that exists at all must still *resolve* — otherwise a caller holding a stored or
#: passed-in label silently gets None (all months) instead of the season it named.
_ALL_SEASONS: dict[str, tuple[int, ...] | None] = {
    label: months for (label, months) in _CANON_SEASONS.values()
}


def season_months(label: str) -> tuple[int, ...] | None:
    """1-based month numbers for a season label; None = Yearly (all months).

    Falls back to the full canonical registry for a label this deployment does not offer.
    ``SEASONS`` is narrowed at import time, so without it a process that reads the config
    late — a test, or anything importing settings before the deployment defaults are
    seeded — would resolve every season to None and silently score the whole year.
    """
    if label in SEASONS:
        return SEASONS[label]
    return _ALL_SEASONS.get(label)

# ── Available regions — driven by utils/regions.py (narrowed to the ───────
# deployment country when VERIFICATION_COUNTRY is set, else the full list) ──
REGIONS: list[str] = _available_regions()
DEFAULT_REGION: str = _default_region()

# ── Brand palette — 8 colors from AfriClimate AI identity guide ───────────
MODEL_PALETTE: list[str] = [
    "#0a9938",  # Jungle Green   (primary)
    "#213b7d",  # Blue           (primary)
    "#f2b029",  # Sun Gold       (primary)
    "#2e7d33",  # Fern Green     (primary)
    "#305429",  # Dark Green     (secondary)
    "#f0d161",  # Lightning      (secondary)
    "#70ba6e",  # Moss Green     (secondary)
    "#bac7c9",  # Silver         (secondary)
]

# ── Comparison-chart series colours ───────────────────────────────────────
# Warm = "the thing being compared against". Miscalibration already reads that
# way (its raw baseline is AMBER #f5b029), so Discrimination's references use the
# same amber family and its models take the cool half of the brand palette.
#
# Two measured constraints shaped this:
#   * #f2b029 and #f0d161 are dropped from the model list — #f2b029 is only
#     ΔE 0.5 from the reference amber, so a model could land on it and read as a
#     reference.
#   * A SECOND amber cannot be made colour-blind-safe against green models:
#     every dark-amber candidate scored CVD ΔE ≤ 4.5 against them (dark orange
#     and mid green converge under protan/deutan). Reference amber vs the nearest
#     model is ΔE 6.9 CVD — the skill's 6–8 floor, legal only with secondary
#     encoding.
#     Reference lines are SOLID (they read as first-class series, not annotations),
#     so the redundant channel is SHAPE, not dash: references carry
#     REFERENCE_SYMBOLS markers and a heavier stroke against the models' circles,
#     and the legend names every series. Shape survives both colour blindness and
#     greyscale print, which a hue pair in the floor band does not.
# ORDER MATTERS: charts take colours in sequence, so the first N must be the
# most separable N. With two models the old order was fine (green vs blue,
# ΔE 34), but a third model landed on Fern Green — ΔE 8.3 from Jungle Green,
# i.e. the same colour on a line. Reordered so three series sit at ΔE 17.9
# (17.4 under protanopia/deuteranopia), keeping the muted Silver last: a pale
# 3px line reads as a disabled series, not as a model. Four or more distinct
# series needs colours this six-colour brand set does not contain.
SERIES_PALETTE: list[str] = [
    "#0a9938",  # Jungle Green
    "#213b7d",  # Blue
    "#305429",  # Dark Green
    "#70ba6e",  # Moss Green
    "#2e7d33",  # Fern Green
    "#bac7c9",  # Silver
]
REFERENCE_PALETTE: list[str] = ["#f5b029", "#b25c00"]
#: Marker shapes that distinguish reference series from models (circles).
REFERENCE_SYMBOLS: list[str] = ["diamond", "square", "triangle-up", "x"]
#: Stroke widths: references sit slightly heavier than the models.
SERIES_WIDTH: float = 3.0
REFERENCE_WIDTH: float = 4.0


# ── Map palette — one source of truth for gridded AND station maps ────────
# Gridded fields render through matplotlib/cartopy (ui.utils.map_plot) and
# station fields through Plotly Scattergeo (ui.components.spatial_map,
# score_decomposition). They used to carry independent palettes, so the same
# quantity — a miscalibration map, say — came out brand green/amber over grids
# and PuOr purple/orange over stations. Both now read these constants, which live
# here (presentation only, no heavy imports) so neither renderer owns them.

#: Diverging stops, low → high: green (below baseline) through white to amber
#: (above). Used symmetrically about 0 for signed fields (bias, Δ vs a baseline,
#: miscalibration). Consumed as a matplotlib colormap and, via
#: :func:`plotly_diverging_scale`, as a Plotly colorscale.
MAP_DIVERGING_STOPS: list[str] = [
    "#2b522c", "#2f7e34", "#8fc896", "#ffffff", "#fadda4", "#f5b029", "#b25c00",
]
#: Signed metrics whose ideal is 0 (bias), where NEITHER direction is good. A
#: deliberately value-neutral cool→warm pair: under-forecast reads cool, over
#: reads warm, and the eye is not told one of them is better. Distinct from
#: MAP_DIVERGING_STOPS on purpose — green there means "better", which would make
#: a −3 °C bias look good when it is exactly as wrong as +3 °C.
#: Poles validated for protan/deutan/tritan separation and contrast.
MAP_SIGNED_STOPS: list[str] = [
    "#2166ac", "#6ba3cd", "#c6dbef", "#ffffff", "#fbc9b4", "#d6604d", "#b2182b",
]
#: Error metrics (lower is better) read forward: pale = low error, deep red = high.
MAP_SEQUENTIAL_SCALE: str = "YlOrRd"
#: Correlation (higher is better, can be negative): red = low/bad, green = high/good.
MAP_CORRELATION_SCALE: str = "RdYlGn"


def map_scale_for(metric: str) -> str:
    """Which scale FAMILY a metric's map needs — the one place that decides.

    Both renderers (cartopy grids, Plotly station points) read this so a metric
    is coloured the same way whichever map draws it:

    ``sequential``  error metrics (mae, crps): 0 is best, one direction.
    ``signed``      bias: 0 is best, both directions wrong → neutral cool/warm.
    ``correlation`` pearsonr/spearmanr: higher is better, can be negative.
    ``delta``       a difference against a baseline (miscalibration,
                    discrimination), where green genuinely means "better".
    """
    m = (metric or "").lower()
    if m in ZERO_IS_BEST_METRICS:
        return "signed"
    if m in CORRELATION_METRICS:
        return "correlation"
    return "sequential"

# Basemap context, shared by both renderers so the two map styles sit together.
MAP_OCEAN: str = "#cde6f5"
MAP_LAND: str = "#f0ede8"
MAP_COASTLINE: str = "#1f2937"
MAP_COUNTRY: str = "#6b7280"


def _plotly_scale(stops: list) -> list:
    """Colour stops in Plotly's ``[[pos, color], ...]`` form, evenly spaced.

    Same order as the matplotlib colormap built from the same list, so a gridded
    map and a station map of the same field are coloured identically.
    """
    last = len(stops) - 1
    return [[i / last, c] for i, c in enumerate(stops)]


def plotly_diverging_scale() -> list:
    """Better/worse ramp (green → amber) for differences against a baseline."""
    return _plotly_scale(MAP_DIVERGING_STOPS)


def plotly_signed_scale() -> list:
    """Value-neutral cool → warm ramp for signed metrics whose ideal is 0."""
    return _plotly_scale(MAP_SIGNED_STOPS)


# ── Direction of goodness — the single source of truth ────────────────────
# Every component that ranks models, colors a difference map, or runs a
# one-sided test derives its direction from these three sets (via the helpers
# below), so "which way is better" can never disagree between tabs.
LOWER_IS_BETTER_METRICS: set[str] = {"mae", "crps", "rmse", "seeps"}
# Correlation metrics: higher = better, and their spatial map / daily series follow the
# correlation "two-flavor" design (temporal map + spatial-pattern daily/monthly series).
CORRELATION_METRICS: set[str] = {"pearsonr", "spearmanr", "acc"}
# Signed metrics whose ideal value is 0 (bias): neither end of the scale is
# "good", so maps need a diverging scale centred at 0, and model-vs-model
# comparison must use the magnitude (|bias| lower = better).
ZERO_IS_BEST_METRICS: set[str] = {"bias"}


def higher_is_better(metric_key: str) -> bool:
    """True when a larger value means a better forecast (correlation family)."""
    return metric_key.lower() in CORRELATION_METRICS


def zero_is_best(metric_key: str) -> bool:
    """True for signed metrics whose ideal value is 0 (bias). Compare by |value|."""
    return metric_key.lower() in ZERO_IS_BEST_METRICS


def direction_note(metric_key: str) -> str:
    """Short user-facing reminder of which direction means better skill."""
    if zero_is_best(metric_key):
        return "closer to 0 is better"
    return "higher is better" if higher_is_better(metric_key) else "lower is better"

# ── Units per variable *kind* (the engine fact: precip is a flux, temp °C) ─
# Keyed by the kind FilterState carries (viz.runs.kind_of output), so
# existing consumers' ``VARIABLE_UNITS.get(f.variable)`` keeps working.
VARIABLE_UNITS: dict[str, str] = {"precip": "mm/hr", "Tmax": "°C", "Tmin": "°C"}

# ── Token → display label maps (presentation only) ────────────────────────
_VARIABLE_LABELS: dict[str, str] = {
    "precip": "Rainfall", "Tmax": "Temperature (MAX)", "Tmin": "Temperature (MIN)",
}
_FORECAST_LABELS: dict[str, str] = {
    "graphcast": "GraphCast", "ifs": "IFS",
    # AIFS is two separate forecasts, never one: the deterministic run and the
    # 26-member ensemble mean. Labelled distinctly so a chart can never imply
    # they are the same model.
    "aifs_single": "AIFS-SINGLE", "aifs_ens": "AIFS-ENS",
    "fuxi": "FUXI", "pangu": "Pangu",
    "chirps_gefs": "CHIRPS-GEFS", "ifs_ens": "IFS-ENS",
    "ifs_ens_dynamical": "IFS-ENS-Dynamical",
}
# Forecasts whose archive holds only a subset of each year's days, token -> days.
# This is the archive's own cadence, not a gap that could be filled: AIFS is
# initialised about eight times a month, so a year holds 91 days rather than 365.
# Every month is represented, so an annual mean is not seasonally skewed — but it
# rests on roughly a quarter of the samples a daily model has, which a reader
# comparing the two numbers has no way to see. Hence `sample_caveat` below.
SPARSE_SAMPLE_FORECASTS: dict[str, int] = {"aifs_single": 91, "aifs_ens": 91}
_DAYS_PER_YEAR = 365

# Forecasts whose daily window does not line up with the observed one, per
# variable. AIFS precipitation arrives pre-aggregated over a 00-00 UTC day and
# cannot be rebuilt on the 06-06 UTC observed window (the archive carries no
# sub-daily precipitation), so it is placed on the right forecast DAY while the
# 24 hours inside it sit 6 h earlier.
_WINDOW_OFFSET_HOURS: dict[tuple[str, str], int] = {
    ("aifs_single", "precip"): 6,
    ("aifs_ens", "precip"): 6,
}


def sample_caveat(tokens, variable: str | None = None) -> str:
    """One sentence naming any selected forecast that is not a full-year sample.

    Returns "" when every selection is comparable, so a caller can append it
    unconditionally. ``tokens`` may carry the ``-idr`` suffix.
    """
    bases = []
    for t in tokens or ():
        base = t[:-4] if str(t).endswith("-idr") else str(t)
        if base in SPARSE_SAMPLE_FORECASTS and base not in bases:
            bases.append(base)
    if not bases:
        return ""
    names = ", ".join(forecast_label(b) for b in bases)
    days = SPARSE_SAMPLE_FORECASTS[bases[0]]
    verb = "is" if len(bases) == 1 else "are"
    note = (f"{names} {verb} verified on {days} of {_DAYS_PER_YEAR} days "
            f"(~8 per month, every month); other models use every available day")
    offs = {_WINDOW_OFFSET_HOURS[(b, variable)]
            for b in bases if (b, variable) in _WINDOW_OFFSET_HOURS}
    if offs:
        h = sorted(offs)[0]
        note += (f". Its rainfall totals cover a 24 h window {h} h earlier than "
                 f"the observed one and cannot be rebuilt on it")
    return note


#: Labels for whole truth-source tokens. Deliberately near-empty: a token is
#: ``{product}-{target}`` and :func:`target_label` composes its label from the two
#: registries that already name each half, so onboarding a product needs no row here.
#: Entries exist only to override a composition that reads badly.
_TARGET_LABELS: dict[str, str] = {}

#: How each data shape is named in the UI. The product half comes from
#: ``verification.store.product_label``.
_TARGET_KIND_LABELS: dict[str, str] = {
    "gridded": "gridded", "station": "stations",
}
_METRIC_LABELS: dict[str, str] = {
    "mae": "MAE", "rmse": "RMSE", "bias": "Bias", "seeps": "SEEPS",
    "crps": "CRPS", "acc": "ACC", "acc_per_cell": "ACC (per-cell)",
    "pearsonr": "Pearson r", "spearmanr": "Spearman ρ",
    # Spatial = pattern correlation per time (a time series); temporal =
    # per-cell correlation over the whole period (a map). Labelled distinctly
    # so they're never read as the same quantity.
    "pearson_spatial":  "Pearson (spatial pattern)",
    "pearson_temporal": "Pearson (temporal, per-cell)",
    "spearman_spatial":  "Spearman (spatial pattern)",
    "spearman_temporal": "Spearman (temporal, per-cell)",
    "roc_auc": "ROC AUC",
}


def variable_label(kind: str) -> str:
    return _VARIABLE_LABELS.get(kind, kind)


def forecast_label(token: str) -> str:
    # An ``-idr`` suffix marks an IDR-calibrated variant: label the base model and
    # tag it ``(IDR)`` so raw and calibrated read as the same model, distinctly named.
    if token.endswith("-idr"):
        return f"{forecast_label(token[:-4])} (IDR)"
    return _FORECAST_LABELS.get(token, token.replace("_", " ").title())


#: Units of the altitude-band edges. They are METRES everywhere — the DEM is in
#: metres and each gauge's `elevation` coord is too — but the band tokens are bare
#: numbers (`0-1000`, `>3000`), so the label adds the unit.
ALTITUDE_UNIT = "m"


def band_label(band: str) -> str:
    """Display label for an altitude band: ``"0-1000"`` -> ``"0-1000 m"``.

    ``all`` is not a range and takes no unit. The token itself is unchanged — it is a
    coordinate value inside every summary and curve file, so it cannot carry the unit
    without invalidating the data; this is presentation only.
    """
    b = str(band)
    return b if b == "all" else f"{b} {ALTITUDE_UNIT}"


def target_label(token: str) -> str:
    """Display label for a truth-source token, e.g. ``chirts-gridded`` -> ``CHIRTS (gridded)``.

    Composed from the product and the data shape rather than looked up whole, so a new
    product needs no edit here. An unknown
    product still renders — :func:`~verification.store.product_label` upper-cases it —
    because the dashboard shows what is on disk, and a missing label must never be the
    reason a real tree is invisible.
    """
    from verification.store import product_label, split_truth_source

    if token in _TARGET_LABELS:
        return _TARGET_LABELS[token]
    try:
        product, kind = split_truth_source(token)
    except ValueError:
        return token.replace("_", " ").upper()
    return f"{product_label(product)} ({_TARGET_KIND_LABELS.get(kind, kind)})"


def metric_label(token: str) -> str:
    return _METRIC_LABELS.get(token, token.replace("_", " ").upper())
