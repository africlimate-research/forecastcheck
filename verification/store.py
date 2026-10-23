# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Everything the pipeline and dashboard need to know about ONE country's data.

A :class:`CountryStore` is the sole authority on where a country's files live and
what they are guaranteed to contain. It answers three questions and nothing else:

* **where** — path builders plus their exact inverse (:meth:`parse_diagnostic`),
* **what exists** — capability, derived by probing the tree rather than declared,
  so it cannot go stale relative to the disk,
* **what it means** — :meth:`open_raw_forecast` returns data in one canonical form
  regardless of which harmoniser wrote the file.

Before this module the scheme was duplicated across five implementations
(``verification.diagnostics`` roots + builders, ``calibration.run``,
``calibration.run_station``, and the UI's scalar and curve discovery). Adding a
country meant editing all of them; worse, the calibration tree carried no country
segment at all, so two agencies' ``idr/station/precipitation/2022/graphcast/24h.nc``
collided. Everything that composes or parses a data path should now come here.

**The layout**, country outermost, one shape for every agency::

    {data_root}/{domain}/forecast/{target}/{variable}/{year}/{model}/{lead}h.nc
    {data_root}/{domain}/truth/{product}/{target}/{variable}/{year}.nc
    {data_root}/{domain}/idr/{product}/{target}/{variable}/{year}/{model}/{lead}h.nc
    {data_root}/{domain}/diagnostics/{model}/{variable}/{product}/{target}/{metric}/...
    {data_root}/{domain}/assets/{dem.nc,regions/}

**Two axes, and the one rule that places them.** ``target`` (``gridded``/``station``)
is the *shape* of the data; ``product`` (``chirts``, ``era5``) names *which observational
dataset* the truth came from. The rule:

    Anything derived from observations names the product it came from.
    Forecasts do not — a forecast is truth-agnostic.

So ``forecast/`` carries the target alone while ``truth/``, ``idr/`` and
``diagnostics/`` carry both: a country can hold several products side by side, each with
its own calibration, while the forecast is stored once.

Product is deliberately **not** a per-country constant, and neither axis is ever a
branch on the country's name. Which products a country *has* is probed from disk
(:meth:`products`). Which one a *run writes to* is read off the same disk by
:func:`resolve_product` unless it is explicitly configured — **per target**, so
**specifying the country is enough** whenever a target has one product. Senegal holds
two gridded products (``chirts`` and ``era5``), so a run there names one with ``--product``.

**Ground truth is exposed at the ENGINE tier only** — narrow, single-variable,
QC-applied. The archive tier (all variables, QC flags, provenance) is deliberately
outside this interface so verification cannot depend on it.

**Capability differences are values, not branches.** A country with no gridded
product simply reports ``targets() == {"station"}`` and the menus built from
discovery show nothing for it. No caller may name a country.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import xarray as xr

from verification import config, settings

#: The two observation targets — the SHAPE of the data, not its source.
#: ``gridded`` is a field on a lat/lon grid; ``station`` is point gauges.
#:
#: The token and its on-disk segment are now the same string, on purpose. They used to
#: differ (``station`` in code, ``stations`` on disk) purely to describe the old tree,
#: and that translation layer is gone with it — one spelling, nothing to map.
GRIDDED = "gridded"
STATION = "station"
TARGETS = (GRIDDED, STATION)


@dataclass(frozen=True)
class TruthProduct:
    """One observational dataset a country can be verified against.

    Onboarding a product is a row in :data:`PRODUCTS`, not a branch anywhere. Note there
    is no ``country`` and no ``targets`` field: which countries have a product, and
    whether it supplies grids or gauges, is probed from disk (:meth:`CountryStore.products`)
    rather than declared here, so this table can never claim something the disk lacks.
    """

    name: str
    label: str


#: The public truth products this build knows by name (for their display label). It is
#: NOT the list of products a run may use: any truth folder a country holds on disk is a
#: product too (:meth:`CountryStore._check_product`), so an agency's own national data —
#: a gauge-corrected grid, its rain gauges — works without being named here.
PRODUCTS: dict[str, TruthProduct] = {
    "chirts": TruthProduct(name="chirts", label="CHIRTS"),   # CHC gridded temperature
    # ECMWF's reanalysis: a model product, not an observation. Verifying against it
    # measures agreement with another model's analysis — a second opinion, not a
    # substitute for CHIRTS.
    "era5": TruthProduct(name="era5", label="ERA5"),
    # CHC satellite-gauge rainfall (v2.0). No values over water, and its rain day runs
    # 00-00 UTC, six hours earlier than the forecasts' 06-06 UTC window.
    "chirps": TruthProduct(name="chirps", label="CHIRPS"),
}

#: Written when nothing configures a product and the disk does not decide it.
DEFAULT_PRODUCT = "chirts"


#: Separator joining a product and a target into one UI-facing token. A hyphen is safe
#: because neither vocabulary contains one — products are lowercase words and the two
#: targets are ``gridded``/``station`` — while model tokens (``aifs_single``) use ``_``.
_SOURCE_SEP = "-"


def truth_source(product: str, target: str) -> str:
    """One token naming a selectable truth source, e.g. ``chirts-gridded``.

    The dashboard's truth-source facet is a single string — it keys menus, caches and
    labels — but on disk the source is a *pair*. Composing it here (rather than in each
    of the UI's two discovery families) keeps one definition of the join, so the scalar
    index and the curve index can never disagree about what a token means.
    """
    return f"{product}{_SOURCE_SEP}{target}"


def split_truth_source(token: str) -> tuple[str, str]:
    """Inverse of :func:`truth_source`: ``"chirts-gridded"`` -> ``("chirts", "gridded")``.

    Splits from the RIGHT on the known target names, so a product containing the
    separator would still resolve. Raises on a token that names no known target, because
    a silently mis-split token would build a path into a tree that does not exist.
    """
    for target in TARGETS:
        suffix = f"{_SOURCE_SEP}{target}"
        if token.endswith(suffix):
            return token[: -len(suffix)], target
    raise ValueError(
        f"truth-source token {token!r} does not end in one of "
        f"{[f'{_SOURCE_SEP}{t}' for t in TARGETS]}")


@lru_cache(maxsize=256)
def resolve_product(data_root: Path, domain: str, target: str = GRIDDED) -> str:
    """The product to WRITE to for a country that has not configured one.

    Resolved **per target**, off the country's own ``truth/`` tree, because grids and
    gauges can come from different providers.

    Exactly one product supplying this target is unambiguous and is used. Zero (a fresh
    tree) or several (genuinely ambiguous — say which with ``--product``) fall back to
    :data:`DEFAULT_PRODUCT`.

    Cached because it touches the filesystem and ``CountryStore.current()`` is built per
    use, inside loops. The cache is per process, matching how the dashboard already
    caches discovery; a product appearing mid-run is not worth a stat call on every
    path build.
    """
    truth = Path(data_root) / domain / "truth"
    if not truth.is_dir():
        return DEFAULT_PRODUCT
    found = sorted(d.name for d in truth.iterdir()
                   if d.is_dir() and any((d / target).rglob("*.nc")))
    return found[0] if len(found) == 1 else DEFAULT_PRODUCT


def require_unambiguous_product(store, target: str = GRIDDED) -> str | None:
    """The product a run will use, or a message saying why it cannot be chosen.

    Returns None when the choice is unambiguous. Otherwise returns a ready-to-print
    error naming what the country actually holds.

    A country with two products for a target and nothing configured resolves to
    :data:`DEFAULT_PRODUCT`, which may not be one it has; a pinned product
    (``--product``) must be one this build knows or this country holds. Failing at
    argument-parsing time, naming the options, beats a FileNotFoundError minutes into a
    run.
    """
    if store.product is not None:                  # pinned: nothing to guess, but real?
        try:
            store._check_product(store.product)
        except ValueError as exc:
            return str(exc)
        return None
    have = store.products()
    if len(have) < 2:
        return None                                # 0 or 1 is unambiguous
    return (f"{store.country} holds {len(have)} truth products {sorted(have)} and none "
            f"is configured, so the run would fall back to {DEFAULT_PRODUCT!r}"
            + ("" if DEFAULT_PRODUCT in have else
               f" — which this country does not have, so every path would point at "
               f"nothing")
            + ". Pass --product, or set VERIFICATION_TRUTH_PRODUCT.")


def product_label(name: str) -> str:
    """Display label for a product, falling back to the name upper-cased.

    Falls back rather than raising so a tree containing a product this build has not
    onboarded still renders in the dashboard — discovery is driven by the disk, and a
    label is presentation, never a gate on what can be shown.
    """
    known = PRODUCTS.get(name)
    return known.label if known else str(name).upper()

#: The scalar-summary segment. Curve metrics omit it.
TRANSFORM_SUBDIR = "annual_monthly_averages"

#: The additive per-region count store, a sibling of the whole-country curve files.
BY_REGION_SUBDIR = "by_region"

#: Every segment that names a FAMILY of output rather than a region or a season.
#:
#: This is a closed vocabulary for the same reason season slugs are: the trailing path
#: segments are positional-free, so the parser tells them apart by what they are called.
#: A family segment left out of this set is read as a REGION — which is exactly what
#: happened when ``by_region`` was added without registering it, giving the dashboard a
#: phantom region whose files have entirely different dimensions.
SUBDIR_SEGMENTS = frozenset({TRANSFORM_SUBDIR, BY_REGION_SUBDIR})

#: The one place the ``{lead}h.nc`` leaf is written, and the one place it is read.
_LEAD_FILE_RE = re.compile(r"^(\d+)h\.nc$")


# ── the facet path, relative to a diagnostics root ───────────────────────────
#
# The segment ORDER below is the interface between the pipeline that writes
# diagnostics and the dashboard that discovers them. It is layout-independent — the
# layout only decides where the root is and how the target is spelled — so it lives
# here as plain functions that both sides call. Before this, the order was written out
# by hand in four places (the engine's output_path, the UI's scalar builder, its curve
# builder, and its curve parser) and kept in agreement only by convention.


def facet_relpath(*, model: str, variable: str, product: str, target: str, metric: str,
                  year, lead, subdir: str | None = None,
                  region: str | None = None, season: str | None = None) -> Path:
    """Diagnostics path for one file, relative to the diagnostics root.

    ``product`` then ``target`` — the same order and the same pair of segments the
    ``truth/`` and ``idr/`` trees use, so one rule describes all three and a reader can
    move between them without a translation table.

    ``region`` / ``season`` of None — or a season equal to the default slug — omit their
    segment, which is what keeps a whole-country yearly path free of empty placeholders.
    """
    base = (Path(model) / variable / product / target / metric / str(year))
    if subdir:
        base = base / subdir
    if region:
        base = base / region
    if season and season != _default_season():
        base = base / season
    return base / f"{lead}h.nc"


def parse_facet_relpath(path, root) -> dict | None:
    """Inverse of :func:`facet_relpath`, or None if ``path`` is not such a file.

    Returns ``subdir``/``region``/``season`` as None when their segment is absent —
    structural truth, not display defaults, so a caller can tell "whole country" from
    "a region literally named after the country". The 0-3 trailing segments are
    disambiguated by controlled vocabulary: output-family subdirs are known literals
    (:data:`SUBDIR_SEGMENTS`) and season slugs are a closed set, which is precisely why
    neither may collide with a region name.

    ``product`` needs no such vocabulary because it is **positional** — a fixed segment,
    not one of the optional trailing ones — so a product this build has never heard of
    still parses. That is deliberate: discovery must surface whatever is on disk.
    """
    p = Path(path)
    m = _LEAD_FILE_RE.match(p.name)
    if m is None:
        return None
    try:
        rel = p.parent.resolve().relative_to(Path(root).resolve()).parts
    except (ValueError, OSError):
        return None
    if not 6 <= len(rel) <= 10:
        return None
    model, variable, product, target, metric, year = rel[:6]
    subdir = region = season = None
    for seg in rel[6:]:
        if seg in SUBDIR_SEGMENTS:
            subdir = seg
        elif seg in _season_slugs():
            season = seg
        else:
            region = seg
    return {"model": model, "variable": variable, "product": product, "target": target,
            "metric": metric, "year": year, "subdir": subdir, "region": region,
            "season": season, "lead": int(m.group(1))}


@dataclass(frozen=True)
class VariableSpec:
    """Unit- and in-file-name facts about one verifiable variable.

    Every field here describes what is *inside* a file, never where it sits: the tree is
    keyed by the variable's own name, so this table no longer carries any path fragments.
    (It used to hold ``obs_subdir``/``obs_prefix``/``raw_fc_subdir``, which existed only
    to spell out the pre-product tree's directory and filename conventions.)

    ``raw_fc_var`` is the variable name a harmonised forecast file stores, and all three
    temperature reductions store the SAME one, ``temperature_2m`` — the harmoniser
    publishes max/min/mean as three separate trees whose files are indistinguishable by
    content. The path is the only thing that says which reduction you are reading, which
    is why the variable segment is authoritative. ``obs_var`` is the canonical
    observation field name (see :func:`truth_field` for why a product may differ);
    ``to_rate`` marks an obs that is a daily flux needing ÷24.
    """

    name: str
    obs_var: str
    to_rate: bool
    raw_fc_var: str
    #: Units *as written in the raw forecast file*, checked by :func:`raw_contract`.
    #: Not the units the calibration stamps on its own output — temperature raw files
    #: say "°C" while the IDR output is labelled "degC".
    raw_units: str
    #: Decision thresholds for exceedance, **in display units** — the values the
    #: calibration emits ``p_exceed`` over and the verification pools ROC / reliability
    #: / cost-loss curves at. Empty means this variable has no exceedance, and the
    #: threshold metrics simply do not apply to it.
    #:
    #: This is the ONE definition. The calibration writes ``p_exceed`` over exactly
    #: these values and the verification selects them back by label, so a second copy
    #: anywhere would let the writer and the reader disagree about which event a curve
    #: describes — silently, since a mismatched label just yields an empty selection.
    exceedance: tuple[float, ...] = ()
    #: Units of :attr:`exceedance`, for the coord attr and the dashboard's labels.
    #: Not always the variable's own units: precipitation thresholds are mm/**day**
    #: while the calibrated field is mm/hr, because a decision threshold is a daily
    #: total. :func:`exceedance_internal` does that conversion.
    exceedance_units: str = ""


#: The onboarded variables. Adding a variable is a row here, not a branch anywhere.
VARIABLES: dict[str, VariableSpec] = {
    "precipitation": VariableSpec(
        name="precipitation", obs_var="precip", to_rate=True,
        raw_fc_var="precipitation", raw_units="mm/hr",
        exceedance=(1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60),
        exceedance_units="mm/day"),
    # Hot-day thresholds. 37-40 degC brackets the range where heat starts to carry a
    # health and crop cost, so the curves answer "how well is a day above X forecast".
    "temperature_max": VariableSpec(
        name="temperature_max", obs_var="temp", to_rate=False,
        raw_fc_var="temperature_2m", raw_units="°C",
        exceedance=(37, 38, 39, 40), exceedance_units="degC"),
    # No exceedance thresholds, deliberately: warm-night exceedance tracks cloud cover
    # more than heat. Tmin keeps crps and the deterministic metrics.
    "temperature_min": VariableSpec(
        name="temperature_min", obs_var="temp", to_rate=False,
        raw_fc_var="temperature_2m", raw_units="°C"),
}


def exceedance_thresholds(variable: str,
                          country: str | None = None) -> tuple[float, ...]:
    """Decision thresholds for ``variable`` in ``country``, in display units; () if none.

    Read at CALL time, like every other per-country fact, so a second deployment in the
    same process sees its own climate rather than whichever was current at import.

    ``config/verification.toml`` (see :mod:`verification.settings`) first, then the
    variable's default in :data:`VARIABLES`.
    """
    if country is None:
        country = config.country()
    configured = settings.thresholds(variable, country)
    if configured is not None:
        # () is a real answer — "no threshold metrics for this variable" — so the miss
        # is None. Returning () for a miss would silently disable every curve.
        return configured
    return VARIABLES[variable].exceedance


def has_exceedance(variable: str, country: str | None = None) -> bool:
    """Whether the threshold metrics (roc / reliability / cost-loss) apply at all.

    The single test the pipeline and the CLI both ask, so "which variables support
    threshold curves" is answered by the registry rather than by a name comparison.
    It used to be ``variable == "precipitation"`` in two places.
    """
    return bool(exceedance_thresholds(variable, country))


def exceedance_internal(variable: str, thresholds) -> list[float]:
    """Display-unit thresholds converted to the units the calibrated field is in.

    Precipitation thresholds are mm/**day** (a decision is about a daily total) while
    the calibrated field is mm/hr, so they divide by 24. Temperature is already in the
    field's own units and passes through. Keeping the conversion next to the registry
    that declares the units is what stops the two drifting apart.
    """
    per_hour = 1.0 / 24.0 if VARIABLES[variable].to_rate else 1.0
    return [t * per_hour for t in thresholds]


@dataclass(frozen=True)
class Contract:
    """What a raw forecast file must contain, and the dim order callers get.

    ``dims`` is the **canonical order** the verification engine is built around. It is per target, not global: a station
    file leads with ``station``, a gridded file with ``prediction_timedelta``.

    Deliberately absent: ``cell_methods`` (harmonisers spell the same reduction
    differently) and ``lead_hours`` (the *filename* is the only authority on lead time —
    ``24h.nc`` may carry ``lead_hours = "30"``).
    """

    var: str
    dims: tuple[str, ...]
    units: str


def raw_contract(target: str, variable: str) -> Contract:
    """The contract for a raw forecast file of this (target, variable).

    Contracts have **no country dimension** on purpose: if a country ever needed its
    own, that would mean its harmoniser is wrong, not that the store needs a knob.
    """
    spec = VARIABLES[variable]
    dims = ((STATION, "prediction_timedelta", "time") if target == STATION
            else ("prediction_timedelta", "time", "lat", "lon"))
    return Contract(var=spec.raw_fc_var, dims=dims, units=spec.raw_units)


def truth_field(ds, variable: str) -> str:
    """Name of the observation field inside a truth file.

    ``VARIABLES[...].obs_var`` is the canonical name, but a product may name its field
    differently (the gauge files carry ``total_precipitation_rate``).

    A truth file holds exactly one observation field, so when the canonical name is absent
    and there is exactly one data variable, that variable is unambiguous and is used.
    Anything else raises rather than guessing — a file with several fields and no
    canonical name is a harmonisation error worth surfacing, not something to resolve by
    picking the first one.
    """
    want = VARIABLES[variable].obs_var
    if want in ds.data_vars:
        return want
    names = list(ds.data_vars)
    if len(names) == 1:
        return names[0]
    raise KeyError(
        f"truth file has no {want!r} and {len(names)} data variables {names}; "
        f"cannot tell which holds the observations for {variable!r}")


# ── the layout ────────────────────────────────────────────────────────────────
#
# The layout owns the directory *shape* — where a country's root is and how the
# segments below it are ordered. There is exactly one; `CountryStore` holds it rather
# than inlining it so the shape is stated once and can be swapped in a test.

@dataclass(frozen=True)
class _StandardLayout:
    """The uniform per-country layout: country outermost, one shape for all agencies."""

    data_root: Path
    domain: str

    @property
    def country_dir(self) -> Path:
        return self.data_root / self.domain

    def forecast_dir(self, target, variable, year, model) -> Path:
        # No product segment: the same forecast is scored against every product a
        # country holds, so duplicating it per product would store identical files twice.
        return self.country_dir / "forecast" / target / variable / str(year) / model

    def calibrated_dir(self, product, target, variable, year, model) -> Path:
        # IDR is FITTED against observations, so a calibration belongs to one product.
        return (self.country_dir / "idr" / product / target / variable
                / str(year) / model)

    def truth(self, product, target, variable, year) -> Path:
        return (self.country_dir / "truth" / product / target / variable
                / f"{year}.nc")

    def dem(self) -> Path | None:
        p = self.country_dir / "assets" / "dem.nc"
        return p if p.exists() else None

    def region_assets_dir(self) -> Path:
        return self.country_dir / "assets" / "regions"

    def diagnostics_root(self) -> Path:
        return self.diagnostics_root_under(self.country_dir / "diagnostics")

    def diagnostics_root_under(self, base: Path) -> Path:
        """The diagnostics root when the caller supplies its own base directory.

        The country is already above ``diagnostics/`` in this layout, so a
        caller-supplied base is the root as-is.
        """
        return Path(base)

    def truth_root(self) -> Path:
        return self.country_dir / "truth"

    def calibrated_root(self) -> Path:
        return self.country_dir / "idr"


@dataclass(frozen=True)
class CountryStore:
    """One country's data locations, capability, and canonical read form.

    Construct with :meth:`current` (from the deployment config) or
    :meth:`for_country` (explicit, for tests and tools). Facet arguments are
    keyword-only throughout so a call site reads as a description of what it wants.

    **The product is store state, not a parameter every caller must thread.** A store
    is bound to one truth product — the one a run reads and writes — so the dozens of
    existing call sites that name only a target keep working and land in the configured
    product. A caller that genuinely spans products (discovery, a cross-product
    comparison) passes ``product=`` per call or takes a second store via
    :meth:`for_product`. That way the common case cannot silently write to the wrong
    tree by forgetting an argument, because there is no argument to forget.
    """

    country: str | None
    data_root: Path
    layout: _StandardLayout
    #: The truth product this store reads and writes, or **None to derive it per
    #: target** from the country's own tree (:func:`resolve_product`) — the normal case,
    #: since grids and gauges may come from different products.
    product: str | None = None

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def for_country(cls, country: str | None, *, data_root,
                    product: str | None = None):
        """A store for an explicitly named country and root.

        ``product`` None derives it per target from the disk, which is what a caller
        wants unless it is deliberately pinning one.
        """
        root = Path(data_root)
        return cls(country=country, data_root=root,
                   layout=_StandardLayout(data_root=root, domain=config.slug(country)),
                   product=product)

    @classmethod
    def current(cls) -> "CountryStore":
        """The store for this deployment, read from the environment at CALL time.

        Nothing is resolved at import: a root resolved at import is a trap, because the
        environment has to be seeded before it.

        **The country is the only thing a deployment must specify.** An explicitly
        configured product (:func:`verification.config.truth_product`) pins every
        target; otherwise the product is left None and resolved PER TARGET at each
        call, so grids and gauges from different products need nothing configured.
        """
        country = config.country()
        root = Path(config.data_root())
        return cls.for_country(
            country, data_root=root,
            product=config.truth_product(),
        )

    def for_product(self, product: str) -> "CountryStore":
        """The same country and root, bound to a different truth product.

        How a caller holds two products at once without either store mixing them up.
        """
        self._check_product(product)
        return CountryStore(country=self.country, data_root=self.data_root,
                            layout=self.layout, product=product)

    @property
    def domain(self) -> str:
        """The path slug for this country (``senegal``; ``africa`` when unscoped)."""
        return self.layout.domain

    # ── inputs ───────────────────────────────────────────────────────────────

    def forecast_dir(self, *, target: str, variable: str, year, model) -> Path:
        """Directory of raw (uncalibrated) forecast files for one run.

        Takes no product: a forecast is truth-agnostic and is stored once however many
        products it is later scored against.
        """
        self._check_target(target)
        return self.layout.forecast_dir(target, variable, year, model)

    def calibrated_dir(self, *, target: str, variable: str, year, model,
                       product: str | None = None) -> Path:
        """Directory of IDR-calibrated forecast files for one run."""
        self._check_target(target)
        return self.layout.calibrated_dir(self._product(product, target), target, variable,
                                          year, model)

    def truth(self, *, target: str, variable: str, year,
              product: str | None = None) -> Path:
        """The engine-tier observation file: narrow, single-variable, QC applied."""
        self._check_target(target)
        return self.layout.truth(self._product(product, target), target, variable, year)

    def dem(self) -> Path | None:
        """Elevation grid for gridded altitude banding, or None if this country has none.

        Station banding never uses it — each gauge carries its own ``elevation`` — so
        a country without a DEM simply writes summaries with no altitude bands.
        """
        p = self.layout.dem()
        return p if p is not None and Path(p).exists() else None

    # ── outputs ──────────────────────────────────────────────────────────────

    def diagnostics_root(self) -> Path:
        """Root of this country's diagnostics tree — what UI discovery walks."""
        return self.layout.diagnostics_root()

    def diagnostics_root_under(self, base) -> Path:
        """Where a diagnostics tree begins under a caller-supplied ``base``.

        ``verification.run`` keeps an ``--outdir`` flag. Callers pass a base and get a
        root; they never place the domain segment themselves.
        """
        return self.layout.diagnostics_root_under(Path(base))

    def diagnostic(self, *, model: str, variable: str, target: str, metric: str,
                   year, lead, subdir: str | None = None,
                   region: str | None = None, season: str | None = None,
                   product: str | None = None) -> Path:
        """Path of one diagnostic file.

        ``subdir`` inserts the scalar-summary segment; curve metrics leave it None.
        ``region`` is an admin-1 name, omitted for the whole country; ``season`` is a
        slug from :mod:`verification.seasons`, omitted for ``yearly``. Those two
        omissions keep a whole-country yearly path free of empty placeholder segments.
        """
        return self.diagnostics_root() / self.diagnostic_relative(
            model=model, variable=variable, target=target, metric=metric, year=year,
            lead=lead, subdir=subdir, region=region, season=season, product=product)

    def diagnostic_relative(self, *, model: str, variable: str, target: str,
                            metric: str, year, lead, subdir: str | None = None,
                            region: str | None = None,
                            season: str | None = None,
                            product: str | None = None) -> Path:
        """The facet segment order, relative to :meth:`diagnostics_root`.

        Split out from :meth:`diagnostic` so the order is defined exactly once and can
        also be rendered under a caller-supplied root — ``verification.run`` keeps an
        ``--outdir`` flag. Only the root differs; the facets below never do.
        """
        self._check_target(target)
        return facet_relpath(
            model=model, variable=variable, product=self._product(product, target),
            target=target, metric=metric,
            year=year, lead=lead, subdir=subdir, region=region, season=season)

    def parse_diagnostic(self, path) -> dict | None:
        """The exact inverse of :meth:`diagnostic`, or None if ``path`` is not one.

        Returns the ``product`` found on disk, which is **not** required to equal this
        store's own — discovery walks one root and must report every product present,
        not just the one a run would write to.

        The 0–3 trailing segments are disambiguated by controlled vocabulary — the
        output-family subdirs are known literals and season slugs are a closed set —
        which is why a season slug must never be allowed to collide with an admin-1 name.
        """
        got = parse_facet_relpath(path, self.diagnostics_root())
        if got is None or got["target"] not in TARGETS:
            return None
        return got

    # ── the {lead}h.nc leaf, in one place ────────────────────────────────────

    def lead_file(self, directory, lead: int) -> Path:
        """The ``{lead}h.nc`` file inside ``directory``."""
        return Path(directory) / f"{lead}h.nc"

    def leads_in(self, directory) -> tuple[int, ...]:
        """Ascending lead hours for which a file exists in ``directory`` (() if absent)."""
        d = Path(directory)
        if not d.is_dir():
            return ()
        return tuple(sorted(int(m.group(1)) for f in d.glob("*h.nc")
                            if (m := _LEAD_FILE_RE.match(f.name))))

    def leads(self, *, target: str, variable: str, year, model,
              calibrated: bool = False) -> tuple[int, ...]:
        """Lead hours available for one run, from the raw or calibrated tree."""
        d = (self.calibrated_dir(target=target, variable=variable, year=year, model=model)
             if calibrated else
             self.forecast_dir(target=target, variable=variable, year=year,
                               model=model))
        return self.leads_in(d)

    # ── capability, probed from disk ─────────────────────────────────────────

    def products(self) -> frozenset[str]:
        """Which truth products this country actually holds observations for.

        Probed from ``truth/``, never declared, so a country gains a product the moment
        its files land and loses it the moment they do not — no registry edit, no
        redeploy. Returns names verbatim, including ones :data:`PRODUCTS` has not
        onboarded, because the disk is the authority on what exists.
        """
        root = self.layout.truth_root()
        if not root.is_dir():
            return frozenset()
        return frozenset(
            d.name for d in root.iterdir()
            if d.is_dir() and any(d.rglob("*.nc")))

    def targets(self) -> frozenset[str]:
        """Which observation targets this country actually has data for.

        Probed, never declared, so it cannot claim something the disk lacks. A
        station-only agency reports ``{"station"}`` and every gridded facet simply
        does not appear downstream.
        """
        found = set()
        for target in TARGETS:
            for base in ("forecast", "idr"):
                if self._target_tree_populated(target, base):
                    found.add(target)
                    break
        return frozenset(found)

    def variables(self, target: str) -> frozenset[str]:
        """Onboarded variables with data present for ``target``."""
        self._check_target(target)
        found = set()
        for name in VARIABLES:
            for calibrated in (False, True):
                d = (self.calibrated_dir if calibrated else self.forecast_dir)
                if any(self.leads_in(p) for p in self._year_model_dirs(d, target, name)):
                    found.add(name)
                    break
        return frozenset(found)

    # ── canonical read ───────────────────────────────────────────────────────

    def open_raw_forecast(self, *, target: str, variable: str, year, model,
                          lead: int) -> xr.Dataset:
        """Open one raw forecast file in the canonical dim order.

        The transpose is a **no-op when the file is already canonical**; only a file that
        genuinely differs (e.g. ``(prediction_timedelta, time, station)``) is reordered. This matters
        because two consumers trust producer order with no transpose of their own —
        ``pool_pairs`` and ``corr_annual_monthly_averages`` both
        ``reshape(n_time, -1)`` assuming time is axis 0 — so a mis-ordered file there
        yields plausible wrong numbers rather than an error.

        Raises ``ValueError`` on a missing variable, an unexpected dim set, or a units
        mismatch: scoring the wrong quantity silently is the failure to avoid.
        """
        contract = raw_contract(target, variable)
        path = self.lead_file(
            self.forecast_dir(target=target, variable=variable, year=year,
                              model=model), lead)
        ds = xr.open_dataset(path)
        if contract.var not in ds:
            raise ValueError(f"{path}: expected variable {contract.var!r}, "
                             f"found {sorted(ds.data_vars)}")
        da = ds[contract.var]
        if set(da.dims) != set(contract.dims):
            raise ValueError(f"{path}: expected dims {set(contract.dims)}, "
                             f"found {set(da.dims)}")
        units = da.attrs.get("units")
        if units != contract.units:
            raise ValueError(f"{path}: expected units {contract.units!r}, "
                             f"found {units!r}")
        return ds.transpose(*contract.dims, missing_dims="ignore")

    # ── internals ────────────────────────────────────────────────────────────

    def _check_target(self, target: str) -> None:
        """Reject anything that is not one of the two data shapes.

        In particular a *product* name: source and shape are separate axes.
        """
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, not {target!r}")

    def _check_product(self, product: str) -> None:
        """A product is usable if this build knows it by name or this country holds it.

        The second half lets an agency verify against data that cannot be published (its
        own gauges, a gauge-corrected national grid): copied into ``truth/{name}/``, it is
        a product, and nothing about it has to appear in this repository. A name that is
        neither is a typo, and fails here rather than as missing files minutes later.
        """
        if product in PRODUCTS or product in self.products():
            return
        held = sorted(self.products())
        raise ValueError(
            f"unknown truth product {product!r}: {self.country or 'this country'} holds "
            f"{held if held else 'no truth data yet'} (known by name: {sorted(PRODUCTS)})")

    def _product(self, override: str | None, target: str = GRIDDED) -> str:
        """This call's product: an explicit override, else the store's, else the disk.

        ``target`` matters for the last of those: a country whose grids and gauges come
        from different providers has no single right answer, so an unpinned store asks
        the disk per target rather than picking one and being wrong for the other.
        """
        if override is None and self.product is None:
            return resolve_product(self.data_root, self.layout.domain, target)
        if override is None:
            return self.product
        self._check_product(override)
        return override

    def _target_tree_populated(self, target: str, base: str) -> bool:
        """True if any ``{lead}h.nc`` exists anywhere under this target's subtree."""
        builder = self.calibrated_dir if base == "idr" else self.forecast_dir
        for name in VARIABLES:
            for d in self._year_model_dirs(builder, target, name):
                if self.leads_in(d):
                    return True
        return False

    def _year_model_dirs(self, builder, target: str, variable: str) -> list[Path]:
        """Every {year}/{model} directory that ``builder`` could produce, from disk.

        Walks up two levels from a probe path so years and models are discovered
        rather than enumerated — the store must not hold a year or model list.
        """
        probe = builder(target=target, variable=variable, year="Y", model="M")
        variable_root = probe.parent.parent
        if not variable_root.is_dir():
            return []
        return [m for y in sorted(variable_root.iterdir()) if y.is_dir()
                for m in sorted(y.iterdir()) if m.is_dir()]


def _default_season() -> str:
    """The season slug that is written as no segment at all."""
    from verification.seasons import DEFAULT_SEASON_SLUG

    return DEFAULT_SEASON_SLUG


def _season_slugs() -> frozenset[str]:
    """The closed season vocabulary that makes a season segment distinguishable."""
    from verification.seasons import SEASON_SLUGS

    return SEASON_SLUGS
