"""Tests for the per-country data store (``verification.store``).

The store is the single owner of the on-disk path scheme. These tests pin down:

  P1-P3  path composition and its exact inverse, across every tree,
  P4     the two axes stay separate — ``target`` is the data SHAPE
         (``gridded``/``station``) and ``product`` is the observational SOURCE
         (``chirts``/``era5``), and a product name is rejected as a target,
  P5     capability is derived from disk, so a station-only country simply has
         fewer facets rather than a special case,
  P6     ``open_raw_forecast`` canonicalises dim order and is deliberately
         indifferent to ``cell_methods`` and ``lead_hours``,
  P7     nothing resolves a data root at import time.

Runs standalone (``python tests/test_store.py``) or under pytest.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.store import CountryStore

_TRAIN_YEAR = 2022
_MODEL = "graphcast"


# ── helpers ───────────────────────────────────────────────────────────────────

def _senegal(data_root="/d", product="chirts"):
    """A Senegal store pinned to one truth product."""
    return CountryStore.for_country("Senegal", data_root=data_root, product=product)


def _raw_station_ds(dim_order, *, units="mm/hr", cell_methods="whatever",
                    lead_attr=None, n_station=3, n_time=4):
    """A raw station forecast Dataset in a caller-chosen dim order.

    Values are unique per (station, time) so a wrong transpose cannot pass.
    """
    base = np.arange(n_station * n_time, dtype="float32").reshape(n_station, n_time)
    canonical = base[:, None, :]                      # (station, ptd, time)
    da = xr.DataArray(
        canonical, dims=("station", "prediction_timedelta", "time"),
        coords={"station": np.arange(n_station),
                "prediction_timedelta": [np.timedelta64(24, "h")],
                "time": np.array([np.datetime64("2022-01-01") + np.timedelta64(i, "D")
                                  for i in range(n_time)])},
        attrs={"units": units, "cell_methods": cell_methods},
    )
    da = da.assign_coords(name=("station", [f"s{i}" for i in range(n_station)]))
    ds = xr.Dataset({"precipitation": da.transpose(*dim_order)})
    if lead_attr is not None:
        ds.attrs["lead_hours"] = lead_attr
    return ds


# ── P1: path composition ──────────────────────────────────────────────────────

def test_p1_input_paths_place_the_product_only_where_truth_is_involved():
    """P1: forecast/ carries no product; truth/ and idr/ carry it above the target.

    This is the whole rule the layout rests on. A forecast is truth-agnostic, so it is
    stored once however many products it is scored against; an IDR fit and an
    observation file belong to exactly one product and say so. Getting this wrong in
    either direction is silent — a product segment on forecast/ duplicates 28 GB per
    product, and a missing one on idr/ makes a second product overwrite the first.
    """
    s = _senegal()
    # forecast: no product segment, keyed by the TARGET VARIABLE (not a forecast subdir)
    assert str(s.forecast_dir(target="gridded", variable="precipitation",
                              year=2022, model="ifs")
               ).endswith("d/senegal/forecast/gridded/precipitation/2022/ifs")
    assert str(s.forecast_dir(target="gridded", variable="temperature_max",
                              year=2022, model=_MODEL)
               ).endswith("d/senegal/forecast/gridded/temperature_max/2022/graphcast")
    assert str(s.forecast_dir(target="station", variable="precipitation",
                              year=2022, model=_MODEL)
               ).endswith("d/senegal/forecast/station/precipitation/2022/graphcast")
    # truth + idr: {product}/{target}, product first
    assert str(s.truth(target="gridded", variable="precipitation", year=2020)
               ).endswith("d/senegal/truth/chirts/gridded/precipitation/2020.nc")
    assert str(s.truth(target="station", variable="precipitation", year=2022)
               ).endswith("d/senegal/truth/chirts/station/precipitation/2022.nc")
    assert str(s.calibrated_dir(target="gridded", variable="temperature_max",
                                year=2021, model="ifs")
               ).endswith("d/senegal/idr/chirts/gridded/temperature_max/2021/ifs")
    # DEM is exists-gated (P5 pins the None case), so prove its *location* on a real tree
    with tempfile.TemporaryDirectory() as tmp:
        assets = Path(tmp) / "senegal" / "assets"
        assets.mkdir(parents=True)
        (assets / "dem.nc").touch()
        assert str(_senegal(data_root=tmp).dem()).endswith("senegal/assets/dem.nc")


def test_p1c_two_products_never_share_a_path():
    """P1c: the same country, variable and year under two products cannot collide.

    The reason the product axis exists. Before it, a country had one gridded tree and a
    second product could only be written on top of the first — so this is the assertion
    that would have failed, and the one that keeps failing if the segment is ever dropped.
    """
    chirts, era5 = _senegal(), _senegal().for_product("era5")
    v = "temperature_max"
    assert chirts.truth(target="gridded", variable=v, year=2022) != \
        era5.truth(target="gridded", variable=v, year=2022)
    assert chirts.calibrated_dir(target="gridded", variable=v, year=2022, model=_MODEL) != \
        era5.calibrated_dir(target="gridded", variable=v, year=2022, model=_MODEL)
    assert chirts.diagnostic(model=_MODEL, variable=v, target="gridded",
                             metric="mae", year=2022, lead=24) != \
        era5.diagnostic(model=_MODEL, variable=v, target="gridded",
                        metric="mae", year=2022, lead=24)
    # ...but the FORECAST they are both scored against is one file, not two
    assert chirts.forecast_dir(target="gridded", variable=v, year=2022, model=_MODEL) == \
        era5.forecast_dir(target="gridded", variable=v, year=2022, model=_MODEL)


def test_p1b_lead_filename_scheme_lives_in_one_place():
    """P1b: lead_file appends {lead}h.nc; leads_in is its inverse over a directory."""
    s = _senegal()
    d = s.forecast_dir(target="station", variable="precipitation",
                       year=2022, model=_MODEL)
    assert str(s.lead_file(d, 24)).endswith("/24h.nc")
    assert str(s.lead_file(d, 168)).endswith("/168h.nc")
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("24h.nc", "72h.nc", "168h.nc", "notalead.nc", "6h.txt"):
            (Path(tmp) / name).touch()
        assert s.leads_in(tmp) == (24, 72, 168)
        assert s.leads_in(Path(tmp) / "missing") == ()


def test_p2_standard_layout_is_country_rooted():
    """P2: the new layout puts the country outermost and is uniform across subtrees."""
    s = _senegal(product="era5")
    assert str(s.forecast_dir(target="station", variable="precipitation",
                              year=2022, model=_MODEL)
               ).endswith("d/senegal/forecast/station/precipitation/2022/graphcast")
    assert str(s.calibrated_dir(target="station", variable="precipitation",
                                year=2022, model=_MODEL)
               ).endswith("d/senegal/idr/era5/station/precipitation/2022/graphcast")
    assert str(s.truth(target="station", variable="precipitation", year=2022)
               ).endswith("d/senegal/truth/era5/station/precipitation/2022.nc")
    assert str(s.diagnostics_root()).endswith("d/senegal/diagnostics")
    # the archive tier is NOT part of the interface
    assert not hasattr(s, "archive")


# ── P3: diagnostics path + its exact inverse ──────────────────────────────────

def test_p3_diagnostic_path_matches_output_path():
    """P3: the store and verification.diagnostics.output_path build the same path.

    Two writers of the same scheme would drift silently — the pipeline would write
    somewhere discovery does not walk and the dashboard would simply show less.
    """
    from verification.diagnostics import output_path
    s = _senegal()
    cases = [
        dict(subdir="annual_monthly_averages", region=None, season=None),
        dict(subdir=None, region=None, season=None),
        dict(subdir=None, region=None, season="jaso"),
        dict(subdir=None, region="Pikine", season=None),
        dict(subdir=None, region="Pikine", season="jaso"),
        dict(subdir=None, region=None, season="yearly"),      # yearly == season-less
    ]
    prev = os.environ.get("VERIFICATION_COUNTRY")
    os.environ["VERIFICATION_COUNTRY"] = "Senegal"
    try:
        for kw in cases:
            want = output_path("roc", 120,
                               str(Path(s.data_root) / "senegal" / "diagnostics"),
                               year=2022, model="graphcast-idr", target="gridded",
                               native_var="precipitation", **kw)
            got = s.diagnostic(model="graphcast-idr", variable="precipitation",
                               target="gridded", metric="roc", year=2022,
                               lead=120, **kw)
            assert Path(got) == Path(want), f"{kw}: {got} != {want}"
    finally:
        if prev is None:
            os.environ.pop("VERIFICATION_COUNTRY", None)
        else:
            os.environ["VERIFICATION_COUNTRY"] = prev


def test_p3b_parse_diagnostic_is_the_exact_inverse():
    """P3b: parse_diagnostic round-trips the whole region x season matrix."""
    for s in (_senegal(), _senegal().for_product("era5"), _senegal(product="era5")):
        root = s.diagnostics_root()
        # Scalar summaries are always whole-domain and whole-year, so `subdir` never
        # co-occurs with a region or season segment. Only pin what is producible.
        combos = [("annual_monthly_averages", None, None)]
        combos += [(None, r, sea) for r in (None, "Pikine")
                   for sea in (None, "jaso")]
        for subdir, region, season in combos:
            p = s.diagnostic(model="graphcast-idr", variable="precipitation",
                             target="station", metric="roc", year=2022,
                             subdir=subdir, region=region, season=season, lead=72)
            got = s.parse_diagnostic(p)
            assert got is not None, p
            assert got["model"] == "graphcast-idr"
            assert got["variable"] == "precipitation"
            assert got["target"] == "station"
            assert got["metric"] == "roc"
            assert got["year"] == "2022"
            assert got["lead"] == 72
            assert got["subdir"] == subdir
            assert got["region"] == region
            assert got["season"] == season
            assert Path(root) in Path(p).parents


def test_p3c_parse_diagnostic_rejects_foreign_paths():
    """P3c: a path outside the tree, or with a non-lead leaf, parses as None."""
    s = _senegal()
    assert s.parse_diagnostic(
        "/elsewhere/graphcast/precipitation/chirts/gridded/roc/2022/24h.nc") is None
    good = s.diagnostic(model="graphcast", variable="precipitation", target="gridded",
                        metric="roc", year=2022, lead=24)
    assert s.parse_diagnostic(Path(good).parent / "summary.nc") is None
    assert s.parse_diagnostic(Path(good).parent) is None


# ── P4: target vocabulary ──────────────────────────────────

def test_p3d_output_families_never_read_as_regions():
    """P3d: every output-family subdir parses as a subdir, not as a region.

    The trailing path segments carry no positional marker, so the parser tells them apart
    by name: a segment in SUBDIR_SEGMENTS is a family, one in SEASON_SLUGS is a season,
    anything else is a region. A family left out of that set is therefore read as a
    REGION — which is what happened when the by-region count store was added without
    registering it, giving the dashboard a phantom region called ``by_region`` whose files
    have entirely different dimensions.
    """
    from verification.store import BY_REGION_SUBDIR, SUBDIR_SEGMENTS, parse_facet_relpath

    root = "/r"
    base = "/r/graphcast-idr/precipitation/chirts/gridded/roc/2022"

    assert BY_REGION_SUBDIR in SUBDIR_SEGMENTS
    from verification.store import TRANSFORM_SUBDIR
    assert TRANSFORM_SUBDIR in SUBDIR_SEGMENTS

    for subdir in sorted(SUBDIR_SEGMENTS):
        got = parse_facet_relpath(f"{base}/{subdir}/24h.nc", root)
        assert got["subdir"] == subdir, subdir
        assert got["region"] is None, f"{subdir} was read as a region"
        assert got["season"] is None, subdir

    # a genuine region still reads as one, with or without a season under it
    got = parse_facet_relpath(f"{base}/Pikine/24h.nc", root)
    assert (got["region"], got["subdir"], got["season"]) == ("Pikine", None, None), got
    got = parse_facet_relpath(f"{base}/Pikine/ndjf/24h.nc", root)
    assert (got["region"], got["season"]) == ("Pikine", "ndjf"), got

    # and no family name may collide with a season slug, for the same reason
    from verification.seasons import SEASON_SLUGS

    assert not (SUBDIR_SEGMENTS & set(SEASON_SLUGS)), SUBDIR_SEGMENTS & set(SEASON_SLUGS)


def test_p4_target_and_product_are_separate_axes():
    """P4: the data SHAPE and the observational SOURCE never stand in for each other.

    Passing a product name where a target belongs must raise rather than quietly build a
    path into a tree that does not exist.
    """
    chirts, era5 = _senegal(), _senegal(product="era5")

    # the shape segment is the shape, for every product
    for s in (chirts, era5):
        for target in ("gridded", "station"):
            p = str(s.diagnostic(model="graphcast", variable="precipitation",
                                 target=target, metric="mae", year=2022, lead=24))
            assert f"/{s.product}/{target}/" in p, p

    # ...and it round-trips back to the same pair
    got = chirts.parse_diagnostic(chirts.diagnostic(
        model="graphcast", variable="precipitation", target="gridded",
        metric="mae", year=2022, lead=24))
    assert (got["product"], got["target"]) == ("chirts", "gridded"), got

    # a product name used as a target is a programming error, not a silent path
    for s in (chirts, era5):
        for bogus in ("chirts", "era5", "stations"):
            try:
                s.diagnostic(model="graphcast", variable="precipitation", target=bogus,
                             metric="mae", year=2022, lead=24)
                raise AssertionError(f"expected ValueError for target={bogus!r}")
            except ValueError:
                pass

    # ...and an unknown product is rejected too, rather than inventing a tree
    try:
        chirts.for_product("nosuch")
        raise AssertionError("expected ValueError for an unknown product")
    except ValueError:
        pass


def test_p4c_country_alone_resolves_the_product():
    """P4c: specifying the COUNTRY is enough — the product is read off its own tree.

    A constant default would misfile silently: a run falling back to a product the
    country does not have would write a phantom tree beside the real one.
    """
    import os

    from verification.store import DEFAULT_PRODUCT, resolve_product

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # one product each, never named anywhere
        for domain, product in (("senegal", "chirts"), ("testland", "other")):
            d = root / domain / "truth" / product / "gridded" / "precipitation"
            d.mkdir(parents=True)
            (d / "2022.nc").touch()
        resolve_product.cache_clear()
        assert resolve_product(root, "senegal") == "chirts"
        assert resolve_product(root, "testland") == "other"

        # a country with NOTHING yet falls back rather than raising — a fresh tree has
        # no fact to read, and the first run has to be able to create one.
        assert resolve_product(root, "nowhere") == DEFAULT_PRODUCT

        # ...and one holding SEVERAL is genuinely ambiguous, so it falls back too and
        # the caller must say which with --product. Picking one would be a coin flip.
        d = root / "testland" / "truth" / "second" / "gridded" / "precipitation"
        d.mkdir(parents=True)
        (d / "2022.nc").touch()
        resolve_product.cache_clear()
        assert resolve_product(root, "testland") == DEFAULT_PRODUCT

        # end to end through the real config path, with NOTHING configured
        saved = {k: os.environ.get(k) for k in
                 ("VERIFICATION_COUNTRY", "VERIFICATION_DATA_ROOT",
                  "VERIFICATION_TRUTH_PRODUCT")}
        try:
            os.environ.pop("VERIFICATION_TRUTH_PRODUCT", None)
            os.environ.update({"VERIFICATION_COUNTRY": "Senegal",
                               "VERIFICATION_DATA_ROOT": str(root)})
            resolve_product.cache_clear()
            # `product` is left None and resolved PER TARGET at each call, so assert
            # on the path the store actually builds rather than on the field.
            assert CountryStore.current().product is None
            assert "/chirts/" in str(CountryStore.current().truth(
                target="gridded", variable="precipitation", year=2022))
            # an explicit setting still wins over the disk, for every target
            os.environ["VERIFICATION_TRUTH_PRODUCT"] = "era5"
            assert CountryStore.current().product == "era5"
            assert "/era5/" in str(CountryStore.current().truth(
                target="gridded", variable="precipitation", year=2022))
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.update({k: v})
            resolve_product.cache_clear()


def test_p4d_product_resolves_per_target():
    """P4d: a country whose grids and gauges come from different providers.

    With two products present, a single per-country answer is wrong for one target no
    matter which it picks, so resolution is per TARGET. Without that, "specifying the
    country is enough" quietly stops being true the moment a second product appears —
    the store falls back to the default and reads the wrong tree for one of the two
    targets, with no error.
    """
    from verification.store import resolve_product

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for product, target in (("chirts", "gridded"), ("gauges", "station")):
            d = root / "testland" / "truth" / product / target / "temperature_max"
            d.mkdir(parents=True)
            (d / "2022.nc").touch()
        resolve_product.cache_clear()

        assert resolve_product(root, "testland", "gridded") == "chirts"
        assert resolve_product(root, "testland", "station") == "gauges"

        s = CountryStore.for_country("Testland", data_root=root)    # nothing pinned
        assert s.product is None
        assert "/chirts/gridded/" in str(
            s.truth(target="gridded", variable="temperature_max", year=2022))
        assert "/gauges/station/" in str(
            s.truth(target="station", variable="temperature_max", year=2022))
        # ...and the same split holds for the calibration and the diagnostics, which
        # are the trees a wrong product would actually corrupt
        assert "/gauges/station/" in str(
            s.calibrated_dir(target="station", variable="temperature_max",
                             year=2022, model=_MODEL))
        assert "/gauges/station/" in str(
            s.diagnostic(model=_MODEL, variable="temperature_max", target="station",
                         metric="mae", year=2022, lead=24))
        assert "/chirts/gridded/" in str(
            s.diagnostic(model=_MODEL, variable="temperature_max", target="gridded",
                         metric="mae", year=2022, lead=24))
        resolve_product.cache_clear()


def test_p4b_truth_source_token_round_trips():
    """P4b: the UI's one-string facet and the two path segments are exact inverses.

    The dashboard keys menus and caches on a single token while the disk holds a pair,
    so this join is the seam between them. A token that split wrong would build a
    plausible path into the wrong product's tree.
    """
    from verification.store import split_truth_source, truth_source

    for product in ("chirts", "era5"):
        for target in ("gridded", "station"):
            token = truth_source(product, target)
            assert split_truth_source(token) == (product, target), token

    # a token naming no known target must raise, never split on a guess
    for bad in ("chirts", "chirts-grid", "gridded", ""):
        try:
            split_truth_source(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass


def test_p8_writer_and_discovery_agree_for_every_product():
    """P8: a written path always sits under the root discovery walks.

    The worst failure mode here is silent: if the writer places a segment differently
    from the reader, nothing raises — discovery just returns an empty frame and the
    dashboard quietly offers fewer options. Has to hold for every product, with and
    without an explicit --outdir base.
    """
    for s in (_senegal(), _senegal().for_product("era5"), _senegal(product="era5")):
        root = s.diagnostics_root()
        p = s.diagnostic(model="graphcast-idr", variable="precipitation",
                         target="station", metric="roc", year=2022, lead=24)
        assert root in Path(p).parents, f"{p} is not under {root}"
        assert s.parse_diagnostic(p) is not None

        # an explicit base must relocate the whole tree consistently
        base = Path("/somewhere/else")
        alt_root = s.diagnostics_root_under(base)
        alt = alt_root / s.diagnostic_relative(
            model="graphcast-idr", variable="precipitation", target="station",
            metric="roc", year=2022, lead=24)
        assert alt_root in alt.parents
        assert base in alt_root.parents or alt_root == base


# ── P5: capability derived from disk ──────────────────────────────────────────

def test_p5_capability_is_probed_not_declared():
    """P5: a station-only tree reports only the station target; absences return None."""
    with tempfile.TemporaryDirectory() as tmp:
        s = _senegal(data_root=tmp)
        assert s.targets() == frozenset()          # nothing on disk yet
        d = s.forecast_dir(target="station", variable="precipitation",
                           year=2022, model=_MODEL)
        d.mkdir(parents=True, exist_ok=True)
        (d / "24h.nc").touch()
        assert s.targets() == frozenset({"station"})
        assert s.variables("station") == frozenset({"precipitation"})
        assert s.variables("gridded") == frozenset()
        # no DEM on disk: None, never a raise
        assert s.dem() is None
        assert s.leads(target="station", variable="precipitation",
                       year=2022, model=_MODEL) == (24,)


# ── P6: canonicalising open ───────────────────────────────────────────────────

def test_p6_open_raw_forecast_canonicalises_dim_order():
    """P6: a non-canonical file is transposed onto the canonical order, values intact."""
    canonical = ("station", "prediction_timedelta", "time")
    other_order = ("prediction_timedelta", "time", "station")
    with tempfile.TemporaryDirectory() as tmp:
        s = _senegal(data_root=tmp)
        d = s.forecast_dir(target="station", variable="precipitation",
                           year=2022, model=_MODEL)
        d.mkdir(parents=True, exist_ok=True)
        want = _raw_station_ds(canonical)
        _raw_station_ds(other_order).to_netcdf(d / "24h.nc")
        got = s.open_raw_forecast(target="station", variable="precipitation",
                                  year=2022, model=_MODEL, lead=24)
        assert got["precipitation"].dims == canonical
        np.testing.assert_array_equal(got["precipitation"].values,
                                      want["precipitation"].values)


def test_p6b_open_raw_forecast_is_a_noop_when_already_canonical():
    """P6b: a canonical file is returned untransposed and unchanged."""
    canonical = ("station", "prediction_timedelta", "time")
    with tempfile.TemporaryDirectory() as tmp:
        s = _senegal(data_root=tmp)
        d = s.forecast_dir(target="station", variable="precipitation",
                           year=2022, model=_MODEL)
        d.mkdir(parents=True, exist_ok=True)
        want = _raw_station_ds(canonical)
        want.to_netcdf(d / "24h.nc")
        got = s.open_raw_forecast(target="station", variable="precipitation",
                                  year=2022, model=_MODEL, lead=24)
        assert got["precipitation"].dims == canonical
        np.testing.assert_array_equal(got["precipitation"].values,
                                      want["precipitation"].values)


def test_p6c_open_ignores_cell_methods_and_lead_hours():
    """P6c: both harmoniser cell_methods strings pass, and lead_hours is not consulted.

    Harmonisers spell the same reduction differently, so gating on it would falsely
    reject valid files. The filename is the only authority on lead, so
    lead_hours="30" on 24h.nc is fine.
    """
    cm_a = "prediction_timedelta: sum (4 × 6hr steps) / 24"
    cm_b = "prediction_timedelta: trapezoidal sum (5 x 6hr samples) / 24"
    canonical = ("station", "prediction_timedelta", "time")
    with tempfile.TemporaryDirectory() as tmp:
        s = _senegal(data_root=tmp)
        d = s.forecast_dir(target="station", variable="precipitation",
                           year=2022, model=_MODEL)
        d.mkdir(parents=True, exist_ok=True)
        for lead, cm, attr in ((24, cm_a, None), (72, cm_b, "30")):
            _raw_station_ds(canonical, cell_methods=cm, lead_attr=attr).to_netcdf(
                d / f"{lead}h.nc")
            got = s.open_raw_forecast(target="station", variable="precipitation",
                                      year=2022, model=_MODEL, lead=lead)
            assert got["precipitation"].dims == canonical


def test_p6d_open_rejects_a_units_mismatch():
    """P6d: wrong units fail loudly rather than silently scoring the wrong quantity."""
    canonical = ("station", "prediction_timedelta", "time")
    with tempfile.TemporaryDirectory() as tmp:
        s = _senegal(data_root=tmp)
        d = s.forecast_dir(target="station", variable="precipitation",
                           year=2022, model=_MODEL)
        d.mkdir(parents=True, exist_ok=True)
        _raw_station_ds(canonical, units="mm/day").to_netcdf(d / "24h.nc")
        try:
            s.open_raw_forecast(target="station", variable="precipitation",
                                year=2022, model=_MODEL, lead=24)
            raise AssertionError("expected a units mismatch to raise")
        except ValueError as exc:
            assert "mm/day" in str(exc) and "mm/hr" in str(exc)


# ── P7: no import-time root resolution ────────────────────────────────────────

def test_p7_no_root_resolved_at_import():
    """P7: importing the store with no env set must not resolve or require a root.

    ui/backend/config/data_path.py already resolves DATA_DIR at import, which is why
    ui/app.py must seed the env first; the store must not add a second such trap or
    tests/test_shim.py::test_ui_import_smoke starts failing.
    """
    import importlib
    saved = {k: os.environ.get(k)
             for k in ("VERIFICATION_COUNTRY", "VERIFICATION_DATA_ROOT")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        import verification.store as st
        importlib.reload(st)
        # module-level names must not include a resolved Path root
        for name, val in vars(st).items():
            if isinstance(val, Path) and not name.startswith("__"):
                raise AssertionError(f"{name} resolves a Path at import: {val}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_p7b_current_reads_env_at_call_time():
    """P7b: CountryStore.current() follows VERIFICATION_COUNTRY at call time."""
    saved = {k: os.environ.get(k)
             for k in ("VERIFICATION_COUNTRY", "VERIFICATION_DATA_ROOT")}
    try:
        os.environ["VERIFICATION_DATA_ROOT"] = "/tmp/root-a"
        os.environ["VERIFICATION_COUNTRY"] = "Senegal"
        a = CountryStore.current()
        os.environ["VERIFICATION_COUNTRY"] = "New Country"
        b = CountryStore.current()
        assert a.country == "Senegal" and b.country == "New Country"
        assert a.domain == "senegal" and b.domain == "new_country"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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



def test_p8_a_product_held_on_disk_is_usable_without_being_named():
    """An agency's own data — say rain gauges in truth/natgauges/ — is a product.

    It cannot be published, so it is never listed in PRODUCTS: holding it on disk is
    enough. A name the country does not hold, and this build does not know, is a typo
    and still fails, naming what the country does hold.
    """
    from verification.store import PRODUCTS, STATION, require_unambiguous_product

    assert "natgauges" not in PRODUCTS
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = root / "testland" / "truth" / "natgauges" / "station" / "precipitation"
        d.mkdir(parents=True)
        (d / "2022.nc").touch()
        s = CountryStore.for_country("Testland", data_root=root)

        held = s.for_product("natgauges")                    # held on disk: accepted
        assert "/natgauges/station/" in str(
            held.truth(target=STATION, variable="precipitation", year=2022))
        assert s.for_product("chirps").product == "chirps"  # known by name: accepted

        try:
            s.for_product("natguages")                       # a typo: refused
            raise AssertionError("expected ValueError for a misspelt product")
        except ValueError as exc:
            assert "natgauges" in str(exc), str(exc)        # ...naming what is held

        # The early CLI check (require_unambiguous_product) applies the same rule to a
        # product pinned with --product, so a typo stops before any work is done.
        pinned = CountryStore.for_country("Testland", data_root=root, product="natguages")
        assert "unknown truth product" in (require_unambiguous_product(pinned, STATION) or "")
        pinned = CountryStore.for_country("Testland", data_root=root, product="natgauges")
        assert require_unambiguous_product(pinned, STATION) is None

if __name__ == "__main__":
    sys.exit(1 if _run() else 0)


def test_p4e_thresholds_and_bands_are_per_country():
    """P4e: thresholds and strata resolve per country, at CALL time.

    ``calibration.run._VARIABLE_SPEC`` is built at import, so it deliberately does not
    carry thresholds: freezing them there would give every deployment whichever country
    happened to be configured when the module loaded.
    """
    import os

    from verification.diagnostics import ALTITUDE_BANDS, altitude_bands, thresholds_for
    from verification.store import VARIABLES, exceedance_thresholds, has_exceedance

    assert exceedance_thresholds("temperature_max", "Senegal") == (
        37, 38, 39, 40, 41, 42, 43, 44, 45)
    # a country with nothing configured falls back to the variable's own row
    assert exceedance_thresholds("temperature_max", "Nowhere") == \
        VARIABLES["temperature_max"].exceedance
    # ...and one with no thresholds anywhere stays that way for every country
    assert not has_exceedance("temperature_min", "Senegal")
    assert not has_exceedance("temperature_min", "Nowhere")

    # the spec dict must NOT carry thresholds, or they freeze at import
    from calibration.run import _VARIABLE_SPEC
    assert "exceedance" not in _VARIABLE_SPEC["temperature_max"]

    # end to end through the environment, both axes at once
    saved = os.environ.get("VERIFICATION_COUNTRY")
    try:
        os.environ["VERIFICATION_COUNTRY"] = "Senegal"
        assert thresholds_for("temperature_max") == [37, 38, 39, 40, 41, 42, 43, 44, 45]
        assert ">100" in altitude_bands()
        os.environ["VERIFICATION_COUNTRY"] = "Nowhere"
        assert altitude_bands() == ALTITUDE_BANDS
    finally:
        if saved is None:
            os.environ.pop("VERIFICATION_COUNTRY", None)
        else:
            os.environ["VERIFICATION_COUNTRY"] = saved
