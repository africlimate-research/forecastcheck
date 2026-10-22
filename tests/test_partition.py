# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""The region partition — the precondition the whole roll-up rests on.

Runs standalone (``python tests/test_partition.py``) or under pytest.

Summing children's counts to get a parent is exact **only** if the children are disjoint
and exhaustive over the parent. These tests assert exactly that, first structurally on
synthetic geometry and then on a 0.05 degree grid over Senegal (the CHIRTS resolution),
where the Natural Earth country mask and the department polygons disagree along the border.

The country mask is **not** redefined — it is partitioned, with the border cells that
fall in no department going to the reserved ``unassigned`` region. If the labelled set ever
stops matching the Natural Earth mask cell-for-cell, every national number has moved.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import admin
from verification import regions as rg

_ENV_KEYS = ("VERIFICATION_COUNTRY", "VERIFICATION_DATA_ROOT", "VERIFICATION_TRUTH_PRODUCT")


class _env:
    """Set deployment env vars for a block, restoring them and the caches after."""

    def __init__(self, **kwargs):
        self.want = kwargs

    def __enter__(self):
        self.prev = {k: os.environ.get(k) for k in _ENV_KEYS}
        for key, value in self.want.items():
            os.environ[key] = str(value)
        admin.clear_cache()
        return self

    def __exit__(self, *exc):
        for key, value in self.prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        admin.clear_cache()
        return False


def _senegal_grid():
    """(lon, lat) of a regular 0.05 degree grid covering Senegal — no data tree needed."""
    return (np.round(np.arange(-17.6, -11.3, 0.05), 4),
            np.round(np.arange(12.3, 16.75, 0.05), 4))


def _boxes_catalogue():
    """A synthetic 2-leaf catalogue of adjacent unit squares, no real geodata."""
    import shapely

    rows = (admin.Region(id="left", label="Left", level=1, parent=None),
            admin.Region(id="right", label="Right", level=1, parent=None),
            admin.Region(id=admin.UNASSIGNED_ID, label="Unassigned", level=1, parent=None))
    geoms = {"left": shapely.box(0, 0, 1, 1), "right": shapely.box(1, 0, 2, 1)}
    return admin.RegionCatalogue("Testland", rows, geoms, pool_level=1)


def test_p1_leaves_are_disjoint_and_exhaustive_over_the_scope():
    """Structural invariant, on geometry simple enough to reason about by hand.

    Every in-scope point gets exactly one label; points inside a polygon get that
    polygon; in-scope points outside every polygon fall to ``unassigned``; out-of-scope
    points are never labelled at all. Together these are 'disjoint and exhaustive'.
    """
    cat = _boxes_catalogue()
    index = admin.leaf_index(cat)

    #        in left     in right    in neither   in neither
    xs = np.array([0.5, 1.5, 5.0, 0.5])
    ys = np.array([0.5, 0.5, 0.5, 9.0])
    in_scope = np.array([True, True, True, False])

    labels = admin._assign(cat, xs, ys, in_scope)
    assert labels[0] == index["left"], labels
    assert labels[1] == index["right"], labels
    assert labels[2] == index[admin.UNASSIGNED_ID], labels
    assert labels[3] == admin.OUTSIDE, labels

    # exhaustive: the labelled set is exactly the in-scope set
    assert np.array_equal(labels != admin.OUTSIDE, in_scope)
    # disjoint: each point carries one label, and every label is a valid leaf axis
    valid = set(index.values()) | {admin.OUTSIDE}
    assert set(labels.tolist()) <= valid


def test_p2_senegal_grid_partitions_the_country_mask_exactly():
    """The gate: the labelled set IS the Natural Earth mask, cell for cell.

    Not 'close to' — identical. If this fails, `country = union of children` has crept
    back in and every national number has shifted by the border cells.
    """
    lon, lat = _senegal_grid()
    with _env(VERIFICATION_COUNTRY="Senegal"):
        cat = admin.catalogue_for()
        labels = admin.assign_grid(cat, lon, lat)
        country = np.asarray(rg._country_keep("Senegal", lon, lat, "lat", "lon"), bool)

        assert labels.dtype == np.int16, labels.dtype
        assert labels.shape == (lat.size, lon.size), labels.shape
        assert np.array_equal(labels != admin.OUTSIDE, country), \
            "labelled cells must be exactly the Natural Earth country mask"

        index = admin.leaf_index(cat)
        counts = {r.id: int((labels == index[r.id]).sum()) for r in cat.leaves()}
        empty = [k for k, n in counts.items() if n == 0 and k != admin.UNASSIGNED_ID]
        # Guediawaye (~13 km2, in Dakar) is smaller than one 0.05 degree cell
        assert set(empty) <= {"dakar.guediawaye"}, f"departments with no cell: {empty}"
        # the leaves account for the whole national pool, with nothing left over
        assert sum(counts.values()) == int(country.sum())


def test_p3_cells_outside_the_country_are_never_labelled():
    """Out-of-domain cells stay OUTSIDE, so they cannot leak into any region's pool.

    A department polygon may claim cells the Natural Earth mask excludes; those must not be
    picked up just because a shapefile claims them — the domain is the country mask.
    """
    lon, lat = _senegal_grid()
    with _env(VERIFICATION_COUNTRY="Senegal"):
        cat = admin.catalogue_for()
        labels = admin.assign_grid(cat, lon, lat)
        country = np.asarray(rg._country_keep("Senegal", lon, lat, "lat", "lon"), bool)

        import shapely
        lon2d, lat2d = np.meshgrid(lon, lat)
        union = shapely.union_all([cat.geometry(r.id) for r in cat.leaves()
                                   if cat.geometry(r.id) is not None])
        claimed_but_outside = shapely.contains_xy(union, lon2d, lat2d) & ~country
        assert (labels[claimed_but_outside] == admin.OUTSIDE).all(), \
            "a cell outside the country mask must stay unlabelled"


def test_p4_gauges_land_in_their_department():
    """Stations use the same partition: a gauge inside Senegal gets its department."""
    with _env(VERIFICATION_COUNTRY="Senegal"):
        cat = admin.catalogue_for()
        index = admin.leaf_index(cat)
        lons = np.array([-14.96, -12.18, -16.58])       # Podor, Kedougou, Banjul
        lats = np.array([16.50, 12.56, 13.45])
        labels = admin.assign_points(cat, lons, lats)
        assert labels[0] == index["saint_louis.podor"], labels
        assert labels[1] == index["kedougou.kedougou"], labels
        assert labels[2] == admin.OUTSIDE, labels       # The Gambia, inside the outline


def test_p5_assignment_is_deterministic_and_cached():
    """Same grid twice gives the same labels, and the second call is a cache hit.

    Determinism matters because the labels decide which pool each sample joins: a
    reordering would change every count. The cache matters because the pipeline asks once
    per lead and the dashboard once per render.
    """
    lon, lat = _senegal_grid()
    with _env(VERIFICATION_COUNTRY="Senegal"):
        cat = admin.catalogue_for()
        first = admin.assign_grid(cat, lon, lat)
        second = admin.assign_grid(cat, lon, lat)
        assert first is second, "second call should hit the label cache"
        assert not first.flags.writeable, "a shared cached array must be read-only"

        admin.clear_cache()
        cat = admin.catalogue_for()
        recomputed = admin.assign_grid(cat, lon, lat)
        assert recomputed is not first
        assert np.array_equal(recomputed, first), "assignment must be deterministic"


def test_p6_concurrent_assignment_does_not_crash():
    """Several threads assigning at once must not race the lazy GEOS index build.

    Prepared-geometry spatial indexes build on first query, and two threads racing that
    build have segfaulted this process before. Distinct grids are used so the label
    cache cannot mask the race.
    """
    from concurrent.futures import ThreadPoolExecutor

    lon, lat = _senegal_grid()
    with _env(VERIFICATION_COUNTRY="Senegal"):
        admin.clear_cache()
        cat = admin.catalogue_for()

        def one(n):
            # a distinct coarse grid per worker, so every call does real GEOS work
            sub_lon = np.linspace(float(lon[0]), float(lon[-1]), 40 + n)
            sub_lat = np.linspace(float(lat[0]), float(lat[-1]), 40 + n)
            return admin.assign_grid(cat, sub_lon, sub_lat)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(one, range(8)))
        assert len(results) == 8
        for labels in results:
            assert labels.dtype == np.int16
            assert (labels >= admin.OUTSIDE).all()


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
