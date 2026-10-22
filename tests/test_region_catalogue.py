# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""The region catalogue: ids, hierarchy, and the invariants that keep data safe.

Runs standalone (``python tests/test_region_catalogue.py``) or under pytest.

The ids these tests pin are written into diagnostics files as a coordinate, so a change
to the slug rule silently orphans every file already on disk. The golden literals here
are therefore a compatibility contract, not a convenience.

Senegal's manifest ships with the code, so every test here runs without a data tree; the
multi-level rules are checked on a small synthetic shapefile.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import admin
from verification.seasons import SEASON_SLUGS

#: Senegal's 14 regions -> their departments (geoBoundaries ADM2, as committed).
_SEN_DEPARTMENTS = {
    "Dakar": ("Dakar", "Guediawaye", "Pikine", "Rufisque"),
    "Diourbel": ("Bambey", "Diourbel", "Mbacke"),
    "Fatick": ("Fatick", "Foundiougne", "Gossas"),
    "Kaffrine": ("Birkelane", "Kaffrine", "Koungheul", "Malem Hodar"),
    "Kaolack": ("Guinguineo", "Kaolack", "Nioro Du Rip"),
    "Kedougou": ("Kedougou", "Salemata", "Saraya"),
    "Kolda": ("Kolda", "Medina Yoroufoula", "Velingara"),
    "Louga": ("Kebemer", "Linguere", "Louga"),
    "Matam": ("Kanel", "Matam", "Ranerou"),
    "Saint Louis": ("Dagana", "Podor", "Saint Louis"),
    "Sedhiou": ("Bounkiling", "Goudomp", "Sedhiou"),
    "Tambacounda": ("Bakel", "Goudiry", "Koumpentoum", "Tambacounda"),
    "Thies": ("Mbour", "Thies", "Tivaoune"),
    "Ziguinchor": ("Bignona", "Oussouye", "Ziguinchor"),
}
_SEN_LEAVES = tuple(sorted(f"{admin.slugify(r)}.{admin.slugify(d)}"
                           for r, deps in _SEN_DEPARTMENTS.items() for d in deps))

_ENV_KEYS = ("VERIFICATION_COUNTRY", "VERIFICATION_DATA_ROOT", "VERIFICATION_TRUTH_PRODUCT")


class _env:
    """Set deployment env vars for a block, restoring them and the catalogue cache after."""

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


def _tiny_shapefile(directory, names, *, name_field="NAME", parents=None,
                    parent_fields=()):
    """Write a 2-column shapefile of unit squares plus a matching manifest.

    Lets the structural tests run with no real boundary data: geometry correctness is
    not what they are checking, the id/hierarchy rules are.
    """
    import geopandas as gpd
    import shapely

    rows = []
    for i, name in enumerate(names):
        record = {name_field: name,
                  "geometry": shapely.box(i, 0, i + 1, 1)}
        for field, values in zip(parent_fields, parents or ()):
            record[field] = values[i]
        rows.append(record)
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    shp = Path(directory) / "units.shp"
    gdf.to_file(shp)
    manifest = Path(directory) / admin.MANIFEST_NAME
    lines = [f'shapefile = "units.shp"', f'name_field = "{name_field}"',
             f"pool_level = {len(parent_fields) + 1}"]
    if parent_fields:
        joined = ", ".join(f'"{f}"' for f in parent_fields)
        lines.append(f"parent_fields = [{joined}]")
    manifest.write_text("\n".join(lines) + "\n")
    return directory


def test_c1_senegal_is_14_regions_over_45_departments():
    """Senegal pools at department level, with its 14 regions as the parent level."""
    with _env(VERIFICATION_COUNTRY="Senegal"):
        cat = admin.catalogue_for()
        assert cat.pool_level == 2, cat.pool_level
        assert cat.levels() == (1, 2), cat.levels()
        assert sorted(r.id for r in cat.at_level(1)) == sorted(
            admin.slugify(r) for r in _SEN_DEPARTMENTS)
        leaf_ids = tuple(r.id for r in cat.leaves())
        assert leaf_ids == (*_SEN_LEAVES, admin.UNASSIGNED_ID), leaf_ids
        # unassigned sorts last so a reader can spot it without a name lookup
        assert leaf_ids[-1] == admin.UNASSIGNED_ID
        assert cat.display_label("dakar.pikine") == "Pikine"
        assert cat.id_for("Pikine") == "dakar.pikine"
        # a department named after its region is qualified by its level noun
        assert cat.display_label("dakar") == "Dakar (region)"
        assert cat.display_label("dakar.dakar") == "Dakar (department)"
        assert cat.id_for("Dakar (department)") == "dakar.dakar"
        assert sorted(cat.leaves_under("dakar")) == [
            "dakar.dakar", "dakar.guediawaye", "dakar.pikine", "dakar.rufisque"]


def _two_level(directory):
    """A synthetic province/district catalogue with one duplicated district name."""
    _tiny_shapefile(directory, ["Alpha", "Beta", "Alpha", "Gamma"], name_field="DIST",
                    parent_fields=("PROV",), parents=(["North", "North", "South", "South"],))
    return _catalogue_from(directory, "X")


def test_c2_duplicate_leaf_names_get_distinct_ids():
    """Two districts named ``Alpha`` in different provinces must not share an id."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _two_level(tmp)
        assert cat.pool_level == 2 and cat.levels() == (1, 2), cat.levels()
        alphas = sorted(r.id for r in cat.at_level(2) if r.label == "Alpha")
        assert alphas == ["north.alpha", "south.alpha"], alphas
        assert cat.display_label("north.alpha") == "Alpha (North)"
        assert cat.display_label("north.beta") == "Beta"      # unambiguous stays bare


def test_c3_parent_chain_partitions_every_level():
    """Each parent's children partition it exactly — the roll-up's structural premise."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _two_level(tmp)
        real = [r for r in cat.leaves() if r.id != admin.UNASSIGNED_ID]
        for leaf in real:
            parent = cat.get(leaf.parent)
            assert parent is not None and parent.level == 1 and parent.parent is None
        seen = [leaf for prov in cat.at_level(1) for leaf in cat.leaves_under(prov.id)]
        assert sorted(seen) == sorted(r.id for r in real), "levels do not partition"
        assert len(seen) == len(set(seen)) == 4, seen
        # leaves_under(None) is the national total: every leaf, including unassigned
        assert sorted(cat.leaves_under(None)) == sorted(r.id for r in cat.leaves())


def test_c4_unassigned_is_reserved_and_unproducible():
    """A real region must never be able to claim the reserved id.

    ``unassigned`` carries the cells that fall inside the country mask but in no polygon.
    If a genuine region slugged to the same token the two pools would merge silently, so
    the build rejects it rather than trusting that no country has such a name.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_shapefile(tmp, ["Unassigned", "Elsewhere"])
        try:
            admin._build_rows(_read(tmp), _manifest(tmp))
        except ValueError as exc:
            assert "reserved" in str(exc) and admin.UNASSIGNED_ID in str(exc), exc
        else:
            raise AssertionError(
                "a region slugging to 'unassigned' must be rejected — left to run it is "
                "OVERWRITTEN by the reserved row and its samples merge silently")

    # a name that merely contains the token is fine; only an exact slug collides
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_shapefile(tmp, ["Unassigned District", "Elsewhere"])
        rows, _, _ = admin._build_rows(_read(tmp), _manifest(tmp))
        ids = {r.id for r in rows}
        assert "unassigned_district" in ids, ids
        assert admin.UNASSIGNED_ID in ids, ids


def test_c4b_national_is_not_the_sum_of_the_named_regions():
    """``leaves_under(None)`` must include ``unassigned`` — the national total needs it.

    Border cells inside the Natural Earth outline but outside every department polygon belong
    to the national score; anything that sums only the *named* regions is short by them. This test pins the contract so the reserved region cannot
    quietly fall out of the national roll-up.
    """
    with _env(VERIFICATION_COUNTRY="Senegal"):
        cat = admin.catalogue_for()
        national = set(cat.leaves_under(None))
        named = {r.id for r in cat.at_level(cat.pool_level) if r.id != admin.UNASSIGNED_ID}

        assert admin.UNASSIGNED_ID in national, \
            "the national roll-up must include the unassigned border cells"
        assert national == named | {admin.UNASSIGNED_ID}, national ^ (named | {admin.UNASSIGNED_ID})
        assert national - named == {admin.UNASSIGNED_ID}, \
            "national differs from the named regions by exactly the reserved region"

        # and the reserved region is reachable by name, so a caller can report the gap
        assert cat.get(admin.UNASSIGNED_ID) is not None
        assert cat.get(admin.UNASSIGNED_ID).level == cat.pool_level


def test_c4c_every_offered_region_actually_masks():
    """Every region the UI offers must filter data — a region that keeps every station
    would show national numbers under a regional heading."""
    import numpy as np

    with _env(VERIFICATION_COUNTRY="Senegal"):
        from verification.regions import (available_regions, default_region,
                                          stations_in_region)

        offered = available_regions()
        assert len(offered) == 1 + 14 + 45, len(offered)
        assert offered[0] == default_region() == "Senegal"

        lons = np.array([-14.96, -12.18, -16.27])         # Podor, Kedougou, Ziguinchor
        lats = np.array([16.50, 12.56, 12.57])
        for region, expected in (("Podor", 0), ("Kedougou (department)", 1),
                                 ("Ziguinchor (region)", 2)):
            keep = np.asarray(stations_in_region(lons, lats, region), dtype=bool)
            assert keep.sum() == 1 and keep[expected], (region, keep)
        assert np.asarray(stations_in_region(lons, lats, "Senegal")).all()


def test_c5_no_region_id_collides_with_a_season_slug():
    """Region and season vocabularies must stay disjoint.

    Curve discovery used to tell a season path segment from a region one by membership in
    SEASON_SLUGS. Ids are coordinates now rather than path segments, but the guarantee is
    cheap to keep and a collision would still confuse any reader that pattern-matches.
    """
    with _env(VERIFICATION_COUNTRY="Senegal"):
        cat = admin.catalogue_for()
        clashes = {r.id for r in cat.leaves()} & set(SEASON_SLUGS)
        assert not clashes, clashes


def test_c6_no_manifest_degrades_to_the_country_itself():
    """An agency with no boundaries onboarded gets one region, not an error.

    This is the normal state for a newly-added country, so it must degrade rather than
    fail: the dashboard offers the country only and the pipeline pools whole-domain.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with _env(VERIFICATION_COUNTRY="Nowhereland", VERIFICATION_DATA_ROOT=tmp):
            cat = admin.catalogue_for()
            assert len(cat) == 1, len(cat)
            assert cat.pool_level == 0, cat.pool_level
            only = cat.leaves()[0]
            assert only.label == "Nowhereland", only
            assert only.parent is None
            # no leaves to pool by, so the partition is empty rather than wrong
            import numpy as np
            labels = admin.assign_grid(cat, np.array([30.0, 31.0]), np.array([5.0, 6.0]))
            assert (labels == admin.OUTSIDE).all(), labels


def test_c7_slugs_and_ids_are_stable():
    """The slug rule is a compatibility contract — ids are written into data files.

    Golden literals, including accent-folding and punctuation cases. A change here
    orphans every file already on disk.
    """
    cases = {
        "Pikine": "pikine",
        "Saint Louis": "saint_louis",
        "Nioro Du Rip": "nioro_du_rip",
        "Thiès": "thies",
        "Guinea-Bissau": "guinea_bissau",
        "Côte d'Ivoire": "cote_d_ivoire",
        "  spaced  out  ": "spaced_out",
        "São Tomé": "sao_tome",
    }
    for raw, want in cases.items():
        got = admin.slugify(raw)
        assert got == want, f"{raw!r} -> {got!r}, expected {want!r}"
    # idempotent: slugging a slug changes nothing, which is what _validate relies on
    for want in cases.values():
        assert admin.slugify(want) == want, want


# ── helpers ──────────────────────────────────────────────────


def _read(directory):
    import geopandas as gpd

    return gpd.read_file(Path(directory) / "units.shp")


def _manifest(directory):
    import tomllib

    with open(Path(directory) / admin.MANIFEST_NAME, "rb") as fh:
        return tomllib.load(fh)


def _catalogue_from(directory, country):
    rows, geoms, pool = admin._build_rows(_read(directory), _manifest(directory))
    return admin.RegionCatalogue(country, rows, geoms, pool)


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


def test_c8_a_name_at_two_levels_is_disambiguated():
    """C8: a child sharing its parent's name must not collide with it in the picker.

    The picker presents every level in one flat namespace, so a district named after its
    region would otherwise look like a duplicate and be dropped.
    """
    rows = (
        admin.Region(id="alpha", label="Alpha", level=1, parent=None),
        admin.Region(id="alpha.alpha", label="Alpha", level=2, parent="alpha"),
        admin.Region(id="alpha.beta", label="Beta", level=2, parent="alpha"),
    )
    # with level nouns: the qualifier names what actually distinguishes them
    cat = admin.RegionCatalogue("X", rows, {}, pool_level=2,
                                level_names=("region", "department"))
    assert cat.display_label("alpha") == "Alpha (region)"
    assert cat.display_label("alpha.alpha") == "Alpha (department)"
    # an unambiguous name stays bare, so only the clash pays the cost
    assert cat.display_label("alpha.beta") == "Beta"
    # every label distinct — which is the property the picker depends on
    labels = [cat.display_label(r.id) for r in rows]
    assert len(set(labels)) == len(labels), labels

    # without level nouns it must STILL disambiguate rather than emit "Alpha (Alpha)"
    bare = admin.RegionCatalogue("X", rows, {}, pool_level=2)
    assert bare.display_label("alpha.beta") == "Beta"
    assert bare.display_label("alpha.alpha") != "Alpha (Alpha)"
