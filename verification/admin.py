# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""A country's administrative regions, as one resolved catalogue.

This module answers two questions and hides everything behind them:

* **what regions does this country have** — :class:`RegionCatalogue`, a hierarchy of
  :class:`Region` rows with stable ids, display labels and parent links;
* **which region does each grid cell / gauge belong to** — :func:`assign_grid` and
  :func:`assign_points`, which return a *partition* of the country's domain.

Everything downstream (the pipeline's pooling, the path segments, the dashboard's
selector, the masks) deals only in region **ids**. Shapefile paths, differing name
fields, differing numbers of admin levels and text encoding quirks are confined to the
manifest loader here, so onboarding a country at admin level 3 rather than level 1 is a
data change, not a code change.

**Only the finest level's shapefile is read.** Coarser levels are synthesised from the
parent-name columns that the same file already carries (GADM's ``NAME_1``/``NAME_2``),
and a parent's geometry is the union of its children's. That is not a shortcut: it makes
the displayed boundary of a province *identical by construction* to the set of
municipalities whose counts are summed to score it, which is the invariant the whole
roll-up depends on.

**The partition, and why the country mask is not redefined.** Verification pools samples
per region and sums them back to get any coarser region. That is exact only if the
children are disjoint *and* exhaustive over the parent. The Natural Earth country mask
and the union of the admin polygons disagree on border cells (different datasets, and
fractional overlap vs centre containment), so redefining the country as the union of its
children would silently move every national number.

So the country mask is left exactly as it is and *partitioned* instead: every cell inside
it is assigned to one leaf, and the ones matching no polygon go to the reserved
``unassigned`` region. Children are then disjoint and exhaustive over the **existing**
country mask, and the roll-up is exact.

A gridded obs grid is a bounding box that reaches beyond the country; summing regions
scores the national curves over the country only, matching the scalar metrics.
"""
from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verification import config, regions as _regions

#: Label array value for a cell/gauge inside the country but in no leaf polygon.
#: A real region in the catalogue (so it carries counts and is summed into the
#: national total) rather than a dropped sample — see the module docstring.
UNASSIGNED_ID = "unassigned"

#: Label array value for a cell/gauge outside the country's domain entirely.
#: Excluded from every pool, exactly as such cells are today.
OUTSIDE = -1

#: Manifest filename, read from the country's region-assets directory.
MANIFEST_NAME = "regions.toml"


def slugify(name: str) -> str:
    """One path- and coordinate-safe token for an administrative name.

    Deterministic and stable: accents are folded to ASCII, everything outside
    ``[a-z0-9]`` collapses to a single ``_``, and the result is stripped of leading and
    trailing separators. Stability matters more than prettiness — the slug ends up in
    region ids that are written into data files, so a change to this function silently
    orphans every file already on disk.
    """
    folded = unicodedata.normalize("NFKD", str(name))
    ascii_only = folded.encode("ascii", "ignore").decode("ascii").lower()
    out = []
    for ch in ascii_only:
        out.append(ch if ch.isalnum() else "_")
    return "_".join(part for part in "".join(out).split("_") if part)


@dataclass(frozen=True)
class Region:
    """One administrative unit, at any level of one country's hierarchy.

    ``id`` is the dotted chain of slugs from the top level down
    (``province.district.unit``), which makes it unique **by construction** — a bare-name
    id would let two same-named units overwrite each other's diagnostics. ``label``
    is the leaf name alone, for display; :meth:`RegionCatalogue.display_label` adds the
    parent in parentheses only where the bare name is ambiguous.

    ``level`` is the source's own admin level (2 = Senegal's departments).
    """

    id: str
    label: str
    level: int
    parent: str | None


class RegionCatalogue:
    """One country's region hierarchy, resolved from its manifest.

    The **leaves** are the level the pipeline pools at; every coarser level is derived,
    so adding a level to the dashboard costs no recomputation — a parent's counts are the
    sum of its children's. A country with no manifest gets a degenerate one-node
    catalogue (the country itself, no leaves to pool by), which is the correct behaviour
    for an agency whose boundaries are not onboarded yet.

    Geometries exist for leaves only; a parent is drawn as the union of its children
    (:meth:`geometry`), computed lazily and cached.
    """

    def __init__(self, country: str | None, rows: tuple[Region, ...],
                 geometries: dict, pool_level: int,
                 level_names: tuple[str, ...] = ()):
        self.country = country
        self.pool_level = pool_level
        #: Optional per-level nouns from the manifest ("region", "department"), used to
        #: disambiguate a name that exists at two levels. Empty is fine — the label then
        #: falls back to the parent, and to the bare name.
        self.level_names = tuple(level_names)
        self._by_id = {r.id: r for r in rows}
        self._leaf_geoms = geometries
        self._children: dict[str | None, list[str]] = {}
        for r in rows:
            self._children.setdefault(r.parent, []).append(r.id)
        self._union_cache: dict[str, object] = {}
        self._lock = threading.RLock()

    # ── the hierarchy ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, region_id: str) -> Region | None:
        """The row for ``region_id``, or None if this country has no such region."""
        return self._by_id.get(region_id)

    def leaves(self) -> tuple[Region, ...]:
        """The regions the pipeline pools by, in a fixed order.

        The order is the ``region`` coordinate written into every diagnostics file, so it
        must not depend on iteration order of anything: it is sorted by id, with
        ``unassigned`` last so a reader can spot it without a name lookup.
        """
        leaves = [r for r in self._by_id.values() if r.level == self.pool_level]
        return tuple(sorted(leaves, key=lambda r: (r.id == UNASSIGNED_ID, r.id)))

    def at_level(self, level: int) -> tuple[Region, ...]:
        """Every region at one admin level, sorted by id."""
        return tuple(sorted((r for r in self._by_id.values() if r.level == level),
                            key=lambda r: r.id))

    def levels(self) -> tuple[int, ...]:
        """Admin levels this catalogue exposes, coarsest first."""
        return tuple(sorted({r.level for r in self._by_id.values()}))

    def children_of(self, region_id: str | None) -> tuple[str, ...]:
        """Ids of the regions directly below ``region_id`` (None = the top level)."""
        return tuple(sorted(self._children.get(region_id, ())))

    def leaves_under(self, region_id: str | None) -> tuple[str, ...]:
        """Leaf ids belonging to ``region_id``, at any depth — the set to sum.

        This is the roll-up rule in one call: a province's counts are the sum of these
        leaves' counts. ``None`` returns every leaf, i.e. the national total.
        """
        if region_id is None:
            return tuple(r.id for r in self.leaves())
        row = self._by_id.get(region_id)
        if row is None:
            return ()
        if row.level == self.pool_level:
            return (region_id,)
        out: list[str] = []
        for child in self.children_of(region_id):
            out.extend(self.leaves_under(child))
        return tuple(out)

    def id_for(self, name: str) -> str | None:
        """Resolve a user-facing region name to its id, or None if this country has none.

        Accepts an id outright, a display label (``"Pikine"`` — what the CLI offers
        and what the pre-existing on-disk region directories are named), or the qualified
        label :meth:`display_label` produces. Keeping the translation here means no caller
        has to know that ids are slugged and labels are not.
        """
        if name in self._by_id:
            return name
        for row in self._by_id.values():
            if name in (row.label, self.display_label(row.id)):
                return row.id
        slug = slugify(name)
        return slug if slug in self._by_id else None

    def display_label(self, region_id: str) -> str:
        """Human label, qualified only when the bare name is ambiguous.

        Ambiguity is checked across **every level**, not within one. The picker presents
        all levels in a single flat namespace — the coarse ones as pills, the rest in a
        dropdown — so a name that exists at two levels is ambiguous to the reader even
        though the ids differ (a district inside a same-named region).

        The qualifier is the level noun when the manifest supplies one ("X (department)"),
        else the parent's name.
        """
        row = self._by_id.get(region_id)
        if row is None:
            return region_id
        same = [r for r in self._by_id.values() if r.label == row.label]
        if len(same) < 2:
            return row.label
        # level_names is indexed from the TOP level present, not from 0
        top = min(r.level for r in self._by_id.values() if r.id != UNASSIGNED_ID)
        i = row.level - top
        if 0 <= i < len(self.level_names):
            return f"{row.label} ({self.level_names[i]})"
        parent = self._by_id.get(row.parent) if row.parent else None
        if parent is not None and parent.label != row.label:
            return f"{row.label} ({parent.label})"
        return row.label

    # ── geometry ─────────────────────────────────────────────────────────────

    def geometry(self, region_id: str):
        """Polygon for ``region_id`` — a leaf's own, or the union of its leaves.

        Deriving a parent from its children is what keeps the drawn boundary and the
        summed counts describing the same area. ``unassigned`` has no geometry (it is
        defined by exclusion, not by a polygon) and returns None.
        """
        if region_id in self._leaf_geoms:
            return self._leaf_geoms[region_id]
        with self._lock:
            if region_id in self._union_cache:
                return self._union_cache[region_id]
            import shapely

            parts = [self._leaf_geoms[i] for i in self.leaves_under(region_id)
                     if i in self._leaf_geoms]
            geom = shapely.union_all(parts) if parts else None
            self._union_cache[region_id] = geom
            return geom


# ── manifest loading ─────────────────────────────────────────────────────────


def _read_manifest(directory: Path) -> dict | None:
    """Parse ``regions.toml`` in ``directory``, or None when there isn't one.

    A malformed manifest is a hard error: unlike a missing one (a country simply not
    onboarded yet), a broken one means the agency *intended* regions and would otherwise
    silently get whole-country-only output.
    """
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        return None
    import tomllib

    with open(path, "rb") as fh:
        manifest = tomllib.load(fh)
    for key in ("shapefile", "name_field"):
        if key not in manifest:
            raise ValueError(f"{path}: missing required key {key!r}")
    return manifest


def _build_rows(gdf, manifest: dict) -> tuple[tuple[Region, ...], dict, int]:
    """Turn one shapefile's rows into the full hierarchy plus the leaf geometries.

    ``parent_fields`` names the ancestor columns in the same table, ordered coarsest
    first, so a leaf's id chain is ``[*parent_fields, name_field]`` slugged and joined.
    Ancestors are synthesised from those columns — no extra shapefile is read, and a
    parent cannot disagree with the children it is derived from.
    """
    name_field = manifest["name_field"]
    parent_fields = list(manifest.get("parent_fields", ()))
    pool_level = int(manifest.get("pool_level", len(parent_fields) + 1))
    top_level = pool_level - len(parent_fields)
    if top_level < 1:
        raise ValueError(
            f"pool_level {pool_level} is too shallow for {len(parent_fields)} "
            f"parent field(s); the top level would be {top_level}")

    rows: dict[str, Region] = {}
    geometries: dict = {}
    for _, record in gdf.iterrows():
        chain = [slugify(record[f]) for f in parent_fields]
        chain.append(slugify(record[name_field]))
        if not all(chain):
            raise ValueError(
                f"empty name in the id chain for row {dict(record.drop('geometry'))!r}; "
                f"check {name_field!r} and {parent_fields!r}")
        # Register every ancestor on the way down, then the leaf itself.
        for depth in range(len(chain)):
            region_id = ".".join(chain[:depth + 1])
            if region_id in rows:
                continue
            field = parent_fields[depth] if depth < len(parent_fields) else name_field
            rows[region_id] = Region(
                id=region_id,
                label=str(record[field]),
                level=top_level + depth,
                parent=".".join(chain[:depth]) if depth else None,
            )
        leaf_id = ".".join(chain)
        if leaf_id == UNASSIGNED_ID:
            # Caught here rather than in _validate because the reserved row is inserted
            # below into the same dict: left to run, it would OVERWRITE this region and
            # silently merge its samples into the unassigned pool.
            raise ValueError(
                f"region {record[name_field]!r} slugs to the reserved id "
                f"{UNASSIGNED_ID!r}; rename it in the source or add a parent level")
        geometries[leaf_id] = record.geometry

    # The reserved sibling that makes the leaves exhaustive over the country mask.
    rows[UNASSIGNED_ID] = Region(id=UNASSIGNED_ID, label="Unassigned",
                                 level=pool_level, parent=None)
    return tuple(rows.values()), geometries, pool_level


def _validate(catalogue: RegionCatalogue) -> None:
    """Fail loudly at build time on anything that would corrupt data later.

    Each of these would otherwise surface as wrong numbers rather than an error: a
    duplicate id silently merges two regions' diagnostics; a slug colliding with a season
    slug corrupts path parsing; a region id containing ``/`` or ``.`` beyond the chain
    separator breaks the coordinate round-trip.
    """
    from verification.seasons import SEASON_SLUGS

    ids = [r.id for r in catalogue.at_level(catalogue.pool_level)]
    if len(ids) != len(set(ids)):
        seen, dupes = set(), set()
        for i in ids:
            (dupes if i in seen else seen).add(i)
        raise ValueError(f"duplicate region ids: {sorted(dupes)}")

    for region in catalogue.at_level(catalogue.pool_level):
        if region.id == UNASSIGNED_ID:
            continue
        for part in region.id.split("."):
            if not part or part != slugify(part):
                raise ValueError(f"region id {region.id!r} is not stably slugged")
            if part == UNASSIGNED_ID:
                raise ValueError(
                    f"region id {region.id!r} collides with the reserved "
                    f"{UNASSIGNED_ID!r} region")
        if region.id in SEASON_SLUGS:
            raise ValueError(
                f"region id {region.id!r} collides with a season slug; season slugs are "
                f"a closed vocabulary precisely so this cannot happen")


# ── the public entry point, cached per country ───────────────────────────────

_CACHE: dict[tuple, RegionCatalogue] = {}
_CACHE_LOCK = threading.RLock()


def region_assets_dir(country: str | None, store=None) -> Path | None:
    """Directory holding this country's ``regions.toml``, or None if there is none.

    The **data tree wins** over the repo: an agency can drop their own official boundary
    into ``{country}/assets/regions/`` and override a copy bundled with the code, which
    is the opposite of what the old registry allowed.
    """
    if not country:
        return None
    if store is None:
        from verification.store import CountryStore

        store = CountryStore.current()
    candidates = []
    layout_dir = getattr(store.layout, "region_assets_dir", None)
    if layout_dir is not None:
        candidates.append(Path(layout_dir()))
    candidates.append(
        Path(__file__).resolve().parent / "assets" / "admin_boundaries"
        / config.slug(country))
    for directory in candidates:
        if (directory / MANIFEST_NAME).exists():
            return directory
    return None


def catalogue_for(country: str | None = None, *, store=None) -> RegionCatalogue:
    """The resolved region catalogue for ``country`` (default: the deployment country).

    Cached per country — the shapefile read and the geometry build are expensive, and
    every pooling pass and every dashboard render wants the same answer. ``geopandas`` is
    imported lazily so merely listing region names stays cheap.
    """
    name = country if country is not None else config.country()
    key = (name,)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit

        directory = region_assets_dir(name, store=store)
        manifest = _read_manifest(directory) if directory else None
        if manifest is None:
            # No boundaries onboarded: the country is its own only region. Degrades
            # rather than failing, which is the right answer for a new agency.
            top = Region(id=slugify(name) if name else "africa",
                         label=str(name or "Africa"), level=0, parent=None)
            catalogue = RegionCatalogue(name, (top,), {}, pool_level=0)
        else:
            import geopandas as gpd

            with _regions._GEOS_LOCK:
                gdf = gpd.read_file(Path(directory) / manifest["shapefile"])
            rows, geometries, pool_level = _build_rows(gdf, manifest)
            catalogue = RegionCatalogue(name, rows, geometries, pool_level,
                                        level_names=tuple(manifest.get("level_names", ())))
            _validate(catalogue)

        _CACHE[key] = catalogue
        return catalogue


def clear_cache() -> None:
    """Drop the cached catalogues and label arrays (tests that vary the country)."""
    with _CACHE_LOCK:
        _CACHE.clear()
        _LABEL_CACHE.clear()


# ── the partition ────────────────────────────────────────────────────────────


def leaf_index(catalogue: RegionCatalogue) -> dict[str, int]:
    """Leaf id -> its position on the ``region`` axis of every diagnostics file.

    The label arrays this module returns hold these indices, and the ``region``
    coordinate written to disk is ``[r.id for r in catalogue.leaves()]`` in the same
    order, so a reader recovers ids without consulting the catalogue.
    """
    return {r.id: i for i, r in enumerate(catalogue.leaves())}


def _has_leaves(catalogue: RegionCatalogue) -> bool:
    """Whether this country can be partitioned at all.

    False for a country with no boundaries onboarded, whose catalogue is a single level-0
    node. Both assign functions answer ``OUTSIDE`` everywhere in that case rather than
    consulting the Natural Earth table, which would raise for a country it has never
    heard of — an unonboarded agency is a normal state, not an error.
    """
    return catalogue.pool_level > 0 and any(
        r.id != UNASSIGNED_ID for r in catalogue.leaves())


def _assign(catalogue: RegionCatalogue, xs: np.ndarray, ys: np.ndarray,
            in_scope: np.ndarray) -> np.ndarray:
    """Label flat coordinates already known to be in/out of the country's domain.

    The shared core of :func:`assign_grid` and :func:`assign_points` — only the
    definition of "in the country" differs between a grid and a gauge list, and that is
    settled by the caller. Keeping one implementation is what guarantees a station and a
    grid cell at the same coordinate land in the same region.

    Uses an R-tree over the leaf polygons and a single ``within`` query, so cost scales
    with the points that actually fall near a polygon rather than with
    ``len(points) x len(leaves)``. GEOS work serialises on the shared lock: prepared
    geometry indexes build lazily on first query and two threads racing that build
    segfault the process.
    """
    import shapely

    index = leaf_index(catalogue)
    labels = np.full(xs.shape, OUTSIDE, dtype=np.int16)
    labels[in_scope] = index[UNASSIGNED_ID]

    leaves = [r for r in catalogue.leaves() if r.id != UNASSIGNED_ID]
    if not leaves:
        return labels

    flat_scope = np.flatnonzero(in_scope.reshape(-1))
    if flat_scope.size == 0:
        return labels

    with _regions._GEOS_LOCK:
        geoms = [catalogue.geometry(r.id) for r in leaves]
        keep = [i for i, g in enumerate(geoms) if g is not None]
        if not keep:
            return labels
        tree = shapely.STRtree([geoms[i] for i in keep])
        # Leaf position on the region axis, per tree entry.
        axis_of = np.array([index[leaves[i].id] for i in keep], dtype=np.int16)

        points = shapely.points(xs.reshape(-1)[flat_scope], ys.reshape(-1)[flat_scope])
        point_pos, tree_pos = tree.query(points, predicate="within")

    flat = labels.reshape(-1)
    # Polygons are disjoint, so a point matches at most one; if a source ever ships
    # overlapping units the last write wins, which _validate cannot detect and
    # tests/test_partition.py::p1 is there to catch.
    flat[flat_scope[point_pos]] = axis_of[tree_pos]
    return flat.reshape(labels.shape)


def assign_grid(catalogue: RegionCatalogue, lon, lat, *,
                lat_name: str = "lat", lon_name: str = "lon") -> np.ndarray:
    """Partition a lat/lon grid into leaf regions, as an ``int16`` label array.

    Returns shape ``(len(lat), len(lon))`` holding a leaf index, the index of
    ``unassigned``, or :data:`OUTSIDE` for cells beyond the country's domain. The
    country's domain is the **unchanged** Natural Earth mask
    (:func:`verification.regions._country_keep`), so this function only ever decides
    *which* region a cell already in scope belongs to — never whether it is in scope.
    That is what keeps national numbers identical to the ones already on disk.

    Cells are assigned by their centre, so no cell lands in two leaves and the partition
    is exact. Cached per (country, pool level, grid) — the pipeline asks once per lead
    and the dashboard once per render, for the same answer every time.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if not _has_leaves(catalogue):
        return np.full((lat.size, lon.size), OUTSIDE, dtype=np.int16)

    key = (catalogue.country, catalogue.pool_level, _grid_key(lat), _grid_key(lon))
    with _CACHE_LOCK:
        hit = _LABEL_CACHE.get(key)
    if hit is not None:
        return hit

    in_country = _regions._country_keep(catalogue.country, lon, lat, lat_name, lon_name)
    if in_country is None:
        return np.full((lat.size, lon.size), OUTSIDE, dtype=np.int16)

    lon2d, lat2d = np.meshgrid(lon, lat)
    labels = _assign(catalogue, lon2d, lat2d, np.asarray(in_country, dtype=bool))
    labels.flags.writeable = False          # cached and shared — never mutate in place
    with _CACHE_LOCK:
        _LABEL_CACHE[key] = labels
    return labels


def assign_points(catalogue: RegionCatalogue, lons, lats) -> np.ndarray:
    """Partition station coordinates into leaf regions — the point analogue of a grid.

    Same contract as :func:`assign_grid`: a gauge outside the country's domain is
    :data:`OUTSIDE`, one inside it but in no leaf polygon is ``unassigned``. Not cached —
    a station list is small and, unlike a grid, is not reused across leads unchanged.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if not _has_leaves(catalogue):
        return np.full(lons.shape, OUTSIDE, dtype=np.int16)
    in_country = np.asarray(
        _regions.stations_in_region(lons, lats, catalogue.country), dtype=bool)
    return _assign(catalogue, lons, lats, in_country)


def _grid_key(axis: np.ndarray) -> tuple:
    """Identity of a coordinate axis for the label cache: size and endpoints.

    Same idiom as the dashboard's mask cache. Two different grids that agree on all three
    would have to be the same regular axis, and an irregular axis that collided would
    still have to share both endpoints and its length.
    """
    return (int(axis.size), float(axis[0]), float(axis[-1])) if axis.size else (0,)


#: (country, pool level, lat key, lon key) -> read-only label array.
_LABEL_CACHE: dict[tuple, np.ndarray] = {}
