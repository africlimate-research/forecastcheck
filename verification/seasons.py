# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Season definitions (shared by the pipeline and the UI).

Each season is keyed by a short filesystem-safe **slug** — the segment written into the
threshold-curve path (``.../{year}/[{region}/][{slug}/]{lead}h.nc``). The slugs are a
closed vocabulary disjoint from region and department names, which is what lets discovery tell a season
path segment apart from a region one. ``months`` is a tuple of 1-based calendar months
(``None`` for "Yearly" = no month filter). A cross-year season such as NDJF means
Nov/Dec + Jan/Feb of the *same* verification year.

``config/verification.toml`` may redefine or add seasons; they are folded in at import.
"""
from __future__ import annotations

# slug -> (display label, 1-based months | None)
SEASONS: dict[str, tuple[str, tuple[int, ...] | None]] = {
    "yearly": ("Yearly",         None),
    "ndjf":   ("NDJF (Nov–Feb)", (11, 12, 1, 2)),     # Senegal: cool dry season (harmattan)
    "mamj":   ("MAMJ (Mar–Jun)", (3, 4, 5, 6)),       # Senegal: hot dry season, heat peak
    "jaso":   ("JASO (Jul–Oct)", (7, 8, 9, 10)),      # Senegal: rainy season (hivernage)
    # Sierra Leone and Kenya use the same windows: one row serves both.
    "ondj":   ("ONDJ (Oct–Jan)", (10, 11, 12, 1)),    # SL: end of the rains · KE: short rains
    "fmam":   ("FMAM (Feb–May)", (2, 3, 4, 5)),       # SL: heat peak · KE: long rains
    "jjas":   ("JJAS (Jun–Sep)", (6, 7, 8, 9)),       # SL: monsoon · KE: cool dry season
}

#: Which seasons each country offers. ``yearly`` is implicit and always first; a country
#: absent here gets ``yearly`` only.
SEASONS_BY_COUNTRY: dict[str, tuple[str, ...]] = {
    "Senegal": ("ndjf", "mamj", "jaso"),
    # Checked against CHIRTS: fmam isolates the heat season, jjas the monsoon.
    "Sierra Leone": ("ondj", "fmam", "jjas"),
    # Bimodal rains (April, November): fmam pools the driest month with the wettest.
    "Kenya": ("ondj", "fmam", "jjas"),
}

DEFAULT_SEASON_SLUG: str = "yearly"


def _apply_configured_seasons() -> None:
    """Fold ``config/verification.toml``'s seasons into the two tables above.

    Read once, at import, because other modules ``from ... import`` these tables. Which
    seasons a run uses is still chosen at call time by :func:`seasons_for_country`.
    The per-country offer list is REPLACED, so an agency can drop a window.
    """
    from verification import settings
    for country, defined in settings.all_seasons().items():
        for slug, entry in defined.items():
            SEASONS[slug] = entry
        # Drop a code row under a different spelling first, or the exact-match lookup
        # in seasons_for_country would find it and answer with the replaced windows.
        folded = country.strip().lower()
        for existing in [k for k in SEASONS_BY_COUNTRY if k.lower() == folded]:
            del SEASONS_BY_COUNTRY[existing]
        SEASONS_BY_COUNTRY[country.strip()] = tuple(defined)


_apply_configured_seasons()

#: Every season slug, built after the configured seasons are folded in, so the path
#: parser recognises them. ``tests/test_region_catalogue.py`` asserts this stays
#: disjoint from every region id.
SEASON_SLUGS: frozenset[str] = frozenset(SEASONS)


def seasons_for_country(country: str | None) -> tuple[str, ...]:
    """Season slugs this country offers, ``yearly`` first. Matched ignoring case."""
    name = (country or "").strip()
    if name in SEASONS_BY_COUNTRY:
        return (DEFAULT_SEASON_SLUG, *SEASONS_BY_COUNTRY[name])
    folded = name.lower()
    for key, extra in SEASONS_BY_COUNTRY.items():
        if key.lower() == folded:
            return (DEFAULT_SEASON_SLUG, *extra)
    return (DEFAULT_SEASON_SLUG,)


def season_months(slug: str) -> tuple[int, ...] | None:
    """1-based calendar months for a season slug; None = Yearly (no month filter)."""
    entry = SEASONS.get(slug)
    return entry[1] if entry else None


def season_label(slug: str) -> str:
    """Display label for a season slug (falls back to the slug itself if unknown)."""
    entry = SEASONS.get(slug)
    return entry[0] if entry else slug
