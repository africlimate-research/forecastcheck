# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""The agency-editable study parameters — ``config/verification.toml``.

What these gates protect:

* the file is OPTIONAL and OVERRIDING — remove it and the code defaults return, so a
  branch without one is not a different platform;
* a malformed file STOPS the run instead of falling back, because a silent fallback
  produces a complete set of curves scored against events nobody asked for;
* an int threshold stays an int, because curves are selected back by label and a
  float coord would not match a tree of int-keyed files;
* a configured season slug reaches the global vocabulary, or curve discovery reads it
  as a region name.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from verification import settings


def _write(tmp_path: Path, body: str) -> Path:
    f = tmp_path / "verification.toml"
    f.write_text(textwrap.dedent(body), encoding="utf-8")
    return f


@pytest.fixture()
def config_file(tmp_path, monkeypatch):
    """Point the loader at a throwaway file and clear its (path, mtime) cache."""
    def use(body: str) -> Path:
        f = _write(tmp_path, body)
        monkeypatch.setenv("VERIFICATION_SETTINGS_FILE", str(f))
        settings._load.cache_clear()
        return f
    yield use
    settings._load.cache_clear()


# --------------------------------------------------------------------------- s1
def test_s1_absent_file_is_no_configuration(tmp_path, monkeypatch):
    """No file at all ⇒ every lookup misses, and the code defaults stand.

    This is the gate that keeps the file optional: a checkout without one must behave
    exactly as the platform did before the file existed.
    """
    monkeypatch.setenv("VERIFICATION_SETTINGS_FILE", str(tmp_path / "nope.toml"))
    settings._load.cache_clear()
    assert settings.configured_countries() == ()
    assert settings.thresholds("temperature_max", "Senegal") is None
    assert settings.altitude_bands("Senegal") is None
    assert settings.seasons("Senegal") is None
    assert settings.all_seasons() == {}


# --------------------------------------------------------------------------- s2
def test_s2_thresholds_override_and_empty_means_off(config_file):
    """A list replaces the default; ``[]`` is a real answer, not a miss.

    ``()`` means "no threshold metrics for this variable" — how a heat study switches
    Tmin curves off. It must be distinguishable from "not configured", which is why
    the miss is None.
    """
    config_file("""
        [Testland.thresholds]
        temperature_max = [30, 35]
        temperature_min = []
    """)
    assert settings.thresholds("temperature_max", "Testland") == (30, 35)
    assert settings.thresholds("temperature_min", "Testland") == ()
    assert settings.thresholds("precipitation", "Testland") is None


# --------------------------------------------------------------------------- s3
def test_s3_int_thresholds_stay_int(config_file):
    """Curves are selected BY LABEL, so the coord dtype must not drift to float."""
    import numpy as np
    config_file("""
        [Testland.thresholds]
        temperature_max = [37, 38, 39]
    """)
    got = settings.thresholds("temperature_max", "Testland")
    assert all(isinstance(v, int) for v in got), got
    assert np.asarray(got).dtype == np.int64
    # a genuinely fractional threshold is still allowed
    config_file("""
        [Testland.thresholds]
        precipitation = [0.5, 1.5]
    """)
    assert settings.thresholds("precipitation", "Testland") == (0.5, 1.5)


# --------------------------------------------------------------------------- s4
def test_s4_country_lookup_ignores_case(config_file):
    """An agency should not have to know how the code spells its own country."""
    config_file("""
        ["new country".thresholds]
        temperature_max = [41]
    """)
    for spelling in ("New Country", "new country", "NEW COUNTRY", " New Country "):
        assert settings.thresholds("temperature_max", spelling) == (41,), spelling


# --------------------------------------------------------------------------- s5
@pytest.mark.parametrize("body,fragment", [
    ("[Testland.thresholds]\ntemperature_max = [40, 38]",        "ascend"),
    ("[Testland.thresholds]\ntemperature_max = [38, 38]",        "duplicate"),
    ("[Testland.thresholds]\ntemperature_max = 38",              "expected a list"),
    ('[Testland.thresholds]\ntemperature_max = ["hot"]',         "expected a number"),
    ("[Testland.altitude_bands]\n\"0-10\" = [10, 0]",            "must exceed"),
    ("[Testland.altitude_bands]\n\"all\" = [0, 10]",             "automatically"),
    ("[Testland.altitude_bands]\n\"0-10\" = 10",                 "expected [low, high]"),
    ("[Testland.seasons.WINTER]\nmonths = [1]",                  "lowercase"),
    ("[Testland.seasons.yearly]\nmonths = [1]",                  "reserved"),
    ("[Testland.seasons.wet]\nmonths = []",                      "non-empty"),
    ("[Testland.seasons.wet]\nmonths = [13]",                    "1-12"),
    ("[Testland.seasons.wet]\nmonths = [1, 1]",                  "duplicate"),
    ("[Testland.seasons.wet]\nmonths = [1]\nmonth = [2]",        "unknown key"),
    ("[Testland.thresholdz]\ntemperature_max = [1]",             "unknown section"),
    ("[Testland.thresholds]\ntemperature_max = [1\n",            "not valid TOML"),
])
def test_s5_a_bad_file_raises_rather_than_falling_back(config_file, body, fragment):
    """Every malformed shape stops the run, naming the country and key.

    A fallback here would be invisible: the run completes, writes a full set of curves,
    and scores events the agency did not choose.
    """
    config_file(body)
    with pytest.raises(settings.SettingsError) as exc:
        settings.configured_countries()
    assert fragment in str(exc.value), (fragment, str(exc.value))


# --------------------------------------------------------------------------- s6
def test_s6_altitude_bands_gain_all_and_open_ended_high(config_file):
    """``all`` is inserted for the author, and the top band may be left open."""
    config_file("""
        [Testland.altitude_bands]
        "0-50"  = [0, 50]
        ">50"   = [50, "inf"]
        ">>999" = [999]
    """)
    bands = settings.altitude_bands("Testland")
    assert list(bands) == ["all", "0-50", ">50", ">>999"], list(bands)
    assert bands["all"] is None
    assert bands["0-50"] == (0.0, 50.0)
    assert bands[">50"] == (50.0, float("inf"))
    assert bands[">>999"] == (999.0, float("inf"))


# --------------------------------------------------------------------------- s7
def test_s7_configured_season_reaches_the_global_vocabulary(tmp_path, monkeypatch):
    """A configured slug must be in SEASON_SLUGS, or discovery reads it as a region.

    ``verification.seasons`` folds the file in at IMPORT, so this reimports the module
    under a file of its own rather than mutating the live tables.
    """
    import importlib
    f = _write(tmp_path, """
        [Testland.seasons.wet]
        months = [5, 6, 7]
        label  = "Wet (May-Jul)"
        [Testland.seasons.dry]
        months = [11, 12, 1]
    """)
    monkeypatch.setenv("VERIFICATION_SETTINGS_FILE", str(f))
    settings._load.cache_clear()
    import verification.seasons as seasons_mod
    seasons = importlib.reload(seasons_mod)
    try:
        assert "wet" in seasons.SEASON_SLUGS and "dry" in seasons.SEASON_SLUGS
        assert seasons.season_months("wet") == (5, 6, 7)
        assert seasons.season_label("wet") == "Wet (May-Jul)"
        # no label ⇒ the slug, upper-cased, rather than a blank selector entry
        assert seasons.season_label("dry") == "DRY"
        assert seasons.seasons_for_country("Testland") == ("yearly", "wet", "dry")
        # the code-defined slugs stay in the vocabulary alongside the configured ones
        for slug in ("ndjf", "mamj", "jaso"):
            assert slug in seasons.SEASON_SLUGS, slug
    finally:
        settings._load.cache_clear()
        monkeypatch.delenv("VERIFICATION_SETTINGS_FILE", raising=False)
        importlib.reload(seasons_mod)


# --------------------------------------------------------------------------- s8
def test_s8_configured_seasons_replace_a_code_row_whatever_its_spelling(
        tmp_path, monkeypatch):
    """Replacing Senegal's windows must not leave the code row winning the lookup.

    The merge deletes any case-insensitive duplicate first. Without that, an exact-match
    lookup on "Senegal" would find the code row sitting beside the configured "senegal" one
    and answer with the very windows the file had just replaced.
    """
    import importlib
    f = _write(tmp_path, """
        [senegal.seasons.onerain]
        months = [6, 7, 8]
    """)
    monkeypatch.setenv("VERIFICATION_SETTINGS_FILE", str(f))
    settings._load.cache_clear()
    import verification.seasons as seasons_mod
    seasons = importlib.reload(seasons_mod)
    try:
        assert seasons.seasons_for_country("Senegal") == ("yearly", "onerain")
        keys = [k for k in seasons.SEASONS_BY_COUNTRY if k.lower() == "senegal"]
        assert len(keys) == 1, keys
        # the slugs it replaced stay in the vocabulary — curves written under them exist
        for old in ("ndjf", "mamj", "jaso"):
            assert old in seasons.SEASON_SLUGS, old
    finally:
        settings._load.cache_clear()
        monkeypatch.delenv("VERIFICATION_SETTINGS_FILE", raising=False)
        importlib.reload(seasons_mod)


# --------------------------------------------------------------------------- s9
def test_s9_the_committed_file_parses_and_holds_senegal():
    """The file in the repo is valid and is what the pipeline actually reads for Senegal.

    The file overrides the code defaults silently, so this pins its content: a typo here
    would otherwise produce a full set of plausible curves at the wrong rungs.
    """
    from verification import diagnostics, seasons, store

    settings._load.cache_clear()
    assert settings.configured_countries() == ("senegal",)
    assert settings.thresholds("temperature_max", "Senegal") == (
        37, 38, 39, 40, 41, 42, 43, 44, 45)
    assert settings.thresholds("temperature_min", "Senegal") == ()
    assert store.exceedance_thresholds("temperature_max", "Senegal") == (
        37, 38, 39, 40, 41, 42, 43, 44, 45)
    assert not store.has_exceedance("temperature_min", "Senegal")
    assert list(diagnostics.altitude_bands("Senegal")) == [
        "all", "0-25", "25-50", "50-100", ">100"]
    assert tuple(settings.seasons("Senegal")) == seasons.SEASONS_BY_COUNTRY["Senegal"] \
        == ("ndjf", "mamj", "jaso")


# --------------------------------------------------------------------------- s10
@pytest.mark.parametrize("country", ["Senegal", "Nowhere"])
def test_s10_every_reader_of_the_thresholds_gets_the_same_answer(country, monkeypatch):
    """The CLI and the store must not disagree about what the thresholds are.

    `verification.run` takes the --threshold default AND its validation list from
    `diagnostics.thresholds_for`, while the calibration writes p_exceed from
    `store.exceedance_thresholds`. They were two different readers: thresholds_for read
    the global registry directly, so the config file's rungs stopped at the CLI.

    It cannot fail loudly either, further down: the curves select p_exceed BY LABEL, so
    a writer and a reader that disagree produce an empty selection, not an error.
    """
    from verification import diagnostics, store

    monkeypatch.setenv("VERIFICATION_COUNTRY", country)
    settings._load.cache_clear()
    for variable in ("precipitation", "temperature_max", "temperature_min"):
        assert (tuple(diagnostics.thresholds_for(variable))
                == tuple(store.exceedance_thresholds(variable))), (country, variable)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    print(f"Running {len(fns)} tests in {__file__}")
    sys.exit(pytest.main([__file__, "-q"]))
