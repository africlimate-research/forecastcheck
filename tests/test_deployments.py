# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Countries are files: ``deployments/{name}.toml``, selected per machine.

What these gates protect:

* the machine's selection (``.deployment`` / ``VERIFICATION_DEPLOYMENT``) picks exactly
  one country file, and ``none`` means the pan-African build;
* a misspelt or missing country STOPS the launch, naming the countries that exist —
  it must never run unconfigured or show another country's data;
* every committed country file is complete: a known country, its admin boundaries and
  its study settings, and no key the code does not read (a typo would be ignored).
"""
from __future__ import annotations

import os
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import config, regions, settings  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent

#: Every key a country file may hold; anything else is a typo the code would ignore.
_KNOWN_KEYS = frozenset({"country", "data_root", "truth_product"})

_ENV = ("VERIFICATION_DEPLOYMENT", "VERIFICATION_COUNTRY", "VERIFICATION_DATA_ROOT",
        "VERIFICATION_TRUTH_PRODUCT")


@contextmanager
def _env(**values):
    """Set (or, with None, unset) the deployment env vars, restoring them afterwards."""
    saved = {k: os.environ.get(k) for k in _ENV}
    try:
        for k in _ENV:
            os.environ.pop(k, None)
        for k, v in values.items():
            if v is not None:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_d1_selection_picks_the_country_file():
    """``VERIFICATION_DEPLOYMENT`` names a file; its country seeds the environment."""
    with _env(VERIFICATION_DEPLOYMENT="senegal"):
        assert config.deployment_name() == "senegal"
        assert config.deployment_file() == _REPO / "deployments" / "senegal.toml"
        config.apply_deployment_defaults()
        assert os.environ["VERIFICATION_COUNTRY"] == "Senegal"
        assert config.domain() == "senegal"


def test_d2_none_is_the_pan_african_build():
    """``none`` selects nothing: no file, no seeded country, domain ``africa``."""
    with _env(VERIFICATION_DEPLOYMENT="none"):
        assert config.deployment_name() is None
        assert config.deployment_file() is None
        assert config.deployment_defaults() == {}
        config.apply_deployment_defaults()
        assert "VERIFICATION_COUNTRY" not in os.environ
        assert config.domain() == "africa"


def test_d3_an_unknown_country_stops_the_launch():
    """A misspelt name raises, listing the countries that do exist."""
    with _env(VERIFICATION_DEPLOYMENT="senegall"):
        with pytest.raises(ValueError) as exc:
            config.apply_deployment_defaults()
    assert "senegall" in str(exc.value) and "senegal" in str(exc.value)


def test_d4_a_broken_country_file_stops_the_launch(tmp_path):
    """A file that does not parse raises rather than launching unconfigured."""
    bad = tmp_path / "broken.toml"
    bad.write_text('country = "Senegal\n')
    with pytest.raises(ValueError, match="broken.toml"):
        config.deployment_defaults(bad)


def test_d5_a_real_env_var_wins_over_the_file():
    """The file only fills gaps: an exported country is never overridden."""
    with _env(VERIFICATION_DEPLOYMENT="senegal", VERIFICATION_COUNTRY="Elsewhere"):
        config.apply_deployment_defaults()
        assert os.environ["VERIFICATION_COUNTRY"] == "Elsewhere"


@pytest.mark.parametrize("name", config.available_deployments())
def test_d6_every_country_file_is_complete(name):
    """Each committed country is known, has boundaries and settings, and no stray key."""
    with open(config.deployment_path(name), "rb") as fh:
        d = tomllib.load(fh)
    assert set(d) <= _KNOWN_KEYS, f"{name}: unknown key(s) {set(d) - _KNOWN_KEYS}"
    country = d.get("country")
    assert country, f"{name}: no country"
    assert country in regions.REGION_COUNTRIES, f"{name}: {country!r} not in REGION_COUNTRIES"
    boundaries = (_REPO / "verification" / "assets" / "admin_boundaries"
                  / config.slug(country) / "regions.toml")
    assert boundaries.exists(), f"{name}: no {boundaries.relative_to(_REPO)}"
    assert country.strip().lower() in settings.configured_countries(), (
        f"{name}: no [{country}.*] section in config/verification.toml")


def test_d7_the_shipped_countries():
    """The countries this repository ships; a new country adds its name here."""
    assert config.available_deployments() == ("kenya", "senegal", "sierra-leone")


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    print(f"Running {len(fns)} tests in {__file__}")
    sys.exit(pytest.main([__file__, "-q"]))
