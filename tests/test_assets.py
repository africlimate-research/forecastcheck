# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Every image the dashboard loads is in the repository.

``chrome._b64_svg`` returns an empty string for a missing file, so a missing asset never
fails a page — it just renders without its icon or logo, and no test notices. CI works
from a fresh clone, so this is where a file that never reached git shows up: a
``.gitignore`` pattern once matched ``ui/assets/icons/`` and kept every section icon out.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.components import chrome  # noqa: E402

#: Files loaded by name outside the section-card table.
_NAMED = ("AfriClimate Symbol.svg", "forecastcheck_wordmark_dark.svg",
          "icons/seasonal_forecast.svg")


def _wanted() -> list[str]:
    return sorted({f"icons/{meta[0]}.svg" for meta in chrome._CARD_META.values()}
                  | set(_NAMED))


@pytest.mark.parametrize("name", _wanted())
def test_a1_every_dashboard_asset_exists(name):
    """Each section icon and logo the pages ask for is a real, non-empty file."""
    path = chrome._ASSETS / name
    assert path.is_file(), f"missing {path.relative_to(Path(__file__).parents[1])}"
    assert chrome._b64_svg(name), f"empty {name}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    print(f"Running {len(fns)} tests in {__file__}")
    sys.exit(pytest.main([__file__, "-q"]))
