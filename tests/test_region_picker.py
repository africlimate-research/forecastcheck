# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""The region picker, exercised by clicking it.

Runs standalone (``python tests/test_region_picker.py``) or under pytest.

Booting the app and getting HTTP 200 proves a page renders once. It does not touch a
widget, and every defect this picker has shipped needed a click to appear: assigning a
widget's state inline raised only when a user chose from the dropdown, and the
default-plus-session-state warning fires only on the run after a selection. So these
tests drive it through Streamlit's own ``AppTest`` harness — select something, rerun,
assert.

``AppTest`` needs a script, so each test writes a tiny one that calls the picker and
records what it returned. That is the whole app surface under test; the pages add only a
key prefix.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

_SCRIPT = f'''
import sys
sys.path.insert(0, {str(REPO)!r})
import streamlit as st
from ui.components.region_picker import render_region_picker

st.session_state["picked_region"] = render_region_picker("t")
'''


def _app():
    """An AppTest over a one-line script that renders the picker, or None if unavailable."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001
        return None
    tmp = Path(tempfile.mkdtemp()) / "picker_app.py"
    tmp.write_text(_SCRIPT)
    return AppTest.from_file(str(tmp), default_timeout=30)


class _env:
    """Point the picker at a deployment for the duration of a block."""

    KEYS = ("VERIFICATION_COUNTRY", "VERIFICATION_DATA_ROOT", "VERIFICATION_TRUTH_PRODUCT")

    def __init__(self, **kwargs):
        self.want = kwargs

    def __enter__(self):
        self.prev = {k: os.environ.get(k) for k in self.KEYS}
        os.environ.update({k: str(v) for k, v in self.want.items()})
        _clear_caches()
        return self

    def __exit__(self, *exc):
        for key, value in self.prev.items():
            os.environ.pop(key, None) if value is None else os.environ.update({key: value})
        _clear_caches()
        return False


def _clear_caches():
    """Drop every cache that outlives a deployment switch, and reload the frozen ones.

    ``ui.app_config.settings`` resolves REGIONS at IMPORT time, so a test that changes
    the country without reloading it silently gets the previous deployment's list.
    """
    import importlib

    from verification import admin, regions

    admin.clear_cache()
    regions._SUBREGIONS = None          # cached per country, not per data root
    for name in ("ui.utils.regions", "ui.app_config.settings", "ui.components.region_picker"):
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)
    from ui.components import region_picker

    region_picker._with_data.cache_clear()


def _one_level_root():
    """A data root whose Senegal manifest has a single level (4 departments, no parents).

    The data-tree manifest overrides the bundled two-level one, so this exercises the
    single-level path, which renders no dropdown.
    """
    import geopandas as gpd
    import shapely

    root = Path(tempfile.mkdtemp())
    regions = root / "senegal" / "assets" / "regions"
    regions.mkdir(parents=True)
    rows = [{"DEP": dep, "geometry": shapely.box(-16 + i, 13, -15 + i, 14)}
            for i, dep in enumerate(["Alpha", "Beta", "Gamma", "Delta"])]
    gpd.GeoDataFrame(rows, crs="EPSG:4326").to_file(regions / "units.shp")
    (regions / "regions.toml").write_text(
        'shapefile = "units.shp"\nname_field = "DEP"\npool_level = 1\n')
    return str(root)


def test_w1_picker_renders_without_warnings_or_exceptions():
    """W1: a first render must be clean — no exception, and no Streamlit warning.

    The warning guarded here is "created with a default value but also had its value set
    via the Session State API". It is cosmetic, but it means the widget has two owners of
    its value and the default is dead code — and these widgets must write their own state
    in order to clear one another.
    """
    app = _app()
    if app is None:
        print("SKIP test_w1 (streamlit AppTest unavailable)")
        return
    with _env(VERIFICATION_COUNTRY="Senegal"):
        app.run()
        assert not app.exception, [str(e) for e in app.exception]
        assert not app.warning, [w.value for w in app.warning]
        assert app.session_state["picked_region"] == "Senegal"


def test_w2_choosing_a_region_from_the_dropdown_does_not_raise():
    """W2: the bug that shipped — selecting from the dropdown used to raise.

    Clearing the pills was done inline, below the pills widget, and Streamlit refuses to
    let a widget's state be written once it has been instantiated that run. The page
    loaded fine and died on the first click.
    """
    app = _app()
    if app is None:
        print("SKIP test_w2 (streamlit AppTest unavailable)")
        return
    with _env(VERIFICATION_COUNTRY="Senegal"):
        app.run()
        assert not app.exception, [str(e) for e in app.exception]

        box = app.selectbox[0]
        target = next(o for o in box.options if o not in ("—", "Senegal"))
        box.set_value(target).run()

        assert not app.exception, [str(e) for e in app.exception]
        assert app.session_state["picked_region"] == target, app.session_state["picked_region"]


def test_w3_the_two_widgets_never_disagree():
    """W3: picking in one selector clears the other, so only one answer exists.

    Without this the page renders one region while the other widget still shows a
    different one highlighted — the user cannot tell which the chart is for.
    """
    app = _app()
    if app is None:
        print("SKIP test_w3 (streamlit AppTest unavailable)")
        return
    with _env(VERIFICATION_COUNTRY="Senegal"):
        app.run()
        box = app.selectbox[0]
        fine = next(o for o in box.options if o not in ("—", "Senegal"))

        # dropdown wins, pills are cleared
        box.set_value(fine).run()
        assert app.session_state["picked_region"] == fine
        assert app.session_state["t_region_pick"] is None, app.session_state["t_region_pick"]

        # now a pill wins, and the dropdown falls back to its placeholder
        province = next(p for p in app.get("button_group")[0].options if p != "Senegal")
        app.get("button_group")[0].set_value(province).run()
        assert not app.exception, [str(e) for e in app.exception]
        assert app.session_state["picked_region"] == province
        assert app.session_state["t_region_more"] == "—", app.session_state["t_region_more"]


def test_w4_reset_returns_to_the_whole_country():
    """W4: Reset must clear BOTH widgets, not just the one the user last touched."""
    app = _app()
    if app is None:
        print("SKIP test_w4 (streamlit AppTest unavailable)")
        return
    with _env(VERIFICATION_COUNTRY="Senegal"):
        app.run()
        box = app.selectbox[0]
        fine = next(o for o in box.options if o not in ("—", "Senegal"))
        box.set_value(fine).run()
        assert app.session_state["picked_region"] == fine

        app.button[0].click().run()
        assert not app.exception, [str(e) for e in app.exception]
        assert app.session_state["picked_region"] == "Senegal"
        assert app.session_state["t_region_more"] == "—"


def test_w5_a_single_level_country_has_no_dropdown():
    """W5: when the finest level is the top one there is nothing to hide.

    The page must look exactly as it did before this widget existed — and that has to
    fall out of the catalogue, not a country check.
    """
    app = _app()
    if app is None:
        print("SKIP test_w5 (streamlit AppTest unavailable)")
        return
    with _env(VERIFICATION_COUNTRY="Senegal", VERIFICATION_DATA_ROOT=_one_level_root()):
        app.run()
        assert not app.exception, [str(e) for e in app.exception]
        assert len(app.selectbox) == 0, "a single-level country should render no dropdown"
        options = app.get("button_group")[0].options
        assert list(options) == ["Senegal", "Alpha", "Beta", "Delta", "Gamma"], options


def test_w6_senegal_regions_are_pills_and_departments_the_dropdown():
    """W6: the real catalogue splits by level — 14 regions as pills, 45 departments below."""
    app = _app()
    if app is None:
        print("SKIP test_w6 (streamlit AppTest unavailable)")
        return
    with _env(VERIFICATION_COUNTRY="Senegal"):
        app.run()
        assert not app.exception, [str(e) for e in app.exception]
        pills = app.get("button_group")[0].options
        assert pills[0] == "Senegal" and len(pills) == 1 + 14, pills
        assert "Dakar (region)" in pills and "Pikine" not in pills
        departments = [o for o in app.selectbox[0].options if o != "—"]
        assert len(departments) == 45, departments
        assert "Pikine" in departments and "Dakar (department)" in departments


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


# --------------------------------------------------------------------------- rp9
def test_rp9_a_gridded_country_offers_every_region_whatever_its_gauges(monkeypatch):
    """Gauge GEOGRAPHY must not decide which regions the picker offers.

    The narrowing exists so a station-only deployment is not offered regions holding no
    gauge. It used to switch off only when a country had NO gauges at all, which made
    the answer depend on where the gauges happened to be rather than on coverage: a
    country whose few gauges sat in one region lost every other region from the picker
    despite gridded truth covering all of them.
    """
    from ui.components.region_picker import _with_data

    class _Store:
        def __init__(self, targets): self._t = frozenset(targets)
        def targets(self): return self._t

    seen = {}

    def fake_catalogue(country=None):
        return _FakeCatalogue()

    class _Leaf:
        def __init__(self, i): self.id = i

    class _FakeCatalogue:
        pool_level = 1
        def levels(self): return (1,)
        def at_level(self, lv): return [_Leaf("a"), _Leaf("b")]
        def leaves(self): return self.at_level(1)
        def leaves_under(self, i): return [i]
        def display_label(self, i): return i.title()

    import verification.admin as admin_mod
    import verification.store as store_mod
    monkeypatch.setattr(admin_mod, "catalogue_for", fake_catalogue)

    # gridded present -> no opinion, every region offered
    monkeypatch.setattr(store_mod.CountryStore, "current",
                        classmethod(lambda cls: _Store({"gridded", "station"})))
    _with_data.cache_clear()
    assert _with_data("Testland") == frozenset(), \
        "a gridded country must not be narrowed by where its gauges are"

    # station-only -> the narrowing still applies (it reaches the gauge probe)
    monkeypatch.setattr(store_mod.CountryStore, "current",
                        classmethod(lambda cls: _Store({"station"})))
    _with_data.cache_clear()
    _with_data("Testland")   # must not raise; result depends on the gauge probe
    _with_data.cache_clear()
