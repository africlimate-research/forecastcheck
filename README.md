# ForecastCheck — Senegal (ANACIM)

Verifies AI weather forecasts (**GraphCast**, **IFS**) for Senegal's national met agency:
raw skill, IDR-calibrated skill and threshold curves, per region, per department and per
season, in a local Streamlit dashboard. Everything runs from this repository on plain
NetCDF under `data/` — no cloud credentials at runtime.

> Part of the [Forecast4Africa](https://forecast4africa.org) initiative by AfriClimate AI.
> Each agency runs its own branch; what differs between them is `deployment.toml`,
> `config/verification.toml`, the boundary files and the data — not the code.

> **Want to contribute?** Please read [CONTRIBUTING.md](CONTRIBUTING.md) first: it explains
> how we branch, open pull requests and keep `main` up to date.

**Step-by-step setup: [`senegal-setup-run.md`](senegal-setup-run.md).** This page is the
overview and the short version.

---

## What is verified against what

**Daily maximum and minimum temperature** (`temperature_max`, `temperature_min`), at lead
days 1, 3, 5 and 7. No precipitation, no gauges — everything is gridded.

| | source | years |
|---|---|---|
| **Gridded truth** | `chirts` — CHIRTS, 0.05° | 2016–2024 |
| | `era5` — ECMWF reanalysis, regridded to the same grid | 2016–2024 |
| **Forecasts** | `graphcast` | 2016–2024 |
| | `ifs` | 2018–2024 |

Settings live in **`config/verification.toml`** under `[Senegal.*]`:

| | |
|---|---|
| `temperature_max` thresholds | 37, 38 … 45 °C (nine rungs) |
| `temperature_min` thresholds | none — crps and the deterministic metrics only |
| Altitude bands | 0–25 / 25–50 / 50–100 / >100 m |
| Seasons | `ndjf` Nov–Feb · `mamj` Mar–Jun · `jaso` Jul–Oct (plus `yearly`) |
| Regions | 14 regions over 45 departments (`verification/assets/admin_boundaries/senegal/`) |

---

## Quick start

Needs [Miniforge](https://github.com/conda-forge/miniforge) (conda); no cloud account.

```bash
git clone https://github.com/africlimate-research/Local_ForecastCheck.git
cd Local_ForecastCheck && git checkout senegal-lforecastcheck

conda env create -f environment.yml
conda activate forecastcheck
```

`deployment.toml` already says `country = "Senegal"`, so nothing else needs setting.

**Data** (`data/` is not in git — details, and the `--from` option for a local copy, in the
runbook, §2):

```bash
./scripts/fetch_country.sh SEN senegal --apply                    # forecasts + CHIRTS + DEM
./scripts/fetch_country.sh SEN senegal --product ERA5 --apply     # + ERA5
```

**Pipeline** — three stages, in order; each reads what the previous one wrote. Senegal holds
two truth products, so every run names one with `--product` (shown for `chirts`; repeat
with `--product era5`). `--variable` is required too: the CLI default is precipitation.
The runbook (§4) lists every command; this does all of them for both products:

```bash
./scripts/run_pipeline.sh
```

`temperature_min` has no thresholds, so the calibrated pass computes `crps` only for it.

**Dashboard:**

```bash
streamlit run ui/app.py                   # http://localhost:8501
```

Restart Streamlit after generating new files — discovery is cached per process.

---

## How it works

Three stages connected only by the on-disk path scheme (`verification/store.py` owns it):

```
data/senegal/forecast/gridded/{variable}/{year}/{model}/{lead}h.nc
data/senegal/truth/{chirts,era5}/gridded/{variable}/{year}.nc
data/senegal/idr/{product}/gridded/{variable}/{test_year}/{model}/{lead}h.nc
data/senegal/diagnostics/{model}/{variable}/{product}/gridded/{metric}/{year}/…/{lead}h.nc
```

- **`calibration/`** — Isotonic Distributional Regression, one fit per grid cell. Writes
  calibrated `crps`, quantiles, `pit` and `p_exceed` at the configured thresholds.
- **`verification/`** — `verification.run`. Raw tokens (`graphcast`) get `mae`, `bias`,
  `pearsonr`, `spearmanr`; `-idr` tokens get `crps` and the `roc` / `reliability` /
  `cost_loss` curves.
- **`ui/`** — the dashboard offers only what exists on disk, and shows both truth products
  side by side.

Curves are pooled **once**, per department, and every region (and the country) is summed
from those counts at read time, so regions and seasons need no extra runs. To write
per-region / per-season curve files explicitly:

```bash
python -m verification.run --model graphcast-idr --variable temperature_max --metric roc \
    --year 2022 --product chirts \
    --region Senegal "Dakar (region)" Podor --season yearly ndjf mamj jaso
```

A department named like its region is qualified: `Dakar (region)` / `Dakar (department)`.

### When you must re-run

| changed in `config/verification.toml` | re-run |
|---|---|
| `thresholds` | `calibration.run` **and** `verification.run` |
| `altitude_bands` | `verification.run` |
| `seasons` | `verification.run --season …` |

A section **replaces** the default rather than merging, and a malformed entry stops the run
naming the country and key. Old result files are not deleted; they stop being selected.

---

## Repository layout

| Path | What it is |
|---|---|
| `calibration/` | IDR calibration — `calibration.run` (gridded), `calibration.run_station` (gauges) |
| `verification/` | diagnostics CLI and engine; `store.py` paths, `admin.py` region hierarchy, `curves.py` region/season roll-up |
| `ui/` | Streamlit dashboard (`ui/app.py`) |
| `config/verification.toml` | thresholds, altitude bands, seasons |
| `deployment.toml` | this branch's country |
| `scripts/` | data fetch (`fetch_country.sh`), whole pipeline (`run_pipeline.sh`) |
| `tools/compare_runs.py` | zero-tolerance diff of two output trees |
| `tests/` | test suite |

## Tests

```bash
python -m pytest tests/                 # or run any file directly: python tests/test_ui.py
```

Checks that need real data skip unless `VERIFICATION_DATA_ROOT` points at it.

## Onboarding another country

No code changes: a `deployment.toml` (`country = "<Country>"`, a Natural Earth name, plus a
row in `verification/regions.py::REGION_COUNTRIES`), its boundaries and a `regions.toml`
manifest under `verification/assets/admin_boundaries/<country>/`, a `[<Country>.*]` section
in `config/verification.toml`, its truth products in `verification/store.py::PRODUCTS`, and
the `data/<country>/` tree above.

## Known limits of the Senegal data

Read once before quoting a number — full list in the
[runbook](senegal-setup-run.md#known-limits-of-this-data).

- **ERA5 is a model, not an observation**, and runs ~1.3 °C below CHIRTS here; GraphCast's
  Tmin bias even changes sign between the two. Always say which truth a figure came from.
- **A rare rung's AUC flatters** — read reliability and cost-loss for the top thresholds.
- **IFS starts in 2018**, GraphCast in 2016: compare calibrated models on the same
  training years.
- The Dakar departments are only a few 0.05° cells each (Guediawaye perhaps none), too
  few to carry a threshold curve.

## Contact

**[research@africlimate.ai](mailto:research@africlimate.ai)** — developed and maintained by
**AfriClimate AI** as part of the **Forecast4Africa** initiative.
