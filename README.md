# ForecastCheck

Local verification and calibration of AI weather forecasts for national met agencies.

ForecastCheck scores AI weather models (**GraphCast**, **IFS**, …) against observations
for one country: raw skill, skill after **IDR calibration** (Isotonic Distributional
Regression), and threshold curves (ROC, reliability, cost–loss), per region, per season
and per altitude band, in a local Streamlit dashboard. Everything runs on your own
machine from plain NetCDF files under `data/` — no cloud service at runtime.

> Part of the [Forecast4Africa](https://forecast4africa.org) initiative by AfriClimate AI.

> **Want to contribute?** Please read [CONTRIBUTING.md](CONTRIBUTING.md) first: it explains
> how we branch, open pull requests and keep `main` up to date.

---

## Countries

Every country runs the **same code**. What makes an installation Senegal's is one file,
`deployments/{name}.toml`, plus its boundaries and settings — not a separate branch.

| Country | Agency | File | Details |
|---|---|---|---|
| Senegal | ANACIM | `deployments/senegal.toml` | [docs/countries/senegal.md](docs/countries/senegal.md) |
| Sierra Leone | — | `deployments/sierra-leone.toml` | [docs/countries/sierra-leone.md](docs/countries/sierra-leone.md) |
| Kenya | KMD | `deployments/kenya.toml` | [docs/countries/kenya.md](docs/countries/kenya.md) |

Each country page says what is verified against what, the study settings, how to run the
pipeline for that country, and the known limits of its data. Read it before quoting a
number.

---

## Installation

Needs [Miniforge](https://github.com/conda-forge/miniforge) (conda). Works on Linux
(including the DGX Spark) and macOS.

```bash
git clone https://github.com/africlimate-research/forecastcheck.git
cd forecastcheck

conda env create -f environment.yml
conda activate forecastcheck
python -m pytest tests/          # confirms the installation; needs no data
```

Without conda, Python ≥ 3.11 and `pip install -r requirements.txt` work too.

### Choose your country, once per machine

```bash
echo senegal > .deployment       # the name of a file in deployments/
```

`.deployment` is not in git, so `git pull` never changes it. Every command and the
dashboard then run as that country. To run as another country in one shell only, set
`VERIFICATION_DEPLOYMENT=<name>`. A misspelt name stops with the list of countries that
exist.

---

## Running it

**1. Data.** `data/` is not in git. The country page explains how to fetch the forecasts,
the observations and the elevation map for that country.

**2. Pipeline** — three stages, in order; each reads what the previous one wrote. For
example, Senegal's GraphCast Tmax against CHIRTS:

```bash
python -m verification.run --model graphcast --variable temperature_max --product chirts \
    --metric mae bias pearsonr spearmanr --lead-day 1 3 5 7 --year 2022      # raw skill
python -m calibration.run  --model graphcast --variable temperature_max --product chirts \
    --leads 1,3,5,7 --train-years 2016,2017,2018,2019,2020,2021 --test-year 2022  # IDR
python -m verification.run --model graphcast-idr --variable temperature_max --product chirts \
    --metric crps roc reliability cost_loss --lead-day 1 3 5 7 --year 2022   # calibrated
```

`--product` names the truth when a country holds more than one (the run stops and lists
them if you leave it out). The country page gives the exact commands — years, models and
products — for that country, and `./scripts/run_pipeline.sh` runs them all.

**3. Dashboard:**

```bash
streamlit run ui/app.py          # http://localhost:8501
```

Restart Streamlit after generating new files — discovery is cached per process. On a
remote machine, forward the port: `ssh -L 8501:localhost:8501 <user>@<host>`.

---

## How it works

Three stages, connected only by the on-disk path scheme (`verification/store.py` owns it):

```
data/{country}/forecast/gridded/{variable}/{year}/{model}/{lead}h.nc
data/{country}/truth/{product}/gridded/{variable}/{year}.nc
data/{country}/idr/{product}/gridded/{variable}/{test_year}/{model}/{lead}h.nc
data/{country}/diagnostics/{model}/{variable}/{product}/gridded/{metric}/{year}/…/{lead}h.nc
```

- **`calibration/`** — Isotonic Distributional Regression, one fit per grid cell (or per
  gauge). Writes calibrated `crps`, quantiles, `pit` and `p_exceed` at the configured
  thresholds.
- **`verification/`** — `verification.run`. Raw model tokens (`graphcast`) get `mae`,
  `bias`, `pearsonr`, `spearmanr`; calibrated `-idr` tokens get `crps` and the `roc` /
  `reliability` / `cost_loss` curves.
- **`ui/`** — the dashboard offers only what exists on disk.

Curves are pooled **once**, per smallest admin unit, and every region (and the country) is
summed from those counts at read time, so regions and seasons need no extra runs.

### Study settings — `config/verification.toml`

Thresholds, altitude bands and seasons, per country under `[<Country>.*]`, editable
without touching Python. They are stored axes, so a change needs a re-run:

| changed | re-run |
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
| `deployments/` | one file per country |
| `config/verification.toml` | thresholds, altitude bands, seasons, per country |
| `docs/countries/` | one page per country |
| `scripts/` | data fetch; `run_pipeline.sh` (the whole pipeline for this machine's country) |
| `tools/compare_runs.py` | zero-tolerance diff of two output trees |
| `tests/` | test suite |

## Tests

```bash
python -m pytest tests/                 # or run a file directly: python tests/test_ui.py
```

The suite runs as no country, whatever `.deployment` says, so it means the same on every
machine. Checks that need real data skip unless `VERIFICATION_DATA_ROOT` points at it.

## Adding a country

No code changes — one pull request with:

1. `deployments/<name>.toml` with `country = "<Country>"` (a Natural Earth name), and the
   name added to `tests/test_deployments.py::test_d7`;
2. a row in `verification/regions.py::REGION_COUNTRIES`;
3. its boundaries and a `regions.toml` manifest under
   `verification/assets/admin_boundaries/<country>/`;
4. a `[<Country>.*]` section in `config/verification.toml`;
5. its truth products in `verification/store.py::PRODUCTS`, if new;
6. a page in `docs/countries/`, and a row in the table above.

`tests/test_deployments.py` checks that every country file is complete. Open an issue with
the *Add a country* template first if you would like help.

## Contact

**[research@africlimate.ai](mailto:research@africlimate.ai)** — developed and maintained by
**AfriClimate AI** as part of the **Forecast4Africa** initiative.
