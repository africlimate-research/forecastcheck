# Kenya (KMD)

```bash
echo kenya > .deployment
```

## What is verified against what

**Precipitation**, at lead days 1, 3, 5 and 7, on a 0.1° grid.

| | source | years |
|---|---|---|
| **Gridded truth** | `chirps` — CHIRPS v2.0, averaged onto a 0.1° grid | 2016–2025 |
| **Forecasts** | `graphcast` | 2016–2026 |
| | `ifs` | 2018–2026, **no 2023** |

## Get the data

`deployments/kenya.toml` points at `gs://aim4scale_training_2026/demo5/KMD-local-engine`,
so the fetch script downloads straight from the bucket. First as a dry run (it lists what
it would fetch), then with `--apply`:

```bash
./scripts/fetch_country.sh
./scripts/fetch_country.sh --apply
```

The bucket needs credentials: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`
(e.g. on the DGX Spark), or `gcloud auth login`. Without access, fetch from a local copy of
the country folder instead:

```bash
./scripts/fetch_country.sh --from /path/to/KMD-local-engine --apply
```

## Settings

In **`config/verification.toml`** under `[Kenya.*]`:

| | |
|---|---|
| Exceedance thresholds | 1, 5, 10 … 60 mm/day (13 rungs) |
| Altitude bands | 0–1000 / 1000–2000 / 2000–3000 / >3000 m |
| Seasons | `ondj` Oct–Jan (short rains) · `fmam` Feb–May (long rains) · `jjas` Jun–Sep (plus `yearly`) |
| Regions | 47 counties (`verification/assets/admin_boundaries/kenya/`) |

The boundaries are geoBoundaries gbOpen KEN ADM1 (2020, public domain). Sub-counties are
too small (~17 cells each) to carry a threshold curve, so counties are the only level.

Kenya's rainfall is bimodal (peaks in April and November), so `fmam` pools the driest
month with the wettest. A MAM / JJAS / OND / JF split would separate the rains better.

## Running the pipeline

```bash
bash docker/run.sh pipeline           # quick: CHIRPS, test year 2022, lead days 1 and 3
bash docker/run.sh pipeline --full    # everything: 2022 and 2024, lead days 1–7 (about 20 min)
```

Training follows one rule for every country: the years 2016–2021, each model on the years it has
(IFS starts in 2018, so it trains on 2018–2021). `--year 2025` tests one more year with the
same training. Without Docker, use `./scripts/run_pipeline.sh` with the same options. The commands
each stage runs are in the lesson (`docs/demo5/`, "Under the hood").

Curves are pooled once per county and summed to the country at read time, so regions and
seasons need no extra runs.

## Known limits of this data

Read once before quoting a number.

- **CHIRPS has no values over water** — about 9 % of the cells (the Indian Ocean and Lake
  Victoria) are empty every day, so they drop out of every score.
- **CHIRPS's rain day is 00–00 UTC**, six hours earlier than the forecasts' 06–06 UTC day.
- **2023 is excluded** because IFS lacks it upstream; scoring it would compare unequal
  samples.
- **CHIRPS runs to 2025**; the 2026 forecasts have no truth yet.
- **Small counties cannot carry a threshold curve** — Mombasa has 2 grid cells, Vihiga 4,
  Nairobi 6, against a median of 33. The sample box under each chart gives the event
  count.
