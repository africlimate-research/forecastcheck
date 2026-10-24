# Senegal (ANACIM)

```bash
echo senegal > .deployment
```

## What is verified against what

**Daily maximum and minimum temperature** (`temperature_max`, `temperature_min`), at lead
days 1, 3, 5 and 7. No precipitation, no gauges — everything is gridded.

| | source | years |
|---|---|---|
| **Gridded truth** | `chirts` — CHIRTS, 0.05° | 2016–2026 |
| | `era5` — ECMWF reanalysis, regridded to the same grid | 2016–2026 |
| **Forecasts** | `graphcast` | 2016–2026 |
| | `ifs` | 2018–2026, **no 2023** |

## Get the data

`data/` is not in git. On the DGX Spark, point the Google Cloud tools at the read-only
key **by its path** (never copy or paste the key itself), then fetch — first as a dry run:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/etc/fleet-gcs-key.json
./scripts/fetch_country.sh            # lists what would be fetched
./scripts/fetch_country.sh --apply    # GraphCast + IFS, CHIRTS + ERA5, elevation
```

Source: `gs://aim4scale_training_2026/demo5/SEN-local-engine`, about 3.5 GB. Re-running resumes; `--with-aifs` adds the AIFS models;
`--from <folder>` reads a local copy of the folder instead of the bucket.

## Settings

In **`config/verification.toml`** under `[Senegal.*]`:

| | |
|---|---|
| `temperature_max` thresholds | 37, 38 … 45 °C (nine rungs) |
| `temperature_min` thresholds | none — crps and the deterministic metrics only |
| Altitude bands | 0–25 / 25–50 / 50–100 / >100 m |
| Seasons | `ndjf` Nov–Feb · `mamj` Mar–Jun · `jaso` Jul–Oct (plus `yearly`) |
| Regions | 14 regions over 45 departments (`verification/assets/admin_boundaries/senegal/`) |

## Running the pipeline

```bash
bash docker/run.sh pipeline           # quick: Tmax against CHIRTS, test year 2022, lead days 1 and 3
bash docker/run.sh pipeline --full    # everything: both variables and truths, 2022 and 2024, days 1–7 (about 3 h)
```

Training follows one rule for every country: the years 2016–2021, each model on the years it has
(IFS starts in 2018, so it trains on 2018–2021). `--year 2025` tests one more year with the
same training. Without Docker, use `./scripts/run_pipeline.sh` with the same options. The commands
each stage runs are in the lesson (`docs/demo5/`, "Under the hood").

`temperature_min` has no thresholds, so the calibrated pass computes `crps` only for it.

Curves are pooled once per department and summed to regions at read time. To write
per-region / per-season curve files explicitly:

```bash
python -m verification.run --model graphcast-idr --variable temperature_max --metric roc \
    --year 2022 --product chirts \
    --region Senegal "Dakar (region)" Podor --season yearly ndjf mamj jaso
```

A department named like its region is qualified: `Dakar (region)` / `Dakar (department)`.

## Known limits of this data

Read once before quoting a number.

- **ERA5 is a model, not an observation**, and runs ~1.3 °C below CHIRTS here; GraphCast's
  Tmin bias even changes sign between the two. Always say which truth a figure came from.
- **A rare rung's AUC flatters** — read reliability and cost-loss for the top thresholds.
- **IFS starts in 2018**, GraphCast in 2016: compare calibrated models on the same
  training years.
- The Dakar departments are only a few 0.05° cells each (Guediawaye perhaps none), too
  few to carry a threshold curve.
