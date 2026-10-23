# Rwanda (Meteo Rwanda)

```bash
echo rwanda > .deployment
```

## What is verified against what

**Precipitation**, at lead days 1, 3, 5 and 7, on a 0.05° grid.

| | source | years |
|---|---|---|
| **Gridded truth** | `chirps` — CHIRPS v2.0, 0.05° | 2016–2025 |
| **Forecasts** | `graphcast` | 2016–2025 |
| | `ifs` | 2018–2025, **no 2023** |

## Get the data

`deployments/rwanda.toml` points at `gs://aim4scale_training_2026/demo5/RWA-local-engine`,
so the fetch script downloads straight from the bucket. First as a dry run (it lists what
it would fetch), then with `--apply`:

```bash
./scripts/fetch_country.sh
./scripts/fetch_country.sh --apply
```

About 160 MB. The bucket needs credentials: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`
(e.g. on the DGX Spark), or `gcloud auth login`. Without access, fetch from a local copy of
the country folder instead:

```bash
./scripts/fetch_country.sh --from /path/to/RWA-local-engine --apply
```

## Settings

In **`config/verification.toml`** under `[Rwanda.*]`:

| | |
|---|---|
| Exceedance thresholds | 1, 5, 10 … 60 mm/day (13 rungs, the same as Kenya) |
| Altitude bands | 0–1400 / 1400–1700 / 1700–2000 / >2000 m |
| Seasons | `djf` Dec–Feb (short dry) · `mam` Mar–May (long rains) · `jja` Jun–Aug (long dry) · `son` Sep–Nov (short rains) (plus `yearly`) |
| Regions | 5 provinces and 30 districts (`verification/assets/admin_boundaries/rwanda/`) |

The boundaries are geoBoundaries gbOpen RWA ADM2 (2012, CC BY 4.0, Open Data Rwanda). Each
district carries its province, taken from geoBoundaries RWA ADM1 (2020, CC BY 4.0, The
Rwanda Geo Portal); the provinces are
unions of their districts. Curves are pooled per district.

The altitude bands are cut so each holds a usable share of the land cells
(16 / 46 / 20 / 17 %); equal 500 m steps would leave the top band with 2 %.

## Running the pipeline

The models are tested on **2022** (trained on 2018–2021) and **2024** (trained on
2018–2022); 2023 is left out because IFS has none. One script runs every stage:

```bash
./scripts/run_pipeline.sh
```

Or step by step:

```bash
python -m verification.run --model graphcast ifs --variable precipitation \
    --metric mae bias pearsonr spearmanr --lead-day 1 3 5 7 \
    --year 2022 2024 --product chirps

for m in graphcast ifs; do
  python -m calibration.run --variable precipitation --model $m --leads 1,3,5,7 \
      --train-years 2018,2019,2020,2021 --test-year 2022 --product chirps
  python -m calibration.run --variable precipitation --model $m --leads 1,3,5,7 \
      --train-years 2018,2019,2020,2021,2022 --test-year 2024 --product chirps
done

python -m verification.run --model graphcast-idr ifs-idr --variable precipitation \
    --metric crps roc reliability cost_loss --lead-day 1 3 5 7 \
    --year 2022 2024 --product chirps
```

Curves are pooled once per district and summed to provinces and the country at read time,
so regions and seasons need no extra runs.

## Known limits of this data

Read once before quoting a number.

- **Rwanda is small**: 827 grid cells fall inside it, out of 1,672 in the data's box (the
  rest is neighbouring countries and Lake Kivu). National scores rest on those 827 cells.
- **Kigali's districts cannot carry a threshold curve** — Nyarugenge has 2 cells,
  Kicukiro 7, Gasabo 14, against a median of 22. Use the Kigali province or the national
  view; the sample box under each chart gives the event count.
- **CHIRPS has no values over water**, so Lake Kivu drops out of every score.
- **CHIRPS's rain day is 00–00 UTC**, six hours earlier than the forecasts' 06–06 UTC day.
- **2023 is excluded** because IFS lacks it upstream; scoring it would compare unequal
  samples.
- **CHIRPS runs to 2025**, the last forecast year.
