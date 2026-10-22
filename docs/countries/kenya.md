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
