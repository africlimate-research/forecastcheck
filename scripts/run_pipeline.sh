#!/usr/bin/env bash
# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
#
# Run the whole verification pipeline for SENEGAL (years and products below are Senegal's), for both truth products:
# raw metrics -> IDR calibration -> calibrated metrics and threshold curves. It is the
# runbook's §4 in one file, for when you want results rather than the step-by-step lesson.
#
# The years are written here, not read from the shell, so the run is the same on every
# machine. Run it inside the conda environment (`conda activate forecastcheck`); data must
# already be under data/ (runbook §2). Takes a while: IDR fits one model per grid cell.
#
# Usage:  ./scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

for product in chirts era5; do
  for variable in temperature_max temperature_min; do
    python -m verification.run --model graphcast ifs --variable "$variable" \
        --metric mae bias pearsonr spearmanr --lead-day 1 3 5 7 --year 2022 --product "$product"

    # IFS starts in 2018, so it trains on four years where GraphCast trains on six.
    python -m calibration.run --variable "$variable" --model graphcast --leads 1,3,5,7 \
        --train-years 2016,2017,2018,2019,2020,2021 --test-year 2022 --product "$product"
    python -m calibration.run --variable "$variable" --model ifs --leads 1,3,5,7 \
        --train-years 2018,2019,2020,2021 --test-year 2022 --product "$product"

    python -m verification.run --model graphcast-idr ifs-idr --variable "$variable" \
        --metric crps roc reliability cost_loss --lead-day 1 3 5 7 --year 2022 --product "$product"
  done
done
echo "done — restart Streamlit to see the new results"
