#!/usr/bin/env bash
# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
#
# Run the whole pipeline for this machine's country (the one named in .deployment, or
# VERIFICATION_DEPLOYMENT): raw metrics -> IDR calibration -> calibrated metrics and
# threshold curves. The country page (docs/countries/) explains each step; this is the
# same commands in one file, for when you want results rather than the lesson.
#
# The years are written here, not read from the shell, so the run is the same on every
# machine. Run it inside the conda environment (`conda activate forecastcheck`); data must
# already be under data/. Takes a while: IDR fits one model per grid cell. On a shared
# machine, run it inside tmux so it survives the SSH session closing.
#
# Usage:  ./scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# The country slug, resolved exactly as the pipeline resolves it (a misspelt
# .deployment stops here with the list of countries that exist).
country=$(python -c "from verification import config
config.apply_deployment_defaults()
print(config.domain())")

# Senegal and Sierra Leone: Tmax / Tmin against two gridded truths (CHIRTS, ERA5).
temperature_study() {
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
}

# Kenya and Rwanda: rainfall against CHIRPS, tested on 2022 and 2024. 2023 is left out everywhere:
# IFS has no 2023, and scoring it would compare unequal samples.
rainfall_study() {
  python -m verification.run --model graphcast ifs --variable precipitation \
      --metric mae bias pearsonr spearmanr --lead-day 1 3 5 7 --year 2022 2024 --product chirps

  for model in graphcast ifs; do
    python -m calibration.run --variable precipitation --model "$model" --leads 1,3,5,7 \
        --train-years 2018,2019,2020,2021 --test-year 2022 --product chirps
    python -m calibration.run --variable precipitation --model "$model" --leads 1,3,5,7 \
        --train-years 2018,2019,2020,2021,2022 --test-year 2024 --product chirps
  done

  python -m verification.run --model graphcast-idr ifs-idr --variable precipitation \
      --metric crps roc reliability cost_loss --lead-day 1 3 5 7 --year 2022 2024 --product chirps
}

case "$country" in
  senegal|sierra_leone) temperature_study ;;
  kenya|rwanda)         rainfall_study ;;
  africa)
    echo "!! no country selected: write one to .deployment first, e.g. echo senegal > .deployment" >&2
    exit 2 ;;
  *)
    echo "!! no pipeline is written for '$country' yet: run the steps from its page in docs/countries/" >&2
    exit 2 ;;
esac
echo "done — restart Streamlit to see the new results"
