#!/usr/bin/env bash
# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
#
# Run the pipeline for this machine's country (the one named in .deployment, or
# VERIFICATION_DEPLOYMENT): raw metrics -> IDR calibration -> calibrated metrics and
# threshold curves. The lesson (docs/demo5/) explains each stage; this runs them.
#
# By default it runs a QUICK slice, a few minutes even for a big country, so a class can
# run it and see the dashboard: test year 2022, lead days 1 and 3, and for the Heat track
# daily maximum temperature against CHIRTS only. --full runs everything (a trainer's run,
# up to a few hours): test years 2022 and 2024, lead days 1, 3, 5 and 7, every variable.
#
#   ./scripts/run_pipeline.sh                         quick: 2022, days 1 and 3
#   ./scripts/run_pipeline.sh --year 2024             the same slice, another test year
#   ./scripts/run_pipeline.sh --year 2025 --leads 1,3,5,7
#   ./scripts/run_pipeline.sh --variable temperature_min --product era5
#   ./scripts/run_pipeline.sh --full                  everything
#
# Training is ONE rule for every country and model: the years 2016-2021. Each model trains
# on the ones it has data for (IFS starts in 2018, so it trains on 2018-2021); a test year
# must come after them. Every truth on disk is scored: the public one(s), and national
# data added with fetch_country.sh --from, gridded or at gauges.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_YEARS=2016,2017,2018,2019,2020,2021
LAST_TRAIN_YEAR=2021

usage="usage: run_pipeline.sh [--year Y[,Y]] [--leads D[,D]] [--variable V] [--product P] [--full]"
YEARS=""; LEADS=""; VARIABLE=""; PRODUCT=""; FULL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --year)     YEARS="${2:?--year needs a year, e.g. 2024}"; shift ;;
    --leads)    LEADS="${2:?--leads needs lead days, e.g. 1,3}"; shift ;;
    --variable) VARIABLE="${2:?--variable needs a name, e.g. temperature_min}"; shift ;;
    --product)  PRODUCT="${2:?--product needs a truth, e.g. era5}"; shift ;;
    --full)     FULL=1 ;;
    -h|--help)  echo "$usage"; exit 0 ;;
    *)          echo "$usage" >&2; exit 2 ;;
  esac
  shift
done

# The country slug and its data folder, resolved exactly as the pipeline resolves them
# (a misspelt .deployment stops here with the list of countries that exist).
resolved=$(python -c "from verification import config
config.apply_deployment_defaults()
print(config.domain())
print(config.data_root() / config.domain())")
country=$(sed -n 1p <<< "$resolved")
data=$(sed -n 2p <<< "$resolved")

die() { echo "!! $*" >&2; exit 2; }

# Each country's study: the variables it verifies, and what the quick slice keeps.
case "$country" in
  senegal|sierra_leone)
    all_variables="temperature_max temperature_min"
    quick_variables="temperature_max"; quick_product="chirts" ;;
  kenya|rwanda)
    all_variables="precipitation"
    quick_variables="precipitation"; quick_product="" ;;   # every truth: they are small
  africa) die "no country selected: write one to .deployment first, e.g. echo senegal > .deployment" ;;
  *)      die "no pipeline is written for '$country' yet: run the steps from its page in docs/countries/" ;;
esac

if [ "$FULL" = 1 ]; then
  YEARS="${YEARS:-2022,2024}"; LEADS="${LEADS:-1,3,5,7}"
  VARIABLES="${VARIABLE:-$all_variables}"; PRODUCT_FILTER="$PRODUCT"
else
  YEARS="${YEARS:-2022}"; LEADS="${LEADS:-1,3}"
  VARIABLES="${VARIABLE:-$quick_variables}"; PRODUCT_FILTER="${PRODUCT:-$quick_product}"
fi
for v in $VARIABLES; do
  case " $all_variables " in
    *" $v "*) ;;
    *) die "'$v' is not studied for $country (its variables: $all_variables)" ;;
  esac
done
years=(${YEARS//,/ })
for y in "${years[@]}"; do
  [ "$y" -gt "$LAST_TRAIN_YEAR" ] 2>/dev/null \
    || die "test year '$y' must come after the training years (2016-$LAST_TRAIN_YEAR): try 2022, 2024 or 2025"
done
lead_days="${LEADS//,/ }"

# The truth products on disk for one shape (gridded | station), narrowed by --product.
truths() {
  local d name
  for d in "$data"/truth/*/"$1"; do
    [ -d "$d" ] || continue
    name=$(basename "$(dirname "$d")")
    [ -z "$PRODUCT_FILTER" ] || [ "$name" = "$PRODUCT_FILTER" ] || continue
    echo "$name"
  done
  return 0
}

# Every requested test year must have this truth; say which years it does have if not.
check_years() {   # check_years <product> <shape> <variable>
  local y have
  for y in "${years[@]}"; do
    if [ ! -f "$data/truth/$1/$2/$3/$y.nc" ]; then
      have=$(ls "$data/truth/$1/$2/$3" 2>/dev/null | sed 's/\.nc$//' | paste -sd' ' -)
      die "no $1 truth for $3 in $y ($2). Years on disk: ${have:-none}"
    fi
  done
}

echo "country: $country | test years: ${years[*]} | lead days: $lead_days | variables: $VARIABLES"
echo "training: 2016-$LAST_TRAIN_YEAR, each model on the years it has (IFS: from 2018)"

ran=0
for shape in gridded station; do
  # Gauges need the forecasts read at the gauges too.
  if [ "$shape" = station ] && [ ! -d "$data/forecast/station" ]; then continue; fi
  for product in $(truths "$shape"); do
    for variable in $VARIABLES; do
      [ -d "$data/truth/$product/$shape/$variable" ] || continue
      check_years "$product" "$shape" "$variable"
      echo "── $shape, $variable, against $product"
      if [ "$shape" = gridded ]; then
        station_flag=(); raw_metrics=(mae bias pearsonr spearmanr)
      else
        # A single point has no spatial correlation: gauges get mae and bias only.
        station_flag=(--is-station); raw_metrics=(mae bias)
      fi

      python -m verification.run ${station_flag[@]+"${station_flag[@]}"} \
          --model graphcast ifs --variable "$variable" --metric "${raw_metrics[@]}" \
          --lead-day $lead_days --year "${years[@]}" --product "$product"

      for model in graphcast ifs; do
        for y in "${years[@]}"; do
          if [ "$shape" = gridded ]; then
            python -m calibration.run --variable "$variable" --model "$model" \
                --leads "$LEADS" --train-years "$TRAIN_YEARS" --test-year "$y" --product "$product"
          else
            python -m calibration.run_station --model "$model" \
                --leads "$LEADS" --train-years "$TRAIN_YEARS" --test-year "$y" --product "$product"
          fi
        done
      done

      python -m verification.run ${station_flag[@]+"${station_flag[@]}"} \
          --model graphcast-idr ifs-idr --variable "$variable" \
          --metric crps roc reliability cost_loss \
          --lead-day $lead_days --year "${years[@]}" --product "$product"
      ran=$((ran + 1))
    done
  done
done

[ "$ran" -gt 0 ] || die "nothing to run: no ${PRODUCT_FILTER:-truth} data for $VARIABLES under $data/truth (fetch it first)"
echo "done — restart the dashboard to see the new results"
