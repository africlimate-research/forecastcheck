#!/usr/bin/env bash
# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
#
# Fetch this machine's country data (the country named in .deployment) into data/.
#
#   {SOURCE}/raw/gridded/{bucket_var}/{year}/{model}/{lead}h.nc
#     -> data/{country}/forecast/gridded/{variable}/{year}/{model}/{lead}h.nc
#   {SOURCE}/target/gridded/{bucket_var}/{PRODUCT}/{year}.nc
#     -> data/{country}/truth/{product}/gridded/{variable}/{year}.nc
#   {SOURCE}/assets/dem.nc
#     -> data/{country}/assets/dem.nc
#
# {SOURCE} is the country file's `data_source` (deployments/{name}.toml), or a local copy
# of that folder passed with --from (a USB drive, a shared disk). The source is only ever
# read. Three renames happen on the way in, none of which touches a file's contents:
#   temperature_2m_max -> temperature_max   (the bucket uses the harmoniser's names)
#   raw/               -> forecast/
#   CHIRTS             -> chirts            (path segments are lowercase)
#
# Models: GraphCast and IFS, the two the pipeline runs; --with-aifs adds AIFS (single and
# ensemble). Other folders in the bucket are not fetched: ForecastCheck cannot read them.
#
# Access. A private bucket is read with Google Cloud credentials, referenced by PATH only:
#   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json   # e.g. on the DGX Spark
# The script then also points CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE at the same file so
# gcloud uses it. It never reads, prints or copies the key. Without that variable, gcloud
# uses whoever is signed in (`gcloud auth login`). Needs the Google Cloud CLI (`gcloud`),
# except with --from.
#
# Re-running is safe: rsync copies only what is missing or changed, and never leaves a
# half-written file behind, so an interrupted download simply resumes.
#
# Usage:  ./scripts/fetch_country.sh                 # dry run: list what would be fetched
#         ./scripts/fetch_country.sh --apply
#         ./scripts/fetch_country.sh --apply --with-aifs
#         ./scripts/fetch_country.sh --from /path/to/SEN-local-engine --apply
set -euo pipefail

usage="usage: fetch_country.sh [--apply] [--with-aifs] [--from DIR]"
APPLY=0
FROM=""
MODELS="graphcast|ifs"
while [ $# -gt 0 ]; do
  case "$1" in
    --apply)     APPLY=1 ;;
    --with-aifs) MODELS="graphcast|ifs|aifs_single|aifs_ens" ;;
    --from)      FROM="${2:?--from needs a folder}"; shift ;;
    -h|--help)   echo "$usage"; exit 0 ;;
    *)           echo "$usage" >&2; exit 2 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."

# The country, its data folder and its source, resolved exactly as the pipeline resolves
# them (a misspelt .deployment stops here with the list of countries that exist).
resolved=$(python -c "from verification import config
config.apply_deployment_defaults()
print(config.domain())
print(config.data_root() / config.domain())
print(config.deployment_defaults().get('data_source', '-'))")
COUNTRY=$(sed -n 1p <<< "$resolved")
DEST=$(sed -n 2p <<< "$resolved")
SOURCE=$(sed -n 3p <<< "$resolved")
if [ "$COUNTRY" = africa ]; then
  echo "!! no country selected: write one to .deployment first, e.g. echo senegal > .deployment" >&2
  exit 2
fi

# Bucket variable name -> ForecastCheck variable name. A variable not listed is skipped.
local_var() {
  case "$1" in
    temperature_2m_max) echo temperature_max ;;
    temperature_2m_min) echo temperature_min ;;
    precipitation)      echo precipitation ;;
    *)                  echo "" ;;
  esac
}

if [ -n "$FROM" ]; then
  [ -d "$FROM/raw" ] && [ -d "$FROM/target" ] || {
    echo "!! $FROM does not look like a country data folder (expected raw/ and target/ inside)" >&2
    exit 1; }
  SOURCE=$(cd "$FROM" && pwd)
  ls_dirs() { find "$SOURCE/$1" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort; }
  # rsync filter: every year folder, but only the chosen model folders inside it.
  sync_forecasts() {
    local rules=(--include='/*/')
    local m; for m in ${MODELS//|/ }; do rules+=(--include="/*/$m/***"); done
    rsync -a ${DRY:+--dry-run} --out-format='  %n' "${rules[@]}" --exclude='*' "$1/" "$2/"
  }
  sync_tree() { rsync -a ${DRY:+--dry-run} --out-format='  %n' "$1/" "$2/"; }
  has_file() { [ -f "$1" ]; }
  copy_file() { if [ -n "$DRY" ]; then echo "  would copy $1"; else mkdir -p "$(dirname "$2")"; cp "$1" "$2"; fi; }
else
  [ "$SOURCE" != "-" ] || {
    echo "!! deployments for '$COUNTRY' has no data_source: pass --from <folder> instead" >&2
    exit 1; }
  command -v gcloud >/dev/null || {
    echo "!! needs the Google Cloud CLI (gcloud), or a local copy passed with --from" >&2
    exit 1; }
  if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    [ -r "$GOOGLE_APPLICATION_CREDENTIALS" ] || {
      echo "!! GOOGLE_APPLICATION_CREDENTIALS points at a file this account cannot read:" >&2
      echo "   $GOOGLE_APPLICATION_CREDENTIALS" >&2
      exit 1; }
    export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="${CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE:-$GOOGLE_APPLICATION_CREDENTIALS}"
  fi
  ls_dirs() {
    gcloud storage ls "$SOURCE/$1/" | sed -E 's#/$##; s#.*/##' | sort
  }
  # gcloud's --exclude is a Python regex on the path relative to the source folder.
  sync_forecasts() {
    gcloud storage rsync -r ${DRY:+--dry-run} --exclude="^[0-9]{4}/(?!($MODELS)/)" "$1" "$2"
  }
  sync_tree() { gcloud storage rsync -r ${DRY:+--dry-run} "$1" "$2"; }
  has_file() { gcloud storage ls "$1" >/dev/null 2>&1; }
  copy_file() { if [ -n "$DRY" ]; then echo "  would copy $1"; else mkdir -p "$(dirname "$2")"; gcloud storage cp "$1" "$2"; fi; }
fi

DRY=""; [ "$APPLY" -eq 1 ] || DRY=1
echo "country: $COUNTRY | source: $SOURCE | into: $DEST | models: ${MODELS//|/, }"

# Listed into a variable, not inside `for … in $(…)`: there a failure (no access, wrong
# path) would be ignored and the run would "succeed" having fetched nothing.
variables=$(ls_dirs raw/gridded) || {
  echo "!! cannot read $SOURCE/raw/gridded — check the path and your access (see the" >&2
  echo "   header of this script: GOOGLE_APPLICATION_CREDENTIALS or gcloud auth login)" >&2
  exit 1; }
[ -n "$variables" ] || { echo "!! no forecasts under $SOURCE/raw/gridded" >&2; exit 1; }

for bucket_var in $variables; do
  var=$(local_var "$bucket_var")
  if [ -z "$var" ]; then echo "  note: skipping unknown variable '$bucket_var'"; continue; fi
  echo "── forecasts: $var"
  [ -n "$DRY" ] || mkdir -p "$DEST/forecast/gridded/$var"
  sync_forecasts "$SOURCE/raw/gridded/$bucket_var" "$DEST/forecast/gridded/$var"
  products=$(ls_dirs "target/gridded/$bucket_var")
  for product in $products; do
    lower=$(printf '%s' "$product" | tr '[:upper:]' '[:lower:]')
    echo "── truth: $var, $lower"
    [ -n "$DRY" ] || mkdir -p "$DEST/truth/$lower/gridded/$var"
    sync_tree "$SOURCE/target/gridded/$bucket_var/$product" "$DEST/truth/$lower/gridded/$var"
  done
done

# Elevation, for the altitude bands. Optional: without it everything still runs and the
# dashboard simply offers no altitude selector.
echo "── elevation"
if has_file "$SOURCE/assets/dem.nc"; then
  copy_file "$SOURCE/assets/dem.nc" "$DEST/assets/dem.nc"
else
  echo "  note: no assets/dem.nc in the source — altitude bands will be unavailable"
fi

if [ -n "$DRY" ]; then
  echo "DRY RUN — nothing was downloaded. Re-run with --apply."
else
  echo "done — data is in $DEST"
fi
