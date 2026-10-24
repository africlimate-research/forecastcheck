#!/usr/bin/env bash
# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
#
# Fetch this machine's country data (the country named in .deployment) into data/.
#
#   {SOURCE}/raw/{shape}/{bucket_var}/{year}/{model}/{lead}h.nc
#     -> data/{country}/forecast/{shape}/{variable}/{year}/{model}/{lead}h.nc
#   {SOURCE}/target/{shape}/{bucket_var}/{PRODUCT}/{year}.nc
#     -> data/{country}/truth/{product}/{shape}/{variable}/{year}.nc
#   {SOURCE}/assets/dem.nc
#     -> data/{country}/assets/dem.nc
#
# {shape} is `gridded` (a grid of cells) or `station` (a list of gauges); each is fetched
# when the source has it.
#
# {SOURCE} is the country file's `data_source` (deployments/{name}.toml), or a local copy
# of a country folder passed with --from (a USB drive, a shared disk). The source is only
# ever read, and a --from copy is ADDED to what is already in data/: your own national data
# (rain gauges, a gauge-corrected grid), kept on a local disk because it cannot be
# published, completes what the bucket gave. Three renames happen on the way in, none of
# which touches a file's contents:
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
# half-written file behind, so an interrupted download simply resumes. After --apply the
# script counts what arrived against what the source holds, and fails if anything is
# missing rather than reporting "done".
#
# Usage:  ./scripts/fetch_country.sh                 # dry run: list what would be fetched
#         ./scripts/fetch_country.sh --apply
#         ./scripts/fetch_country.sh --apply --with-aifs
#         ./scripts/fetch_country.sh --from /path/to/SEN-local-engine --apply
#         ./scripts/fetch_country.sh --from /media/$USER/USB/<folder> --apply   # your own data
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
  ls_files() { find "$1" -type f -name '*.nc'; }
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
  # gcloud's own libraries warn that Python 3.11 will lose support in 2027; nothing
  # to act on here, and it buries the messages that matter. Silenced for gcloud only.
  gcloud() { PYTHONWARNINGS=ignore::FutureWarning command gcloud "$@"; }
  type -P gcloud >/dev/null || {          # the program on PATH, not the function above
    echo "!! needs the Google Cloud CLI (gcloud), or a local copy passed with --from" >&2
    exit 1; }
  if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    [ -r "$GOOGLE_APPLICATION_CREDENTIALS" ] || {
      echo "!! GOOGLE_APPLICATION_CREDENTIALS points at a file this account cannot read:" >&2
      echo "   $GOOGLE_APPLICATION_CREDENTIALS" >&2
      exit 1; }
    export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="${CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE:-$GOOGLE_APPLICATION_CREDENTIALS}"
  fi
  # Download every file whole. On a machine with 4+ CPUs (the DGX Spark) gcloud splits a
  # file above 10 MB into slices, which it can verify only with the google-crc32c
  # library; without it, it SKIPS the file. Senegal's and Kenya's forecasts are 12-15 MB.
  # Whole files are verified by MD5, which needs nothing extra. A value you set wins.
  export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD="${CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD:-0}"
  ls_dirs() {
    gcloud storage ls "$SOURCE/$1/" | sed -E 's#/$##; s#.*/##' | sort
  }
  ls_files() { gcloud storage ls "$1/**" | { grep '\.nc$' || true; }; }
  # gcloud's --exclude is a Python regex on the path relative to the source folder.
  sync_forecasts() {
    gcloud storage rsync -r ${DRY:+--dry-run} --exclude="^[0-9]{4}/(?!($MODELS)/)" "$1" "$2"
  }
  sync_tree() { gcloud storage rsync -r ${DRY:+--dry-run} "$1" "$2"; }
  has_file() { gcloud storage ls "$1" >/dev/null 2>&1; }
  copy_file() { if [ -n "$DRY" ]; then echo "  would copy $1"; else mkdir -p "$(dirname "$2")"; gcloud storage cp "$1" "$2"; fi; }
fi

DRY=""; [ "$APPLY" -eq 1 ] || DRY=1
# Inside Docker (docker/run.sh) the folders are mounted under other names; the wrapper
# passes the host's names so these messages show what you would type.
SHOWN_DEST="$DEST"; SHOWN_SOURCE="$SOURCE"
[ -z "${FETCH_SHOWN_DATA:-}" ] || SHOWN_DEST="$FETCH_SHOWN_DATA/$COUNTRY"
[ -z "${FETCH_SHOWN_FROM:-}" ] || SHOWN_SOURCE="$FETCH_SHOWN_FROM"
echo "country: $COUNTRY | source: $SHOWN_SOURCE | into: $SHOWN_DEST | models: ${MODELS//|/, }"

# Listed into variables, not inside `for … in $(…)`: there a failure (no access, wrong
# path) would be ignored and the run would "succeed" having fetched nothing.
raw_shapes=$(ls_dirs raw) && truth_shapes=$(ls_dirs target) || {
  echo "!! cannot read $SHOWN_SOURCE — check the path and your access (see the header of" >&2
  echo "   this script: GOOGLE_APPLICATION_CREDENTIALS or gcloud auth login)" >&2
  exit 1; }

# After --apply, every file the source holds must be here. A download tool that skips a
# file it cannot verify must not end in "done": count what the source has (forecasts
# through the same model filter) and what arrived, and remember any shortfall.
FORECAST_FILE="/[0-9]{4}/($MODELS)/[^/]+\.nc$"
missing=0
synced=0
check_arrived() {   # check_arrived <source folder> <destination folder> <path regex>
  synced=$((synced + 1))
  [ -z "$DRY" ] || return 0
  local want have
  want=$(ls_files "$1" | { grep -Ec "$3" || true; })
  have=$(find "$2" -type f -name '*.nc' 2>/dev/null | { grep -Ec "$3" || true; })
  if [ "$have" -lt "$want" ]; then
    echo "  !! only $have of $want files arrived in $2" >&2
    missing=$((missing + want - have))
  fi
}
listed() { grep -qx "$1" <<< "$2"; }   # listed <name> <newline-separated list>

# Two shapes: gridded (a grid of cells) and station (a list of gauges). A source may hold
# forecasts without truth or truth without forecasts for a shape — a local folder of
# national data added to what the bucket already gave — so both raw/ and target/ are
# walked, each for whatever it holds.
for shape in gridded station; do
  in_raw=""; in_target=""
  listed "$shape" "$raw_shapes" && in_raw=1
  listed "$shape" "$truth_shapes" && in_target=1
  [ -n "$in_raw$in_target" ] || continue
  fvars=""; tvars=""
  [ -z "$in_raw" ] || fvars=$(ls_dirs "raw/$shape")
  [ -z "$in_target" ] || tvars=$(ls_dirs "target/$shape")
  for bucket_var in $(printf '%s\n%s\n' "$fvars" "$tvars" | sort -u); do
    var=$(local_var "$bucket_var")
    if [ -z "$var" ]; then echo "  note: skipping unknown variable '$bucket_var'"; continue; fi
    if listed "$bucket_var" "$fvars"; then
      echo "── forecasts: $var ($shape)"
      [ -n "$DRY" ] || mkdir -p "$DEST/forecast/$shape/$var"
      sync_forecasts "$SOURCE/raw/$shape/$bucket_var" "$DEST/forecast/$shape/$var"
      check_arrived "$SOURCE/raw/$shape/$bucket_var" "$DEST/forecast/$shape/$var" "$FORECAST_FILE"
    fi
    listed "$bucket_var" "$tvars" || continue
    products=$(ls_dirs "target/$shape/$bucket_var")
    for product in $products; do
      lower=$(printf '%s' "$product" | tr '[:upper:]' '[:lower:]')
      echo "── truth: $var, $lower ($shape)"
      [ -n "$DRY" ] || mkdir -p "$DEST/truth/$lower/$shape/$var"
      sync_tree "$SOURCE/target/$shape/$bucket_var/$product" "$DEST/truth/$lower/$shape/$var"
      check_arrived "$SOURCE/target/$shape/$bucket_var/$product" \
                    "$DEST/truth/$lower/$shape/$var" '\.nc$'
    done
  done
done
[ "$synced" -gt 0 ] || { echo "!! nothing to fetch under $SOURCE/raw or $SOURCE/target" >&2; exit 1; }

# Elevation, for the altitude bands. Optional: without it everything still runs and the
# dashboard simply offers no altitude selector.
echo "── elevation"
if has_file "$SOURCE/assets/dem.nc"; then
  copy_file "$SOURCE/assets/dem.nc" "$DEST/assets/dem.nc"
elif [ -f "$DEST/assets/dem.nc" ]; then
  echo "  already here (this source has none)"
else
  echo "  note: no assets/dem.nc in the source — altitude bands will be unavailable"
fi

if [ -n "$DRY" ]; then
  echo "DRY RUN — nothing was downloaded. Re-run with --apply."
elif [ "$missing" -gt 0 ]; then
  echo "!! $missing file(s) did not arrive (see above). Re-run the same command to resume;" >&2
  echo "   if they are skipped again, read the messages above this one." >&2
  exit 1
else
  echo "done — data is in $SHOWN_DEST"
fi
