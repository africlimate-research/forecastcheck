#!/usr/bin/env bash
# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
#
# ForecastCheck with Docker: every command a workshop needs, without typing docker flags.
#
#   bash docker/run.sh build              build the image (once; ~10 min)
#   bash docker/run.sh load FILE          ...or load one a trainer saved (see: save)
#   bash docker/run.sh start              start the dashboard; it stays up
#   bash docker/run.sh fetch [ARGS]       scripts/fetch_country.sh (dry run unless --apply)
#   bash docker/run.sh fetch --from DIR --apply    add a local folder (a flash disk), read-only
#   bash docker/run.sh run CMD ...        one command, e.g. run python -m verification.run ...
#   bash docker/run.sh pipeline [OPTS]    scripts/run_pipeline.sh in the background: a quick
#                                         slice (2022, days 1 and 3); --year 2024, --leads 1,3,5,7,
#                                         --full (everything) ... see scripts/run_pipeline.sh
#   bash docker/run.sh logs [pipeline]    follow the dashboard's (or the pipeline's) log
#   bash docker/run.sh status             what is running, for which country, on which port
#   bash docker/run.sh shell              a terminal inside the image
#   bash docker/run.sh restart            re-read results and .deployment (same as start)
#   bash docker/run.sh stop               stop the dashboard (data/ is kept)
#   bash docker/run.sh update             git pull, then rebuild and restart
#   bash docker/run.sh save               write forecastcheck-image.tar.gz for another machine
#
# Options, set before the command:
#   PORT=8503   host port of the dashboard (default 8502; the course's own app uses 8501)
#   DATA=/path  the data folder (default: data/ in this checkout)
#   DEV=1       use the code in this checkout instead of the image's copy (edits apply live)
#
# The country is the one in .deployment (echo kenya > .deployment). The dashboard is a
# long-running container; every other command runs in a short-lived one from the same
# image, with the same data/, so it always sees the current .deployment and key. A Google
# Cloud key is used only by its path: if GOOGLE_APPLICATION_CREDENTIALS is set, that one
# file is mounted read-only at the same path. Nothing else from your home is visible.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${FC_IMAGE:-forecastcheck}"
NAME=forecastcheck
PIPELINE="$NAME-pipeline"
PORT="${PORT:-8502}"
DATA="${DATA:-$HERE/data}"

die() { echo "!! $*" >&2; exit 1; }

# This machine's country, as verification.config reads it: the env var, else .deployment.
country() {
  if [ -n "${VERIFICATION_DEPLOYMENT:-}" ]; then echo "$VERIFICATION_DEPLOYMENT"; return; fi
  if [ -f "$HERE/.deployment" ]; then
    grep -v '^[[:space:]]*\(#\|$\)' "$HERE/.deployment" | head -1 | tr -d '[:space:]'
    return
  fi
  echo none
}

need_image() {
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "no '$IMAGE' image yet: run 'bash docker/run.sh build' (or 'load FILE')"
}

# Flags shared by every container: the data, the country, the key, and who writes files.
common_args() {
  mkdir -p "$DATA/logs"
  ARGS=(-v "$DATA:/app/data" -e "VERIFICATION_DEPLOYMENT=$(country)" -e "FETCH_SHOWN_DATA=$DATA")
  if [ "${DEV:-0}" = 1 ]; then ARGS+=(-v "$HERE:/app"); fi
  if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    [ -r "$GOOGLE_APPLICATION_CREDENTIALS" ] \
      || die "GOOGLE_APPLICATION_CREDENTIALS points at a file you cannot read: $GOOGLE_APPLICATION_CREDENTIALS"
    ARGS+=(-v "$GOOGLE_APPLICATION_CREDENTIALS:$GOOGLE_APPLICATION_CREDENTIALS:ro"
           -e "GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS")
  fi
  # Rootless Docker (the Spark) maps the container's root to you, and Docker Desktop
  # maps files to you as well. Only rootful Docker on Linux needs to be told who you are,
  # or every file it writes to data/ would belong to root.
  if [ "$(uname -s)" = Linux ] \
     && ! docker info --format '{{.SecurityOptions}}' 2>/dev/null | grep -q rootless; then
    ARGS+=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
  fi
}

# A terminal is attached only when there is one, so the commands also work from scripts.
tty_args() { if [ -t 0 ] && [ -t 1 ]; then echo "-it"; else echo "-i"; fi; }

# One short-lived container from the image, with the shared flags.
in_image() {
  need_image
  common_args
  docker run --rm $(tty_args) "${ARGS[@]}" "$@" "$IMAGE" "${CMD[@]}"
}

start() {
  need_image
  common_args
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --restart unless-stopped \
    -p "127.0.0.1:$PORT:8501" "${ARGS[@]}" "$IMAGE" >/dev/null
  echo "ForecastCheck dashboard for '$(country)': http://localhost:$PORT"
  echo "From your laptop:  ssh -L $PORT:localhost:$PORT <user>@<this machine>"
}

pipeline() {
  need_image
  if docker container inspect "$PIPELINE" >/dev/null 2>&1; then
    die "a pipeline is already running: follow it with 'bash docker/run.sh logs pipeline'"
  fi
  common_args
  local log="logs/pipeline-$(country).log"
  # The options (--year, --leads, --full, ...) reach the script as real arguments, not
  # pasted into the command string, so nothing in them is re-interpreted by a shell.
  docker run -d --rm --name "$PIPELINE" "${ARGS[@]}" "$IMAGE" bash -c '
    set -o pipefail; log="data/$1"; shift
    ./scripts/run_pipeline.sh "$@" 2>&1 | tee "$log"
    echo "pipeline finished with exit status $?" | tee -a "$log"' _ "$log" "$@" >/dev/null
  # When it ends, restart the dashboard so it sees the new results. Run on this machine,
  # detached, so closing the SSH session does not stop it.
  nohup bash -c "docker wait '$PIPELINE' >/dev/null 2>&1;
                 docker container inspect '$NAME' >/dev/null 2>&1 && docker restart '$NAME'" \
    >/dev/null 2>&1 &
  disown || true
  echo "Pipeline started for '$(country)' ${*:-(quick: 2022, lead days 1 and 3)}. Follow it:  bash docker/run.sh logs pipeline"
  echo "It runs on if you log out; the dashboard restarts by itself when it ends."
}

status() {
  echo "country (.deployment): $(country)"
  if docker container inspect "$NAME" >/dev/null 2>&1; then
    echo "dashboard: $(docker inspect -f '{{.State.Status}}' "$NAME") at http://localhost:$(docker port "$NAME" 8501 | head -1 | sed 's/.*://')"
  else
    echo "dashboard: not running (bash docker/run.sh start)"
  fi
  if docker container inspect "$PIPELINE" >/dev/null 2>&1; then
    echo "pipeline: running (bash docker/run.sh logs pipeline)"
  else
    echo "pipeline: not running"
  fi
}

cmd="${1:-help}"; [ $# -gt 0 ] && shift
case "$cmd" in
  build)    docker build -t "$IMAGE" -f "$HERE/docker/Dockerfile" "$HERE" ;;
  start|restart) start ;;
  stop)     docker rm -f "$NAME" >/dev/null 2>&1 && echo "dashboard stopped (data/ is kept)" \
              || echo "dashboard was not running" ;;
  status)   status ;;
  fetch)
    # Reading the bucket needs a key by its path; a gcloud login on this machine is not
    # visible inside the container. Say so now, not after a container has started.
    case " $* " in
      *" --from "*) ;;
      *) [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || die "the data bucket needs a Google Cloud key, given by its path:
     export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json     (DGX Spark: /etc/fleet-gcs-key.json)
   then run this command again. (Or fetch a local copy: bash docker/run.sh fetch --from <folder>)" ;;
    esac
    # --from DIR: that folder is mounted read-only at /import, and only for this command.
    extra=(); CMD=(./scripts/fetch_country.sh)
    while [ $# -gt 0 ]; do
      if [ "$1" = --from ]; then
        dir="$(cd "${2:?--from needs a folder}" && pwd)" || die "no such folder: ${2:-}"
        extra=(-v "$dir:/import:ro" -e "FETCH_SHOWN_FROM=$dir"); CMD+=(--from /import); shift 2
      else
        CMD+=("$1"); shift
      fi
    done
    in_image ${extra[@]+"${extra[@]}"} ;;   # bash 3.2 (macOS): an empty array needs this form
  run)      [ $# -gt 0 ] || die "usage: bash docker/run.sh run COMMAND ..."
            CMD=("$@"); in_image ;;
  shell)    CMD=(bash); in_image ;;
  pipeline) pipeline "$@" ;;
  logs)
    if [ "${1:-}" = pipeline ]; then
      log="$DATA/logs/pipeline-$(country).log"
      [ -f "$log" ] || die "no pipeline log yet for '$(country)' ($log)"
      tail -n 50 -f "$log"
    else
      docker logs -f "$NAME"
    fi ;;
  update)
    git -C "$HERE" pull --ff-only
    if [ "${DEV:-0}" = 1 ]; then start; else docker build -t "$IMAGE" -f "$HERE/docker/Dockerfile" "$HERE" && start; fi ;;
  save)     need_image; docker save "$IMAGE" | gzip > forecastcheck-image.tar.gz
            echo "wrote forecastcheck-image.tar.gz — on the other machine: bash docker/run.sh load forecastcheck-image.tar.gz" ;;
  load)     docker load < "${1:?usage: bash docker/run.sh load FILE}" ;;
  help|-h|--help) awk 'NR >= 5 && /^set -euo/ {exit} NR >= 5 {sub(/^# ?/, ""); print}' "$0" ;;
  *)        die "unknown command '$cmd' — see: bash docker/run.sh help" ;;
esac
