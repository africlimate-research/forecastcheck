# ForecastCheck in one image: the conda environment, the code, the Google Cloud CLI and
# the map data the dashboard draws. No data, no credentials and no country are baked in:
# docker/run.sh mounts data/ and the key, and passes the country, at run time.
#
#   bash docker/run.sh build        (or: docker build -t forecastcheck -f docker/Dockerfile .)
#   bash docker/run.sh start        see docker/run.sh for every command
#
# Runs on the DGX Spark (arm64) and on laptops and servers (amd64). CPU only: ForecastCheck
# needs no GPU. The container runs as root, which rootless Docker (the Spark) maps to the
# user who started it, so files written to data/ belong to that user on the host.
FROM condaforge/miniforge3:26.7.2-0

SHELL ["/bin/bash", "-c"]
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The environment first, from the same environment.yml as a non-Docker install, so the
# two never drift. Copied on its own so a code change does not rebuild it.
COPY environment.yml requirements.txt /tmp/env/
RUN mamba env create -f /tmp/env/environment.yml \
 && mamba clean -afy \
 && rm -rf /tmp/env

ENV PATH=/opt/conda/envs/forecastcheck/bin:$PATH \
    CARTOPY_DATA_DIR=/opt/cartopy \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

# The Natural Earth layers the dashboard's maps draw (ui/utils/map_plot.py), fetched now so
# the maps work on a machine with no internet during class. CARTOPY_DATA_DIR above points
# cartopy at them for any user.
RUN python -c "\
import cartopy, cartopy.io.shapereader as sr; \
cartopy.config['data_dir'] = '/opt/cartopy'; \
[sr.natural_earth(resolution='50m', category=c, name=n) for c, n in ( \
    ('physical', 'ocean'), ('physical', 'land'), ('physical', 'coastline'), \
    ('cultural', 'admin_0_boundary_lines_land'))]" \
 && chmod -R a+rX /opt/cartopy

# rsync: scripts/fetch_country.sh --from copies a local folder (a flash disk) with it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends rsync \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]
