---
title: "Demo 5"
teaching: 45
exercises: 75
questions:
- "Over years of past forecasts, how good are GraphCast and IFS for daily temperature in my country, my regions and my seasons?"
- "How do we run a local verification on our own machine?"
- "What does calibration with IDR change in a forecast, how is it trained, and how do we know it helped?"
- "How do we read MAE, bias, correlation, CRPS, ROC, reliability and cost-loss value, and which question does each answer?"
objectives:
- "Set up ForecastCheck on the DGX Spark with conda and fetch Senegal's or Sierra Leone's forecasts and truth data from the cloud bucket."
- "Run the three pipeline stages (raw metrics, IDR calibration, calibrated metrics) from the command line."
- "Explain how IDR turns a single forecast value into a calibrated probability distribution, and why it is trained and tested on different years."
- "Interpret deterministic, probabilistic and threshold diagnostics in the ForecastCheck dashboard, per region, season and altitude band."
- "Identify what the results depend on (truth product, years, thresholds, sample size) and carry those questions into the ground truth challenge."
keypoints:
- "A single forecast cannot tell you whether a model is trustworthy. Years of past forecasts scored against observations can."
- "ForecastCheck runs locally in three stages that talk only through files: raw metrics, IDR calibration, calibrated metrics. The dashboard reads what they wrote."
- "IDR learns, per grid cell and lead time, how the observed temperature is distributed given the forecast value. It must be trained and tested on different years."
- "Calibration fixes bias and reliability. It cannot add discrimination the raw forecast does not have."
- "Every number depends on the truth product. ERA5 and CHIRTS disagree by more than the models' own errors, which is the question of Demo 6."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Putting Your Forecast to the Test

In Demo 3 you ran AI models and produced one forecast for one event. In Demo 4 you benchmarked rainy-season onset and met calibration with IDR for the first time. This demo answers the question a forecaster asks before using any of it operationally: **over years of past forecasts, how good is this model here, in my regions, in my seasons, and for the thresholds my warnings use?**

You will do this with **ForecastCheck**, a local verification framework. Each agency runs its own copy, on its own machine, with its own regions and thresholds. You will fetch Senegal's or Sierra Leone's data from the cloud, score GraphCast and IFS daily maximum and minimum temperature against observations, calibrate them with IDR, and read the results in a dashboard.

> #### Prerequisites
>
> - An account on the DGX Spark and the SSH tunnel from Demo 1. You do **not** need Docker for this demo, and you do **not** need a Google Cloud account.
> - The temperature-exceedance use case from Demo 3 (what a lead time is, what a hot-day threshold is).
> - The IDR section of Demo 4 (what calibration can and cannot fix).
{: .prereq}

## Learning Objectives

By the end of this lesson, you will be able to:

- Set up ForecastCheck with conda and fetch a country's data from the cloud bucket.
- Run raw verification, IDR calibration and calibrated verification from the command line.
- Explain how IDR is trained and what it produces.
- Read MAE, bias, correlation, CRPS, ROC/AUC, reliability diagrams and cost-loss value curves.
- Use the dashboard to compare models by region, season, altitude band and lead time.
- State what each result depends on before quoting it.

## Lesson Roadmap

| Section | Purpose |
| --- | --- |
| 1. From One Forecast to a Track Record | The question, the two country teams, what is verified |
| 2. ForecastCheck in One Picture | The three stages and the files that connect them |
| 3. Set Up on the DGX Spark | conda environment and your country |
| 4. Get the Data from the Cloud | What is in the bucket, and how to fetch it |
| 5. The Country Settings | Thresholds, seasons, altitude bands and regions |
| 6. Stage 1: Raw Forecast Metrics | MAE, bias and correlation |
| 7. Stage 2: Calibration with IDR | How the calibration is trained and applied |
| 8. Stage 3: Calibrated Metrics | CRPS, ROC, reliability and cost-loss value |
| 9. The Dashboard | A guided tour, page by page |
| 10. What the Results Say | Senegal and Sierra Leone, side by side |
| 11. Before You Quote a Number | Limits of the data, and the bridge to Demo 6 |
| Exercises and Discussion Sheet | Hands-on practice in country teams |

## 1. From One Forecast to a Track Record

A forecast of 41 °C in Matam for next Tuesday is only useful if you know how often this model has been right about days like that. Verification answers that by scoring **many past forecasts** (here, several years of daily forecasts) against **what was observed**.

You will work in two country teams. Both teams run the same commands; only the data and the settings differ.

| | Team Senegal | Team Sierra Leone |
| --- | --- | --- |
| Country (`.deployment`) | `senegal` | `sierra-leone` |
| Bucket folder | `SEN-local-engine` | `SLE-local-engine` |
| Variables | daily max (`temperature_max`) and min (`temperature_min`) temperature | same |
| Models | GraphCast (2016–2026), IFS (2018–2026), no 2023 | same |
| Lead times | days 1, 3, 5, 7 | same |
| Truth products | CHIRTS (0.05°, observation-based), ERA5 (reanalysis) | same |
| Regions | 14 regions over 45 departments | 4 provinces over 14 districts |
| Seasons | NDJF, MAMJ, JASO, plus the whole year | ONDJ, FMAM, JJAS, plus the whole year |
| Hot-day thresholds (Tmax) | 37, 38 … 45 °C (nine) | 37, 38, 39, 40 °C |

> #### Why temperature, and why two truths?
>
> Neither country has gauge data in ForecastCheck yet, so both are verified against gridded products. **CHIRTS** is built from station and satellite observations. **ERA5** is a reanalysis: a weather model constrained by observations. They do not agree, and the disagreement is large enough to change conclusions. Keep that in mind; it is the subject of Demo 6.
{: .tip}

## 2. ForecastCheck in One Picture

ForecastCheck has three stages. They never call each other: each one writes NetCDF files under `data/`, and the next one finds them there. The dashboard only shows what exists on disk.

```
                 cloud bucket  gs://aim4scale_training_2026/demo5/{SEN|SLE}-local-engine
                      │
                      │  scripts/fetch_country.sh
                      ▼
 data/{country}/forecast/…    raw GraphCast / IFS forecasts
 data/{country}/truth/{chirts|era5}/…   observations
 data/{country}/assets/dem.nc           elevation, for the altitude bands
                      │
     ┌────────────────┼──────────────────────────────┐
     ▼                ▼                              │
 Stage 1          Stage 2                            │
 verification.run calibration.run (IDR)              │
 raw metrics      train 2016–2021, apply to 2022     │
 MAE, bias,       → data/{country}/idr/…             │
 correlation                │                        │
     │                      ▼                        │
     │               Stage 3                         │
     │               verification.run  --model *-idr │
     │               CRPS, ROC, reliability, cost-loss
     ▼                      ▼                        │
 data/{country}/diagnostics/…   ◄────────────────────┘
                      │
                      ▼
               streamlit run ui/app.py   (the dashboard)
```

Two naming rules make the files self-describing:

- A model token without a suffix (`graphcast`, `ifs`) is the **raw** forecast. The suffix `-idr` (`graphcast-idr`) is the **calibrated** one.
- Anything derived from observations carries the name of the truth product it came from (`chirts` or `era5`). A raw forecast does not.

## 3. Set Up on the DGX Spark

Log in to the Spark with the port forward from Demo 1, so the dashboard will reach your laptop later:

```bash
ssh -L 8501:localhost:8501 <user>@<spark>
```

ForecastCheck runs in a **conda** environment. If `conda --version` already prints a version, skip the Miniforge install.

```bash
# 1. Install Miniforge (once per account)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
~/miniforge3/bin/conda init
source ~/.bashrc
```

Close the terminal, open a new one (with the same `ssh -L` command), then:

```bash
# 2. Get the code
git clone https://github.com/africlimate-research/forecastcheck.git
cd forecastcheck

# 3. Create and activate the environment (about 5 minutes)
conda env create -f environment.yml
conda activate forecastcheck

# 4. Choose your country (Team Sierra Leone: echo sierra-leone > .deployment)
echo senegal > .deployment
```

Every command in this demo runs inside `forecastcheck`. A new terminal needs `conda activate forecastcheck` again.

> #### Which country am I?
>
> The one-line file `.deployment` decides: `senegal` or `sierra-leone`. Both teams run the same code from the same `main`; the country file it names (`deployments/senegal.toml`, `deployments/sierra-leone.toml`) sets the map, the regions, the seasons and where the data comes from. `.deployment` is not in git, so `git pull` never changes it. There is nothing else to set.
{: .tip}

## 4. Get the Data from the Cloud

### 4.1 What is in the bucket

Each country has one folder in the bucket. The forecasts and the truth have already been put on the same 0.05° grid and in the same units (°C), so ForecastCheck only has to match dates.

```
gs://aim4scale_training_2026/demo5/SEN-local-engine/
├── raw/gridded/temperature_2m_max/{year}/{graphcast|ifs|…}/{24,72,120,168,216}h.nc forecasts
├── raw/gridded/temperature_2m_min/…
├── target/gridded/temperature_2m_max/{CHIRTS|ERA5}/{year}.nc                       truth
├── target/gridded/temperature_2m_min/…
└── assets/dem.nc                                                                   elevation
```

A file such as `raw/gridded/temperature_2m_max/2022/graphcast/72h.nc` holds every GraphCast forecast started in 2022, read at a lead time of 72 hours (day 3), for every grid cell in the country.

### 4.2 Fetch it

The bucket is read-only and private to the training. On the DGX Spark a key that can read it is already installed; tell the Google Cloud tools where it is, **by its path**:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/etc/fleet-gcs-key.json
```

Never open the key, paste its contents into a file or a chat, or copy it elsewhere: the path is all anyone needs. The variable lasts for the current terminal; set it again in a new one.

`scripts/fetch_country.sh` knows your country from `.deployment` and where its data lives from the country file. It downloads the GraphCast and IFS forecasts, both truth products (CHIRTS and ERA5) and the elevation map, and lays the files out the way ForecastCheck expects. Always look before you download: without `--apply` it only lists what it would fetch.

```bash
./scripts/fetch_country.sh            # dry run: lists what would be fetched
./scripts/fetch_country.sh --apply    # downloads
```

Both teams run the same two commands.

| | Senegal | Sierra Leone |
| --- | --- | --- |
| Download (forecasts + both truths + elevation) | about 3.5 GB | about 1.7 GB |
| Files | 235 | 235 |

Re-running is safe: it copies only what is missing or changed, and never leaves a half-written file behind, so an interrupted download simply resumes.

> #### If the fetch stops
>
> - *"cannot read gs://…"*: the key is not set in this terminal. Run the `export` line above again. (The script also sets `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE` to the same path, which some setups need.)
> - *"needs the Google Cloud CLI"*: the conda environment is not active; run `conda activate forecastcheck`.
> - No network, or a slow one? Your trainer can give you a copy of the country folder (for example on a shared disk). Point the script at it; nothing is downloaded and the result is identical:
>
> ```bash
> ./scripts/fetch_country.sh --from /path/to/SEN-local-engine --apply
> ```
{: .warning}

### 4.3 Check it landed

```bash
ls data/senegal/truth data/senegal/assets
```

Expect `chirts` and `era5` under `truth`, and `dem.nc` under `assets` (Team Sierra Leone: `data/sierra_leone/…`).

## 5. The Country Settings

Everything a forecaster is likely to change lives in one file, `config/verification.toml`. No Python.

```toml
[Senegal.thresholds]
temperature_max = [37, 38, 39, 40, 41, 42, 43, 44, 45]   # °C, one "hot day" event per value
temperature_min = []                                      # no threshold events for Tmin

[Senegal.altitude_bands]
"0-25"  = [0, 25]          # metres
">100"  = [100, "inf"]

[Senegal.seasons.ndjf]
months = [11, 12, 1, 2]
label  = "NDJF (Nov–Feb)"
```

- A **threshold** defines an event ("Tmax > 40 °C"). Every threshold produces its own ROC, reliability and cost-loss curves in Stage 3.
- A **season** is a set of months. Seasons, regions and departments all come out of the same run.
- An **altitude band** groups grid cells by elevation from `dem.nc`.

> #### Changing a setting means re-running
>
> Change `thresholds` and you must re-run Stage 2 and Stage 3. Change `seasons` and you must re-run Stage 3. A mistake in the file stops the run with a message naming the country and the key; it never silently falls back to a default.
{: .warning}

## 6. Stage 1: Raw Forecast Metrics

### 6.1 How a forecast is matched to an observation

A forecast started on 1 March at lead day 3 is **valid** on 4 March. It is compared with the observation for 4 March, in the same grid cell. Every (forecast, observation) pair for one lead time, one model and one year is scored, then averaged over grid cells and days, per month, per region and per altitude band.

### 6.2 The metrics

For $N$ pairs of forecast $f_i$ and observation $o_i$:

| Metric | Formula | Units | Perfect | Question it answers |
| --- | --- | --- | --- | --- |
| **MAE** (mean absolute error) | $\frac{1}{N}\sum_i \lvert f_i - o_i \rvert$ | °C | 0 | How far off is a typical forecast? |
| **Bias** (mean error) | $\frac{1}{N}\sum_i (f_i - o_i)$ | °C | 0 | Is the model systematically too warm (+) or too cold (−)? |
| **Pearson correlation** | $\frac{\mathrm{cov}(f,o)}{\sigma_f\,\sigma_o}$ | none | 1 | Does the forecast go up and down with the observations? |
| **Spearman correlation** | Pearson correlation of the ranks of $f$ and $o$ | none | 1 | Same, but insensitive to outliers and to non-linear relations |

> #### Read them together
>
> MAE mixes two things: a constant offset (bias) and day-to-day errors. A model with a bias of +2 °C and a high correlation is easy to fix with calibration. A model with no bias and a low correlation is not. That is why we look at bias and correlation before we calibrate.
{: .tip}

### 6.3 Run it

```bash
python -m verification.run --model graphcast ifs --variable temperature_max --metric mae bias pearsonr spearmanr --lead-day 1 3 5 7 --year 2022 --product chirts
python -m verification.run --model graphcast ifs --variable temperature_min --metric mae bias pearsonr spearmanr --lead-day 1 3 5 7 --year 2022 --product chirts
```

| Flag | Meaning |
| --- | --- |
| `--model graphcast ifs` | Raw forecasts of both models |
| `--variable` | Required: the default is precipitation, which these countries do not have |
| `--lead-day 1 3 5 7` | Lead times to score |
| `--year 2022` | The verification year; several are allowed (`--year 2020 2021 2022`) |
| `--product chirts` | Which truth to score against. Required: there are two |

Each command takes 1–3 minutes. For every model and lead time it prints one line, then the files it wrote:

```text
graphcast | lead 1d (24h) | temperature_max | gridded 2022
  wrote data/senegal/diagnostics/graphcast/temperature_max/chirts/gridded/mae/2022/annual_monthly_averages/24h.nc
  wrote data/senegal/diagnostics/graphcast/temperature_max/chirts/gridded/bias/2022/annual_monthly_averages/24h.nc
  …
```

The path tells you everything about a result: model, variable, truth product, metric, year and lead time.

> #### Two kinds of correlation
>
> The **map** of correlation shows, for each grid cell, how well the day-to-day ups and downs of the forecast follow the observations over the year. The **single national number** is different: for each day it measures how well the forecast *map* matches the observed *map* (a pattern correlation), then averages over the days. Both are useful, but they answer different questions. Say which one you quote.
{: .warning}

## 7. Stage 2: Calibration with IDR

### 7.1 The idea

A raw forecast says "Tmax will be 39 °C". Past data tells us how often, when this model said 39 °C in this grid cell, the observation was 37, 39 or 42 °C. **Isotonic Distributional Regression (IDR)** learns that relationship and turns every new forecast value into a full **probability distribution** for the observation.

![IDR on one grid cell near Matam, Senegal. Left: six years of GraphCast forecasts against CHIRTS observations, with the calibrated median and 5–95 % range. Right: April–May 2022, the raw forecast is about 3 °C too cold; the calibrated median follows the observations.](../fig/demo5_idr_concept.png)

*Figure 1. IDR on one real grid cell near Matam (GraphCast, day 1, against CHIRTS). On the left, every grey dot is a past day: what GraphCast said (x) and what was observed (y). Almost all dots lie above the dashed line, so GraphCast is too cold here. IDR learns the blue staircase: for each forecast value, the median and spread of what was then observed. On the right, the same mapping applied to forecasts in 2022, a year it never saw: the raw forecast (orange) is about 3.2 °C too cold, the calibrated median (blue) is off by about 0.7 °C.*

IDR makes one assumption only: **a higher forecast should never mean a colder outcome.** Mathematically, the predicted distribution must shift upward (or stay put) as the forecast increases. It does not assume a bell curve or any other shape. The fit is done with the pool-adjacent-violators algorithm.

### 7.2 How it is trained here

| Choice | In ForecastCheck | Why |
| --- | --- | --- |
| One fit per… | grid cell, lead time, model and truth product | Errors depend on place (coast vs. inland) and grow with lead time |
| Predictor | the raw forecast value only | Simple, monotone, needs no extra data |
| Training years | GraphCast 2016–2021, IFS 2018–2021 | IFS data starts in 2018 |
| Test year | 2022 | **Never** in the training years, otherwise the skill is overstated |
| Seasons | not separated in the fit | One fit uses all days of the training years |

### 7.3 What it produces

For each day of the test year, each grid cell and each lead time, `calibration.run` writes to `data/{country}/idr/{product}/…`:

| Output | Meaning |
| --- | --- |
| `q5`, `q50`, `q95`, `q99` | Quantiles of the calibrated distribution (°C). `q50` is the calibrated "best guess". |
| `p_exceed` | Probability that Tmax exceeds each threshold in the settings (37 °C, 38 °C, …) |
| `crps` | The score of the whole distribution against the observation (see Section 8) |
| `pit` | Where the observation fell in the distribution (0 to 1); used to check calibration |

### 7.4 Run it

```bash
python -m calibration.run --variable temperature_max --model graphcast --leads 1,3,5,7 --train-years 2016,2017,2018,2019,2020,2021 --test-year 2022 --product chirts
python -m calibration.run --variable temperature_max --model ifs --leads 1,3,5,7 --train-years 2018,2019,2020,2021 --test-year 2022 --product chirts
python -m calibration.run --variable temperature_min --model graphcast --leads 1,3,5,7 --train-years 2016,2017,2018,2019,2020,2021 --test-year 2022 --product chirts
python -m calibration.run --variable temperature_min --model ifs --leads 1,3,5,7 --train-years 2018,2019,2020,2021 --test-year 2022 --product chirts
```

For each lead it prints how many grid cells received a calibrated forecast:

```text
  lead 1d (24h) -> data/senegal/idr/chirts/gridded/temperature_max/2022/graphcast/24h.nc  (calibrated cells: 15476)
```

Calibration is the slow stage, because it fits one model per grid cell (about 15,000 in Senegal, 7,000 in Sierra Leone). Times measured for one variable and one truth product:

| Country | GraphCast, 4 leads | IFS, 4 leads | Whole pipeline, both truths | Calibrated files on disk |
| --- | --- | --- | --- | --- |
| Senegal | about 13 min | about 6.5 min | about 1.5 h | about 19 GB |
| Sierra Leone | about 6 min | about 3 min | about 45 min | about 7 GB |

The calibrated files are large, mostly because of the threshold probabilities (`p_exceed`), and Senegal has nine thresholds. Check the free disk space before running everything.

> #### In class: run a small slice, read the full results
>
> The complete pipeline has already been run on the Spark for both countries, so the dashboard has everything. In the session, each team runs **one** calibration live, to see it work:
>
> ```bash
> python -m calibration.run --variable temperature_max --model graphcast --leads 1,3 --train-years 2016,2017,2018,2019,2020,2021 --test-year 2022 --product chirts
> ```
>
> It takes about 5–10 minutes. Use the time for Exercise 3.
{: .tip}

> #### What IDR can and cannot do (recap from Demo 4)
>
> IDR removes bias and fixes over- and under-confidence. It cannot create skill the raw forecast does not have: if the raw forecast does not move with the observations (low correlation), the calibrated distribution will just be wide.
{: .tip}

## 8. Stage 3: Calibrated Metrics

```bash
python -m verification.run --model graphcast-idr ifs-idr --variable temperature_max --metric crps roc reliability cost_loss --lead-day 1 3 5 7 --year 2022 --product chirts
python -m verification.run --model graphcast-idr ifs-idr --variable temperature_min --metric crps roc reliability cost_loss --lead-day 1 3 5 7 --year 2022 --product chirts
```

`temperature_min` has no thresholds, so for it the run prints a note and computes CRPS only. Then repeat Stages 1–3 with `--product era5`, or run everything at once with `./scripts/run_pipeline.sh`.

### 8.1 CRPS: the score of a whole distribution

The **Continuous Ranked Probability Score** compares the predicted cumulative distribution $F$ with the observation $o$:

$$\mathrm{CRPS}(F, o) = \int_{-\infty}^{\infty} \big(F(x) - \mathbb{1}\{x \ge o\}\big)^2 \, dx$$

- It is in °C, and lower is better.
- For a single-value forecast, $F$ is a step at $f$ and the CRPS **equals the absolute error** $\lvert f - o \rvert$. So the raw MAE and the calibrated CRPS sit on the same axis, and the dashboard compares them directly. If CRPS(IDR) < MAE(raw), calibration helped.

### 8.2 Events and probabilities

For each threshold $t$ (for example 40 °C) an **event** is a day where the observed Tmax > $t$. The calibrated forecast gives a probability $p$ for that event. To turn $p$ into a yes/no warning, choose a cut-off $p^*$ and warn when $p \ge p^*$. Each cut-off gives a contingency table:

| | Event observed | Not observed |
| --- | --- | --- |
| **Warned** | Hit (H) | False alarm (FA) |
| **Not warned** | Miss (M) | Correct negative (CN) |

### 8.3 ROC curve and AUC: can the model tell event days from other days?

For 500 cut-offs between 0.01 and 1, ForecastCheck plots the hit rate $\mathrm{TPR} = H/(H+M)$ against the false-alarm rate $\mathrm{FPR} = FA/(FA+CN)$. The area under that curve (**AUC**) is 0.5 for a coin toss and 1 for a perfect model. AUC measures **discrimination**, and calibration barely changes it.

### 8.4 Reliability diagram: can I trust the probability?

Forecast probabilities are put into 10 bins. For each bin, the diagram plots the mean forecast probability (x) against how often the event actually happened (y). On a reliable forecast the points lie on the diagonal: when it says 30 %, it happens 30 % of the time.

- **Below the diagonal**: the event happens less often than forecast. The forecast is too sure of the event (over-forecasting).
- **Above the diagonal**: the event happens more often than forecast (under-forecasting).
- **Error bars**: a 95 % range from 50 bootstrap resamples of whole days. A point whose bar crosses the diagonal is not clearly unreliable.

The bins are not equally wide. After calibration, most days get a probability very close to 0 (a cool day is clearly not a 40 °C day), so ForecastCheck keeps one bin for those near-zero probabilities and splits the rest so that each bin holds the same number of forecasts. That is why the points are not evenly spaced along the x-axis.

### 8.5 Cost-loss value: is it worth acting on?

A user who protects against a hot day pays a cost $C$; if they do not protect and the event happens, they lose $L$. Their cost-loss ratio is $\alpha = C/L$. With $s$ the base rate (how often the event happens) and $n$ the number of cases:

$$E_{\text{clim}} = \min(\alpha, s), \qquad E_{\text{perfect}} = \alpha s, \qquad E_{\text{forecast}} = \alpha\,\frac{H + FA}{n} + \frac{M}{n}$$

$$V(\alpha) = \frac{E_{\text{clim}} - E_{\text{forecast}}}{E_{\text{clim}} - E_{\text{perfect}}}$$

ForecastCheck picks, for each $\alpha$, the cut-off $p^*$ that gives the best value. $V = 1$ is as good as a perfect forecast, $V = 0$ is no better than always or never acting based on climatology, and $V < 0$ means the forecast costs money.

> #### Read the event count first
>
> Under every curve the dashboard shows how many events are behind it. A curve built on a few hundred events can look excellent by chance. AUC in particular is blind to how rare an event is.
{: .warning}

## 9. The Dashboard

```bash
streamlit run ui/app.py
```

Open <http://localhost:8501> on your laptop (the SSH tunnel from Section 3 forwards it). If you produce new results while the dashboard is open, restart it: it only discovers files at start-up.

The screenshots below are from Team Senegal's results. Team Sierra Leone sees the same pages with its own provinces, districts and thresholds.

### 9.1 Choosing where: the region picker

Every page starts with the same picker. The top row is the whole country and its first-level areas (Senegal's 14 regions, Sierra Leone's 4 provinces). **More regions** opens a search box for the second level (Senegal's 45 departments, Sierra Leone's 14 districts). Everything below the picker is recomputed for the area you select.

![The region picker on the Senegal dashboard](../fig/demo5_sen_region_picker.png)

### 9.2 Every page has a configuration panel

On the left of each chart, the **Configuration** panel has three blocks:

| Block | What you choose |
| --- | --- |
| Forecast & Truth | Variable (Tmax or Tmin), models, truth product (CHIRTS or ERA5) |
| Score | The metric |
| Conditioning | Year, season, lead time, altitude band |

The menus only offer combinations that exist on disk. If a model or a metric is missing, the pipeline did not produce it (see Troubleshooting).

### 9.3 National Overview: raw forecast metrics

Open **National Overview** and select the **Raw Forecast Metrics** tab. This is Stage 1.

- **Graph View** plots a metric against lead time or month, one line per model.
- **Country View** maps it, grid cell by grid cell, for one model and one lead time.

![Raw bias of GraphCast and IFS against CHIRTS, by lead time](../fig/demo5_sen_no_raw_lead.png)

*Figure 2. Bias of the raw Tmax forecasts against CHIRTS, Senegal 2022. Both models are about 3 °C too cold at every lead time. IFS (blue) is the colder of the two.*

![Map of the raw GraphCast bias over Senegal](../fig/demo5_sen_no_raw_map.png)

*Figure 3. The same bias as a map (GraphCast, day 5). It is negative everywhere, slightly stronger on the coast. A bias that is this uniform is exactly what calibration can remove.*

### 9.4 National Overview: calibrated forecast metrics

The **Calibrated Forecast Metrics** tab shows the CRPS of the `-idr` forecasts (Stage 3). The **Month** view shows when in the year the calibrated forecast struggles.

![CRPS of the calibrated forecasts by month](../fig/demo5_sen_no_cal_month.png)

*Figure 4. CRPS of calibrated GraphCast and IFS by month, day 5. The errors are largest in June–July for GraphCast, and in October–December for IFS. A likely reason for the summer peak is the arrival of the monsoon, when day-to-day cloudiness makes Tmax harder to predict. Exercise 5 asks you to check this by season.*

![Map of the CRPS of calibrated GraphCast](../fig/demo5_sen_no_cal_map.png)

*Figure 5. CRPS of calibrated GraphCast, day 5. After calibration the remaining error is below 1 °C almost everywhere, and largest in the centre-north.*

### 9.5 Score Decomposition: did calibration help?

**Score Decomposition → Miscalibration** plots, for each model, $\mathrm{CRPS}_{\text{IDR}} - \mathrm{MAE}_{\text{raw}}$: the error removed by calibration. **Below zero means calibration helped.** With one model selected, you can compare both truth products on the same chart.

![Error removed by calibration for GraphCast, against CHIRTS and against ERA5](../fig/demo5_sen_sd_miscal.png)

*Figure 6. Calibration removes about 2.5 °C of GraphCast's error against CHIRTS, but only about 1 °C against ERA5. The model and the forecasts are the same; only the truth changed. Keep this figure in mind for Demo 6.*

### 9.6 Threshold Diagnostics: hot-day warnings

**Threshold Diagnostics** has three tabs, one per curve from Section 8, for one threshold, lead time, season and altitude band at a time. Under every chart, the **sample box** gives the number of events, the number of forecasts evaluated and the event's share.

![Reliability diagram for Tmax > 37 °C at day 5](../fig/demo5_sen_td_reliability.png)

*Figure 7. Reliability for Tmax > 37 °C at day 5. Both calibrated models are close to the diagonal. GraphCast sits slightly below it (a little over-forecasting), IFS slightly above. 37 °C is common in Senegal (43 % of cell-days), so the curve rests on almost a million events.*

![ROC curve for Tmax > 37 °C at day 5](../fig/demo5_sen_td_roc.png)

*Figure 8. ROC for the same event. Both curves hug the top-left corner: AUC 0.95 for GraphCast and 0.94 for IFS at day 5.*

![Cost-loss value for Tmax > 37 °C at day 5](../fig/demo5_sen_td_costloss.png)

*Figure 9. Value for the same event. The forecast is worth acting on for almost every user, and most valuable to users whose cost-loss ratio is close to the event's frequency (about 0.43 here), where climatology alone gives no guidance.*


## 10. What the Results Say

These are the reference results for 2022, computed with exactly the commands in this lesson. Use them to check your own run, and as the starting point for the discussion sheet. All numbers are national averages for daily maximum temperature (Tmax) against CHIRTS, unless stated otherwise.

### 10.1 Side by side

| Day 1 (24 h) | Senegal GraphCast | Senegal IFS | Sierra Leone GraphCast | Sierra Leone IFS |
| --- | --- | --- | --- | --- |
| Raw MAE | 3.20 °C | 3.42 °C | 4.39 °C | 4.40 °C |
| Raw bias | −3.17 °C | −3.38 °C | −4.37 °C | −4.38 °C |
| Pattern correlation | 0.89 | 0.86 | 0.60 | 0.56 |
| Calibrated CRPS | **0.66 °C** | 0.74 °C | **0.66 °C** | 0.71 °C |
| AUC, Tmax > 40 °C | 0.98 | 0.98 | 0.98 | 0.96 |
| Events > 40 °C in 2022 | 393,650 | 410,884 | 4,325 | 4,337 |

| Day 7 (168 h) | Senegal GraphCast | Senegal IFS | Sierra Leone GraphCast | Sierra Leone IFS |
| --- | --- | --- | --- | --- |
| Raw MAE | 2.90 °C | 3.55 °C | 4.35 °C | 4.40 °C |
| Calibrated CRPS | 0.88 °C | 0.91 °C | 0.78 °C | 0.81 °C |
| AUC, Tmax > 40 °C | 0.97 | 0.97 | 0.97 | 0.98 |

*Event counts are cell-days: one grid cell on one day. The two models have slightly different counts because each is scored only on days and cells where it has a forecast.*

### 10.2 Five findings

1. **Both models are much too cold against CHIRTS, everywhere.** About −3 °C in Senegal and −4.4 °C in Sierra Leone, almost the same at every lead time and in every grid cell (Figures 3 and 10). Most of the raw MAE is this bias.

2. **Calibration removes almost all of it.** After IDR the error is about 0.7 °C at day 1 in both countries, from very different starting points. In Sierra Leone, IDR removes about 3.7 °C of error against CHIRTS (Figure 11).

3. **For Tmax, after calibration, GraphCast is slightly better than IFS at every lead time,** by 0.03–0.09 °C of CRPS. This difference is small compared with the effect of calibration or of the truth product, and it does not hold for every variable: for Tmin in Senegal, calibrated IFS is better from day 5 on.

4. **Sierra Leone's maps are harder to get right.** The pattern correlation is 0.60 there, against 0.89 in Senegal. A likely reason: Sierra Leone is small and its temperature varies less from place to place, so the daily pattern is made of small details that a 0.25° model resolves poorly.

5. **The rarest thresholds need care.** In Sierra Leone, 40 °C happens on 0.5 % of cell-days against CHIRTS, and on 77 cell-days in the whole year against ERA5 (Figure 12). ERA5 almost never reaches 40 °C there, so the curve has nothing to learn from: AUC is exactly 0.5 and the value is 0. That threshold must be read against CHIRTS, or removed if ERA5 is the chosen truth.

![Raw GraphCast bias over Sierra Leone](../fig/demo5_sle_no_raw_map.png)

*Figure 10. Raw GraphCast Tmax bias against CHIRTS over Sierra Leone, day 5: about −4 °C over the whole country.*

![Error removed by calibration in Sierra Leone, against CHIRTS and ERA5](../fig/demo5_sle_sd_miscal.png)

*Figure 11. Error removed by calibration for GraphCast in Sierra Leone: about 3.7 °C against CHIRTS, 1.3 °C against ERA5.*

![ROC for Tmax > 40 °C against ERA5 in Sierra Leone: a straight line, AUC 0.5](../fig/demo5_sle_td_roc_era5_40.png)

*Figure 12. Tmax > 40 °C against ERA5 in Sierra Leone, day 5: 77 events in 780,364 cell-days. The ROC curve is the diagonal (AUC 0.5). This is not a model failure; the truth dataset almost never contains the event.*

## 11. Before You Quote a Number

Every number in the dashboard is true *for a given truth product, year, area and sample*. Read this list once before you present a result.

| Watch out for | Example from this demo | What to do |
| --- | --- | --- |
| **The truth product changes the answer** | Senegal GraphCast Tmax bias, day 1: −3.2 °C against CHIRTS, −1.6 °C against ERA5. For Tmin the **sign flips**: −0.9 °C against CHIRTS, +0.5 °C against ERA5. | Always say which truth you used. Never mix the two in one comparison. |
| **ERA5 is a model, not an observation** | ERA5 is smoother and clips the extremes: it runs about 1.3 °C below CHIRTS in Senegal and about 2.7 °C below in Sierra Leone. | Agreeing with ERA5 means agreeing with another model's analysis. |
| **One test year** | All calibrated scores are for 2022 only. 2023 is missing from the bucket. | Present them as "one year of evidence", not as a climatology of skill. |
| **AUC flatters rare events** | Senegal, ERA5, 45 °C: only 901 events in the whole year. AUC stays near 0.95–0.99 while the value at day 7 falls to about 0.6. | Read the event count in the sample box first; read reliability and value for rare thresholds. |
| **Small areas, few cells** | The Dakar departments and Sierra Leone's Western Area Urban are only a few 0.05° cells. The Gambia sits inside Senegal's outline and is excluded from every pool. | Prefer the region or province view for small areas. |
| **Calibration narrows the gap between models** | Raw Tmax MAE at day 1: GraphCast 3.2 °C, IFS 3.4 °C. After IDR: 0.66 °C and 0.74 °C. | Rank models on calibrated scores if you plan to calibrate operationally; on raw scores if you will use the raw output. |
| **Tmin has no threshold curves** | Warm nights follow cloud cover more than heat. | Tmin is judged on MAE, bias, correlation and CRPS only. |

> #### The bridge to Demo 6
>
> Figure 6 is the key result of this demo. The same GraphCast forecasts look like they have 2.5 °C of correctable error against CHIRTS, and 1 °C against ERA5. Before choosing a model, a team has to decide **which truth it trusts**, and ideally check both against its own stations. That is the ground truth challenge.
{: .tip}

## Troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `ModuleNotFoundError` | The conda environment is not active | `conda activate forecastcheck` |
| *"cannot read gs://…"* when fetching | The key is not set in this terminal | `export GOOGLE_APPLICATION_CREDENTIALS=/etc/fleet-gcs-key.json` (Section 4.2) |
| *"no country selected"* | `.deployment` is missing | `echo senegal > .deployment` (or `sierra-leone`) |
| *"… holds 2 truth products and none is configured"* | `--product` is missing | Add `--product chirts` or `--product era5` |
| The run looks for `precipitation` files | `--variable` is missing | Add `--variable temperature_max` or `temperature_min` |
| `SKIP — missing file` for IFS before 2018 | Expected: IFS starts in 2018 | Nothing |
| New results do not appear in the dashboard | The dashboard caches what it found at start-up | Stop and restart `streamlit run ui/app.py` |
| No altitude selector | `data/{country}/assets/dem.nc` is missing | Re-run the fetch; it prints a note when the DEM is absent |
| The dashboard does not open on the laptop | No port forward | Reconnect with `ssh -L 8501:localhost:8501 <user>@<spark>` |

## Exercises

Work in your country team. Exercises 1–3 run while the pipeline computes; Exercises 4–7 use the dashboard.

> #### Exercise 1: Read Before You Download (5 min)
>
> 1. Run the fetch script **without** `--apply`. How many files would it fetch, and how many megabytes?
> 2. Which lead times are in the bucket? Which one is not in the pipeline commands, and why might that be?
> 3. Open `config/verification.toml`. Write down your country's thresholds, seasons and altitude bands.
{: .exercise}

> #### Exercise 2: Predict, Then Measure (10 min)
>
> Before running Stage 1, write down your predictions, then compare them with the dashboard (National Overview, Raw Forecast Metrics).
>
> 1. Will MAE grow or shrink from day 1 to day 7? By roughly how much?
> 2. Will GraphCast or IFS have the lower MAE at day 1? At day 7?
> 3. Will the models be too warm or too cold for Tmax? For Tmin?
{: .exercise}

> #### Exercise 3: Why Not Train on 2022? (5 min, discussion)
>
> A colleague proposes training IDR on 2016–2022 and testing on 2022, "to use all the data".
>
> 1. What would happen to the CRPS of the calibrated forecast?
> 2. Why is that number misleading for next year's forecasts?
{: .exercise}

<details>
<summary><strong>Check your answer</strong></summary>
<br>

> #### Solution
>
> 1. The CRPS would drop, because IDR would be scored on days it was fitted to. With one fit per grid cell it can bend to the quirks of those exact days.
> 2. Next year's forecasts are days IDR has never seen. Only a test year kept out of training tells you how the calibrated forecast will behave in operations. The same rule appeared in Demo 4: calibrating and scoring on the same years overstates skill.
{: .solution}

</details>

> #### Exercise 4: Did Calibration Help? (15 min)
>
> Open **Score Decomposition**.
>
> 1. For Tmax at day 1, how much did IDR reduce the error (CRPS of `-idr` vs. MAE of raw) for each model?
> 2. Is the improvement the same at day 7? Explain using what IDR can and cannot fix.
> 3. Go back to the raw bias. Is the model with the largest bias the one that gained most from calibration?
{: .exercise}

> #### Exercise 5: Where and When (15 min)
>
> Use the region picker, the season filter and the altitude bands on National Overview.
>
> 1. Find the region (Senegal) or district (Sierra Leone) where each model's MAE is highest. Is it coastal or inland?
> 2. In which season are the errors largest? Can you think of a physical reason?
> 3. Does the ranking of the two models change between regions or seasons?
{: .exercise}

> #### Exercise 6: A Heat Warning (15 min)
>
> Open **Threshold Diagnostics** and pick the threshold your agency would use for a heat warning.
>
> 1. Read the AUC for each model at day 1, 3, 5 and 7. When does it fall below 0.8?
> 2. On the reliability diagram, is the calibrated forecast over- or under-confident at high probabilities? Check the event count under the chart before answering.
> 3. On the cost-loss curve, for which range of users (cost-loss ratios) is the forecast worth acting on at day 3?
{: .exercise}

> #### Exercise 7: Same Forecast, Another Truth (10 min)
>
> In Score Decomposition, select one model and compare it across **both** truth products (CHIRTS and ERA5).
>
> 1. How different are the MAE and the bias?
> 2. Which conclusion from Exercises 2–6 would change if you had only used ERA5?
> 3. Write one question about the truth data that you want answered in Demo 6.
{: .exercise}

## Discussion Sheet on Metrics

Fill one sheet per team and share it at the end.

| Question | Our answer |
| --- | --- |
| Which model would we use at day 1? At day 5? | |
| Which metric convinced us most? Which one could have misled us? | |
| Where (region, season, altitude) is the forecast least reliable? | |
| At which lead time does the heat-warning forecast stop being useful (AUC, value)? | |
| How large is the difference between CHIRTS and ERA5, compared with the model errors? | |
| What would we need (data, stations, years) to trust these results operationally? | |

## Summary

> #### Key Points
>
> - A single forecast cannot tell you whether a model is trustworthy. Years of past forecasts scored against observations can.
> - ForecastCheck runs in three stages that communicate only through files: raw metrics, IDR calibration, calibrated metrics. The dashboard reads what they wrote.
> - IDR learns, per grid cell and lead time, how the observation is distributed given the forecast value. Train and test on different years.
> - CRPS equals MAE for a single-value forecast, so raw and calibrated forecasts can be compared on one axis.
> - Read AUC (discrimination), reliability (trust in the probability) and cost-loss value (worth acting on) together, and always with the event count.
> - Every number depends on the truth product. Say which one you used.
{: .keypoints}
