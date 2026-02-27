# Weekly US Inflation Nowcasting (Free-API version)

This is a practical replication scaffold for the **"Weekly Nowcasting US Inflation with Enhanced Random Forest"** idea using only free/public API data.

## What this implementation does

- Pulls macro and market series from **FRED public CSV endpoints** (no API key needed).
- Converts mixed-frequency data into weekly observable features with publication-lag assumptions.
- Trains an enhanced random forest (feature lags + nonlinear interactions) to predict next-week CPI YoY.
- Saves:
  - `data/weekly_inflation_nowcast.csv`
  - `data/model_metrics.json`
- Includes a GitHub Action to refresh the nowcast every 6 hours.

## Data sources (all free)

Configured in `nowcasting/config.yaml`:

- CPIAUCSL (target)
- PPIACO
- CUSR0000SA0L1E
- MICH
- T10YIE
- DCOILWTICO
- UNRATE

You can add/remove series there without changing code.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r nowcasting/requirements.txt
python nowcasting/inflation_nowcast.py
```

## Continuous updating

- `.github/workflows/update-nowcast.yml` runs on a schedule and commits refreshed outputs.
- If you run outside GitHub Actions, use cron/systemd to call `python nowcasting/inflation_nowcast.py` at your preferred cadence.

## Notes on replication fidelity

This is an **open-data approximation** of the paper’s framework, not a full archival-data replication.
To get closer to the paper's exact setup, you can extend this with:

- Real-time vintage data (ALFRED).
- Richer weekly high-frequency feature set.
- Alternative loss/objective tuning and rolling-origin evaluation.
- Hyperparameter search and quantile prediction intervals.
