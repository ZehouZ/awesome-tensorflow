#!/usr/bin/env python3
"""Weekly inflation nowcasting pipeline using only free API data.

This script:
1) Downloads data from FRED's public CSV endpoints.
2) Applies simple publication lag assumptions.
3) Builds weekly features and trains an Enhanced Random Forest style nowcast.
4) Exports updated nowcast and backtest metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_fred_series(series_id: str) -> pd.Series:
    response = requests.get(FRED_CSV_URL.format(series_id=series_id), timeout=30)
    response.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(response.text))
    df.columns = ["DATE", "VALUE"]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    out = df.set_index("DATE")["VALUE"].sort_index().dropna()
    out.name = series_id
    return out


def transform_series(series: pd.Series, transform: str) -> pd.Series:
    if transform == "level":
        return series
    if transform == "yoy":
        return (series / series.shift(12) - 1.0) * 100.0
    if transform == "pct_change_4w":
        return (series / series.shift(20) - 1.0) * 100.0
    raise ValueError(f"Unsupported transform: {transform}")


def to_weekly_observable(series: pd.Series, lag_weeks: int, freq: str) -> pd.Series:
    weekly = series.resample(freq).last().ffill()
    if lag_weeks > 0:
        weekly = weekly.shift(lag_weeks)
    return weekly


def build_dataset(config: dict) -> pd.DataFrame:
    freq = config["training"]["resample_frequency"]
    data_cols: Dict[str, pd.Series] = {}

    for series_id, meta in config["series"].items():
        raw = fetch_fred_series(series_id)
        transformed = transform_series(raw, meta["transform"])
        observable = to_weekly_observable(transformed, int(meta["release_lag_weeks"]), freq)
        col_name = "target" if meta["role"] == "target" else series_id
        data_cols[col_name] = observable

    df = pd.DataFrame(data_cols).sort_index()

    feature_cols = [c for c in df.columns if c != "target"]
    for col in feature_cols:
        for lag in (1, 2, 4, 8):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df["target_lead1"] = df["target"].shift(-1)
    return df


def train_and_nowcast(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    min_train = int(config["training"]["min_train_weeks"])
    test_weeks = int(config["training"]["test_weeks"])
    params = config["model"]

    usable = df.dropna(subset=["target_lead1"]).copy()
    feature_cols = [c for c in usable.columns if c not in {"target", "target_lead1"}]
    usable = usable.dropna(subset=feature_cols)

    if len(usable) < (min_train + test_weeks):
        raise ValueError("Not enough data after transformations. Expand history or reduce constraints.")

    train = usable.iloc[: -test_weeks]
    test = usable.iloc[-test_weeks:]

    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=int(params["random_state"]),
        n_jobs=-1,
    )

    model.fit(train[feature_cols], train["target_lead1"])

    test_pred = model.predict(test[feature_cols])
    metrics = {
        "test_weeks": test_weeks,
        "mae": float(mean_absolute_error(test["target_lead1"], test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(test["target_lead1"], test_pred))),
        "r2": float(r2_score(test["target_lead1"], test_pred)),
        "train_start": str(train.index.min().date()),
        "train_end": str(train.index.max().date()),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
    }

    latest_features = df.drop(columns=["target", "target_lead1"], errors="ignore").iloc[[-1]].copy()
    if latest_features.isna().any(axis=None):
        latest_features = latest_features.ffill().bfill()

    nowcast_next_week = float(model.predict(latest_features[feature_cols])[0])

    out = df[["target"]].copy()
    out["predicted_next_week_yoy"] = np.nan
    out.loc[test.index, "predicted_next_week_yoy"] = test_pred
    out.loc[df.index.max(), "predicted_next_week_yoy"] = nowcast_next_week

    return out, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly inflation nowcast with enhanced random forest")
    parser.add_argument("--config", default="nowcasting/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output = config["output"]
    Path(output["data_dir"]).mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config)
    predictions, metrics = train_and_nowcast(dataset, config)

    predictions.to_csv(output["predictions_file"], index_label="date")
    with open(output["metrics_file"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Nowcast updated.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
