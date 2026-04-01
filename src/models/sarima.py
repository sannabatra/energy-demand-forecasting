from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class SarimaConfig:
    label: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: str = "n"
    simple_differencing: bool = False
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False


def fit_sarima(
    series: pd.Series,
    config: SarimaConfig,
    *,
    maxiter: int = 50,
):
    model = SARIMAX(
        series,
        order=config.order,
        seasonal_order=config.seasonal_order,
        trend=config.trend,
        simple_differencing=config.simple_differencing,
        enforce_stationarity=config.enforce_stationarity,
        enforce_invertibility=config.enforce_invertibility,
    )
    return model.fit(disp=False, maxiter=maxiter)


def seasonal_naive_forecast(
    train_series: pd.Series,
    *,
    horizon: int,
    seasonal_period: int,
    forecast_index: pd.Index,
) -> pd.Series:
    if len(train_series) < seasonal_period:
        raise ValueError("train_series must be at least as long as seasonal_period")

    repeated = []
    pattern = train_series.iloc[-seasonal_period:].to_list()
    while len(repeated) < horizon:
        repeated.extend(pattern)

    forecast_values = repeated[:horizon]
    return pd.Series(forecast_values, index=forecast_index, name="seasonal_naive")


def build_forecast_frame(
    fitted_result: Any,
    *,
    forecast_start: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    forecast = fitted_result.get_forecast(steps=horizon).summary_frame()
    forecast_index = pd.date_range(start=forecast_start, periods=horizon, freq="h")
    frame = pd.DataFrame(
        {
            "forecast_mw": forecast["mean"].to_numpy(),
            "lower_95": forecast["mean_ci_lower"].to_numpy(),
            "upper_95": forecast["mean_ci_upper"].to_numpy(),
        },
        index=forecast_index,
    )
    frame.index.name = "Datetime"
    return frame


def save_model_artifacts(
    fitted_result: Any,
    *,
    config: SarimaConfig,
    model_path: Path,
    metadata_path: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as model_file:
        pickle.dump(fitted_result, model_file)

    metadata = asdict(config)
    if extra_metadata:
        metadata.update(extra_metadata)

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def load_model_artifact(model_path: Path):
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)
