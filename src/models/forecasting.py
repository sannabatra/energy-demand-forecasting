from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.data.loaders import load_imputed_series, load_selected_model_config, processed_path
from src.models.sarima import (
    SarimaConfig,
    build_forecast_frame,
    fit_sarima,
    seasonal_naive_forecast,
)


@dataclass(frozen=True)
class ForecastRequest:
    horizon_hours: int = 24
    strategy: str = "champion"
    entry_timestamp: str | None = None
    entry_value_mw: float | None = None
    refit_window_hours: int = 24 * 180
    maxiter: int = 50


@dataclass(frozen=True)
class ForecastResult:
    strategy_requested: str
    strategy_resolved: str
    forecast_frame: pd.DataFrame
    latest_observation_timestamp: pd.Timestamp
    latest_observation_value_mw: float
    custom_entry_applied: bool


def resolve_strategy(requested_strategy: str) -> str:
    if requested_strategy != "champion":
        return requested_strategy

    recommendation_path = processed_path("deployment_recommendation.json")
    if not recommendation_path.exists():
        return "sarima"

    with recommendation_path.open("r", encoding="utf-8") as recommendation_file:
        recommendation = json.load(recommendation_file)

    recommended_model = recommendation.get("recommended_model", "")
    if recommended_model == "seasonal_naive_s24":
        return "seasonal_naive"
    return "sarima"


def apply_manual_entry(
    series: pd.Series,
    *,
    entry_timestamp: str | None,
    entry_value_mw: float | None,
) -> tuple[pd.Series, bool]:
    if entry_timestamp is None and entry_value_mw is None:
        return series.copy(), False

    if entry_timestamp is None or entry_value_mw is None:
        raise ValueError("Both entry_timestamp and entry_value_mw must be provided together.")

    timestamp = pd.Timestamp(entry_timestamp)
    if pd.isna(timestamp):
        raise ValueError("entry_timestamp must be a valid datetime.")

    if timestamp.minute != 0 or timestamp.second != 0 or timestamp.microsecond != 0:
        raise ValueError("entry_timestamp must be aligned to an hourly timestamp.")

    updated = series.copy()
    latest_timestamp = updated.index.max()
    next_expected_timestamp = latest_timestamp + pd.Timedelta(hours=1)

    if timestamp in updated.index:
        updated.loc[timestamp] = float(entry_value_mw)
    elif timestamp == next_expected_timestamp:
        updated.loc[timestamp] = float(entry_value_mw)
    else:
        raise ValueError(
            "entry_timestamp must match an existing hourly observation or the next expected hour "
            f"({next_expected_timestamp})."
        )

    updated = updated.sort_index()
    updated.name = series.name
    return updated, True


def generate_forecast(request: ForecastRequest) -> ForecastResult:
    if request.horizon_hours <= 0:
        raise ValueError("horizon_hours must be greater than 0.")

    full_series = load_imputed_series()
    adjusted_series, custom_entry_applied = apply_manual_entry(
        full_series,
        entry_timestamp=request.entry_timestamp,
        entry_value_mw=request.entry_value_mw,
    )

    strategy = resolve_strategy(request.strategy)
    forecast_start = adjusted_series.index.max() + pd.Timedelta(hours=1)

    if strategy == "sarima":
        label, order, seasonal_order = load_selected_model_config()
        config = SarimaConfig(label=label, order=order, seasonal_order=seasonal_order)

        train_series = adjusted_series.iloc[-request.refit_window_hours :].copy()
        fitted_result = fit_sarima(train_series, config, maxiter=request.maxiter)
        forecast_frame = build_forecast_frame(
            fitted_result,
            forecast_start=forecast_start,
            horizon=request.horizon_hours,
        )
    else:
        seasonal_period = 24
        baseline_forecast = seasonal_naive_forecast(
            adjusted_series,
            horizon=request.horizon_hours,
            seasonal_period=seasonal_period,
            forecast_index=pd.date_range(
                start=forecast_start,
                periods=request.horizon_hours,
                freq="h",
            ),
        )
        forecast_frame = pd.DataFrame(
            {
                "forecast_mw": baseline_forecast,
                "lower_95": pd.NA,
                "upper_95": pd.NA,
            }
        )
        forecast_frame.index.name = "Datetime"

    latest_observation_timestamp = adjusted_series.index.max()
    latest_observation_value_mw = float(adjusted_series.iloc[-1])

    return ForecastResult(
        strategy_requested=request.strategy,
        strategy_resolved=strategy,
        forecast_frame=forecast_frame,
        latest_observation_timestamp=latest_observation_timestamp,
        latest_observation_value_mw=latest_observation_value_mw,
        custom_entry_applied=custom_entry_applied,
    )
