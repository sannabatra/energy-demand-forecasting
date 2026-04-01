from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import artifacts_path, load_imputed_series, processed_path
from src.models.sarima import build_forecast_frame, load_model_artifact, seasonal_naive_forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a future forecast from the saved PJME SARIMA artifact.")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=24,
        help="Number of hours to forecast into the future.",
    )
    parser.add_argument(
        "--strategy",
        choices=["champion", "sarima", "seasonal_naive"],
        default="champion",
        help="Forecasting strategy to use. 'champion' follows the latest holdout recommendation.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    series = load_imputed_series()
    strategy = resolve_strategy(args.strategy)
    forecast_start = series.index.max() + pd.Timedelta(hours=1)

    if strategy == "sarima":
        fitted_result = load_model_artifact(artifacts_path("pjme_sarima.pkl"))
        forecast_frame = build_forecast_frame(
            fitted_result,
            forecast_start=forecast_start,
            horizon=args.horizon_hours,
        )
    else:
        seasonal_period = 24
        baseline_forecast = seasonal_naive_forecast(
            series,
            horizon=args.horizon_hours,
            seasonal_period=seasonal_period,
            forecast_index=pd.date_range(
                start=forecast_start,
                periods=args.horizon_hours,
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

    output_path = processed_path(f"next_{args.horizon_hours}h_forecast.csv")
    forecast_frame.to_csv(output_path)

    print(f"Strategy used: {strategy}")
    print(forecast_frame.head().to_string())
    print(f"Saved forecast to {output_path}")


if __name__ == "__main__":
    main()
