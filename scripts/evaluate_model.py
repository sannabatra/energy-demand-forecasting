from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-cache-"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="xdg-cache-"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.loaders import PROJECT_ROOT as REPO_ROOT, load_imputed_series, load_selected_model_config, processed_path
from src.models.sarima import (
    SarimaConfig,
    build_forecast_frame,
    fit_sarima,
    seasonal_naive_forecast,
)
from src.utils.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the selected PJME SARIMA model on a holdout window.")
    parser.add_argument(
        "--evaluation-window-hours",
        type=int,
        default=24 * 180,
        help="Number of most recent hourly observations to use for evaluation.",
    )
    parser.add_argument(
        "--holdout-hours",
        type=int,
        default=24 * 14,
        help="Length of the final holdout forecast horizon.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=50,
        help="Maximum optimizer iterations for the SARIMA fit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_series = load_imputed_series()
    evaluation_window = full_series.iloc[-args.evaluation_window_hours :].copy()
    train_series = evaluation_window.iloc[: -args.holdout_hours]
    holdout_series = evaluation_window.iloc[-args.holdout_hours :]

    label, order, seasonal_order = load_selected_model_config()
    config = SarimaConfig(label=label, order=order, seasonal_order=seasonal_order)
    fitted_result = fit_sarima(train_series, config, maxiter=args.maxiter)

    forecast_frame = build_forecast_frame(
        fitted_result,
        forecast_start=holdout_series.index.min(),
        horizon=len(holdout_series),
    )
    model_forecast = forecast_frame["forecast_mw"]
    seasonal_period = seasonal_order[3]
    baseline_forecast = seasonal_naive_forecast(
        train_series,
        horizon=len(holdout_series),
        seasonal_period=seasonal_period,
        forecast_index=holdout_series.index,
    )

    sarima_metrics = regression_metrics(holdout_series, model_forecast)
    baseline_metrics = regression_metrics(holdout_series, baseline_forecast)

    metrics_df = pd.DataFrame(
        [
            {
                "model": label,
                "evaluation_window_hours": args.evaluation_window_hours,
                "holdout_hours": args.holdout_hours,
                **sarima_metrics,
            },
            {
                "model": f"seasonal_naive_s{seasonal_period}",
                "evaluation_window_hours": args.evaluation_window_hours,
                "holdout_hours": args.holdout_hours,
                **baseline_metrics,
            },
        ]
    )

    predictions_df = pd.DataFrame(
        {
            "actual_mw": holdout_series,
            "sarima_forecast_mw": model_forecast,
            "seasonal_naive_forecast_mw": baseline_forecast,
            "sarima_lower_95": forecast_frame["lower_95"],
            "sarima_upper_95": forecast_frame["upper_95"],
        }
    )
    predictions_df.index.name = "Datetime"

    metrics_path = processed_path("holdout_metrics.csv")
    predictions_path = processed_path("holdout_predictions.csv")
    summary_path = processed_path("project_summary.json")
    recommendation_path = processed_path("deployment_recommendation.json")

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path)

    recommended_row = metrics_df.sort_values("rmse", ascending=True).iloc[0]
    recommendation_payload = {
        "recommended_model": str(recommended_row["model"]),
        "recommendation_basis": "Lowest holdout RMSE over the final 14-day evaluation window.",
        "sarima_selected_by_bic": label,
        "seasonal_period_baseline": seasonal_period,
    }
    with recommendation_path.open("w", encoding="utf-8") as recommendation_file:
        json.dump(recommendation_payload, recommendation_file, indent=2)

    summary_payload = {
        "raw_series_stationary": False,
        "stationarity_fix": "Applied log transform, first differencing, and 24-hour seasonal differencing before SARIMA order selection.",
        "selected_model_label": label,
        "selected_order": list(order),
        "selected_seasonal_order": list(seasonal_order),
        "evaluation_window_hours": args.evaluation_window_hours,
        "holdout_hours": args.holdout_hours,
        "sarima_metrics": sarima_metrics,
        "seasonal_naive_metrics": baseline_metrics,
        "recommended_deployment_model": recommendation_payload["recommended_model"],
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, indent=2)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(14, 6))
    history_window = train_series.iloc[-24 * 7 :]

    history_window.plot(ax=ax, color="steelblue", label="Training history", linewidth=1.0)
    holdout_series.plot(ax=ax, color="black", label="Actual holdout", linewidth=1.2)
    model_forecast.plot(ax=ax, color="firebrick", label=label, linewidth=1.5)
    baseline_forecast.plot(ax=ax, color="darkorange", label="Seasonal naive", linewidth=1.2)
    ax.fill_between(
        forecast_frame.index,
        forecast_frame["lower_95"],
        forecast_frame["upper_95"],
        color="firebrick",
        alpha=0.12,
        label="SARIMA 95% interval",
    )

    ax.set_title("Holdout forecast comparison")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Demand (MW)")
    ax.legend()

    figure_path = REPO_ROOT / "reports" / "figures" / "12_holdout_forecast.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(metrics_df.to_string(index=False))
    print(f"Saved holdout metrics to {metrics_path}")
    print(f"Saved holdout predictions to {predictions_path}")
    print(f"Saved project summary to {summary_path}")
    print(f"Saved deployment recommendation to {recommendation_path}")
    print(f"Saved holdout figure to {figure_path}")


if __name__ == "__main__":
    main()
