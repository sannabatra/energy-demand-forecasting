from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import artifacts_path, load_imputed_series, load_selected_model_config
from src.models.sarima import SarimaConfig, fit_sarima, save_model_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and save the selected PJME SARIMA model.")
    parser.add_argument(
        "--refit-window-hours",
        type=int,
        default=24 * 180,
        help="Number of most recent hourly observations to use for the deployment refit.",
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
    series = full_series.iloc[-args.refit_window_hours :].copy()
    label, order, seasonal_order = load_selected_model_config()
    config = SarimaConfig(label=label, order=order, seasonal_order=seasonal_order)

    fitted_result = fit_sarima(series, config, maxiter=args.maxiter)

    save_model_artifacts(
        fitted_result,
        config=config,
        model_path=artifacts_path("pjme_sarima.pkl"),
        metadata_path=artifacts_path("pjme_sarima_metadata.json"),
        extra_metadata={
            "full_series_start": str(full_series.index.min()),
            "full_series_end": str(full_series.index.max()),
            "refit_start": str(series.index.min()),
            "refit_end": str(series.index.max()),
            "n_observations_used_for_refit": int(len(series)),
            "optimizer_maxiter": args.maxiter,
        },
    )

    print(f"Saved fitted model for {label}")
    print(f"Refit window: {series.index.min()} to {series.index.max()}")
    print(f"Observations used: {len(series):,}")


if __name__ == "__main__":
    main()
