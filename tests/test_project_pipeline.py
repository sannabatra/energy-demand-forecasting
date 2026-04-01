from __future__ import annotations

import unittest

import pandas as pd

from src.models.sarima import SarimaConfig, seasonal_naive_forecast
from src.utils.metrics import regression_metrics


class MetricsTests(unittest.TestCase):
    def test_regression_metrics_returns_expected_keys(self) -> None:
        actual = pd.Series([10.0, 12.0, 14.0], index=pd.date_range("2024-01-01", periods=3, freq="h"))
        predicted = pd.Series([9.0, 12.0, 15.0], index=actual.index)

        metrics = regression_metrics(actual, predicted)

        self.assertEqual(
            set(metrics),
            {"n_observations", "mae", "rmse", "mape_pct", "smape_pct"},
        )
        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)


class SeasonalNaiveTests(unittest.TestCase):
    def test_seasonal_naive_repeats_last_pattern(self) -> None:
        index = pd.date_range("2024-01-01", periods=6, freq="h")
        train = pd.Series([1, 2, 3, 4, 5, 6], index=index)
        forecast_index = pd.date_range("2024-01-01 06:00:00", periods=5, freq="h")

        forecast = seasonal_naive_forecast(
            train,
            horizon=5,
            seasonal_period=2,
            forecast_index=forecast_index,
        )

        self.assertListEqual(forecast.tolist(), [5, 6, 5, 6, 5])


class ConfigTests(unittest.TestCase):
    def test_sarima_config_stores_orders(self) -> None:
        config = SarimaConfig(
            label="test_model",
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 24),
        )

        self.assertEqual(config.order, (1, 1, 1))
        self.assertEqual(config.seasonal_order[-1], 24)


if __name__ == "__main__":
    unittest.main()
