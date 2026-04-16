from __future__ import annotations

import unittest

import pandas as pd

from src.models.forecasting import ForecastRequest, apply_manual_entry, resolve_strategy


class ManualEntryTests(unittest.TestCase):
    def test_apply_manual_entry_overwrites_existing_timestamp(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00", periods=3, freq="h")
        series = pd.Series([10.0, 11.0, 12.0], index=index, name="PJME_MW")

        updated, applied = apply_manual_entry(
            series,
            entry_timestamp="2024-01-01 01:00:00",
            entry_value_mw=99.0,
        )

        self.assertTrue(applied)
        self.assertEqual(updated.loc[pd.Timestamp("2024-01-01 01:00:00")], 99.0)

    def test_apply_manual_entry_appends_next_hour(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00", periods=3, freq="h")
        series = pd.Series([10.0, 11.0, 12.0], index=index, name="PJME_MW")

        updated, applied = apply_manual_entry(
            series,
            entry_timestamp="2024-01-01 03:00:00",
            entry_value_mw=13.0,
        )

        self.assertTrue(applied)
        self.assertEqual(len(updated), 4)
        self.assertEqual(updated.iloc[-1], 13.0)

    def test_apply_manual_entry_rejects_non_adjacent_new_timestamp(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00", periods=3, freq="h")
        series = pd.Series([10.0, 11.0, 12.0], index=index, name="PJME_MW")

        with self.assertRaises(ValueError):
            apply_manual_entry(
                series,
                entry_timestamp="2024-01-01 05:00:00",
                entry_value_mw=15.0,
            )


class StrategyResolutionTests(unittest.TestCase):
    def test_forecast_request_defaults(self) -> None:
        request = ForecastRequest()

        self.assertEqual(request.horizon_hours, 24)
        self.assertEqual(request.strategy, "champion")

    def test_resolve_strategy_passthrough_for_non_champion(self) -> None:
        self.assertEqual(resolve_strategy("sarima"), "sarima")
        self.assertEqual(resolve_strategy("seasonal_naive"), "seasonal_naive")


if __name__ == "__main__":
    unittest.main()
