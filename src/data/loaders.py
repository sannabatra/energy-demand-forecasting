from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"


def processed_path(filename: str) -> Path:
    return PROCESSED_DIR / filename


def artifacts_path(filename: str) -> Path:
    return ARTIFACTS_DIR / filename


def load_imputed_series(filename: str = "pjme_imputed.csv") -> pd.Series:
    series = pd.read_csv(
        processed_path(filename),
        parse_dates=["Datetime"],
        index_col="Datetime",
    )["PJME_MW"].astype(float)
    series.name = "PJME_MW"
    return series.sort_index()


def load_selected_model_config(
    filename: str = "selected_sarima_model.csv",
) -> tuple[str, tuple[int, int, int], tuple[int, int, int, int]]:
    selected_model = pd.read_csv(processed_path(filename)).iloc[0]
    label = str(selected_model["selected_label"])
    order = tuple(ast.literal_eval(selected_model["selected_order"]))
    seasonal_order = tuple(ast.literal_eval(selected_model["selected_seasonal_order"]))
    return label, order, seasonal_order
