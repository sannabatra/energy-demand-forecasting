from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    aligned = pd.concat(
        [actual.rename("actual"), predicted.rename("predicted")], axis=1
    ).dropna()

    error = aligned["actual"] - aligned["predicted"]
    abs_actual = aligned["actual"].abs()

    safe_actual = abs_actual.replace(0, np.nan)
    smape_denominator = (
        aligned["actual"].abs() + aligned["predicted"].abs()
    ).replace(0, np.nan)

    return {
        "n_observations": float(len(aligned)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mape_pct": float((error.abs() / safe_actual).mean() * 100.0),
        "smape_pct": float((2.0 * error.abs() / smape_denominator).mean() * 100.0),
    }
