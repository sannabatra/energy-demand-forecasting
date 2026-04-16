from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import artifacts_path, processed_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the energy demand forecasting project.")
    parser.add_argument(
        "--rerun-evaluation",
        action="store_true",
        help="Rerun the holdout evaluation before checking outputs.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main() -> None:
    args = parse_args()

    print("1. Running unit tests...")
    run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])

    if args.rerun_evaluation:
        print("2. Re-running holdout evaluation...")
        run_command([sys.executable, "scripts/evaluate_model.py"])

    print("3. Checking generated artifacts...")
    required_paths = [
        processed_path("holdout_metrics.csv"),
        processed_path("deployment_recommendation.json"),
        processed_path("project_summary.json"),
        artifacts_path("pjme_sarima_metadata.json"),
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required outputs:\n" + "\n".join(str(path) for path in missing)
        )

    metrics = pd.read_csv(processed_path("holdout_metrics.csv"))
    best_row = metrics.sort_values("rmse", ascending=True).iloc[0]

    print("4. Validation summary")
    print(metrics.to_string(index=False))
    print()
    print(
        "Champion model:"
        f" {best_row['model']} with RMSE={best_row['rmse']:.2f} and MAE={best_row['mae']:.2f}"
    )
    print("Validation passed.")


if __name__ == "__main__":
    main()
