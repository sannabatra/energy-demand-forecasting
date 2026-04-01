# Energy Demand Forecasting

Hourly PJM East electricity demand forecasting project built around a full time-series workflow:

1. Data collection
2. Exploratory data analysis
3. Missing-value and timestamp-gap handling
4. Stationarity testing
5. Transformations for stationarity
6. Seasonality analysis and decomposition
7. ACF and PACF diagnostics
8. Model-family identification
9. Lag selection
10. AIC and BIC comparison
11. Deployment-ready forecasting pipeline

## Project Outcome

The raw PJME hourly demand series is not stationary.

- Visual rolling statistics in [04_stationarity.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/04_stationarity.ipynb) showed changing mean and variance.
- ADF rejected a unit root on the large sample, but KPSS and the visual diagnostics still supported treating the raw series as non-stationary.
- The project rectified that by using a log transform, first differencing, and 24-hour seasonal differencing during the model-identification stage.

The notebook workflow selected `SARIMA(2,1,1)(1,1,1,24)` by BIC, but the finished project adds a real holdout evaluation step and found that a simple 24-hour seasonal-naive baseline performed better on the final 14-day holdout window.

## Final Findings

- Raw series status: non-stationary
- Stationarity fix used for diagnostics/model selection: `log(PJME_MW)`, first difference, seasonal difference at 24 hours
- Best SARIMA by information criteria: `SARIMA(2,1,1)(1,1,1,24)`
- Best holdout performer by RMSE: `seasonal_naive_s24`
- Recommended deployment strategy in this repo: use the holdout champion by default

Holdout metrics saved in [holdout_metrics.csv](/Users/sannabatra/energy-demand-forecasting/data/processed/holdout_metrics.csv):

- `seasonal_naive_s24`: RMSE `3882.51`, MAE `3110.39`
- `sarima_211_111_24`: RMSE `5418.23`, MAE `4548.32`

## Repository Layout

- [notebooks](/Users/sannabatra/energy-demand-forecasting/notebooks): step-by-step notebook analysis
- [src/data/loaders.py](/Users/sannabatra/energy-demand-forecasting/src/data/loaders.py): reusable dataset and config loading
- [src/models/sarima.py](/Users/sannabatra/energy-demand-forecasting/src/models/sarima.py): SARIMA fitting, persistence, and forecasting helpers
- [src/utils/metrics.py](/Users/sannabatra/energy-demand-forecasting/src/utils/metrics.py): evaluation metrics
- [scripts/train_model.py](/Users/sannabatra/energy-demand-forecasting/scripts/train_model.py): refit and save the selected SARIMA artifact
- [scripts/evaluate_model.py](/Users/sannabatra/energy-demand-forecasting/scripts/evaluate_model.py): holdout evaluation and deployment recommendation
- [scripts/forecast_next_day.py](/Users/sannabatra/energy-demand-forecasting/scripts/forecast_next_day.py): generate future forecasts from the saved artifact or champion strategy
- [artifacts/models](/Users/sannabatra/energy-demand-forecasting/artifacts/models): saved model artifacts
- [reports/figures](/Users/sannabatra/energy-demand-forecasting/reports/figures): generated figures from the analysis and evaluation

## How To Run

Use the project virtual environment:

```bash
venv/bin/python -m unittest discover -s tests -v
venv/bin/python scripts/train_model.py
venv/bin/python scripts/evaluate_model.py
venv/bin/python scripts/forecast_next_day.py
```

Useful forecast modes:

```bash
venv/bin/python scripts/forecast_next_day.py --strategy champion
venv/bin/python scripts/forecast_next_day.py --strategy sarima
venv/bin/python scripts/forecast_next_day.py --strategy seasonal_naive
```

`champion` is the default. It reads [deployment_recommendation.json](/Users/sannabatra/energy-demand-forecasting/data/processed/deployment_recommendation.json) and uses the best holdout strategy automatically.

## Notebook Flowchart Coverage

Every step in the original flowchart is covered:

1. [01_data_collection.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/01_data_collection.ipynb)
2. [02_eda.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/02_eda.ipynb)
3. [03_missing_values.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/03_missing_values.ipynb)
4. [04_stationarity.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/04_stationarity.ipynb)
5. [05_transformations.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/05_transformations.ipynb)
6. [06_seasonality.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/06_seasonality.ipynb)
7. [07_acf_pacf.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/07_acf_pacf.ipynb)
8. [08_model_identification.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/08_model_identification.ipynb)
9. [09_lag_selection.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/09_lag_selection.ipynb)
10. [10_aic_bic.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/10_aic_bic.ipynb)
11. [11_deployment.ipynb](/Users/sannabatra/energy-demand-forecasting/notebooks/11_deployment.ipynb)

## Important Outputs

- [stationarity_results.csv](/Users/sannabatra/energy-demand-forecasting/data/processed/stationarity_results.csv)
- [transformation_selection.csv](/Users/sannabatra/energy-demand-forecasting/data/processed/transformation_selection.csv)
- [selected_sarima_model.csv](/Users/sannabatra/energy-demand-forecasting/data/processed/selected_sarima_model.csv)
- [holdout_metrics.csv](/Users/sannabatra/energy-demand-forecasting/data/processed/holdout_metrics.csv)
- [deployment_recommendation.json](/Users/sannabatra/energy-demand-forecasting/data/processed/deployment_recommendation.json)
- [next_24h_forecast.csv](/Users/sannabatra/energy-demand-forecasting/data/processed/next_24h_forecast.csv)
