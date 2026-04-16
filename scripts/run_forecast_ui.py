from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.forecasting import ForecastRequest, generate_forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple local forecasting UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def _get_first(form_data: dict[str, list[str]], key: str, default: str = "") -> str:
    values = form_data.get(key)
    if not values:
        return default
    return values[0]


def _build_form_values(form_data: dict[str, list[str]]) -> dict[str, str]:
    return {
        "strategy": _get_first(form_data, "strategy", "champion"),
        "horizon_hours": _get_first(form_data, "horizon_hours", "24"),
        "entry_timestamp": _get_first(form_data, "entry_timestamp", ""),
        "entry_value_mw": _get_first(form_data, "entry_value_mw", ""),
    }


def _forecast_table(frame: pd.DataFrame) -> str:
    display = frame.reset_index().copy()
    display["forecast_mw"] = display["forecast_mw"].map(lambda value: f"{float(value):,.2f}")
    for column in ["lower_95", "upper_95"]:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):,.2f}"
        )
    return display.to_html(index=False, classes="forecast-table", border=0)


def render_page(
    *,
    form_values: dict[str, str],
    result_html: str = "",
    error_message: str = "",
) -> bytes:
    strategy = html.escape(form_values["strategy"])
    horizon_hours = html.escape(form_values["horizon_hours"])
    entry_timestamp = html.escape(form_values["entry_timestamp"])
    entry_value_mw = html.escape(form_values["entry_value_mw"])

    message_block = ""
    if error_message:
        message_block = f'<p class="error">{html.escape(error_message)}</p>'

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PJME Forecast UI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      background: #f5f7fa;
      color: #1f2933;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    p {{
      line-height: 1.5;
    }}
    form {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      margin: 20px 0;
    }}
    label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 14px;
      font-weight: 600;
    }}
    input, select, button {{
      font: inherit;
      padding: 10px 12px;
      border: 1px solid #cbd2d9;
      border-radius: 6px;
      background: white;
    }}
    button {{
      cursor: pointer;
      background: #0f766e;
      color: white;
      border: 0;
      font-weight: 700;
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    .panel {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 18px;
      margin-top: 18px;
    }}
    .error {{
      color: #b42318;
      font-weight: 700;
    }}
    .forecast-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .forecast-table th,
    .forecast-table td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px;
      text-align: left;
    }}
    .hint {{
      color: #52606d;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>PJME Energy Demand Forecast</h1>
    <p>
      Use the saved project data to generate the next hourly forecast. You can also enter one
      manual observation to overwrite an existing hour or append the next expected hour before forecasting.
    </p>
    <div class="panel">
      <h2>Forecast Input</h2>
      <form method="post">
        <label>
          Strategy
          <select name="strategy">
            <option value="champion" {"selected" if strategy == "champion" else ""}>champion</option>
            <option value="sarima" {"selected" if strategy == "sarima" else ""}>sarima</option>
            <option value="seasonal_naive" {"selected" if strategy == "seasonal_naive" else ""}>seasonal_naive</option>
          </select>
        </label>
        <label>
          Horizon Hours
          <input type="number" min="1" max="168" name="horizon_hours" value="{horizon_hours}">
        </label>
        <label>
          Entry Timestamp
          <input type="text" name="entry_timestamp" value="{entry_timestamp}" placeholder="2018-08-03 01:00:00">
        </label>
        <label>
          Entry Value MW
          <input type="number" step="0.01" name="entry_value_mw" value="{entry_value_mw}" placeholder="34283">
        </label>
        <button class="full" type="submit">Generate Forecast</button>
      </form>
      <p class="hint">
        Tip: leave the manual entry blank to forecast from the saved dataset. Manual input is useful for
        classroom demos when you want to simulate receiving a new hourly demand reading.
      </p>
      {message_block}
    </div>
    {result_html}
  </main>
</body>
</html>
"""
    return page.encode("utf-8")


def app(environ, start_response):
    form_data: dict[str, list[str]] = {}
    result_html = ""
    error_message = ""

    if environ["REQUEST_METHOD"] == "POST":
        content_length = int(environ.get("CONTENT_LENGTH", "0") or "0")
        body = environ["wsgi.input"].read(content_length).decode("utf-8")
        form_data = parse_qs(body, keep_blank_values=True)

        form_values = _build_form_values(form_data)
        try:
            entry_timestamp = form_values["entry_timestamp"] or None
            entry_value_mw = (
                float(form_values["entry_value_mw"])
                if form_values["entry_value_mw"]
                else None
            )
            request = ForecastRequest(
                strategy=form_values["strategy"],
                horizon_hours=int(form_values["horizon_hours"]),
                entry_timestamp=entry_timestamp,
                entry_value_mw=entry_value_mw,
            )
            result = generate_forecast(request)
            result_html = f"""
            <section class="panel">
              <h2>Forecast Result</h2>
              <p><strong>Requested strategy:</strong> {html.escape(result.strategy_requested)}</p>
              <p><strong>Resolved strategy:</strong> {html.escape(result.strategy_resolved)}</p>
              <p><strong>Latest observation used:</strong> {html.escape(str(result.latest_observation_timestamp))} at {result.latest_observation_value_mw:,.2f} MW</p>
              <p><strong>Manual entry applied:</strong> {"yes" if result.custom_entry_applied else "no"}</p>
              {_forecast_table(result.forecast_frame)}
            </section>
            """
        except Exception as exc:  # pragma: no cover - exercised manually
            error_message = str(exc)

        response_body = render_page(
            form_values=form_values,
            result_html=result_html,
            error_message=error_message,
        )
    else:
        response_body = render_page(form_values=_build_form_values(form_data))

    start_response(
        "200 OK",
        [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(response_body))),
        ],
    )
    return [response_body]


def main() -> None:
    args = parse_args()
    with make_server(args.host, args.port, app) as server:
        print(f"Forecast UI running at http://{args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
