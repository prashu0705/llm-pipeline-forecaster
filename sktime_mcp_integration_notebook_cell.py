"""
Example 5 — Using LLMPipelineForecaster via sktime-mcp Tools
"""

import json

import matplotlib.pyplot as plt
import pandas as pd
from sktime.datasets import load_airline

from sktime_mcp_tools import (
    analyze_timeseries,
    assess_confidence,
    compose_pipeline,
    fit_and_forecast,
)

y = load_airline()

data_json = json.dumps(
    {
        "index": [str(period) for period in y.index],
        "values": y.values.tolist(),
    }
)

print("=" * 60)
print("STEP 1 - analyze_timeseries (MCP tool call)")
print("=" * 60)
summary_json = analyze_timeseries(data_json)
summary = json.loads(summary_json)
print(summary_json)

print("\n" + "=" * 60)
print("STEP 2 - compose_pipeline (MCP tool call)")
print("=" * 60)
prompt = "Forecast airline passengers as accurately as possible for next 6 months"
config_json = compose_pipeline(
    data_summary_json=summary_json,
    prompt=prompt,
    groq_api_key=GROQ_KEY,
)
config = json.loads(config_json)
print(config_json)

print("\n" + "=" * 60)
print("STEP 3 - fit_and_forecast (MCP tool call)")
print("=" * 60)
forecast_json = fit_and_forecast(
    data_json=data_json,
    pipeline_config_json=config_json,
    data_summary_json=summary_json,
    horizon=6,
    session_id="airline_demo",
    prompt=prompt,
    groq_api_key=GROQ_KEY,
)
forecast = json.loads(forecast_json)
print(f"Pipeline : {forecast['pipeline']}")
print(f"Reasoning: {forecast['reasoning']}")
print("\nPredictions:")
for pred in forecast["predictions"]:
    print(f"  Step +{pred['step']}: {pred['value']:.1f}")

print("\n" + "=" * 60)
print("STEP 4 - assess_confidence (MCP tool call)")
print("=" * 60)
confidence_json = assess_confidence(
    session_id="airline_demo",
    groq_api_key=GROQ_KEY,
)
confidence = json.loads(confidence_json)
print(f"Confidence : {confidence['confidence'].upper()}")
print("Reasons:")
for reason in confidence["reasons"]:
    print(f"  - {reason}")
if confidence.get("warning"):
    print(f"Warning    : {confidence['warning']}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    y.index.to_timestamp(),
    y.values,
    color="#2C5F8A",
    linewidth=1.8,
    label="Historical",
)

last_period = y.index[-1]
fh_index = pd.period_range(start=last_period + 1, periods=6, freq="M")
pred_values = [pred["value"] for pred in forecast["predictions"]]
ax.plot(
    fh_index.to_timestamp(),
    pred_values,
    color="#E85D26",
    linewidth=2.2,
    linestyle="--",
    marker="o",
    markersize=6,
    label="Forecast (via MCP tools)",
)

conf_colors = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}
conf_level = confidence.get("confidence", "medium")
ax.annotate(
    f"Confidence: {conf_level.upper()}",
    xy=(0.99, 0.05),
    xycoords="axes fraction",
    ha="right",
    fontsize=9,
    color="white",
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor=conf_colors.get(conf_level, "#888"),
        edgecolor="none",
    ),
)

ax.set_title(
    f"Example 5: Airline via sktime-mcp Tools - {forecast['pipeline']}",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Date")
ax.set_ylabel("Passengers")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_mcp_demo.png", dpi=150)
plt.show()
print("Plot saved as forecast_mcp_demo.png")

print("\n" + "=" * 60)
print("COMPARISON: Direct LLMPipelineForecaster vs MCP tool calls")
print("=" * 60)
print(
    """
+------------------------+----------------------------------------+
| Direct API             | Via MCP tools                          |
+------------------------+----------------------------------------+
| forecaster.fit(y)      | analyze_timeseries(data_json)          |
|                        | compose_pipeline(summary, prompt)      |
| forecaster.predict(fh) | fit_and_forecast(data, config, ...)    |
| get_confidence_*()     | assess_confidence(session_id)          |
+------------------------+----------------------------------------+
| Works in Python only   | Works from any MCP-compatible agent    |
| Monolithic call        | Each step inspectable / overridable    |
|                        | Agent can inject domain knowledge      |
|                        | between steps                          |
+------------------------+----------------------------------------+
"""
)
