"""MCP tools exposing LLMPipelineForecaster as agent-callable primitives."""

import json
import os

import pandas as pd
from mcp.server.fastmcp import FastMCP

from llm_pipeline_forecaster import LLMAgentForecaster

mcp = FastMCP(
    name="sktime-forecasting",
    instructions=(
        "Tools for agentic time series forecasting using sktime. "
        "Use analyze_timeseries first, then compose_pipeline, "
        "then fit_and_forecast, and finally assess_confidence."
    ),
)

_forecaster_cache = {}


def _load_series_from_json(data_json: str) -> pd.Series:
    """Load a Series from supported JSON payload shapes."""
    raw = json.loads(data_json)

    if isinstance(raw, dict) and "values" in raw:
        values = raw["values"]
        index = raw.get("index")
        if index:
            try:
                idx = pd.PeriodIndex(pd.to_datetime(index), freq="M")
            except Exception:
                idx = pd.RangeIndex(len(values))
        else:
            idx = pd.RangeIndex(len(values))
        return pd.Series(values, index=idx, dtype=float)

    if isinstance(raw, list):
        if raw and isinstance(raw[0], (list, tuple)):
            timestamps, values = zip(*raw)
            try:
                idx = pd.PeriodIndex(pd.to_datetime(list(timestamps)), freq="M")
            except Exception:
                idx = pd.RangeIndex(len(values))
            return pd.Series(list(values), index=idx, dtype=float)
        return pd.Series(raw, dtype=float)

    raise ValueError(
        "data_json must be a JSON object with 'index'/'values' keys, "
        "a list of [timestamp, value] pairs, or a plain list of numbers."
    )


def _make_forecaster(prompt: str, groq_api_key: str = "", model: str = "llama-3.3-70b-versatile"):
    """Create a forecaster using the provided or environment API key."""
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("No Groq API key provided.")
    return LLMPipelineForecaster(prompt=prompt, api_key=api_key, model=model)


@mcp.tool()
def analyze_timeseries(data_json: str) -> str:
    """Analyze a time series and return a JSON summary."""
    y = _load_series_from_json(data_json)
    summary = LLMPipelineForecaster(
        prompt="Analyze this time series",
        api_key=os.environ.get("GROQ_API_KEY", "placeholder"),
    )._analyze_series(y)
    return json.dumps(summary, indent=2)


@mcp.tool()
def compose_pipeline(
    data_summary_json: str,
    prompt: str,
    groq_api_key: str = "",
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """Use the LLM to choose a pipeline configuration from the summary."""
    forecaster = _make_forecaster(prompt=prompt, groq_api_key=groq_api_key, model=model)
    data_summary = json.loads(data_summary_json)
    config, reasoning = forecaster._ask_llm(data_summary)
    config["reasoning"] = reasoning
    return json.dumps(config, indent=2)


@mcp.tool()
def fit_and_forecast(
    data_json: str,
    pipeline_config_json: str,
    data_summary_json: str,
    horizon: int,
    session_id: str = "default",
    prompt: str = "Forecast this time series",
    groq_api_key: str = "",
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """Fit the selected pipeline and return forecast values."""
    y = _load_series_from_json(data_json)
    config = json.loads(pipeline_config_json)
    data_summary = json.loads(data_summary_json)

    reasoning = config.pop("reasoning", "Provided by compose_pipeline")
    forecaster = _make_forecaster(prompt=prompt, groq_api_key=groq_api_key, model=model)
    forecaster.data_summary_ = data_summary
    forecaster.pipeline_config_ = config
    forecaster.reasoning_ = reasoning
    forecaster.intent_ = {"intent_summary": prompt}
    forecaster.iteration_log_ = [
        {
            "iteration": 1,
            "config": config,
            "reasoning": reasoning,
            "mae": "not_evaluated",
            "mae_relative": "not_evaluated",
            "failed": False,
        }
    ]
    forecaster.pipeline_ = forecaster._build_pipeline(config)
    fh = list(range(1, horizon + 1))
    forecaster.pipeline_.fit(y, fh=fh)
    forecaster._y = y
    y_pred = forecaster.pipeline_.predict(fh=fh)
    _forecaster_cache[session_id] = forecaster

    result = {
        "session_id": session_id,
        "pipeline": forecaster.get_pipeline_description().splitlines()[0].replace("Pipeline: ", ""),
        "reasoning": reasoning,
        "predictions": [
            {"step": step, "index": str(idx), "value": float(value)}
            for step, (idx, value) in enumerate(y_pred.items(), start=1)
        ],
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def assess_confidence(
    session_id: str = "default",
    groq_api_key: str = "",
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """Assess forecast confidence for a previously fitted session."""
    if session_id not in _forecaster_cache:
        return json.dumps(
            {
                "error": (
                    f"No fitted forecaster cached for session_id='{session_id}'. "
                    "Run fit_and_forecast first."
                )
            },
            indent=2,
        )

    forecaster = _forecaster_cache[session_id]
    if groq_api_key:
        forecaster.api_key = groq_api_key
    if model:
        forecaster.model = model
    return json.dumps(forecaster.get_confidence_assessment(), indent=2)


if __name__ == "__main__":
    mcp.run()
