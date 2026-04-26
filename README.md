# LLMPipelineForecaster

An sktime-compatible forecaster that uses an LLM to automatically 
compose, evaluate, and improve time series forecasting pipelines 
from a natural language prompt.

Built as part of the sktime agentic track for ESoC 2026.

This repo now also includes an MCP integration layer so the forecaster can be
driven tool-by-tool from Claude Desktop, Cursor, or any other MCP-compatible
agent.

## What it does

Given a natural language prompt and time series data, the forecaster:

1. **Analyzes** the data statistically — trend, seasonality, stationarity
2. **Composes** a full sktime pipeline using an LLM — detrending, 
   deseasonalization, model selection
3. **Evaluates** the pipeline on a validation split
4. **Self-corrects** iteratively — if MAE is too high, the LLM reasons 
   about why and tries a different approach
5. **Exogenous Extraction**: Automatically translates unstructured text logs into numeric signals via an internal `LLMTextEventFeaturizer` during the fitting process.
6. **Returns** the best forecast, prediction intervals, and an English 
   explanation of its reasoning

## Quick example
```python
from llm_pipeline_forecaster import LLMPipelineForecaster
from sktime.datasets import load_airline

y = load_airline()

forecaster = LLMPipelineForecaster(
    prompt="Forecast airline passengers as accurately as possible",
    api_key="your_groq_key"
)

forecaster.fit(y)
y_pred = forecaster.predict(fh=[1, 2, 3])

print(forecaster.get_pipeline_description())
print(forecaster.get_confidence_assessment())
```

## Input formats

Accepts CSV path, pandas DataFrame, or sktime-compatible Series:
```python
# CSV
forecaster.fit("sales.csv")

# DataFrame  
forecaster.fit(df)

# sktime Series
forecaster.fit(y)
```

## Prompt-aware behavior

The forecaster extracts intent from your prompt:

| Prompt | Behavior |
|--------|----------|
| "quick rough estimate" | 1 iteration, fast |
| "most accurate possible" | up to 5 iterations |
| "simple baseline" | forces NaiveForecaster |
| "next 3 months" | auto-sets forecast horizon |

## Pipeline composition

The LLM composes from these components:

- **Transformations:** Detrender, Deseasonalizer
- **Models:** NaiveForecaster, AutoARIMA, ExponentialSmoothing
- **Selection:** based on stationarity test (ADF), ACF seasonality 
  detection, trend slope, and previous iteration MAE

## Installation
```bash
pip install -r requirements.txt
```

## MCP integration

The MCP bonus criterion in [sktime issue #9721](https://github.com/sktime/sktime/issues/9721)
asks for a forecaster that either uses `sktime-mcp` tools or exposes a
reasonably defined custom set of MCP tools. This repo now does that with
[`sktime_mcp_tools.py`](./sktime_mcp_tools.py).

It exposes four agent-callable tools:

1. `analyze_timeseries`
2. `compose_pipeline`
3. `fit_and_forecast`
4. `assess_confidence`

These map directly onto the LLMPipelineForecaster workflow, but make each
stage inspectable and remotely callable over MCP.

## Run the MCP server

```bash
python sktime_mcp_tools.py
```

## Claude Desktop config

Add this to your Claude Desktop MCP config:

```json
{
  "mcpServers": {
    "sktime-forecasting": {
      "command": "python",
      "args": [
        "/absolute/path/to/llm-pipeline-forecaster/sktime_mcp_tools.py"
      ],
      "env": {
        "GROQ_API_KEY": "your_groq_key"
      }
    }
  }
}
```

## Notebook example

`demo_cleaned.ipynb` now includes an Example 5 section showing the same
forecasting workflow driven through MCP tool calls. A copy of that cell is also
available in
[`sktime_mcp_integration_notebook_cell.py`](./sktime_mcp_integration_notebook_cell.py)
for easy reuse.

## Standalone Text Featurization

If you just need to turn text logs into exogenous data for other `sktime` models:

```python
from llm_text_featurizer import LLMTextEventFeaturizer

schema = {"sentiment": "float -1.0 to 1.0"}
featurizer = LLMTextEventFeaturizer(api_key="...", text_column="logs", feature_schema=schema)

X_numeric = featurizer.fit_transform(X_text)
```

## Project structure
llm-pipeline-forecaster/
├── text_exogenous_demo.ipynb       # Main tutorial notebook
├── demo_cleaned.ipynb              # General pipeline demo
├── llm_pipeline_forecaster.py      # Core Agentic Forecaster
├── llm_text_featurizer.py          # Context-Aware Transformer
├── sktime_mcp_tools.py             # MCP Server Layer
├── requirements.txt
└── README.md

## Related

- [sktime](https://github.com/sktime/sktime)
- [sktime issue #9721](https://github.com/sktime/sktime/issues/9721)
- [sktime-mcp](https://github.com/sktime/sktime-mcp)

## License

MIT
