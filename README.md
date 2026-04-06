# LLMPipelineForecaster

An sktime-compatible forecaster that uses an LLM to automatically 
compose, evaluate, and improve time series forecasting pipelines 
from a natural language prompt.

Built as part of the sktime agentic track for ESoC 2026.

## What it does

Given a natural language prompt and time series data, the forecaster:

1. **Analyzes** the data statistically — trend, seasonality, stationarity
2. **Composes** a full sktime pipeline using an LLM — detrending, 
   deseasonalization, model selection
3. **Evaluates** the pipeline on a validation split
4. **Self-corrects** iteratively — if MAE is too high, the LLM reasons 
   about why and tries a different approach
5. **Returns** the best forecast, prediction intervals, and an English 
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

## Project structure
llm-pipeline-forecaster/
├── llm_pipeline_forecaster_demo.ipynb   # full demo notebook
├── requirements.txt
└── README.md

## Related

- [sktime](https://github.com/sktime/sktime)
- [sktime issue #9721](https://github.com/sktime/sktime/issues/9721)
- [sktime-mcp](https://github.com/sktime/sktime-mcp)

## License

MIT
