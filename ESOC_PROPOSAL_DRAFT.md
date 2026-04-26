# [ENH] Foundation Model Text-to-Exogenous Transformer & Agentic Forecaster

## Description
This proposal introduces a novel bridge between unstructured textual event logs and structured time-series forecasting within the `sktime` ecosystem.

While `sktime` has made strides in LLM-based forecasting, most current implementations focus on univariate prediction from numbers. However, real-world forecasting (e.g., retail, supply chain) is often driven by events recorded in text: 
- *"Store closed for 2 days due to flash flooding"*
- *"Competitor launched a 20% discount campaign"*
- *"Server downtime from 2 PM to 6 PM"*

I am proposing the implementation of the **`LLMTextEventFeaturizer`**, a transformer that interprets these logs into numerical exogenous signals, and the **`LLMPipelineForecaster`**, an agentic wrapper that autonomously handles data analysis and pipeline selection.

## Proposed Components

### 1. `LLMTextEventFeaturizer` (Transformer)
- **Base class**: `BaseTransformer`
- **Functionality**: Uses a Foundation Model (Groq/OpenAI/etc.) to featurize unstructured text columns into multivariate numerical data based on a provided schema.
- **Innovation**: Native support for "Context-Aware" forecasting without manual feature engineering.

### 2. `LLMAgentForecaster` (Meta-Forecaster)
- **Base class**: `BaseMetaForecaster`
- **Dynamic Registry**: Uses `sktime.registry.all_estimators` to scan the local environment and provide the LLM with a comprehensive list of candidate models (moving beyond hardcoded logic).
- **Validation-Gated Selection**: Every LLM-proposed configuration is validated against a data slice to protect against hallucinations before the full fit.
- **Semantic Intelligence**: Proactively cross-references text logs with numerical trends to flag data integrity anomalies (e.g., text indicates a closure but numbers remain high).
- **Autonomous Benchmarking**: Generates a rich diagnostic report comparing agentic performance against naive baselines with calculated "Accuracy Lift."

## Prototype & Progress
I have already developed a working prototype including:
- [x] Functional `LLMTextEventFeaturizer` with batch support.
- [x] Iterative `LLMPipelineForecaster` logic.
- [x] Tutorial notebook showcasing sales prediction improved by event log featurization.

**Repository Link:** [Your Repo URL Here]

## ESoC 2026 Context
This work is intended as a contribution for the **Agentic Forecaster / Foundation Models** track of ESoC 2026. I am seeking feedback from maintainers on the architectural fit within `sktime.transformations` and `sktime.forecasting`.

---
*Submitted as part of the sktime ESoC 2026 program.*
