"""Core LLMPipelineForecaster implementation."""

import json

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from groq import Groq
from sktime.forecasting.base import BaseForecaster
from statsmodels.tsa.stattools import acf, adfuller


class LLMPipelineForecaster(BaseForecaster):
    """LLM-guided sktime forecaster with iterative pipeline selection."""

    _tags = {
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "ignores-exogeneous-X": True,
        "y_inner_mtype": "pd.Series",
        "capability:pred_int": True,
    }

    def __init__(
        self,
        prompt,
        api_key,
        model="llama-3.3-70b-versatile",
        max_iterations=3,
        mae_threshold=0.2,
        date_col=None,
        value_col=None,
    ):
        self.prompt = prompt
        self.api_key = api_key
        self.model = model
        self.max_iterations = max_iterations
        self.mae_threshold = mae_threshold
        self.date_col = date_col
        self.value_col = value_col
        super().__init__()

    def fit(self, y, X=None, fh=None):
        """Fit the forecaster from CSV, DataFrame, or Series input."""
        y = self._load_input(y)
        return super().fit(y, X=X, fh=fh)

    def _load_input(self, data):
        """Convert supported input formats to an sktime-compatible Series."""
        if isinstance(data, pd.Series):
            if isinstance(data.index, pd.PeriodIndex):
                return data
            try:
                data.index = pd.PeriodIndex(data.index, freq="M")
                return data
            except Exception:
                return data

        if isinstance(data, str):
            print(f"Loading CSV from: {data}")
            data = pd.read_csv(data)

        if isinstance(data, pd.DataFrame):
            date_col = self.date_col
            value_col = self.value_col

            if date_col is None:
                for col in data.columns:
                    if any(
                        kw in col.lower()
                        for kw in ["date", "time", "period", "month", "year", "day"]
                    ):
                        date_col = col
                        break
            if date_col is None:
                date_col = data.columns[0]

            if value_col is None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != date_col]
                if not numeric_cols:
                    raise ValueError("No numeric columns found in DataFrame.")
                value_col = numeric_cols[0]

            print(f"Detected date column: '{date_col}'")
            print(f"Detected value column: '{value_col}'")

            data = data.copy()
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.sort_values(date_col)

            diffs = data[date_col].diff().dropna()
            median_diff = diffs.median().days
            if median_diff <= 1:
                freq = "D"
            elif median_diff <= 8:
                freq = "W"
            elif median_diff <= 32:
                freq = "M"
            elif median_diff <= 93:
                freq = "Q"
            else:
                freq = "Y"

            print(f"Inferred frequency: {freq}")

            period_index = pd.PeriodIndex(data[date_col].dt.to_period(freq))
            return pd.Series(data[value_col].values, index=period_index, name=value_col)

        raise ValueError(
            f"Unsupported data type: {type(data)}. "
            "Pass a CSV path, pd.DataFrame, or pd.Series."
        )

    def _fit(self, y, X=None, fh=None):
        print("Extracting intent from prompt...")
        self.intent_ = self._extract_intent(self.prompt)
        print(f"Intent: {self.intent_}\n")

        priority = self.intent_.get("accuracy_priority", "medium")
        if priority == "low":
            effective_iterations = 1
        elif priority == "high":
            effective_iterations = 5
        else:
            effective_iterations = self.max_iterations

        if self.intent_.get("force_model"):
            effective_iterations = 1

        self.data_summary_ = self._analyze_series(y)

        if self.intent_.get("ignore_seasonality"):
            self.data_summary_["has_seasonality"] = False
            print("Seasonality ignored as per prompt.\n")

        split = int(len(y) * 0.8)
        y_train = y.iloc[:split]
        y_val = y.iloc[split:]
        val_fh = list(range(1, len(y_val) + 1))

        best_mae = float("inf")
        self.iteration_log_ = []
        previous_attempts = []

        for i in range(effective_iterations):
            print(f"\n--- Iteration {i + 1} ---")

            force_model = self.intent_.get("force_model")
            if force_model:
                config = {
                    "detrend": self.data_summary_["has_trend"],
                    "deseasonalize": self.data_summary_["has_seasonality"],
                    "model": force_model,
                }
                reasoning = f"Model forced by prompt intent: {force_model}"
                print(f"Prompt forced model: {force_model}")
            else:
                config, reasoning = self._ask_llm(
                    self.data_summary_, previous_attempts=previous_attempts
                )

            print(f"LLM chose: {config}")
            print(f"Reasoning: {reasoning}")

            pipeline = None
            try:
                pipeline = self._build_pipeline(config)
                pipeline.fit(y_train)
                y_pred_val = pipeline.predict(fh=val_fh)
                y_pred_val.index = y_val.index
                mae = float(np.mean(np.abs(y_val.values - y_pred_val.values)))
                mae_relative = mae / self.data_summary_["std"]
                print(f"MAE: {mae:.2f} (relative: {mae_relative:.3f})")
            except Exception as exc:
                failure_reason = str(exc)[:100]
                print(f"Pipeline failed: {failure_reason}")
                mae = float("inf")
                mae_relative = float("inf")
                reasoning = f"{reasoning} [FAILED: {failure_reason}]"

            attempt = {
                "iteration": i + 1,
                "config": config,
                "reasoning": reasoning,
                "mae": round(mae, 2) if mae != float("inf") else "FAILED",
                "mae_relative": (
                    round(mae_relative, 3)
                    if mae_relative != float("inf")
                    else "FAILED"
                ),
                "failed": mae == float("inf"),
            }
            self.iteration_log_.append(attempt)
            previous_attempts.append(attempt)

            if mae < best_mae and pipeline is not None:
                best_mae = mae
                self.reasoning_ = reasoning
                self.pipeline_config_ = config

            if mae_relative < self.mae_threshold:
                print(f"Accepted pipeline at iteration {i + 1}")
                break

            tried_models = [a["config"]["model"] for a in previous_attempts]
            available_models = ["naive", "arima", "ets"]
            if all(model in tried_models for model in available_models):
                print("All models tried. Stopping early.")
                break

        print(f"\nBest pipeline MAE: {best_mae:.2f}")
        print("Refitting best pipeline on full data...")
        self.pipeline_ = self._build_pipeline(self.pipeline_config_)
        self.pipeline_.fit(y, fh=fh)
        return self

    def _predict(self, fh, X=None):
        return self.pipeline_.predict(fh)

    def _predict_interval(self, fh, X=None, coverage=0.9):
        """Return prediction intervals with a naive fallback where needed."""
        model_name = self.pipeline_config_.get("model", "naive")
        if model_name in ["naive", "arima"]:
            return self.pipeline_.predict_interval(fh=fh, coverage=coverage)

        print(
            f"Note: {model_name.upper()} does not support prediction intervals "
            "natively. Using NaiveForecaster as fallback."
        )
        from sktime.forecasting.naive import NaiveForecaster

        naive = NaiveForecaster(strategy="last")
        naive.fit(self._y)
        return naive.predict_interval(fh=fh, coverage=coverage)

    def _analyze_series(self, y):
        """Analyze time series statistics used in pipeline selection."""
        freq = str(y.index.freqstr) if hasattr(y.index, "freqstr") else "unknown"
        sp_map = {"M": 12, "Q": 4, "W": 52, "D": 7, "H": 24}
        sp = sp_map.get(freq, 1)

        adf_result = adfuller(y.dropna())
        is_stationary = bool(adf_result[1] < 0.05)

        x = np.arange(len(y))
        trend_coef = float(np.polyfit(x, y.values, 1)[0])
        has_trend = abs(trend_coef) > (y.std() * 0.01)

        if sp > 1 and len(y) > sp * 2:
            acf_vals = acf(y.dropna(), nlags=sp, fft=True)
            seasonal_strength = float(abs(acf_vals[sp]))
            has_seasonality = seasonal_strength > 0.3
        else:
            seasonal_strength = 0.0
            has_seasonality = False

        return {
            "length": len(y),
            "mean": round(float(y.mean()), 2),
            "std": round(float(y.std()), 2),
            "min": round(float(y.min()), 2),
            "max": round(float(y.max()), 2),
            "freq": freq,
            "seasonal_period": sp,
            "is_stationary": is_stationary,
            "has_trend": bool(has_trend),
            "trend_slope": round(trend_coef, 4),
            "has_seasonality": has_seasonality,
            "seasonal_strength": round(seasonal_strength, 4),
        }

    def _extract_intent(self, prompt):
        """Extract structured forecasting intent from natural language prompt."""
        system_prompt = """You are a time series forecasting assistant.
Extract structured intent from a user's forecasting prompt.

Respond with ONLY a valid JSON object, no markdown, no explanation.
The JSON must have exactly these fields:
{
    "horizon": null or integer,
    "accuracy_priority": "low", "medium", or "high",
    "force_model": null or one of ["naive", "arima", "ets"],
    "ignore_seasonality": false or true,
    "domain": null or short domain string,
    "intent_summary": "one sentence summary"
}

Rules:
- "quick", "rough", "approximate" -> accuracy_priority = "low"
- "accurate", "precise", "best possible" -> accuracy_priority = "high"
- "next 3 months" -> horizon = 3
- "next year" -> horizon = 12
- "simple" or "baseline" -> force_model = "naive"
- default -> accuracy_priority = "medium", others null/false
"""
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Prompt: {prompt}"},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    def _ask_llm(self, data_summary, previous_attempts=None):
        """Ask the LLM to choose a forecasting pipeline configuration."""
        system_prompt = """You are an expert time series forecasting assistant.
Given a statistical summary of a time series and a user prompt, choose the best sktime pipeline.

Respond with ONLY a valid JSON object. No markdown. No explanation outside JSON.
The JSON must have exactly these fields:
{
    "detrend": true or false,
    "deseasonalize": true or false,
    "model": one of ["naive", "arima", "ets"],
    "reasoning": "one sentence explanation"
}

Decision rules:
- If has_trend is true, set detrend to true
- If has_seasonality is true, set deseasonalize to true
- If length < 50, use "naive"
- If is_stationary is false and has_seasonality is true, use "ets"
- If is_stationary is true, use "arima"
- Otherwise use "arima"
- If previous attempts failed, try a DIFFERENT model
"""
        attempts_str = ""
        if previous_attempts:
            attempts_str = "\nPrevious attempts:\n"
            for attempt in previous_attempts:
                if attempt.get("failed"):
                    attempts_str += (
                        f"- Iteration {attempt['iteration']}: "
                        f"model={attempt['config']['model']} -> FAILED\n"
                    )
                else:
                    attempts_str += (
                        f"- Iteration {attempt['iteration']}: "
                        f"model={attempt['config']['model']}, "
                        f"MAE={attempt['mae']}, "
                        f"relative_MAE={attempt['mae_relative']}\n"
                    )
            attempts_str += "\nAvoid FAILED configurations. Choose a DIFFERENT one."

        user_message = f"""
User prompt: {self.prompt}
Data statistics: {json.dumps(data_summary, indent=2)}
{attempts_str}
Respond with JSON only.
"""
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        config = json.loads(raw)
        reasoning = config.pop("reasoning", "No reasoning provided")
        return config, reasoning

    def _build_pipeline(self, config):
        """Build an sktime forecaster pipeline from the LLM config."""
        from sktime.forecasting.arima import AutoARIMA
        from sktime.forecasting.compose import TransformedTargetForecaster
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.transformations.series.detrend import Detrender, Deseasonalizer

        sp = self.data_summary_.get("seasonal_period", 1)
        model_map = {
            "naive": NaiveForecaster(strategy="last"),
            "arima": AutoARIMA(sp=sp),
            "ets": ExponentialSmoothing(trend="add", seasonal="add", sp=sp),
        }
        model = model_map.get(config.get("model", "naive"))

        steps = []
        if config.get("detrend"):
            steps.append(
                ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1)))
            )
        if config.get("deseasonalize"):
            steps.append(("deseasonalize", Deseasonalizer(sp=sp)))

        if steps:
            steps.append(("forecast", model))
            return TransformedTargetForecaster(steps)
        return model

    def get_pipeline_description(self):
        """Return a human-readable description of the chosen pipeline."""
        if not hasattr(self, "reasoning_"):
            return "Forecaster not fitted yet."

        config = self.pipeline_config_
        steps = []
        if config.get("detrend"):
            steps.append("detrending")
        if config.get("deseasonalize"):
            steps.append("deseasonalization")
        steps.append(f"{config.get('model').upper()} model")
        return (
            f"Pipeline: {' -> '.join(steps)}\n"
            f"Reasoning: {self.reasoning_}\n"
            f"Data: {self.data_summary_['length']} points, "
            f"freq={self.data_summary_['freq']}, "
            f"trend={self.data_summary_['has_trend']}, "
            f"seasonality={self.data_summary_['has_seasonality']}"
        )

    def get_iteration_log(self):
        """Print a summary of all evaluated iterations."""
        if not hasattr(self, "iteration_log_"):
            return "Forecaster not fitted yet."

        print(
            f"{'Iter':<6}{'Model':<8}"
            f"{'Detrend':<10}{'Deseas':<10}"
            f"{'MAE':<12}{'Relative MAE'}"
        )
        print("-" * 55)
        for log in self.iteration_log_:
            print(
                f"{log['iteration']:<6}"
                f"{log['config']['model']:<8}"
                f"{str(log['config']['detrend']):<10}"
                f"{str(log['config']['deseasonalize']):<10}"
                f"{str(log['mae']):<12}"
                f"{log['mae_relative']}"
            )

    def get_confidence_assessment(self):
        """Return an LLM-generated confidence assessment of the forecast."""
        if not hasattr(self, "iteration_log_"):
            return "Forecaster not fitted yet."

        summary = self.data_summary_
        valid_iterations = [
            iteration
            for iteration in self.iteration_log_
            if isinstance(iteration["mae_relative"], (int, float))
        ]
        if not valid_iterations:
            return {
                "confidence": "low",
                "reasons": ["All pipelines failed"],
                "warning": "No valid pipeline found.",
            }

        best_mae_relative = min(
            iteration["mae_relative"] for iteration in valid_iterations
        )

        assessment_prompt = f"""
You are a time series forecasting expert.
Given this information about a fitted forecasting model, assess confidence.

Data summary:
- Length: {summary['length']} points
- Is stationary: {summary['is_stationary']}
- Has trend: {summary['has_trend']}
- Has seasonality: {summary['has_seasonality']}
- Seasonal strength: {summary['seasonal_strength']}

Model performance:
- Best relative MAE: {best_mae_relative}
  (<0.1 excellent, 0.1-0.2 good, 0.2-0.3 fair, >0.3 poor)
- Models tried: {[iteration['config']['model'] for iteration in self.iteration_log_]}
- Any failures: {any(iteration.get('failed') for iteration in self.iteration_log_)}

Respond with ONLY valid JSON, no markdown:
{{
    "confidence": "high", "medium", or "low",
    "reasons": ["reason 1", "reason 2"],
    "warning": null or "one sentence warning"
}}
"""
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": assessment_prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)


def plot_forecast(forecaster, y, y_pred, title=None):
    """Plot the forecast, intervals, and iteration diagnostics."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        y.index.to_timestamp(),
        y.values,
        color="#2C5F8A",
        linewidth=1.8,
        label="Historical data",
    )
    ax1.plot(
        y_pred.index.to_timestamp(),
        y_pred.values,
        color="#E85D26",
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=5,
        label="Forecast",
    )

    try:
        intervals = forecaster.predict_interval(
            fh=list(range(1, len(y_pred) + 1)), coverage=0.9
        )
        lower = intervals.iloc[:, 0].values
        upper = intervals.iloc[:, 1].values
        pred_dates = y_pred.index.to_timestamp()
        ax1.fill_between(
            pred_dates,
            lower,
            upper,
            alpha=0.25,
            color="#E85D26",
            label="90% prediction interval",
        )
    except Exception:
        ax1.axvspan(
            y_pred.index.to_timestamp()[0],
            y_pred.index.to_timestamp()[-1],
            alpha=0.08,
            color="#E85D26",
        )

    try:
        conf = forecaster.get_confidence_assessment()
        conf_colors = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}
        conf_level = conf.get("confidence", "medium")
        ax1.annotate(
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
    except Exception:
        pass

    config = forecaster.pipeline_config_
    pipeline_str = []
    if config.get("detrend"):
        pipeline_str.append("Detrend")
    if config.get("deseasonalize"):
        pipeline_str.append("Deseasonalize")
    pipeline_str.append(config.get("model", "").upper())

    ax1.set_title(
        title or f"LLMPipelineForecaster - {' -> '.join(pipeline_str)}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    iters = forecaster.iteration_log_
    valid = [iteration for iteration in iters if isinstance(iteration["mae"], (int, float))]
    if valid:
        iter_nums = [iteration["iteration"] for iteration in valid]
        maes = [iteration["mae"] for iteration in valid]
        colors = ["#2ecc71" if mae == min(maes) else "#3498db" for mae in maes]
        bars = ax2.bar(iter_nums, maes, color=colors, edgecolor="white", linewidth=0.5)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("MAE")
        ax2.set_title("MAE per iteration", fontsize=11)
        ax2.set_xticks(iter_nums)
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, mae in zip(bars, maes):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{mae:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    summary = forecaster.data_summary_
    intent = getattr(forecaster, "intent_", {})
    info_lines = [
        ("Domain", intent.get("domain") or "-"),
        ("Accuracy priority", intent.get("accuracy_priority", "-")),
        ("Data points", str(summary["length"])),
        ("Frequency", summary["freq"]),
        ("Has trend", str(summary["has_trend"])),
        ("Has seasonality", str(summary["has_seasonality"])),
        ("Seasonal strength", str(summary["seasonal_strength"])),
        ("Is stationary", str(summary["is_stationary"])),
        ("Best model", config.get("model", "-").upper()),
        ("Best MAE", str(min([iteration["mae"] for iteration in valid])) if valid else "-"),
    ]
    y_pos = 0.95
    ax3.text(
        0.0,
        y_pos + 0.05,
        "Run summary",
        fontsize=11,
        fontweight="bold",
        transform=ax3.transAxes,
    )
    for label, value in info_lines:
        ax3.text(
            0.0,
            y_pos,
            f"{label}:",
            fontsize=9,
            color="#555555",
            transform=ax3.transAxes,
        )
        ax3.text(
            0.55,
            y_pos,
            value,
            fontsize=9,
            fontweight="bold",
            transform=ax3.transAxes,
        )
        y_pos -= 0.09

    plt.suptitle(
        f"Intent: {intent.get('intent_summary', '')}",
        fontsize=10,
        color="#666666",
        y=1.01,
    )
    plt.savefig("forecast_plot.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Plot saved as forecast_plot.png")
