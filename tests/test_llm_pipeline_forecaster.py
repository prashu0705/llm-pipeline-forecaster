import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from llm_pipeline_forecaster import LLMPipelineForecaster


@pytest.fixture
def mock_groq_client():
    """Mock the Groq client used inside the forecaster."""
    with patch("llm_pipeline_forecaster.Groq") as mock:
        yield mock


@pytest.fixture
def sample_timeseries():
    """Create a sample time series index and values."""
    idx = pd.PeriodIndex(pd.date_range("2020-01-01", periods=24, freq="M"))
    values = np.linspace(10, 50, 24) + np.sin(np.linspace(0, 4*np.pi, 24)) * 5
    return pd.Series(values, index=idx)


def test_analyze_series(sample_timeseries):
    """Test the statistical analysis of a time series."""
    forecaster = LLMPipelineForecaster("test", "test_key")
    
    summary = forecaster._analyze_series(sample_timeseries)
    
    assert summary["length"] == 24
    assert summary["has_trend"] is True  # np.linspace gives strong trend
    assert summary["min"] < summary["max"]
    assert "freq" in summary
    assert "is_stationary" in summary


def test_extract_intent(mock_groq_client):
    """Test extracting intent from a prompt with mocked LLM."""
    mock_instance = mock_groq_client.return_value
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "horizon": 12,
        "accuracy_priority": "high",
        "force_model": "arima",
        "ignore_seasonality": False,
        "domain": "sales",
        "intent_summary": "High accuracy sales forecast"
    })
    mock_instance.chat.completions.create.return_value = mock_response

    forecaster = LLMPipelineForecaster("Get me a high accuracy sales forecast for next year using arima", "test_key")
    intent = forecaster._extract_intent(forecaster.prompt)
    
    assert intent["horizon"] == 12
    assert intent["accuracy_priority"] == "high"
    assert intent["force_model"] == "arima"


def test_build_pipeline():
    """Test that the correct sktime objects are instantiated based on config."""
    forecaster = LLMPipelineForecaster("test", "test")
    # Provide dummy data summary
    forecaster.data_summary_ = {"seasonal_period": 12}
    
    # 1. Test ETS with detrend and deseasonalize
    config = {
        "detrend": True,
        "deseasonalize": True,
        "model": "ets"
    }
    
    pipeline = forecaster._build_pipeline(config)
    # The returned pipeline is a TransformedTargetForecaster
    steps = [name for name, _ in pipeline.steps]
    assert "detrend" in steps
    assert "deseasonalize" in steps
    assert "forecast" in steps
    
    # 2. Test simple Naive model (no transformers)
    config_naive = {
        "detrend": False,
        "deseasonalize": False,
        "model": "naive"
    }
    naive_pipeline = forecaster._build_pipeline(config_naive)
    # Should be just the model itself, not wrapped in TransformedTargetForecaster
    from sktime.forecasting.naive import NaiveForecaster
    assert isinstance(naive_pipeline, NaiveForecaster)
