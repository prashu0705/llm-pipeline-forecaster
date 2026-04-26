import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from llm_pipeline_forecaster import LLMAgentForecaster

class TestLLMAgentForecaster(unittest.TestCase):
    def setUp(self):
        self.y = pd.Series(np.random.randn(30), index=pd.period_range("2024-01", periods=30, freq="M"))
        self.agent = LLMAgentForecaster(
            prompt="Forecast sales",
            api_key="fake_key"
        )

    @patch('llm_pipeline_forecaster.LLMAgentForecaster._call_llm')
    def test_intent_extraction(self, mock_call):
        # Mock intent response
        mock_call.return_value = '{"horizon": 3, "accuracy_priority": "medium", "force_model": null, "ignore_seasonality": false, "domain": "retail", "intent_summary": "Forecast sales"}'
        
        intent = self.agent._extract_intent("Forecast sales next 3 months")
        self.assertEqual(intent['horizon'], 3)
        self.assertEqual(intent['domain'], 'retail')

    @patch('llm_pipeline_forecaster.LLMAgentForecaster._call_llm')
    def test_ask_llm_selection(self, mock_call):
        # Mock model selection response
        mock_call.return_value = '{"detrend": true, "deseasonalize": false, "model": "NaiveForecaster", "params": {"strategy": "last"}, "reasoning": "Simple data", "anomaly": null}'
        
        summary = {"is_stationary": True, "has_trend": True, "has_seasonality": False, "std": 1.0}
        config, reasoning = self.agent._ask_llm(summary)
        
        self.assertEqual(config['model'], 'NaiveForecaster')
        self.assertTrue(config['detrend'])

    def test_build_pipeline(self):
        config = {
            "detrend": True,
            "deseasonalize": False,
            "model": "NaiveForecaster",
            "params": {"strategy": "mean"}
        }
        # We need data_summary_ for sp
        self.agent.data_summary_ = {"seasonal_period": 12}
        
        pipeline = self.agent._build_pipeline(config)
        self.assertIsNotNone(pipeline)
        # Should be a TransformedTargetForecaster due to detrend=True
        from sktime.forecasting.compose import TransformedTargetForecaster
        self.assertIsInstance(pipeline, TransformedTargetForecaster)

if __name__ == '__main__':
    unittest.main()
