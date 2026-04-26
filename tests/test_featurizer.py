import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from llm_text_featurizer import LLMTextEventFeaturizer

class TestLLMTextEventFeaturizer(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            "logs": ["Sale started", "Rainy day", "Holiday"]
        }, index=pd.date_range("2024-01-01", periods=3))
        
        self.featurizer = LLMTextEventFeaturizer(
            api_key="fake_key",
            text_column="logs",
            feature_schema={"impact": "float"},
            batch_size=2
        )

    @patch('llm_text_featurizer.LLMTextEventFeaturizer._call_llm')
    def test_transform_batching(self, mock_call):
        # Mock batch response (list of objects)
        mock_call.return_value = '[{"impact": 0.5}, {"impact": 0.1}]'
        
        # We need to handle the fact that it calls twice for batch_size=2 with 3 items
        # 1st call for 2 items, 2nd call for 1 item
        mock_call.side_effect = [
            '[{"impact": 0.5}, {"impact": 0.1}]',
            '[{"impact": 0.9}]'
        ]
        
        X_trans = self.featurizer.fit_transform(self.X)
        
        self.assertEqual(len(X_trans), 3)
        self.assertIn("impact", X_trans.columns)
        self.assertEqual(X_trans["impact"].iloc[0], 0.5)
        self.assertEqual(X_trans["impact"].iloc[2], 0.9)
        self.assertNotIn("logs", X_trans.columns)

if __name__ == '__main__':
    unittest.main()
