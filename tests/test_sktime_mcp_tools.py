import json

import pandas as pd
import pytest

from sktime_mcp_tools import _load_series_from_json


def test_load_series_from_json_dict():
    """Test loading data from a dict with values."""
    payload = json.dumps({
        "values": [1.0, 2.0, 3.0]
    })
    
    series = _load_series_from_json(payload)
    assert len(series) == 3
    assert series.iloc[0] == 1.0


def test_load_series_from_json_list():
    """Test loading data from a plain list."""
    payload = json.dumps([1.0, 2.0, 3.0])
    
    series = _load_series_from_json(payload)
    assert len(series) == 3
    assert series.iloc[-1] == 3.0


def test_load_series_from_json_pairs():
    """Test loading data from timestamp-value pairs."""
    payload = json.dumps([
        ["2020-01-01", 100],
        ["2020-02-01", 200]
    ])
    
    series = _load_series_from_json(payload)
    assert len(series) == 2
    assert isinstance(series.index, pd.PeriodIndex)
    assert series.iloc[0] == 100


def test_load_series_invalid():
    """Test invalid payload raises error."""
    payload = json.dumps({"wrong_key": 123})
    with pytest.raises(ValueError, match="must be a JSON object"):
        _load_series_from_json(payload)
