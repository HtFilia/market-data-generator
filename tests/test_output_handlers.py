import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from outputs.csv_handler import CSVHandler
from outputs.memory_handler import MemoryHandler

def test_csv_handler():
    """Test CSV output handler."""
    # Setup
    dates = pd.date_range(start='2023-01-01', end='2023-01-03', freq='B')
    prices = np.array([
        [100.0, 200.0],
        [101.0, 201.0],
        [102.0, 202.0]
    ])
    assets = [
        {'id': 'ASSET_001', 'sector': 'Technology', 'geography': 'North_America'},
        {'id': 'ASSET_002', 'sector': 'Financial', 'geography': 'Europe'}
    ]
    
    # Test CSV output
    csv_path = Path("test_output.csv")
    handler = CSVHandler(str(csv_path))
    handler.save(dates, prices, assets)
    
    # Verify file exists and content
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == len(dates) * len(assets)
    assert set(df.columns) == {'date', 'asset_id', 'close', 'sector', 'geography'}
    assert df['close'].min() > 0
    
    # Clean up
    csv_path.unlink()

def test_memory_handler():
    """Test memory output handler."""
    # Setup
    dates = pd.date_range(start='2023-01-01', end='2023-01-03', freq='B')
    prices = np.array([
        [100.0, 200.0],
        [101.0, 201.0],
        [102.0, 202.0]
    ])
    assets = [
        {'id': 'ASSET_001', 'sector': 'Technology', 'geography': 'North_America'},
        {'id': 'ASSET_002', 'sector': 'Financial', 'geography': 'Europe'}
    ]
    
    # Test memory output
    handler = MemoryHandler()
    handler.save(dates, prices, assets)
    
    # Verify data
    df = handler.get_data()
    assert len(df) == len(dates) * len(assets)
    assert set(df.columns) == {'date', 'asset_id', 'close', 'sector', 'geography'}
    assert df['close'].min() > 0
    assert len(df['asset_id'].unique()) == len(assets)
    assert len(df['date'].unique()) == len(dates)

def test_memory_handler_no_data():
    """Test memory handler error when no data is stored."""
    handler = MemoryHandler()
    with pytest.raises(ValueError):
        handler.get_data() 