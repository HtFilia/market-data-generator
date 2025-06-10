import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from models.black_scholes import BlackScholesModel

def test_black_scholes_parameters():
    """Test Black-Scholes model parameter validation."""
    # Valid parameters
    valid_params = {
        'volatility': np.array([0.2, 0.3]),
        'drift': np.array([0.05, 0.06]),
        'risk_free_rate': 0.02
    }
    model = BlackScholesModel(valid_params)
    assert model.parameters == valid_params
    
    # Invalid parameters
    with pytest.raises(ValueError):
        BlackScholesModel({'volatility': np.array([0.2])})  # Missing parameters
    
    with pytest.raises(TypeError):
        BlackScholesModel({
            'volatility': 0.2,  # Should be array
            'drift': np.array([0.05]),
            'risk_free_rate': 0.02
        })

def test_black_scholes_simulation():
    """Test Black-Scholes price simulation."""
    # Setup
    params = {
        'volatility': np.array([0.2]),
        'drift': np.array([0.05]),
        'risk_free_rate': 0.02
    }
    model = BlackScholesModel(params)
    
    start_prices = np.array([100.0])
    dates = pd.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 10),
        freq='B'
    )
    corr_matrix = np.array([[1.0]])
    seed = 42
    
    # Run simulation
    prices = model.simulate_paths(start_prices, dates, corr_matrix, seed)
    
    # Check results
    assert prices.shape == (len(dates), 1)
    assert np.all(prices > 0)  # Prices should be positive
    assert prices[0, 0] == start_prices[0]  # First price should match start price

def test_black_scholes_volatility():
    """Test Black-Scholes volatility function."""
    params = {
        'volatility': np.array([0.2, 0.3]),
        'drift': np.array([0.05, 0.06]),
        'risk_free_rate': 0.02
    }
    model = BlackScholesModel(params)
    
    # Volatility should be constant in Black-Scholes
    vol1 = model.get_volatility(0.0)
    vol2 = model.get_volatility(1.0)
    assert np.array_equal(vol1, vol2)
    assert np.array_equal(vol1, params['volatility'])
