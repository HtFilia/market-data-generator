import numpy as np
import pandas as pd
from datetime import datetime

from models.black_scholes import BlackScholesModel
from models.market_description import MarketDescription


def test_black_scholes_parameters():
    """Test Black-Scholes model parameter validation."""
    # Create market description
    market_description = MarketDescription(
        n_assets=2, sectors=["Tech", "Finance"], geographical_areas=["US", "EU"]
    )
    model = BlackScholesModel(market_description)

    # Validate model data
    assert model._model_data is not None
    assert len(model._model_data.initial_prices) == 2
    assert len(model._model_data.volatility) == 2
    assert model._model_data.correlation_matrix.shape == (2, 2)
    assert np.all(model._model_data.volatility > 0)


def test_black_scholes_simulation():
    """Test Black-Scholes price simulation."""
    # Setup
    market_description = MarketDescription(
        n_assets=1, sectors=["Tech"], geographical_areas=["US"]
    )
    model = BlackScholesModel(market_description)
    dates = pd.date_range(
        start=datetime(2023, 1, 1), end=datetime(2023, 1, 10), freq="B"
    )
    seed = 42

    # Run simulation
    prices = model.simulate_paths(dates, seed)

    # Check results
    assert prices.shape == (len(dates), 1)
    assert np.all(prices > 0)  # Prices should be positive
    assert (
        prices[0, 0] == model._model_data.initial_prices[0]
    )  # First price should match initial price


def test_black_scholes_volatility():
    """Test Black-Scholes volatility function."""
    market_description = MarketDescription(
        n_assets=2, sectors=["Tech", "Finance"], geographical_areas=["US", "EU"]
    )
    model = BlackScholesModel(market_description)

    # Volatility should be constant in Black-Scholes
    vol1 = model.get_volatility(0.0)
    vol2 = model.get_volatility(1.0)
    assert np.array_equal(vol1, vol2)
    assert np.array_equal(vol1, model._model_data.volatility)
