import numpy as np
import pandas as pd
from datetime import datetime

from models.black_scholes import BlackScholesModel
from models.market_description import MarketDescription


def test_single_asset_same_sector_same_area():
    """Test Black-Scholes model with a single asset in the same sector and area."""
    market_description = MarketDescription(
        n_assets=1, sectors=["Tech"], geographical_areas=["US"]
    )
    model = BlackScholesModel(market_description)

    # Validate model data
    assert model._model_data is not None
    assert len(model._model_data.initial_prices) == 1
    assert len(model._model_data.volatility) == 1
    assert model._model_data.correlation_matrix.shape == (1, 1)
    assert (
        model._model_data.correlation_matrix[0, 0] == 1.0
    )  # Single asset correlation is 1
    assert np.all(model._model_data.volatility > 0)


def test_single_asset_simulation():
    """Test price simulation for a single asset."""
    market_description = MarketDescription(
        n_assets=1, sectors=["Tech"], geographical_areas=["US"]
    )
    model = BlackScholesModel(market_description)

    # Test with different time periods
    for days in [1, 5, 10, 20]:
        dates = pd.date_range(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 1) + pd.Timedelta(days=days),
            freq="B",
        )

        # Test with different seeds
        for seed in [42, 123, 456]:
            prices = model.simulate_paths(dates, seed)

            # Basic shape and positivity checks
            assert prices.shape == (len(dates), 1)
            assert np.all(prices > 0)
            assert prices[0, 0] == model._model_data.initial_prices[0]

            # Check for reasonable price movements
            returns = np.diff(prices, axis=0) / prices[:-1]
            assert np.all(np.abs(returns) < 0.5)  # No extreme daily returns


def test_single_asset_volatility():
    """Test volatility function for a single asset."""
    market_description = MarketDescription(
        n_assets=1, sectors=["Tech"], geographical_areas=["US"]
    )
    model = BlackScholesModel(market_description)

    # Test volatility at different time points
    time_points = [0.0, 0.5, 1.0, 2.0, 5.0]
    for t in time_points:
        vol = model.get_volatility(t)
        assert vol.shape == (1,)
        assert vol[0] > 0
        assert vol[0] == model._model_data.volatility[0]  # Constant volatility in BS
