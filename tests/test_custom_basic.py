import numpy as np
import pandas as pd
from datetime import datetime

from models.custom import CustomModel
from models.market_description import MarketDescription


def test_single_asset_same_sector_same_area():
    """Test single asset in the same sector and geographical area."""
    market_description = MarketDescription(
        n_assets=1, sectors=["Tech"], geographical_areas=["US"]
    )
    model = CustomModel(market_description)

    # Validate model data
    assert model._model_data is not None
    assert len(model._model_data.initial_prices) == 1
    assert len(model._model_data.initial_volatility) == 1
    assert len(model._model_data.long_term_volatility) == 1
    assert len(model._model_data.mean_reversion_speed) == 1
    assert len(model._model_data.mean_reversion_speed_vol) == 1
    assert len(model._model_data.vol_of_vol) == 1
    assert len(model._model_data.long_term_price) == 1
    assert len(model._model_data.market_beta) == 1
    assert len(model._model_data.sector_beta) == 1
    assert len(model._model_data.jump_intensity) == 1
    assert len(model._model_data.jump_mean) == 1
    assert len(model._model_data.jump_std) == 1
    assert len(model._model_data.price_vol_correlation) == 1
    assert model._model_data.correlation_matrix.shape == (1, 1)
    assert model._model_data.correlation_matrix[0, 0] == 1.0

    # Simulate paths
    dates = pd.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 31),  # One month of data
        freq="B",
    )
    prices = model.simulate_paths(dates, seed=42)

    # Basic checks
    assert prices.shape == (len(dates), 1)
    assert np.all(prices > 0)
    assert np.allclose(prices[0], model._model_data.initial_prices)

    # Check volatility function
    vol = model.get_volatility(0.0)  # Get initial volatility
    assert vol.shape == (1,)
    assert np.all(vol > 0)
    assert np.all(vol <= model._model_data.long_term_volatility)  # Volatility should not exceed long-term level


def test_single_asset_simulation():
    """Test price simulation for a single asset."""
    market_description = MarketDescription(
        n_assets=1, sectors=["Tech"], geographical_areas=["US"]
    )
    model = CustomModel(market_description)

    # Test with different time periods
    for days in [5, 10, 20]:
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

            # Check for jumps (should be present due to jump-diffusion)
            log_returns = np.diff(np.log(prices), axis=0)
            assert np.any(np.abs(log_returns) > 0.02)  # Should have some jumps


def test_single_asset_volatility():
    """Test volatility function for a single asset."""
    market_description = MarketDescription(
        n_assets=1, sectors=["Tech"], geographical_areas=["US"]
    )
    model = CustomModel(market_description)

    # Test volatility at different time points
    time_points = [0.0, 0.5, 1.0, 2.0, 5.0]
    for t in time_points:
        vol = model.get_volatility(t)
        assert vol.shape == (1,)
        assert vol[0] > 0
        assert vol[0] == model._model_data.initial_volatility[0]  # Initial volatility 