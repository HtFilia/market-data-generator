import numpy as np
import pandas as pd
from datetime import datetime
import pytest

from models.custom import CustomModel
from models.market_description import MarketDescription


def test_single_asset():
    """Test model with a single asset."""
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


def test_large_number_of_assets():
    """Test model with a large number of assets."""
    n_assets = 100
    market_description = MarketDescription(
        n_assets=n_assets,
        sectors=["Tech"] * n_assets,
        geographical_areas=["US"] * n_assets,
    )
    model = CustomModel(market_description)

    # Validate model data
    assert model._model_data is not None
    assert len(model._model_data.initial_prices) == n_assets
    assert len(model._model_data.initial_volatility) == n_assets
    assert len(model._model_data.long_term_volatility) == n_assets
    assert len(model._model_data.mean_reversion_speed) == n_assets
    assert len(model._model_data.mean_reversion_speed_vol) == n_assets
    assert len(model._model_data.vol_of_vol) == n_assets
    assert len(model._model_data.long_term_price) == n_assets
    assert len(model._model_data.market_beta) == n_assets
    assert len(model._model_data.sector_beta) == n_assets
    assert len(model._model_data.jump_intensity) == n_assets
    assert len(model._model_data.jump_mean) == n_assets
    assert len(model._model_data.jump_std) == n_assets
    assert len(model._model_data.price_vol_correlation) == n_assets
    assert model._model_data.correlation_matrix.shape == (n_assets, n_assets)

    # Check correlation matrix properties
    corr = model._model_data.correlation_matrix
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert np.all(np.linalg.eigvals(corr) > 0)


def test_extreme_parameters():
    """Test model with extreme parameter values."""
    market_description = MarketDescription(
        n_assets=2, sectors=["Tech", "Finance"], geographical_areas=["US", "EU"]
    )
    model = CustomModel(market_description)

    # Set extreme parameters
    model._model_data.initial_volatility = np.array([0.01, 0.5])  # Very low and very high vol
    model._model_data.long_term_volatility = np.array([0.005, 0.6])
    model._model_data.mean_reversion_speed = np.array([0.1, 5.0])  # Slow and fast mean reversion
    model._model_data.mean_reversion_speed_vol = np.array([0.05, 4.0])
    model._model_data.vol_of_vol = np.array([0.1, 0.8])  # Low and high vol of vol
    model._model_data.jump_intensity = np.array([0.01, 0.5])  # Rare and frequent jumps
    model._model_data.jump_mean = np.array([-0.1, 0.1])  # Negative and positive jump means
    model._model_data.jump_std = np.array([0.01, 0.2])  # Small and large jump sizes

    # Simulate paths
    dates = pd.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 31),  # One month of data
        freq="B",
    )
    prices = model.simulate_paths(dates, seed=42)

    # Basic checks
    assert prices.shape == (len(dates), 2)
    assert np.all(prices > 0)
    assert np.allclose(prices[0], model._model_data.initial_prices)

    # Check for extreme movements
    returns = np.diff(prices, axis=0) / prices[:-1]
    assert np.any(np.abs(returns) > 0.1)  # Should have some large moves


def test_invalid_inputs():
    """Test model with invalid inputs."""
    # Test with zero assets
    with pytest.raises(ValueError):
        MarketDescription(n_assets=0, sectors=["Tech"], geographical_areas=["US"])

def test_simulation_edge_cases():
    """Test simulation with edge cases."""
    market_description = MarketDescription(
        n_assets=2, sectors=["Tech", "Finance"], geographical_areas=["US", "EU"]
    )
    model = CustomModel(market_description)

    # Test with single date
    single_date = pd.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 1),
        freq="B",
    )
    prices = model.simulate_paths(single_date, seed=42)
    assert prices.shape == (1, 2)
    assert np.allclose(prices[0], model._model_data.initial_prices)

    # Test with non-business days
    dates = pd.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 7),  # Includes weekend
        freq="D",
    )
    prices = model.simulate_paths(dates, seed=42)
    assert prices.shape == (len(dates), 2)
    assert np.all(prices > 0)

    # Test with very long time period
    long_dates = pd.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),  # Full year
        freq="B",
    )
    prices = model.simulate_paths(long_dates, seed=42)
    assert prices.shape == (len(long_dates), 2)
    assert np.all(prices > 0)

    # Test with different seeds
    prices1 = model.simulate_paths(dates, seed=42)
    prices2 = model.simulate_paths(dates, seed=42)
    prices3 = model.simulate_paths(dates, seed=123)
    assert np.allclose(prices1, prices2)  # Same seed should give same result
    assert not np.allclose(prices1, prices3)  # Different seed should give different result 