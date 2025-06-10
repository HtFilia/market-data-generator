import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from models.black_scholes import BlackScholesModel
from models.market_description import MarketDescription


def test_empty_market():
    """Test behavior with zero assets."""
    with pytest.raises(ValueError):
        MarketDescription(n_assets=0, sectors=[], geographical_areas=[])


def test_single_sector_multiple_assets():
    """Test that a single sector is correctly applied to all assets."""
    market_description = MarketDescription(
        n_assets=2,
        sectors=["Tech"],  # Single sector for two assets
        geographical_areas=["US", "EU"],
    )
    model = BlackScholesModel(market_description)

    # Validate that both assets are in the Tech sector
    assets_metadata = market_description.get_asset_metadata()
    assert len(assets_metadata) == 2
    assert all(asset["sector"] == "Tech" for asset in assets_metadata)

    # Validate model data
    assert model._model_data is not None
    assert len(model._model_data.initial_prices) == 2
    assert len(model._model_data.volatility) == 2
    assert model._model_data.correlation_matrix.shape == (2, 2)


def test_single_day_simulation():
    """Test simulation with only one day."""
    market_description = MarketDescription(
        n_assets=2, sectors=["Tech", "Finance"], geographical_areas=["US", "EU"]
    )
    model = BlackScholesModel(market_description)

    dates = pd.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 1),  # Same start and end date
        freq="B",
    )

    prices = model.simulate_paths(dates)
    assert prices.shape == (1, 2)
    assert np.array_equal(prices[0], model._model_data.initial_prices)


def test_very_long_simulation():
    """Test simulation with a very long time period."""
    market_description = MarketDescription(
        n_assets=2, sectors=["Tech", "Finance"], geographical_areas=["US", "EU"]
    )
    model = BlackScholesModel(market_description)

    dates = pd.date_range(
        start=datetime(2023, 1, 1), end=datetime(2023, 12, 31), freq="B"  # Full year
    )

    prices = model.simulate_paths(dates)
    assert prices.shape == (len(dates), 2)
    assert np.all(prices > 0)

    # Check for reasonable price movements over long period
    returns = np.diff(prices, axis=0) / prices[:-1]
    assert np.all(np.abs(returns) < 0.5)  # No extreme daily returns
    assert np.all(
        np.abs(np.log(prices[-1] / prices[0])) < 5
    )  # No extreme total returns


def test_negative_time_volatility():
    """Test volatility function with negative time."""
    market_description = MarketDescription(
        n_assets=2, sectors=["Tech", "Finance"], geographical_areas=["US", "EU"]
    )
    model = BlackScholesModel(market_description)

    # Should handle negative time gracefully
    vol = model.get_volatility(-1.0)
    assert vol.shape == (2,)
    assert np.all(vol > 0)
    assert np.array_equal(vol, model._model_data.volatility)


def test_reproducibility():
    """Test that simulations are reproducible with the same seed."""
    market_description = MarketDescription(
        n_assets=2, sectors=["Tech", "Finance"], geographical_areas=["US", "EU"]
    )
    model = BlackScholesModel(market_description)

    dates = pd.date_range(
        start=datetime(2023, 1, 1), end=datetime(2023, 1, 10), freq="B"
    )

    # Run same simulation twice with same seed
    prices1 = model.simulate_paths(dates, seed=42)
    prices2 = model.simulate_paths(dates, seed=42)
    assert np.array_equal(prices1, prices2)

    # Run with different seed
    prices3 = model.simulate_paths(dates, seed=43)
    assert not np.array_equal(prices1, prices3)


def test_empty_dates_simulation():
    """Test simulation with empty dates array."""
    market_description = MarketDescription(
        n_assets=2, sectors=["Tech", "Finance"], geographical_areas=["US", "EU"]
    )
    model = BlackScholesModel(market_description)

    # Create empty dates array with proper DatetimeIndex
    dates = pd.DatetimeIndex([], dtype="datetime64[ns]", freq="B")

    # Should return initial prices as a single row
    prices = model.simulate_paths(dates)
    assert prices.shape == (1, 2)  # 1 date (initial), 2 assets
    assert np.array_equal(prices[0], model._model_data.initial_prices)
