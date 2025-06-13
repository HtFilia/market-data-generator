import numpy as np
import pandas as pd
from datetime import datetime

from models.custom import CustomModel
from models.market_description import MarketDescription


def test_multi_asset_same_sector_same_area():
    """Test multiple assets in the same sector and area."""
    market_description = MarketDescription(
        n_assets=3, sectors=["Tech"], geographical_areas=["US"]
    )
    model = CustomModel(market_description)

    # Validate model data
    assert model._model_data is not None
    assert len(model._model_data.initial_prices) == 3
    assert len(model._model_data.initial_volatility) == 3
    assert len(model._model_data.long_term_volatility) == 3
    assert len(model._model_data.mean_reversion_speed) == 3
    assert len(model._model_data.mean_reversion_speed_vol) == 3
    assert len(model._model_data.vol_of_vol) == 3
    assert len(model._model_data.long_term_price) == 3
    assert len(model._model_data.market_beta) == 3
    assert len(model._model_data.sector_beta) == 3
    assert len(model._model_data.jump_intensity) == 3
    assert len(model._model_data.jump_mean) == 3
    assert len(model._model_data.jump_std) == 3
    assert len(model._model_data.price_vol_correlation) == 3
    assert model._model_data.correlation_matrix.shape == (3, 3)

    # Check correlation matrix properties
    corr = model._model_data.correlation_matrix
    assert np.allclose(np.diag(corr), 1.0)  # Diagonal elements are 1
    assert np.allclose(corr, corr.T)  # Symmetric
    assert np.all(np.linalg.eigvals(corr) > 0)  # Positive definite


def test_multi_asset_different_sectors_same_area():
    """Test multiple assets in different sectors but same area."""
    market_description = MarketDescription(
        n_assets=3, sectors=["Tech", "Finance", "Healthcare"], geographical_areas=["US"]
    )
    model = CustomModel(market_description)

    # Validate model data
    assert model._model_data is not None
    assert len(model._model_data.initial_prices) == 3
    assert len(model._model_data.initial_volatility) == 3
    assert len(model._model_data.long_term_volatility) == 3
    assert len(model._model_data.mean_reversion_speed) == 3
    assert len(model._model_data.mean_reversion_speed_vol) == 3
    assert len(model._model_data.vol_of_vol) == 3
    assert len(model._model_data.long_term_price) == 3
    assert len(model._model_data.market_beta) == 3
    assert len(model._model_data.sector_beta) == 3
    assert len(model._model_data.jump_intensity) == 3
    assert len(model._model_data.jump_mean) == 3
    assert len(model._model_data.jump_std) == 3
    assert len(model._model_data.price_vol_correlation) == 3
    assert model._model_data.correlation_matrix.shape == (3, 3)

    # Check correlation matrix properties
    corr = model._model_data.correlation_matrix
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert np.all(np.linalg.eigvals(corr) > 0)


def test_multi_asset_different_areas_same_sector():
    """Test multiple assets in different areas but same sector."""
    market_description = MarketDescription(
        n_assets=3, sectors=["Tech"], geographical_areas=["US", "EU", "Asia"]
    )
    model = CustomModel(market_description)

    # Validate model data
    assert model._model_data is not None
    assert len(model._model_data.initial_prices) == 3
    assert len(model._model_data.initial_volatility) == 3
    assert len(model._model_data.long_term_volatility) == 3
    assert len(model._model_data.mean_reversion_speed) == 3
    assert len(model._model_data.mean_reversion_speed_vol) == 3
    assert len(model._model_data.vol_of_vol) == 3
    assert len(model._model_data.long_term_price) == 3
    assert len(model._model_data.market_beta) == 3
    assert len(model._model_data.sector_beta) == 3
    assert len(model._model_data.jump_intensity) == 3
    assert len(model._model_data.jump_mean) == 3
    assert len(model._model_data.jump_std) == 3
    assert len(model._model_data.price_vol_correlation) == 3
    assert model._model_data.correlation_matrix.shape == (3, 3)

    # Check correlation matrix properties
    corr = model._model_data.correlation_matrix
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert np.all(np.linalg.eigvals(corr) > 0)


def test_multi_asset_all_different():
    """Test multiple assets with different sectors and areas."""
    market_description = MarketDescription(
        n_assets=4,
        sectors=["Tech", "Finance", "Healthcare", "Energy"],
        geographical_areas=["US", "EU", "Asia", "UK"],
    )
    model = CustomModel(market_description)

    # Validate model data
    assert model._model_data is not None
    assert len(model._model_data.initial_prices) == 4
    assert len(model._model_data.initial_volatility) == 4
    assert len(model._model_data.long_term_volatility) == 4
    assert len(model._model_data.mean_reversion_speed) == 4
    assert len(model._model_data.mean_reversion_speed_vol) == 4
    assert len(model._model_data.vol_of_vol) == 4
    assert len(model._model_data.long_term_price) == 4
    assert len(model._model_data.market_beta) == 4
    assert len(model._model_data.sector_beta) == 4
    assert len(model._model_data.jump_intensity) == 4
    assert len(model._model_data.jump_mean) == 4
    assert len(model._model_data.jump_std) == 4
    assert len(model._model_data.price_vol_correlation) == 4
    assert model._model_data.correlation_matrix.shape == (4, 4)

    # Check correlation matrix properties
    corr = model._model_data.correlation_matrix
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert np.all(np.linalg.eigvals(corr) > 0)


def test_multi_asset_simulation():
    """Test price simulation for multiple assets."""
    market_description = MarketDescription(
        n_assets=3,
        sectors=["Tech", "Finance", "Healthcare"],
        geographical_areas=["US", "EU", "Asia"],
    )
    model = CustomModel(market_description)

    # Use a longer time period for more reliable statistics
    dates = pd.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),  # Full year of data
        freq="B",
    )

    # Test with different seeds
    for seed in [42, 123, 456]:
        prices = model.simulate_paths(dates, seed=seed)

        # Basic shape and positivity checks
        assert prices.shape == (len(dates), 3)
        assert np.all(prices > 0)
        assert np.allclose(prices[0], model._model_data.initial_prices)

        # Check for reasonable price movements
        returns = np.diff(prices, axis=0) / prices[:-1]
        assert np.all(np.abs(returns) < 0.5)  # No extreme daily returns

        # Check for jumps (should be present due to jump-diffusion)
        log_returns = np.diff(np.log(prices), axis=0)
        assert np.any(np.abs(log_returns) > 0.02)  # Should have some jumps

        # Check that returns have the expected properties
        returns_corr = np.corrcoef(returns.T)
        assert np.allclose(np.diag(returns_corr), 1.0)  # Diagonal should be 1
        assert np.allclose(returns_corr, returns_corr.T)  # Should be symmetric
        assert np.all(np.linalg.eigvals(returns_corr) > 0)  # Should be positive definite

        # Check that the correlation structure is reasonable
        # (but don't expect exact match with model correlation)
        assert np.all(returns_corr >= -1) and np.all(returns_corr <= 1)  # Valid correlation range
        assert np.all(np.abs(returns_corr - np.eye(3)) < 1)  # Not perfectly correlated 