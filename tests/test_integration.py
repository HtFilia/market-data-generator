import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd

from models.market_description import MarketDescription
from models.factory import MarketModelFactory
from outputs.csv_handler import CSVHandler
from outputs.memory_handler import MemoryHandler


def test_end_to_end_simulation():
    """Test end-to-end simulation with Black-Scholes model."""
    # Setup
    n_assets = 5
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)
    seed = 42

    # Create market description
    market_description = MarketDescription(
        n_assets=n_assets,
        sectors=["Technology"],
        geographical_areas=["North_America"],
        seed=seed,
    )

    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    # Create model (calibration happens in __init__)
    market_model = MarketModelFactory().create_model(
        "black_scholes", market_description
    )

    # Verify model data was created during initialization
    assert market_model._model_data is not None
    assert len(market_model._model_data.initial_prices) == n_assets
    assert market_model._model_data.correlation_matrix.shape == (n_assets, n_assets)

    # Simulate paths
    prices = market_model.simulate_paths(dates)

    # Test CSV output
    csv_path = Path("test_output.csv")
    csv_handler = CSVHandler(str(csv_path))
    csv_handler.save(dates, prices, market_description.get_asset_metadata())
    assert csv_path.exists()
    csv_path.unlink()  # Clean up

    # Test memory output
    mem_handler = MemoryHandler()
    mem_handler.save(dates, prices, market_description.get_asset_metadata())
    df = mem_handler.get_data()

    # Verify output
    assert len(df) == len(dates) * n_assets
    assert set(df.columns) == {"date", "asset_id", "close", "sector", "geography"}
    assert df["close"].min() > 0  # Prices should be positive
    assert len(df["asset_id"].unique()) == n_assets
    assert len(df["date"].unique()) == len(dates)


def test_reproducibility():
    """Test that simulations are reproducible with the same seed."""
    # Setup
    n_assets = 3
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 5)
    seed = 42

    # Create market description
    market_description = MarketDescription(
        n_assets=n_assets,
        sectors=["Technology"],
        geographical_areas=["North_America"],
        seed=seed,
    )

    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    # Create model (calibration happens in __init__)
    market_model = MarketModelFactory().create_model(
        "black_scholes", market_description
    )

    # Run two simulations with the same seed
    prices1 = market_model.simulate_paths(dates, seed=seed)
    prices2 = market_model.simulate_paths(dates, seed=seed)

    # Check reproducibility
    assert np.allclose(prices1, prices2)


def test_market_description_asset_creation():
    """Test that MarketDescription creates assets correctly."""
    # Setup
    n_assets = 5
    sectors = ["Technology", "Finance"]
    areas = ["North_America", "Europe"]
    seed = 42

    # Create market description
    market_description = MarketDescription(
        n_assets=n_assets, sectors=sectors, geographical_areas=areas, seed=seed
    )

    # Get asset metadata
    assets_metadata = market_description.get_asset_metadata()

    # Verify asset creation
    assert len(assets_metadata) == n_assets
    for asset in assets_metadata:
        assert asset["id"].startswith("ASSET_")
        assert asset["sector"] in sectors
        assert asset["geography"] in areas

    # Test reproducibility with same seed
    market_description2 = MarketDescription(
        n_assets=n_assets, sectors=sectors, geographical_areas=areas, seed=seed
    )
    assert market_description2.get_asset_metadata() == assets_metadata

    # Test different assignments with different seed
    market_description3 = MarketDescription(
        n_assets=n_assets, sectors=sectors, geographical_areas=areas, seed=43
    )
    assert market_description3.get_asset_metadata() != assets_metadata
