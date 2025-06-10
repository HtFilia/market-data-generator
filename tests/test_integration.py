import numpy as np
from datetime import datetime
from pathlib import Path

from models.black_scholes import BlackScholesModel
from calibration.correlation_engine import CorrelationEngine, Asset
from calibration.synthetic_calibrator import SyntheticCalibrator
from core.date_generator import BusinessDayGenerator
from outputs.csv_handler import CSVHandler
from outputs.memory_handler import MemoryHandler

def test_end_to_end_simulation():
    """Test end-to-end simulation with Black-Scholes model."""
    # Setup
    n_assets = 5
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)
    seed = 42
    
    # Create assets
    assets = [
        Asset(id=f"ASSET_{i+1:03d}", sector="Technology", geography="North_America")
        for i in range(n_assets)
    ]
    
    # Generate dates
    date_gen = BusinessDayGenerator()
    dates = date_gen.generate_dates(start_date, end_date)
    
    # Generate correlation matrix
    corr_engine = CorrelationEngine()
    corr_matrix = corr_engine.create_matrix(assets, seed)
    
    # Generate parameters
    calibrator = SyntheticCalibrator(seed=seed)
    params = calibrator.generate_parameters(n_assets)
    start_prices = calibrator.generate_start_prices(n_assets)
    
    # Create and run model
    model = BlackScholesModel(params.__dict__)
    prices = model.simulate_paths(start_prices, dates, corr_matrix, seed)
    
    # Test CSV output
    csv_path = Path("test_output.csv")
    csv_handler = CSVHandler(str(csv_path))
    csv_handler.save(dates, prices, [asset.__dict__ for asset in assets])
    assert csv_path.exists()
    csv_path.unlink()  # Clean up
    
    # Test memory output
    mem_handler = MemoryHandler()
    mem_handler.save(dates, prices, [asset.__dict__ for asset in assets])
    df = mem_handler.get_data()
    
    # Verify output
    assert len(df) == len(dates) * n_assets
    assert set(df.columns) == {'date', 'asset_id', 'close', 'sector', 'geography'}
    assert df['close'].min() > 0  # Prices should be positive
    assert len(df['asset_id'].unique()) == n_assets
    assert len(df['date'].unique()) == len(dates)

def test_reproducibility():
    """Test that simulations are reproducible with the same seed."""
    # Setup
    n_assets = 3
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 5)
    seed = 42
    
    # Create assets
    assets = [
        Asset(id=f"ASSET_{i+1:03d}", sector="Technology", geography="North_America")
        for i in range(n_assets)
    ]
    
    # Generate dates
    date_gen = BusinessDayGenerator()
    dates = date_gen.generate_dates(start_date, end_date)
    
    # Generate correlation matrix
    corr_engine = CorrelationEngine()
    corr_matrix = corr_engine.create_matrix(assets, seed)
    
    # Generate parameters
    calibrator = SyntheticCalibrator(seed=seed)
    params = calibrator.generate_parameters(n_assets)
    start_prices = calibrator.generate_start_prices(n_assets)
    
    # Run two simulations with the same seed
    model = BlackScholesModel(params.__dict__)
    prices1 = model.simulate_paths(start_prices, dates, corr_matrix, seed)
    prices2 = model.simulate_paths(start_prices, dates, corr_matrix, seed)
    
    # Check reproducibility
    assert np.allclose(prices1, prices2) 