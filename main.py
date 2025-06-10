import os
from typing import List, Optional
import click
from datetime import datetime
import numpy as np

from models.black_scholes import BlackScholesModel
from calibration.correlation_engine import CorrelationEngine, Asset
from calibration.synthetic_calibrator import SyntheticCalibrator
from core.date_generator import BusinessDayGenerator
from outputs.csv_handler import CSVHandler
from outputs.memory_handler import MemoryHandler

# Predefined sectors and geographies
SECTORS = ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer']
GEOGRAPHIES = ['North_America', 'Europe', 'Asia', 'Emerging']

def create_assets(n_assets: int, seed: Optional[int] = None) -> List[Asset]:
    """
    Create synthetic assets with random sector and geography assignments.
    
    Args:
        n_assets: Number of assets to create
        seed: Random seed for reproducibility
        
    Returns:
        List of Asset objects
    """
    if seed is not None:
        np.random.seed(seed)
    
    assets = []
    for i in range(n_assets):
        asset = Asset(
            id=f'ASSET_{i+1:03d}',
            sector=np.random.choice(SECTORS),
            geography=np.random.choice(GEOGRAPHIES)
        )
        assets.append(asset)
    
    return assets

@click.command()
@click.option('--assets', default=10, help='Number of assets to simulate')
@click.option('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2023-12-31', help='End date (YYYY-MM-DD)')
@click.option('--model', type=click.Choice(['black_scholes', 'heston']), default='black_scholes', help='Market model to use')
@click.option('--output', type=click.Choice(['csv', 'memory']), default='csv', help='Output format')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--corr-matrix-path', type=str, help='Path to save correlation matrix')
def main(
    assets: int,
    start_date: str,
    end_date: str,
    model: str,
    output: str,
    seed: Optional[int],
    corr_matrix_path: Optional[str]
) -> None:
    """
    Run the market simulator.
    
    Args:
        assets: Number of assets to simulate
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        model: Market model to use
        output: Output format
        seed: Random seed for reproducibility
        corr_matrix_path: Path to save correlation matrix
    """
    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Create assets
    assets_list = create_assets(assets, seed)
    
    # Generate business days
    date_gen = BusinessDayGenerator()
    dates = date_gen.generate_dates(start, end)
    
    # Generate correlation matrix
    corr_engine = CorrelationEngine()
    corr_matrix = corr_engine.create_matrix(assets_list, seed)
    
    # Save correlation matrix if requested
    if corr_matrix_path:
        np.save(corr_matrix_path, corr_matrix)
    
    # Generate model parameters
    calibrator = SyntheticCalibrator(seed=seed)
    params = calibrator.generate_parameters(assets)
    start_prices = calibrator.generate_start_prices(assets)
    
    # Create and run model
    if model == 'black_scholes':
        market_model = BlackScholesModel(params.__dict__)
    else:
        raise NotImplementedError(f"Model {model} not implemented yet")
    
    # Simulate prices
    prices = market_model.simulate_paths(start_prices, dates, corr_matrix, seed)
    
    # Save results
    if output == 'csv':
        handler = CSVHandler(os.path.join('data', 'simulation_results.csv'))
    else:
        handler = MemoryHandler()
    
    handler.save(dates, prices, [asset.__dict__ for asset in assets_list])

if __name__ == '__main__':
    main() 