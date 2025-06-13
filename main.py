import os
from typing import Optional, Union, Tuple
import click
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from models.market_description import MarketDescription
from models.factory import MarketModelFactory
from outputs.csv_handler import CSVHandler
from outputs.memory_handler import MemoryHandler

# Type alias for output handlers
OutputHandler = Union[CSVHandler, MemoryHandler]
OUTPUT_PATH = "data/simulation_results.csv"

@click.command()
@click.option(
    "--model",
    type=click.Choice(["black_scholes", "heston", "custom"]),
    default="black_scholes",
    help="Market model to use. Available models: black_scholes (constant volatility), heston (stochastic volatility), custom (multi-factor stochastic volatility with jumps). Default: black_scholes"
)
@click.option(
    "--days",
    type=int,
    default=252,
    help="Number of days to simulate. Default: 252 (one trading year)"
)
@click.option(
    "--assets",
    type=int,
    default=10,
    help="Number of assets to simulate. Default: 10"
)
@click.option(
    "--sectors",
    type=str,
    multiple=True,
    default=(
        "Technology",
        "Healthcare",
        "Financial",
        "Consumer",
        "Energy",
        "Materials",
        "Industrials",
        "Utilities",
        "Real Estate",
        "Communication",
    ),
    help="Sectors to include. Can be specified multiple times. Default: Technology, Healthcare, Financial, Consumer, Energy, Materials, Industrials, Utilities, Real Estate, Communication"
)
@click.option(
    "--areas",
    type=str,
    multiple=True,
    default=("US", "EU", "Asia", "UK"),
    help="Geographical areas to include. Can be specified multiple times. Default: US, EU, Asia, UK"
)
@click.option(
    "--seed",
    type=Optional[int],
    default=None,
    help="Random seed for reproducibility. Default: None"
)
@click.option(
    "--output",
    type=click.Choice(["csv", "memory"]),
    default="csv",
    help="Output format. Default: csv"
)
def main(
    model: str,
    days: int,
    assets: int,
    sectors: Tuple[str, ...],
    areas: Tuple[str, ...],
    seed: int,
    output: str,
) -> None:
    """
    Run the market simulator.

    Args:
        model: Market model to use
        days: Number of days to simulate
        assets: Number of assets to simulate
        sectors: List of sectors
        areas: List of geographical areas
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    # Create market description (this will also create assets with their metadata)
    market_description = MarketDescription(
        n_assets=assets,
        sectors=list(sectors),
        geographical_areas=list(areas),
        seed=seed,
    )

    # Create factory and register models
    factory = MarketModelFactory()

    # Create and use models
    try:
        market_model = factory.create_model(model, market_description)
    except ValueError:
        print(f"Available models: {factory.get_available_models()}")
        return
    # Generate simulation dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    # Simulate paths
    paths = market_model.simulate_paths(dates)

    # Save results
    handler: OutputHandler
    if output == "csv":
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        handler = CSVHandler(OUTPUT_PATH)
    else:
        handler = MemoryHandler()

    # Get asset metadata from market description
    assets_metadata = market_description.get_asset_metadata()

    handler.save(dates, paths, assets_metadata)


if __name__ == "__main__":
    main()
