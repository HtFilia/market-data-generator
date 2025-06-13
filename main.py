import os
from typing import Optional, Union
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


@click.command()
@click.option("--n-assets", default=10, help="Number of assets to simulate")
@click.option(
    "--sectors",
    multiple=True,
    default=[
        "Information Technology",
        "Financials",
        "Healthcare",
        "Consumer Discretionary",
        "Consumer Staples",
        "Industrials",
        "Materials",
        "Energy",
        "Utilities",
        "Real Estate",
        "Communication Services"
    ],
    help="List of sectors (can be specified multiple times)",
)
@click.option(
    "--areas",
    multiple=True,
    default=["US", "EU", "Asia"],
    help="List of geographical areas (can be specified multiple times)",
)
@click.option(
    "--model-type", type=str, default="black_scholes", help="Market model to use"
)
@click.option("--days", type=int, default=252, help="Number of days to simulate")
@click.option(
    "--output",
    type=click.Choice(["csv", "memory"]),
    default="csv",
    help="Output format",
)
@click.option(
    "--output-path",
    type=str,
    default="data/simulation_results.csv",
    help="Path to save results (for CSV output)",
)
@click.option("--seed", type=int, help="Random seed for reproducibility")
def main(
    n_assets: int,
    sectors: tuple[str, ...],
    areas: tuple[str, ...],
    model_type: str,
    days: int,
    output: str,
    output_path: str,
    seed: Optional[int],
) -> None:
    """
    Run the market simulator.

    Args:
        n_assets: Number of assets to simulate
        sectors: List of sectors
        areas: List of geographical areas
        model_type: Market model to use
        days: Number of days to simulate
        output: Output format
        output_path: Path to save results
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    # Create market description (this will also create assets with their metadata)
    market_description = MarketDescription(
        n_assets=n_assets,
        sectors=list(sectors),
        geographical_areas=list(areas),
        seed=seed,
    )

    # Create factory and register models
    factory = MarketModelFactory()

    # Create and use models
    try:
        market_model = factory.create_model(model_type, market_description)
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        handler = CSVHandler(output_path)
    else:
        handler = MemoryHandler()

    # Get asset metadata from market description
    assets_metadata = market_description.get_asset_metadata()

    handler.save(dates, paths, assets_metadata)


if __name__ == "__main__":
    main()
