from typing import Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path
from .base_handler import OutputHandler


class CSVHandler(OutputHandler):
    """Handler for saving simulation results to CSV."""

    def __init__(self, output_path: str):
        """
        Initialize the CSV handler.

        Args:
            output_path: Path to save the CSV file
        """
        self.output_path = Path(output_path)

    def save(
        self, dates: pd.DatetimeIndex, prices: np.ndarray, assets: List[Dict[str, Any]]
    ) -> None:
        """
        Save simulation results to CSV.

        Args:
            dates: Array of dates
            prices: Array of prices with shape (n_dates, n_assets)
            assets: List of asset metadata dictionaries
        """
        # Create DataFrame
        data = []
        for i, date in enumerate(dates):
            for j, asset in enumerate(assets):
                data.append(
                    {
                        "date": date,
                        "asset_id": asset["id"],
                        "close": prices[i, j],
                        "sector": asset["sector"],
                        "geography": asset["geography"],
                    }
                )

        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(self.output_path, index=False)
