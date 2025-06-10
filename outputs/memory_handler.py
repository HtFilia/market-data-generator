from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base_handler import OutputHandler


class MemoryHandler(OutputHandler):
    """Handler for storing simulation results in memory."""

    def __init__(self):
        """Initialize the memory handler."""
        self.data = None

    def save(
        self, dates: pd.DatetimeIndex, prices: np.ndarray, assets: List[Dict[str, Any]]
    ) -> None:
        """
        Store simulation results in memory as a DataFrame.

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

        self.data = pd.DataFrame(data)

    def get_data(self) -> pd.DataFrame:
        """
        Get the stored simulation results.

        Returns:
            DataFrame containing simulation results
        """
        if self.data is None:
            raise ValueError("No data has been stored yet")
        return self.data
