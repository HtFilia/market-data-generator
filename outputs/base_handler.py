from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class OutputHandler(ABC):
    """Abstract base class for output handlers."""

    @abstractmethod
    def save(
        self, dates: pd.DatetimeIndex, prices: np.ndarray, assets: List[Dict[str, Any]]
    ) -> None:
        """
        Save simulation results.

        Args:
            dates: Array of dates
            prices: Array of prices with shape (n_dates, n_assets)
            assets: List of asset metadata dictionaries
        """
        pass
