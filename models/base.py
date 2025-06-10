from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

class MarketModel(ABC):
    """Abstract base class for market models."""
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the market model.
        
        Args:
            parameters: Dictionary of model-specific parameters
        """
        self.parameters = parameters
        self._validate_parameters()
    
    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        pass
    
    @abstractmethod
    def simulate_paths(
        self,
        start_prices: np.ndarray,
        dates: pd.DatetimeIndex,
        correlation_matrix: np.ndarray,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate price paths for multiple assets.
        
        Args:
            start_prices: Array of initial prices for each asset
            dates: Array of dates to simulate
            correlation_matrix: Correlation matrix between assets
            seed: Random seed for reproducibility
            
        Returns:
            Array of simulated prices with shape (n_dates, n_assets)
        """
        pass
    
    @abstractmethod
    def get_volatility(self, t: float) -> np.ndarray:
        """
        Get the volatility for each asset at time t.
        
        Args:
            t: Time point
            
        Returns:
            Array of volatilities for each asset
        """
        pass 