from typing import Optional
import numpy as np
import pandas as pd
from .base import MarketModel

class BlackScholesModel(MarketModel):
    """Black-Scholes model implementation using Geometric Brownian Motion."""
    
    def _validate_parameters(self) -> None:
        """Validate Black-Scholes model parameters."""
        required_params = {'volatility', 'drift', 'risk_free_rate'}
        if not all(param in self.parameters for param in required_params):
            raise ValueError(f"Missing required parameters: {required_params}")
        
        if not isinstance(self.parameters['volatility'], np.ndarray):
            raise TypeError("Volatility must be a numpy array")
        if not isinstance(self.parameters['drift'], np.ndarray):
            raise TypeError("Drift must be a numpy array")
        if not isinstance(self.parameters['risk_free_rate'], float):
            raise TypeError("Risk-free rate must be a float")
    
    def simulate_paths(
        self,
        start_prices: np.ndarray,
        dates: pd.DatetimeIndex,
        correlation_matrix: np.ndarray,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion.
        
        Args:
            start_prices: Array of initial prices for each asset
            dates: Array of dates to simulate
            correlation_matrix: Correlation matrix between assets
            seed: Random seed for reproducibility
            
        Returns:
            Array of simulated prices with shape (n_dates, n_assets)
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_dates = len(dates)
        n_assets = len(start_prices)
        
        # Calculate time steps in years
        dt = np.diff(dates.astype(np.int64)) / (365 * 24 * 60 * 60 * 1e9)
        dt = np.insert(dt, 0, 0)  # Add initial time step of 0
        
        # Generate correlated Brownian motions
        L = np.linalg.cholesky(correlation_matrix)
        Z = np.random.normal(0, 1, (n_dates, n_assets))
        dW = np.sqrt(dt[:, np.newaxis]) * (Z @ L.T)
        
        # Calculate drift and diffusion terms
        drift = (self.parameters['drift'] - 0.5 * self.parameters['volatility']**2) * dt[:, np.newaxis]
        diffusion = self.parameters['volatility'] * dW
        
        # Simulate log returns
        log_returns = drift + diffusion
        
        # Calculate prices
        prices = start_prices * np.exp(np.cumsum(log_returns, axis=0))
        
        return prices
    
    def get_volatility(self, t: float) -> np.ndarray:
        """
        Get the constant volatility for each asset.
        
        Args:
            t: Time point (unused in Black-Scholes as volatility is constant)
            
        Returns:
            Array of constant volatilities for each asset
        """
        return self.parameters['volatility'] 