from typing import Optional, Type
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .base import MarketModel


@dataclass
class BlackScholesData(MarketModel.Data):
    """Data structure for Black-Scholes model parameters."""

    initial_prices: np.ndarray
    volatility: np.ndarray
    correlation_matrix: np.ndarray


class BlackScholesModel(MarketModel[BlackScholesData]):
    """
    Black-Scholes model implementation.
    """

    model_name = "black_scholes"

    class Calibrator(MarketModel.Calibrator[BlackScholesData]):
        """Black-Scholes model calibrator."""

        def calibrate(self) -> BlackScholesData:
            """Calibrate the Black-Scholes model parameters."""
            n_assets = self.market_description.n_assets

            # Generate initial prices (log-normal distribution)
            initial_prices = np.exp(np.random.normal(4, 1, n_assets))

            # Generate constant volatility
            volatility = np.random.uniform(0.1, 0.4, n_assets)

            # Generate correlation matrix
            # Start with random correlations
            raw_corr = np.random.uniform(-0.3, 0.7, (n_assets, n_assets))
            # Make it symmetric
            raw_corr = (raw_corr + raw_corr.T) / 2
            # Set diagonal to 1
            np.fill_diagonal(raw_corr, 1.0)
            # Ensure positive definiteness
            correlation_matrix = self._make_positive_definite(raw_corr)

            return BlackScholesData(
                initial_prices=initial_prices,
                volatility=volatility,
                correlation_matrix=correlation_matrix,
            )

        def _make_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
            """Make a matrix positive definite by adjusting eigenvalues."""
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-6)  # Ensure positive eigenvalues
            return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    @classmethod
    def get_calibrator_class(cls) -> Type[Calibrator]:
        """Get the concrete calibrator class for this model."""
        return cls.Calibrator

    def _validate_parameters(self) -> None:
        """Validate Black-Scholes model parameters."""
        if not isinstance(self._model_data, BlackScholesData):
            raise ValueError("Invalid model data type")
        if not np.all(self._model_data.volatility > 0):
            raise ValueError("All volatilities must be positive")

    def simulate_paths(
        self,
        dates: pd.DatetimeIndex,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate price paths using Black-Scholes model.

        Args:
            dates: Array of dates to simulate
            seed: Random seed for reproducibility

        Returns:
            Array of simulated prices with shape (n_dates, n_assets)
        """
        if seed is not None:
            np.random.seed(seed)

        n_dates = len(dates)
        n_assets = len(self._model_data.initial_prices)

        # Handle empty dates by returning initial prices
        if n_dates == 0:
            return self._model_data.initial_prices.reshape(1, -1)

        # Calculate time steps using day count convention
        dt = np.array([
            self.market_description.day_count.year_fraction(dates[i-1], dates[i])
            for i in range(1, n_dates)
        ])

        # Generate correlated Brownian motions
        z = np.random.normal(0, 1, (n_dates, n_assets))
        z = z @ np.linalg.cholesky(self._model_data.correlation_matrix)

        # Simulate paths
        paths = np.zeros((n_dates, n_assets))
        paths[0] = self._model_data.initial_prices

        for t in range(1, n_dates):
            drift = -0.5 * self._model_data.volatility**2 * dt[t-1]
            diffusion = self._model_data.volatility * np.sqrt(dt[t-1])
            paths[t] = paths[t - 1] * np.exp(drift + diffusion * z[t])

        return paths

    def get_volatility(self, t: float) -> np.ndarray:
        """
        Get the volatility for each asset at time t.

        Args:
            t: Time point

        Returns:
            Array of volatilities for each asset
        """
        return self._model_data.volatility
