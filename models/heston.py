from typing import Optional, Type
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .base import MarketModel


@dataclass
class HestonData(MarketModel.Data):
    """Data structure for Heston model parameters."""

    initial_prices: np.ndarray
    initial_volatility: np.ndarray  # Initial volatility (v0)
    long_term_volatility: np.ndarray  # Long-term mean volatility (theta)
    mean_reversion_speed: np.ndarray  # Mean reversion speed (kappa)
    vol_of_vol: np.ndarray  # Volatility of volatility (sigma)
    correlation_matrix: np.ndarray  # Correlation between price and volatility processes
    price_vol_correlation: np.ndarray  # Correlation between price and volatility


class HestonModel(MarketModel[HestonData]):
    """
    Heston model implementation with stochastic volatility.
    """

    model_name = "heston"

    class Calibrator(MarketModel.Calibrator[HestonData]):
        """Heston model calibrator."""

        def calibrate(self) -> HestonData:
            """Calibrate the Heston model parameters."""
            n_assets = self.market_description.n_assets

            # Generate initial prices (log-normal distribution)
            initial_prices = np.exp(np.random.normal(4, 1, n_assets))

            # Generate initial volatility (v0)
            initial_volatility = np.random.uniform(0.1, 0.3, n_assets)

            # Generate long-term mean volatility (theta)
            long_term_volatility = np.random.uniform(0.15, 0.35, n_assets)

            # Generate mean reversion speed (kappa)
            mean_reversion_speed = np.random.uniform(1.0, 3.0, n_assets)

            # Generate volatility of volatility (sigma)
            vol_of_vol = np.random.uniform(0.1, 0.4, n_assets)

            # Generate correlation between price and volatility
            price_vol_correlation = np.random.uniform(-0.7, -0.3, n_assets)

            # Generate correlation matrix for assets
            raw_corr = np.random.uniform(-0.3, 0.7, (n_assets, n_assets))
            raw_corr = (raw_corr + raw_corr.T) / 2
            np.fill_diagonal(raw_corr, 1.0)
            correlation_matrix = self._make_positive_definite(raw_corr)

            return HestonData(
                initial_prices=initial_prices,
                initial_volatility=initial_volatility,
                long_term_volatility=long_term_volatility,
                mean_reversion_speed=mean_reversion_speed,
                vol_of_vol=vol_of_vol,
                correlation_matrix=correlation_matrix,
                price_vol_correlation=price_vol_correlation,
            )

        def _make_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
            """Make a matrix positive definite by adjusting eigenvalues."""
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    @classmethod
    def get_calibrator_class(cls) -> Type[Calibrator]:
        """Get the concrete calibrator class for this model."""
        return cls.Calibrator

    def _validate_parameters(self) -> None:
        """Validate Heston model parameters."""
        if not isinstance(self._model_data, HestonData):
            raise ValueError("Invalid model data type")
        if not np.all(self._model_data.initial_volatility > 0):
            raise ValueError("All initial volatilities must be positive")
        if not np.all(self._model_data.long_term_volatility > 0):
            raise ValueError("All long-term volatilities must be positive")
        if not np.all(self._model_data.mean_reversion_speed > 0):
            raise ValueError("All mean reversion speeds must be positive")
        if not np.all(self._model_data.vol_of_vol > 0):
            raise ValueError("All volatility of volatility values must be positive")
        if not np.all(np.abs(self._model_data.price_vol_correlation) < 1):
            raise ValueError("Price-volatility correlations must be in (-1, 1)")

    def simulate_paths(
        self,
        dates: pd.DatetimeIndex,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate price paths using Heston model.

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

        dt = 1 / 252  # Assuming daily simulation

        # Generate correlated Brownian motions for prices
        z1 = np.random.normal(0, 1, (n_dates, n_assets))
        z1 = z1 @ np.linalg.cholesky(self._model_data.correlation_matrix)

        # Generate correlated Brownian motions for volatility
        z2 = np.random.normal(0, 1, (n_dates, n_assets))
        z2 = z2 @ np.linalg.cholesky(self._model_data.correlation_matrix)

        # Combine Brownian motions with correlation
        z2 = (
            self._model_data.price_vol_correlation * z1
            + np.sqrt(1 - self._model_data.price_vol_correlation**2) * z2
        )

        # Initialize arrays
        paths = np.zeros((n_dates, n_assets))
        volatility = np.zeros((n_dates, n_assets))
        paths[0] = self._model_data.initial_prices
        volatility[0] = self._model_data.initial_volatility

        # Simulate paths
        for t in range(1, n_dates):
            # Update volatility using CIR process
            vol_drift = (
                self._model_data.mean_reversion_speed
                * (self._model_data.long_term_volatility - volatility[t - 1])
                * dt
            )
            vol_diffusion = (
                self._model_data.vol_of_vol
                * np.sqrt(volatility[t - 1])
                * np.sqrt(dt)
                * z2[t]
            )
            volatility[t] = np.maximum(
                volatility[t - 1] + vol_drift + vol_diffusion, 1e-6
            )

            # Update prices using Heston dynamics
            price_drift = -0.5 * volatility[t] * dt
            price_diffusion = np.sqrt(volatility[t]) * np.sqrt(dt) * z1[t]
            paths[t] = paths[t - 1] * np.exp(price_drift + price_diffusion)

        return paths

    def get_volatility(self, t: float) -> np.ndarray:
        """
        Get the volatility for each asset at time t.
        For Heston model, this returns the current volatility state.

        Args:
            t: Time point

        Returns:
            Array of volatilities for each asset
        """
        # For simplicity, we return the initial volatility
        # In a real implementation, you might want to simulate the volatility path
        # up to time t and return the volatility at that point
        return self._model_data.initial_volatility 