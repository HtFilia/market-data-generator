from typing import Optional, Type
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .base import MarketModel


@dataclass
class CustomModelData(MarketModel.Data):
    """Data structure for multi-factor stochastic volatility jump-diffusion model parameters."""

    initial_prices: np.ndarray
    initial_volatility: np.ndarray  # Initial volatility (v0)
    long_term_volatility: np.ndarray  # Long-term mean volatility (theta_v)
    mean_reversion_speed: np.ndarray  # Mean reversion speed for prices (kappa)
    mean_reversion_speed_vol: np.ndarray  # Mean reversion speed for volatility (kappa_v)
    vol_of_vol: np.ndarray  # Volatility of volatility (sigma_v)
    long_term_price: np.ndarray  # Long-term mean price level (theta)
    market_beta: np.ndarray  # Market factor loading (beta_m)
    sector_beta: np.ndarray  # Sector factor loading (beta_s)
    jump_intensity: np.ndarray  # Jump intensity (lambda)
    jump_mean: np.ndarray  # Mean jump size
    jump_std: np.ndarray  # Standard deviation of jump size
    correlation_matrix: np.ndarray  # Correlation between price and volatility processes
    price_vol_correlation: np.ndarray  # Correlation between price and volatility


class CustomModel(MarketModel[CustomModelData]):
    """
    Multi-factor stochastic volatility jump-diffusion model implementation.
    
    The model follows these SDEs:
    dln(Si(t)) = κi(θi - ln(Si(t)))dt + βm*dFm(t) + βs*dFs(t) + vi(t)*dWi(t) + Ji*dNi(t)
    dvi(t) = κv,i(θv,i - vi(t))dt + σv,i*vi(t)*dWv,i(t)
    """

    model_name = "custom"

    class Calibrator(MarketModel.Calibrator[CustomModelData]):
        """Custom model calibrator."""

        def calibrate(self) -> CustomModelData:
            """Calibrate the model parameters."""
            n_assets = self.market_description.n_assets

            # Generate initial prices (log-normal distribution)
            initial_prices = np.exp(np.random.normal(4, 1, n_assets))

            # Generate initial volatility (v0) - increased range
            initial_volatility = np.random.uniform(0.2, 0.5, n_assets)

            # Generate long-term mean volatility (theta_v) - increased range
            long_term_volatility = np.random.uniform(0.25, 0.45, n_assets)

            # Generate mean reversion speed for prices (kappa) - reduced to allow more persistence
            mean_reversion_speed = np.random.uniform(0.2, 1.0, n_assets)

            # Generate mean reversion speed for volatility (kappa_v) - reduced to allow more persistence
            mean_reversion_speed_vol = np.random.uniform(0.5, 2.0, n_assets)

            # Generate volatility of volatility (sigma_v) - increased to allow more volatility spikes
            vol_of_vol = np.random.uniform(0.2, 0.6, n_assets)

            # Generate long-term mean price level (theta)
            long_term_price = np.log(initial_prices) + np.random.uniform(-0.5, 0.5, n_assets)

            # Generate market and sector factor loadings - increased to allow larger market moves
            market_beta = np.random.uniform(0.5, 1.0, n_assets)
            sector_beta = np.random.uniform(0.3, 0.8, n_assets)

            # Generate jump parameters - increased frequency and size
            jump_intensity = np.random.uniform(0.05, 0.15, n_assets)  # 5-15 jumps per year
            jump_mean = np.random.uniform(-0.05, -0.02, n_assets)  # Larger negative jumps
            jump_std = np.random.uniform(0.02, 0.05, n_assets)  # Larger jump volatility

            # Generate correlation between price and volatility
            price_vol_correlation = np.random.uniform(-0.7, -0.3, n_assets)

            # Generate correlation matrix for assets
            raw_corr = np.random.uniform(-0.3, 0.7, (n_assets, n_assets))
            raw_corr = (raw_corr + raw_corr.T) / 2
            np.fill_diagonal(raw_corr, 1.0)
            
            # Make positive definite and normalize to ensure unit diagonal
            correlation_matrix = self._make_positive_definite(raw_corr)
            diag_sqrt = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / (diag_sqrt[:, np.newaxis] * diag_sqrt[np.newaxis, :])

            return CustomModelData(
                initial_prices=initial_prices,
                initial_volatility=initial_volatility,
                long_term_volatility=long_term_volatility,
                mean_reversion_speed=mean_reversion_speed,
                mean_reversion_speed_vol=mean_reversion_speed_vol,
                vol_of_vol=vol_of_vol,
                long_term_price=long_term_price,
                market_beta=market_beta,
                sector_beta=sector_beta,
                jump_intensity=jump_intensity,
                jump_mean=jump_mean,
                jump_std=jump_std,
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
        """Validate model parameters."""
        if not isinstance(self._model_data, CustomModelData):
            raise ValueError("Invalid model data type")
        if not np.all(self._model_data.initial_volatility > 0):
            raise ValueError("All initial volatilities must be positive")
        if not np.all(self._model_data.long_term_volatility > 0):
            raise ValueError("All long-term volatilities must be positive")
        if not np.all(self._model_data.mean_reversion_speed > 0):
            raise ValueError("All mean reversion speeds must be positive")
        if not np.all(self._model_data.mean_reversion_speed_vol > 0):
            raise ValueError("All volatility mean reversion speeds must be positive")
        if not np.all(self._model_data.vol_of_vol > 0):
            raise ValueError("All volatility of volatility values must be positive")
        if not np.all(self._model_data.jump_intensity >= 0):
            raise ValueError("All jump intensities must be non-negative")
        if not np.all(self._model_data.jump_std > 0):
            raise ValueError("All jump standard deviations must be positive")
        if not np.all(np.abs(self._model_data.price_vol_correlation) < 1):
            raise ValueError("Price-volatility correlations must be in (-1, 1)")

    def simulate_paths(
        self,
        dates: pd.DatetimeIndex,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate price paths using the multi-factor stochastic volatility jump-diffusion model.

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

        # Generate correlated Brownian motions for prices
        z1 = np.random.normal(0, 1, (n_dates, n_assets))
        z1 = z1 @ np.linalg.cholesky(self._model_data.correlation_matrix)

        # Generate correlated Brownian motions for volatility
        z2 = np.random.normal(0, 1, (n_dates, n_assets))
        z2 = z2 @ np.linalg.cholesky(self._model_data.correlation_matrix)

        # Generate market and sector factors
        # Note: We only need factors for t=1 to t=n_dates-1 since we don't use them at t=0
        market_factor = np.random.normal(0, np.sqrt(dt), n_dates-1)
        sector_factor = np.random.normal(0, np.sqrt(dt), n_dates-1)

        # Combine Brownian motions with correlation
        z2 = (
            self._model_data.price_vol_correlation * z1
            + np.sqrt(1 - self._model_data.price_vol_correlation**2) * z2
        )

        # Initialize arrays
        paths = np.zeros((n_dates, n_assets))
        volatility = np.zeros((n_dates, n_assets))
        log_prices = np.zeros((n_dates, n_assets))
        paths[0] = self._model_data.initial_prices
        volatility[0] = self._model_data.initial_volatility
        log_prices[0] = np.log(self._model_data.initial_prices)

        # Simulate paths
        for t in range(1, n_dates):
            # Update volatility using CIR process
            vol_drift = (
                self._model_data.mean_reversion_speed_vol
                * (self._model_data.long_term_volatility - volatility[t - 1])
                * dt[t-1]
            )
            vol_diffusion = (
                self._model_data.vol_of_vol
                * np.sqrt(volatility[t - 1])
                * np.sqrt(dt[t-1])
                * z2[t]
            )
            volatility[t] = np.maximum(
                volatility[t - 1] + vol_drift + vol_diffusion, 1e-6
            )

            # Update log prices
            price_drift = (
                self._model_data.mean_reversion_speed
                * (self._model_data.long_term_price - log_prices[t - 1])
                * dt[t-1]
            )
            price_diffusion = np.sqrt(volatility[t]) * np.sqrt(dt[t-1]) * z1[t]
            
            # Add market and sector factor contributions
            factor_contribution = (
                self._model_data.market_beta * market_factor[t-1]
                + self._model_data.sector_beta * sector_factor[t-1]
            )

            # Generate jumps
            jumps = np.zeros(n_assets)
            for i in range(n_assets):
                if np.random.random() < self._model_data.jump_intensity[i] * dt[t-1]:
                    jumps[i] = np.random.normal(
                        self._model_data.jump_mean[i],
                        self._model_data.jump_std[i]
                    )

            # Update log prices
            log_prices[t] = (
                log_prices[t - 1]
                + price_drift
                + price_diffusion
                + factor_contribution
                + jumps
            )

            # Convert back to prices
            paths[t] = np.exp(log_prices[t])

        return paths

    def get_volatility(self, t: float) -> np.ndarray:
        """
        Get the volatility for each asset at time t.
        For this model, this returns the current volatility state.

        Args:
            t: Time point

        Returns:
            Array of volatilities for each asset
        """
        # For simplicity, we return the initial volatility
        # In a real implementation, you might want to simulate the volatility path
        # up to time t and return the volatility at that point
        return self._model_data.initial_volatility 