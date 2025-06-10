from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelParameters:
    """Container for model parameters."""

    volatility: np.ndarray
    drift: np.ndarray
    risk_free_rate: float
    heston_params: Optional[Dict[str, Any]] = None


class SyntheticCalibrator:
    """Generator for synthetic model parameters."""

    def __init__(
        self,
        vol_mu: float = -2.0,
        vol_sigma: float = 0.4,
        drift_mu: float = 0.05,
        drift_sigma: float = 0.02,
        price_mu: float = 4.605,  # log(100)
        price_sigma: float = 0.5,
        risk_free_min: float = 0.01,
        risk_free_max: float = 0.03,
        seed: Optional[int] = None,
    ):
        """
        Initialize the synthetic calibrator.

        Args:
            vol_mu: Mean for log-normal volatility distribution
            vol_sigma: Standard deviation for log-normal volatility distribution
            drift_mu: Mean for normal drift distribution
            drift_sigma: Standard deviation for normal drift distribution
            price_mu: Mean for log-normal price distribution
            price_sigma: Standard deviation for log-normal price distribution
            risk_free_min: Minimum risk-free rate
            risk_free_max: Maximum risk-free rate
            seed: Random seed for reproducibility
        """
        self.vol_mu = vol_mu
        self.vol_sigma = vol_sigma
        self.drift_mu = drift_mu
        self.drift_sigma = drift_sigma
        self.price_mu = price_mu
        self.price_sigma = price_sigma
        self.risk_free_min = risk_free_min
        self.risk_free_max = risk_free_max

        if seed is not None:
            np.random.seed(seed)

    def generate_parameters(self, n_assets: int) -> ModelParameters:
        """
        Generate synthetic parameters for n assets.

        Args:
            n_assets: Number of assets

        Returns:
            ModelParameters object containing generated parameters
        """
        # Generate volatilities (10-50% range)
        volatilities = np.exp(np.random.normal(self.vol_mu, self.vol_sigma, n_assets))
        volatilities = np.clip(volatilities, 0.1, 0.5)

        # Generate drifts
        drifts = np.random.normal(self.drift_mu, self.drift_sigma, n_assets)

        # Generate risk-free rate
        risk_free_rate = np.random.uniform(self.risk_free_min, self.risk_free_max)

        return ModelParameters(
            volatility=volatilities, drift=drifts, risk_free_rate=risk_free_rate
        )

    def generate_start_prices(self, n_assets: int) -> np.ndarray:
        """
        Generate synthetic starting prices for n assets.

        Args:
            n_assets: Number of assets

        Returns:
            Array of starting prices
        """
        return np.exp(np.random.normal(self.price_mu, self.price_sigma, n_assets))

    def generate_heston_parameters(self, n_assets: int) -> Dict[str, Any]:
        """
        Generate Heston model parameters for n assets.

        Args:
            n_assets: Number of assets

        Returns:
            Dictionary of Heston parameters
        """
        # Long-term variance (theta)
        theta = np.exp(np.random.normal(-2.5, 0.3, n_assets))

        # Mean reversion speed (kappa)
        kappa = np.random.uniform(0.5, 2.0, n_assets)

        # Volatility of volatility (eta)
        eta = np.exp(np.random.normal(-3.0, 0.2, n_assets))

        # Asset-volatility correlation (rho)
        rho = np.random.uniform(-0.7, -0.3, n_assets)

        return {"theta": theta, "kappa": kappa, "eta": eta, "rho": rho}
