from abc import ABC, abstractmethod
from typing import Optional, Type, Generic, TypeVar
import numpy as np
import pandas as pd

from .market_description import MarketDescription

T = TypeVar("T")
U = TypeVar("U")


class MarketModel(Generic[T]):
    """
    Base class for market models.
    Each model must implement its own Calibrator inner class.
    """

    model_name: str = ""  # Override this in subclasses to specify the model name

    class Data(ABC):
        """Abstract base class for market model data."""

        pass

    class Calibrator(Generic[U]):
        """Base class for model calibrators."""

        def __init__(self, market_description: MarketDescription):
            self.market_description = market_description
            self._calibration_data: Optional[U] = None

        def calibrate(self) -> U:
            """
            Calibrate the model parameters.

            Returns:
                Model-specific calibration data
            """
            raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_calibrator_class(cls) -> Type["MarketModel.Calibrator"]:
        """Get the concrete calibrator class for this model."""
        pass

    def __init__(self, market_description: MarketDescription):
        """
        Initialize the market model.

        Args:
            market_description: Description of the market structure
        """
        self.market_description = market_description
        self.calibrator = self.get_calibrator_class()(market_description)
        self._model_data = self.calibrator.calibrate()
        self._validate_parameters()

    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        pass

    @abstractmethod
    def simulate_paths(
        self,
        dates: pd.DatetimeIndex,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate price paths for multiple assets.

        Args:
            dates: Array of dates to simulate
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
