import os
import importlib
import inspect
from typing import Dict, Type
from .market_description import MarketDescription
from .base import MarketModel


class MarketModelFactory:
    """
    Factory for creating market models.
    Automatically discovers and registers all MarketModel subclasses.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the factory and discover all market models.

        Args:
            models_dir: Directory containing model implementations
        """
        self._models: Dict[str, Type[MarketModel]] = {}
        self._discover_models(models_dir)

    def _discover_models(self, models_dir: str) -> None:
        """
        Discover all MarketModel subclasses in the given directory.

        Args:
            models_dir: Directory to search for model implementations
        """
        # Get the absolute path of the models directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_path = os.path.join(base_dir, models_dir)

        # Iterate through all Python files in the directory
        for filename in os.listdir(models_path):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]  # Remove .py extension

                # Skip base.py and factory.py
                if module_name in ["base", "factory"]:
                    continue

                # Import the module
                try:
                    module = importlib.import_module(f"{models_dir}.{module_name}")

                    # Find all MarketModel subclasses in the module
                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, MarketModel)
                            and obj != MarketModel
                        ):
                            # Use the model_name attribute if defined, otherwise use class name
                            model_name = getattr(obj, "model_name", name.lower())
                            if model_name:  # Only register if model_name is set
                                self._models[model_name] = obj

                except ImportError as e:
                    print(f"Warning: Could not import {module_name}: {e}")

    def get_available_models(self) -> list[str]:
        """
        Get a list of available model names.

        Returns:
            List of available model names
        """
        return list(self._models.keys())

    def create_model(
        self, model_name: str, market_description: MarketDescription
    ) -> MarketModel:
        """
        Create a market model instance.

        Args:
            model_name: Name of the model to create
            market_description: Market description to use

        Returns:
            Market model instance

        Raises:
            ValueError: If model_name is not registered
        """
        model_class = self._models.get(model_name.lower())
        if model_class is None:
            available = ", ".join(self.get_available_models())
            raise ValueError(
                f"Unknown model: {model_name}. " f"Available models: {available}"
            )

        return model_class(market_description)
