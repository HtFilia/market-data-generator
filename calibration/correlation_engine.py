from typing import List, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Asset:
    """Asset class representing a financial instrument."""

    id: str
    sector: str
    geography: str


class CorrelationEngine:
    """Engine for generating and managing correlation matrices."""

    # Predefined correlation matrices
    SECTOR_CORRELATIONS = {
        ("Consumer", "Energy"): 0.3,
        ("Consumer", "Healthcare"): 0.6,
        ("Energy", "Energy"): 0.6,
        ("Energy", "Financial"): 0.4,
        ("Energy", "Technology"): 0.3,
        ("Financial", "Financial"): 0.7,
        ("Financial", "Technology"): 0.5,
        ("Healthcare", "Healthcare"): 0.75,
        ("Technology", "Technology"): 0.8,
    }

    GEOGRAPHY_CORRELATIONS = {
        ("Asia", "Asia"): 0.8,
        ("Asia", "Emerging"): 0.65,
        ("Asia", "Europe"): 0.55,
        ("Asia", "North_America"): 0.5,
        ("Europe", "Europe"): 0.85,
        ("Europe", "North_America"): 0.6,
        ("North_America", "North_America"): 0.8,
    }

    def __init__(
        self,
        sector_weight: float = 0.6,
        geography_weight: float = 0.4,
        noise_level: float = 0.05,
    ):
        """
        Initialize the correlation engine.

        Args:
            sector_weight: Weight for sector correlations
            geography_weight: Weight for geography correlations
            noise_level: Level of random noise to add
        """
        self.sector_weight = sector_weight
        self.geography_weight = geography_weight
        self.noise_level = noise_level

        # Validate weights
        if not np.isclose(sector_weight + geography_weight, 1.0):
            raise ValueError("Sector and geography weights must sum to 1.0")

    def _get_base_correlation(self, asset1: Asset, asset2: Asset) -> float:
        """
        Get the base correlation between two assets based on their sectors and geographies.

        Args:
            asset1: First asset
            asset2: Second asset

        Returns:
            Base correlation value
        """
        # Get sector correlation
        sector_key = (
            min(asset1.sector, asset2.sector),
            max(asset1.sector, asset2.sector),
        )
        sector_corr = self.SECTOR_CORRELATIONS.get(sector_key, 0.0)

        # Get geography correlation
        geo_key = (
            min(asset1.geography, asset2.geography),
            max(asset1.geography, asset2.geography),
        )
        geo_corr = self.GEOGRAPHY_CORRELATIONS.get(geo_key, 0.0)

        # Combine correlations with weights
        combined_corr = (
            self.sector_weight * sector_corr + self.geography_weight * geo_corr
        )

        return combined_corr

    def create_matrix(
        self, assets: List[Asset], seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Create a correlation matrix for the given assets.

        Args:
            assets: List of assets
            seed: Random seed for reproducibility

        Returns:
            Positive semi-definite correlation matrix
        """
        if seed is not None:
            np.random.seed(seed)

        n_assets = len(assets)
        corr_matrix = np.ones((n_assets, n_assets))

        # Fill correlation matrix
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                base_corr = self._get_base_correlation(assets[i], assets[j])
                # Add random noise
                noise = np.random.uniform(-self.noise_level, self.noise_level)
                corr_matrix[i, j] = corr_matrix[j, i] = base_corr + noise

        # Ensure positive semi-definite
        return self._adjust_psd(corr_matrix)

    def _adjust_psd(self, matrix: np.ndarray) -> np.ndarray:
        """
        Adjust matrix to ensure positive semi-definite property.

        Args:
            matrix: Input correlation matrix

        Returns:
            Adjusted positive semi-definite correlation matrix
        """
        # Get eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(matrix)

        # Set negative eigenvalues to small positive number
        eigenvals = np.maximum(eigenvals, 1e-6)

        # Reconstruct matrix
        adjusted_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Normalize to ensure unit diagonal
        diag_sqrt = np.sqrt(np.diag(adjusted_matrix))
        return adjusted_matrix / (diag_sqrt[:, np.newaxis] * diag_sqrt[np.newaxis, :])
