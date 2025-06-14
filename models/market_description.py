from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from .day_count import DayCountConvention, get_day_count_convention


@dataclass
class Asset:
    """Represents a single asset in the market."""

    id: str
    sector: str
    geography: str


@dataclass
class MarketDescription:
    """Describes the structure of a market simulation."""

    n_assets: int
    sectors: List[str]
    geographical_areas: List[str]
    day_count_convention: str = "ACT/365"  # Default to ACT/365
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate the market description parameters and create assets."""
        if self.n_assets <= 0:
            raise ValueError("Number of assets must be positive")
        if not self.sectors:
            raise ValueError("At least one sector must be specified")
        if not self.geographical_areas:
            raise ValueError("At least one geographical area must be specified")

        if self.seed is not None:
            np.random.seed(self.seed)

        # Create assets with random sector and geography assignments
        self.assets = [
            Asset(
                id=f"ASSET_{i+1:03d}",
                sector=np.random.choice(self.sectors),
                geography=np.random.choice(self.geographical_areas),
            )
            for i in range(self.n_assets)
        ]
        
        # Initialize day count convention
        self._day_count = get_day_count_convention(self.day_count_convention)

    def get_asset_metadata(self) -> List[Dict[str, str]]:
        """Get metadata about assets including their sectors and geographical areas."""
        return [
            {"id": asset.id, "sector": asset.sector, "geography": asset.geography}
            for asset in self.assets
        ]
        
    @property
    def day_count(self) -> DayCountConvention:
        """Get the day count convention instance."""
        return self._day_count
