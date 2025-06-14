from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Union, Type, Dict, ClassVar


class DayCountConvention(ABC):
    """Abstract base class for day count conventions."""
    
    name: ClassVar[str]
    
    @abstractmethod
    def year_fraction(self, start_date: Union[datetime, date], end_date: Union[datetime, date]) -> float:
        """
        Calculate the year fraction between two dates according to the convention.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Year fraction as a float
        """
        pass


class ActualActual(DayCountConvention):
    """Actual/Actual day count convention."""
    
    name = "ACT/ACT"
    
    def year_fraction(self, start_date: Union[datetime, date], end_date: Union[datetime, date]) -> float:
        """Calculate year fraction using Actual/Actual convention."""
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
            
        # Get the year of each date
        start_year = start_date.year
        end_year = end_date.year
        
        # Calculate days in each year
        days_in_start_year = 366 if self._is_leap_year(start_year) else 365
        days_in_end_year = 366 if self._is_leap_year(end_year) else 365
        
        # If dates are in the same year
        if start_year == end_year:
            return (end_date - start_date).days / days_in_start_year
        
        # Calculate fraction for partial years
        start_year_fraction = (date(start_year + 1, 1, 1) - start_date).days / days_in_start_year
        end_year_fraction = (end_date - date(end_year, 1, 1)).days / days_in_end_year
        
        # Add full years in between
        full_years = end_year - start_year - 1
        
        return start_year_fraction + full_years + end_year_fraction
    
    @staticmethod
    def _is_leap_year(year: int) -> bool:
        """Check if a year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


class Actual360(DayCountConvention):
    """Actual/360 day count convention."""
    
    name = "ACT/360"
    
    def year_fraction(self, start_date: Union[datetime, date], end_date: Union[datetime, date]) -> float:
        """Calculate year fraction using Actual/360 convention."""
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
            
        return (end_date - start_date).days / 360


class Actual365(DayCountConvention):
    """Actual/365 day count convention."""
    
    name = "ACT/365"
    
    def year_fraction(self, start_date: Union[datetime, date], end_date: Union[datetime, date]) -> float:
        """Calculate year fraction using Actual/365 convention."""
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
            
        return (end_date - start_date).days / 365


# Registry of available day count conventions
DAY_COUNT_CONVENTIONS: Dict[str, Type[DayCountConvention]] = {
    "ACT/ACT": ActualActual,
    "ACT/360": Actual360,
    "ACT/365": Actual365,
}


def get_day_count_convention(name: str) -> DayCountConvention:
    """
    Get a day count convention by name.
    
    Args:
        name: Name of the convention (e.g., "ACT/ACT", "ACT/360", "ACT/365")
        
    Returns:
        Instance of the requested day count convention
        
    Raises:
        ValueError: If the convention name is not recognized
    """
    if name not in DAY_COUNT_CONVENTIONS:
        raise ValueError(f"Unknown day count convention: {name}")
    return DAY_COUNT_CONVENTIONS[name]() 