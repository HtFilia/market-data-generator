from typing import List, Optional
import pandas as pd
from datetime import datetime

class BusinessDayGenerator:
    """Generator for business days between start and end dates."""
    
    def __init__(self, holidays: Optional[List[datetime]] = None):
        """
        Initialize the business day generator.
        
        Args:
            holidays: Optional list of holiday dates to exclude
        """
        self.holidays = set(holidays) if holidays else set()
    
    def generate_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        freq: str = 'B'  # 'B' for business days
    ) -> pd.DatetimeIndex:
        """
        Generate business days between start and end dates.
        
        Args:
            start_date: Start date
            end_date: End date
            freq: Frequency string ('B' for business days)
            
        Returns:
            DatetimeIndex of business days
        """
        # Generate all business days
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Remove holidays if provided
        if self.holidays:
            dates = dates[~dates.isin(self.holidays)]
        
        return dates
    
    @staticmethod
    def is_business_day(date: datetime) -> bool:
        """
        Check if a date is a business day.
        
        Args:
            date: Date to check
            
        Returns:
            True if date is a business day, False otherwise
        """
        return date.weekday() < 5  # 0-4 are Monday-Friday 