from datetime import datetime
from core.date_generator import BusinessDayGenerator


def test_business_day_generator():
    """Test business day generation."""
    # Setup
    start_date = datetime(2023, 1, 1)  # Sunday
    end_date = datetime(2023, 1, 7)  # Saturday
    date_gen = BusinessDayGenerator()

    # Generate dates
    dates = date_gen.generate_dates(start_date, end_date)

    # Check results
    assert len(dates) == 5  # Monday to Friday
    assert dates[0] == datetime(2023, 1, 2)  # Monday
    assert dates[-1] == datetime(2023, 1, 6)  # Friday

    # Check all dates are business days
    for date in dates:
        assert date.weekday() < 5  # Monday to Friday


def test_holiday_exclusion():
    """Test holiday exclusion."""
    # Setup
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 7)
    holidays = [datetime(2023, 1, 3)]  # Wednesday
    date_gen = BusinessDayGenerator(holidays=holidays)

    # Generate dates
    dates = date_gen.generate_dates(start_date, end_date)

    # Check results
    assert len(dates) == 4  # Monday, Tuesday, Thursday, Friday
    assert datetime(2023, 1, 3) not in dates  # Holiday should be excluded


def test_is_business_day():
    """Test business day checking."""
    # Test business days
    assert BusinessDayGenerator.is_business_day(datetime(2023, 1, 2))  # Monday
    assert BusinessDayGenerator.is_business_day(datetime(2023, 1, 3))  # Tuesday
    assert BusinessDayGenerator.is_business_day(datetime(2023, 1, 4))  # Wednesday
    assert BusinessDayGenerator.is_business_day(datetime(2023, 1, 5))  # Thursday
    assert BusinessDayGenerator.is_business_day(datetime(2023, 1, 6))  # Friday

    # Test weekends
    assert not BusinessDayGenerator.is_business_day(datetime(2023, 1, 1))  # Sunday
    assert not BusinessDayGenerator.is_business_day(datetime(2023, 1, 7))  # Saturday
