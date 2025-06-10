import numpy as np
from calibration.synthetic_calibrator import SyntheticCalibrator, ModelParameters


def test_parameter_generation():
    """Test parameter generation."""
    # Setup
    n_assets = 5
    seed = 42
    calibrator = SyntheticCalibrator(seed=seed)

    # Generate parameters
    params = calibrator.generate_parameters(n_assets)

    # Check results
    assert isinstance(params, ModelParameters)
    assert params.volatility.shape == (n_assets,)
    assert params.drift.shape == (n_assets,)
    assert isinstance(params.risk_free_rate, float)

    # Check parameter ranges
    assert np.all(params.volatility >= 0.1) and np.all(params.volatility <= 0.5)
    assert np.all(params.drift >= -0.1) and np.all(params.drift <= 0.2)
    assert 0.01 <= params.risk_free_rate <= 0.03


def test_start_price_generation():
    """Test starting price generation."""
    # Setup
    n_assets = 5
    seed = 42
    calibrator = SyntheticCalibrator(seed=seed)

    # Generate prices
    prices = calibrator.generate_start_prices(n_assets)

    # Check results
    assert prices.shape == (n_assets,)
    assert np.all(prices > 0)  # Prices should be positive

    # Check reproducibility
    other_calibrator = SyntheticCalibrator(seed=seed)
    prices2 = other_calibrator.generate_start_prices(n_assets)
    assert np.allclose(prices, prices2)


def test_heston_parameter_generation():
    """Test Heston parameter generation."""
    # Setup
    n_assets = 5
    seed = 42
    calibrator = SyntheticCalibrator(seed=seed)

    # Generate parameters
    params = calibrator.generate_heston_parameters(n_assets)

    # Check results
    assert isinstance(params, dict)
    assert all(key in params for key in ["theta", "kappa", "eta", "rho"])
    assert all(param.shape == (n_assets,) for param in params.values())

    # Check parameter ranges
    assert np.all(params["theta"] > 0)  # Long-term variance
    assert np.all(params["kappa"] >= 0.5) and np.all(
        params["kappa"] <= 2.0
    )  # Mean reversion
    assert np.all(params["eta"] > 0)  # Vol of vol
    assert np.all(params["rho"] >= -0.7) and np.all(
        params["rho"] <= -0.3
    )  # Correlation
