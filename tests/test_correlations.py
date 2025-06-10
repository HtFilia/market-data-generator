import pytest
import numpy as np
from calibration.correlation_engine import CorrelationEngine, Asset


def test_correlation_engine_initialization():
    """Test correlation engine initialization."""
    # Test default weights
    engine = CorrelationEngine()
    assert engine.sector_weight == 0.6
    assert engine.geography_weight == 0.4
    assert engine.noise_level == 0.05

    # Test custom weights
    engine = CorrelationEngine(sector_weight=0.7, geography_weight=0.3, noise_level=0.1)
    assert engine.sector_weight == 0.7
    assert engine.geography_weight == 0.3
    assert engine.noise_level == 0.1

    # Test invalid weights
    with pytest.raises(ValueError):
        CorrelationEngine(sector_weight=0.7, geography_weight=0.4)


def test_correlation_matrix_creation():
    """Test correlation matrix creation."""
    engine = CorrelationEngine()

    # Create test assets
    assets = [
        Asset(id="TECH1", sector="Technology", geography="North_America"),
        Asset(id="TECH2", sector="Technology", geography="Europe"),
        Asset(id="FIN1", sector="Financial", geography="North_America"),
    ]

    # Generate correlation matrix
    corr_matrix = engine.create_matrix(assets, seed=42)

    # Check matrix properties
    assert corr_matrix.shape == (3, 3)

    # Check that each element is a valid correlation
    assert np.allclose(
        corr_matrix, np.clip(corr_matrix, -1.0, 1.0), rtol=1e-10, atol=1e-10
    )

    # Check symmetry
    assert np.allclose(corr_matrix, corr_matrix.T, rtol=1e-10, atol=1e-10)

    # Check diagonal elements are 1.0
    assert np.allclose(np.diag(corr_matrix), 1.0, rtol=1e-10, atol=1e-10)

    # Check positive definiteness
    eigenvalues = np.linalg.eigvals(corr_matrix)
    assert np.allclose(
        eigenvalues, np.clip(eigenvalues, 0, None), rtol=1e-10, atol=1e-10
    )

    # Check specific correlations
    assert np.allclose(
        corr_matrix[0, 1], 0.7074540118847361, rtol=1e-10, atol=1e-10
    )  # Exact value with seed=42
    assert np.allclose(
        corr_matrix[0, 2], 0.6650714306409915, rtol=1e-10, atol=1e-10
    )  # Exact value with seed=42
    assert np.allclose(
        corr_matrix[1, 2], 0.5631993941811407, rtol=1e-10, atol=1e-10
    )  # Exact value with seed=42


def test_base_correlation_calculation():
    """Test base correlation calculation between assets."""
    engine = CorrelationEngine()

    # Test same sector, different geography
    asset1 = Asset(id="TECH1", sector="Technology", geography="Europe")
    asset2 = Asset(id="TECH2", sector="Technology", geography="North_America")
    corr = engine._get_base_correlation(asset1, asset2)
    assert np.isclose(
        corr, 0.72, rtol=1e-10, atol=1e-10
    )  # 0.72 = 0.6 * 0.8 + 0.4 * 0.6

    # Test different sector, same geography
    asset1 = Asset(id="TECH1", sector="Technology", geography="North_America")
    asset2 = Asset(id="FIN1", sector="Financial", geography="North_America")
    corr = engine._get_base_correlation(asset1, asset2)
    assert np.isclose(
        corr, 0.62, rtol=1e-10, atol=1e-10
    )  # 0.62 = 0.6 * 0.5 + 0.4 * 0.8

    # Test different sector, different geography
    asset1 = Asset(id="TECH1", sector="Technology", geography="North_America")
    asset2 = Asset(id="FIN1", sector="Financial", geography="Asia")
    corr = engine._get_base_correlation(asset1, asset2)
    assert np.isclose(corr, 0.5, rtol=1e-10, atol=1e-10)  # 0.5 = 0.6 * 0.5 + 0.4 * 0.5
