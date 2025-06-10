# Market Simulator

A maintainable multi-asset market simulator that generates daily closing prices with realistic sector and geography-based correlations.

## Features

- Modular architecture for easy extension
- Support for multiple market models:
  - Black-Scholes (Geometric Brownian Motion)
  - Heston (Stochastic Volatility) - Coming soon
- Realistic sector and geography-based correlations
- Configurable parameters and output formats
- Reproducible results via seed control
- Business day generation (weekends excluded)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd market-simulator
```

2. Install dependencies:
```bash
make install
```

## Usage

Run the simulator with default settings:
```bash
python main.py
```

Customize the simulation:
```bash
python main.py --assets 20 --start-date 2023-01-01 --end-date 2023-12-31 --model black_scholes --output csv --seed 42
```

### Command Line Options

- `--assets`: Number of assets to simulate (default: 10)
- `--start-date`: Start date in YYYY-MM-DD format (default: 2023-01-01)
- `--end-date`: End date in YYYY-MM-DD format (default: 2023-12-31)
- `--model`: Market model to use (choices: black_scholes, heston) (default: black_scholes)
- `--output`: Output format (choices: csv, memory) (default: csv)
- `--seed`: Random seed for reproducibility
- `--corr-matrix-path`: Optional path to save correlation matrix

## Project Structure

```
market-simulator/
├── models/
│   ├── base.py
│   ├── black_scholes.py
│   └── heston.py
├── calibration/
│   ├── correlation_engine.py
│   └── synthetic_calibrator.py
├── core/
│   ├── asset.py
│   └── date_generator.py
├── outputs/
│   ├── base_handler.py
│   ├── csv_handler.py
│   └── memory_handler.py
├── tests/
│   ├── test_models.py
│   └── test_correlations.py
├── main.py
├── requirements.txt
├── Makefile
└── README.md
```

## Development

### Running Tests

```bash
make test
```

### Code Formatting

```bash
make format
```

### Type Checking

```bash
make lint
```

## Future Extensions

- Heston model implementation
- Additional market models
- Database output handlers
- Real market data calibration
- Volatility surface generation
- Jump diffusion processes

## License

MIT License 