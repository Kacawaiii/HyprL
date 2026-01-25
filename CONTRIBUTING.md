# Contributing to HyprL

Thank you for your interest in contributing to HyprL!

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- (Optional) Rust for native engine

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Kacawaiii/HyprL.git
cd HyprL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev,ml]"

# (Optional) Build Rust engine
cd native/hyprl_supercalc
pip install maturin
maturin develop --release
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/hyprl --cov-report=html

# Run specific test file
pytest tests/broker/test_alpaca_broker_mock.py -v
```

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/Kacawaiii/HyprL/issues)
2. Create a new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment (OS, Python version)

### Suggesting Features

1. Open an issue with `[Feature]` prefix
2. Describe the use case
3. Explain expected behavior

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Format code: `black src/ tests/`
6. Commit with clear message
7. Push and open a PR

### Code Style

- Use [Black](https://github.com/psf/black) for formatting
- Follow PEP 8 guidelines
- Add type hints where possible
- Write docstrings for public functions

### Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring

Example:
```
feat: add Stochastic RSI indicator

- Implement Stochastic RSI calculation
- Add tests for edge cases
- Update documentation
```

## Project Structure

```
src/hyprl/
├── backtest/      # Backtesting engine
├── broker/        # Broker integrations
├── features/      # Feature engineering
├── indicators/    # Technical indicators
├── model/         # ML models
├── risk/          # Risk management
└── strategy/      # Trading strategies
```

## Areas to Contribute

- **Indicators**: Add new technical indicators
- **Brokers**: Support for new brokers (Interactive Brokers, etc.)
- **Features**: New feature engineering methods
- **Documentation**: Improve docs and examples
- **Tests**: Increase test coverage
- **Performance**: Optimize Python or Rust code

## Questions?

Open an issue or discussion for any questions.

---

Thank you for contributing!
