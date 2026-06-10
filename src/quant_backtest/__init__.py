"""Simple SMA crossover backtest package."""

from .backtest import BacktestConfig, BacktestResult, run_sma_backtest
from .experiments import ResearchConfig, ResearchResult, run_research

__version__ = "0.5.0"

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "ResearchConfig",
    "ResearchResult",
    "__version__",
    "run_research",
    "run_sma_backtest",
]
