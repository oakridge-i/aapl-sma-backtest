"""Parallel evaluation of strategy grids.

Sweeps call :func:`evaluate_grid` with a list of (params, variant, cost_bps)
jobs and a shared context (prices, ticker, capital, ...). With ``workers <= 0``
in serial mode the jobs run in-process; otherwise a ``ProcessPoolExecutor``
is used, with the shared context shipped once per worker via the pool
initializer instead of once per job. Result order always matches job order,
so parallel runs are bit-for-bit identical to serial runs.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .research_config import AUTO_WORKERS


@dataclass(frozen=True)
class EvaluationJob:
    params: Any
    variant: str
    cost_bps: float
    label: str


@dataclass(frozen=True)
class EvaluationContext:
    prices: pd.DataFrame
    ticker: str
    initial_capital: float
    market_regime_short_window: int
    market_regime_long_window: int
    cash_proxy: str | None


_WORKER_CONTEXT: EvaluationContext | None = None


def resolve_workers(configured: int, n_jobs: int) -> int:
    """Translate the configured worker count into an effective one.

    Small grids are not worth the process startup cost, so parallelism only
    kicks in past a minimum job count.
    """
    if configured == 0 or n_jobs < 32:
        return 0
    workers = os.cpu_count() or 1 if configured == AUTO_WORKERS else configured
    return max(0, min(workers, n_jobs))


def evaluate_grid(
    jobs: list[EvaluationJob],
    context: EvaluationContext,
    workers: int,
    collect_returns: bool = False,
) -> tuple[list[dict[str, Any]], pd.DataFrame | None]:
    """Evaluate jobs and return (metric rows, optional daily-return matrix).

    The return matrix has one column per job (named ``job_0``, ``job_1``, ...)
    holding the strategy's daily net returns; it feeds PBO analysis.
    """
    effective_workers = resolve_workers(workers, len(jobs))
    if effective_workers <= 1:
        outputs = [_evaluate_job(job, context, collect_returns) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_init_worker,
            initargs=(context,),
        ) as pool:
            outputs = list(pool.map(_evaluate_job_in_worker, jobs, [collect_returns] * len(jobs), chunksize=8))

    rows = [output[0] for output in outputs]
    returns_matrix = None
    if collect_returns:
        returns_matrix = pd.DataFrame(
            {f"job_{idx}": output[1] for idx, output in enumerate(outputs)}
        )
    return rows, returns_matrix


def _init_worker(context: EvaluationContext) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _evaluate_job_in_worker(job: EvaluationJob, collect_returns: bool) -> tuple[dict[str, Any], pd.Series | None]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context is not initialized.")
    return _evaluate_job(job, _WORKER_CONTEXT, collect_returns)


def _evaluate_job(
    job: EvaluationJob,
    context: EvaluationContext,
    collect_returns: bool,
) -> tuple[dict[str, Any], pd.Series | None]:
    from .evaluation import evaluate_strategy

    result = evaluate_strategy(
        prices=context.prices,
        ticker=context.ticker,
        params=job.params,
        variant=job.variant,
        cost_bps=job.cost_bps,
        initial_capital=context.initial_capital,
        label=job.label,
        market_regime_short_window=context.market_regime_short_window,
        market_regime_long_window=context.market_regime_long_window,
        cash_proxy=context.cash_proxy,
    )
    returns = result["curve"]["strategy_return"] if collect_returns else None
    return result["row"], returns
