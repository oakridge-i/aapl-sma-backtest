from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quant_backtest.experiments import load_research_config, run_research
from quant_backtest.reports import save_research_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trend allocation research framework.")
    parser.add_argument("--config", default="configs/research_v5.yaml", help="Path to research YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--fixture-data", action="store_true", help="Use deterministic synthetic data.")
    parser.add_argument("--no-download", action="store_true", help="Alias for --fixture-data for CI/smoke tests.")
    parser.add_argument(
        "--data-snapshot",
        default=None,
        help="Rerun on a saved data_snapshot.csv instead of downloading fresh data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_research_config(Path(args.config))
    if args.output_dir:
        config = dataclasses.replace(config, output_dir=args.output_dir)

    result = run_research(
        config,
        fixture_data=args.fixture_data or args.no_download,
        snapshot_path=Path(args.data_snapshot) if args.data_snapshot else None,
    )
    output_dir = Path(config.output_dir)
    save_research_outputs(result, output_dir)

    print(f"Wrote research outputs to: {output_dir}")
    if not result.v04_comparison.empty:
        columns = [
            "model",
            "selection_status",
            "variant",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover",
            "upside_capture",
            "downside_capture",
            "capture_spread",
        ]
        print(result.v04_comparison[columns].to_string(index=False))
    else:
        print(result.model_leaderboard.head(10).to_string(index=False))

    if not result.v06_comparison.empty:
        print()
        print("v0.6 candidates on the test period:")
        columns = [
            column
            for column in ["model", "selection_status", "cagr", "sharpe", "max_drawdown", "turnover", "exposure"]
            if column in result.v06_comparison.columns
        ]
        print(result.v06_comparison[columns].to_string(index=False))

    if not result.nested_ensemble_summary.empty:
        print()
        print("Nested ensemble walk-forward (stitched OOS):")
        columns = [
            column
            for column in ["name", "cagr", "sharpe", "max_drawdown", "windows", "windows_beating_benchmark"]
            if column in result.nested_ensemble_summary.columns
        ]
        print(result.nested_ensemble_summary[columns].to_string(index=False))

    if not result.significance_results.empty:
        print()
        print("Significance (test period, selection done on train only):")
        columns = [
            column
            for column in [
                "model",
                "observed_sharpe",
                "sharpe_p05",
                "sharpe_p95",
                "prob_negative_sharpe",
                "deflated_sharpe_prob",
                "permutation_p_value",
            ]
            if column in result.significance_results.columns
        ]
        print(result.significance_results[columns].to_string(index=False))


if __name__ == "__main__":
    main()
