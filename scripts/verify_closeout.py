"""Independent daily identities and preserved versus offline-replayed numbers."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from quant_backtest.data import load_price_snapshot
from quant_backtest.data_quality import price_returns


def verify(source: Path, replay: Path) -> dict:
    daily = pd.read_csv(source / 'daily_audit.csv', parse_dates=['Date'])
    prices = load_price_snapshot(source / 'data_snapshot.csv')
    dates = pd.DatetimeIndex(daily[daily.policy == daily.policy.iloc[0]].Date.unique()).sort_values()
    returns = price_returns(prices.loc[dates])
    maxima = {'gross_return_identity': 0., 'net_return_identity': 0.,
              'nav_identity': 0., 'fee_turnover_identity': 0., 'exposure_identity': 0.}
    for (_, cost, _), curve in daily.groupby(['policy', 'cost_bps', 'lag']):
        curve = curve.set_index('Date')
        risky = pd.Series(0., index=curve.index)
        exposure = pd.Series(0., index=curve.index)
        for ticker in prices:
            weights = curve[f'earning_{ticker}']
            risky += weights * returns[ticker]
            exposure += weights
        gross = risky + (1 - exposure) * curve.return_reference
        prior_nav = curve.strategy_equity.shift(1, fill_value=10000.)
        residuals = {'gross_return_identity': curve.gross_return - gross,
                     'net_return_identity': curve.strategy_return - (curve.gross_return - curve.transaction_cost),
                     'nav_identity': curve.strategy_equity - prior_nav * (1 + curve.strategy_return),
                     'fee_turnover_identity': curve.transaction_cost - cost / 10000 * curve.turnover,
                     'exposure_identity': curve.earning_exposure - exposure}
        for name, residual in residuals.items():
            if not np.isfinite(residual).all():
                raise AssertionError(f'Incomplete daily audit: {name}')
            maximum = float(residual.abs().max())
            maxima[name] = max(maxima[name], maximum)
            assert maximum < (1e-8 if name == 'nav_identity' else 1e-12), (name, maximum)
    for filename in ['comparison.csv', 'sensitivity.csv']:
        original = pd.read_csv(source / filename)
        reproduced = pd.read_csv(replay / filename)
        assert list(original.columns) == list(reproduced.columns)
        for column in original:
            if pd.api.types.is_numeric_dtype(original[column]):
                np.testing.assert_allclose(original[column], reproduced[column], rtol=1e-12, atol=1e-10, equal_nan=True)
            else:
                pd.testing.assert_series_equal(original[column], reproduced[column])
    return {'daily_rows_checked': len(daily), 'scenarios': daily.groupby(['policy', 'cost_bps', 'lag']).ngroups,
            'maximum_absolute_residuals': maxima, 'offline_tables_match': True,
            'numeric_tolerance': {'rtol': 1e-12, 'atol': 1e-10}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    result = verify(Path(args.source), Path(args.replay))
    Path(args.receipt).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
