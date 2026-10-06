"""Acquire a separate vendor snapshot; never overwrite the historical snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from quant_backtest.data import save_price_snapshot
from quant_backtest.research_config import load_research_config
from quant_backtest.research_data import download_research_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise ValueError('Refusing to overwrite an existing snapshot.')
    config = load_research_config(Path(args.config))
    prices = download_research_prices(config)
    digest = save_price_snapshot(prices, path)
    receipt = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
               'source': 'Yahoo Finance via yfinance, Adj Close, auto_adjust=False',
               'requested_start': config.start, 'requested_exclusive_end': config.end,
               'actual_start': str(prices.index[0].date()), 'actual_end': str(prices.index[-1].date()),
               'data_sha256': digest, 'sessions': len(prices), 'tickers': list(prices.columns),
               'point_in_time': False}
    path.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
