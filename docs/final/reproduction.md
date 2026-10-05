# Reproducing the closeout

## What is reproducible

Two frozen full searches and two deterministic accounting replays are distinguished:

1. Full search: the unchanged v6 grids, nested annual selection and statistics on an explicit snapshot.
2. Offline replay: already selected and hashed target CSVs, the same snapshot/config and the common engine under all 72 cost/lag scenarios. It does not repeat selection or bootstrap.

Both need external market data. The exact raw snapshots and detailed target CSVs are retained locally in ignored `outputs_closeout/`. This Git repository contains compact derived evidence, not the complete external data bundle. A fresh clone without the requisite inputs cannot reproduce vintage numbers exactly. A new vendor download is a separate dataset; never disable the hash checks to make it pass.

## Recommended path (PowerShell, from repository root)

Use Python **3.14.0**, the interpreter used for the verified release. The package advertises >=3.11; exact dependency resolution and numbers on other interpreters/platforms were not certified by this release.

```powershell
python -m venv .venv-closeout
.\.venv-closeout\Scripts\python.exe -X utf8 -m pip --disable-pip-version-check install --no-cache-dir -r requirements-lock.txt
.\.venv-closeout\Scripts\python.exe -X utf8 -m pip --disable-pip-version-check install --no-deps --no-build-isolation -e .
New-Item -ItemType Directory -Force .pytest_tmp | Out-Null
.\.venv-closeout\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider --basetemp=.pytest_tmp/reproduction
```

UTF-8 avoids Windows console encoding failures on Cyrillic paths. `--no-cache-dir` avoids dependence on a corrupt global installer cache. Pytest's cache is disabled; its temporary parent must exist.

With the locally preserved historical bundle:

```powershell
.\.venv-closeout\Scripts\python.exe -X utf8 scripts/finalize_research.py `
  --config configs/closeout_historical.yaml `
  --snapshot outputs_closeout/historical/data_snapshot.csv `
  --expected-hash ffdb3eb9ed8108e5995e62b471ceab40c61ea3c457102756242fad2b9cbb96cd `
  --replay outputs_closeout/historical `
  --output outputs_closeout/replay_historical_new
```

For the updated bundle:

```powershell
.\.venv-closeout\Scripts\python.exe -X utf8 scripts/finalize_research.py `
  --config configs/closeout_updated.yaml `
  --snapshot outputs_closeout/updated/data_snapshot.csv `
  --expected-hash 87b0315bc31e916696d6261c6db8b66088653bc21343edb0e3223c87b318fea8 `
  --replay outputs_closeout/updated `
  --output outputs_closeout/replay_updated_new
```

Output directories must be new and empty. Compare `sensitivity.csv` and `comparison.csv` with their preserved counterparts. Numeric equality is checked to `rtol=1e-12`, `atol=1e-10`; CSV text, timestamps and platform details need not be byte-identical.

The data content hash canonicalizes the sorted frame to eight decimal places. Target CSVs use exact file hashes. Configurations record both original file-byte hashes and canonical YAML meaning hashes, so changing Windows/Linux line endings does not change the settings. Older target receipts gained the semantic hash as a metadata-only portability upgrade; target files, data and simulation results were untouched.

## Repeating the full search

Use the same command without `--replay` and a new output directory. This performs selection, nested windows, statistical diagnostics and then comparable accounting. It takes substantially longer than replay. Historical training ends 2020-12-31; the search grids are unchanged from `research_v6.yaml`, except explicit cutoff and eight workers. Initial cash/warmed indicators and the same 10 bps engine are used.

If the input extends outside the configured research period, the full runner uses the effective `result.prices` consistently for warmup, target schedules, saved prices and comparisons. New target receipts preserve `input_data_sha256` separately from the effective `data_sha256`. When replaying the saved bundle, pass its effective hash, not the hash of the wider original file. The two published datasets already fit their configured periods, so this correction leaves their numbers unchanged.

The original vintage input is still available locally at `C:\Quantitive\model 1 aapl\outputs_v6_preview\data_snapshot.csv`; the historical run made its own copy. It was not overwritten. The historical full rerun used financial/selection code from A1, on a working tree based on `61466e9`; the new runner was committed as `1605584` after this run. The updated full run started from that commit. Subsequent receipt portability additions do not change decisions or NAV.

## Acquiring a different snapshot

```powershell
.\.venv-closeout\Scripts\python.exe -X utf8 scripts/download_closeout_data.py `
  --config configs/closeout_updated.yaml `
  --output outputs_closeout/inputs/another_updated.csv
```

The config requests an **exclusive** vendor end of 2026-10-06; the preserved download ends at the completed 2026-10-05 close. Acquisition writes a receipt with retrieval time, actual coverage and content hash. Future vendor revisions may change historical adjusted prices. Use the new receipt's hash for a separate full run, audit differences, and label it as a different dataset. Do not reuse frozen target receipts with a different data hash.

## Rebuilding derived evidence

```powershell
.\.venv-closeout\Scripts\python.exe -X utf8 scripts/build_closeout_report.py `
  --historical outputs_closeout/historical `
  --updated outputs_closeout/updated `
  --archive 'C:\Quantitive\model 1 aapl\outputs_v6_preview' `
  --destination docs/final
```

This rebuilds compact tables, fixed-return bootstrap diagnostics, audits and figures. It does not edit the narrative report or choose a new strategy. The archive argument supplies original v06 and nested summary tables; their manifest is preserved in `docs/final/archive_manifest.json`.

## Data use

Raw prices are excluded from the current tracked release tree. Legacy generated exports remain locally and in earlier Git history; that history was not rewritten. Access and redistribution require rights to the vendor data independently of the Python library's license. The [yfinance README](https://github.com/ranaroussi/yfinance) describes research/education use, personal-use limitations and links to Yahoo terms. No redistribution permission is claimed here.

This is a daily-close educational simulator: cash is synthetic BIL remuneration, market impact and taxes are absent, and there is no terminal liquidation. Hashes prove input identity within the declared precision; they do not certify market truth or point-in-time availability.

