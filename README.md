# Pipelining

Lightweight project that runs an EDA profiling pipeline and an AI "brain" pipeline on a CSV dataset. The main script (`app.py`) loads data, runs profiling, converts profiling results into metadata, and feeds that into the AI pipeline. Outputs are printed to the console.

## Features

- Run full EDA profiling pipeline (EDA/Profiling).
- Generate column-level metadata from EDA results.
- Run AI analysis/prediction pipeline (AI_Brain).
- Simple CLI usage: prints profiling, metadata and AI outputs.

## Repo structure

- `app.py` — main entrypoint.
- `Data/` — CSV datasets (e.g. `ai4i2020.csv`, optional `uci-secom.csv`).
- `EDA/` — profiling pipeline implementation.
  - `Profiling/` — profiling pipeline module (`profiling_pipeline.py`).
- `AI_Brain/` — AI pipeline implementation (`brain_pipeline.py`).
- `README.md` — this file.

## Requirements

- Python 3.8+
- pandas
- other dependencies required by `EDA` and `AI_Brain` modules (add to `requirements.txt` if desired)

Install basic deps:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas
# pip install -r requirements.txt
```

## Usage

From project root:
```bash
python app.py
```

By default `app.py` loads `Data/ai4i2020.csv`. To use the alternative dataset, edit `app.py` to point to `uci-secom.csv` (example commented in file).

Options you may modify in `app.py`:
- `run_eda_pipeline(df, save_plots=False)` — set `save_plots=True` to save EDA plots (if implemented).
- Use returned `eda_results` with `create_metadata_from_eda_results(df, eda_results)` to inspect generated metadata.

## Development notes

- `create_metadata_from_eda_results` in `app.py` demonstrates converting profiling outputs (`column_types`, `missing_report`, `outlier_report`, `skewness_report`, `cardinality_report`) into per-column metadata.
- Ensure `PROJECT_ROOT` and module import paths are correct when running from different working directories (the script appends project folders to `sys.path`).

## Contributing

Open an issue or submit a PR. Keep changes small and document added dependencies in `requirements.txt`.

## License

Add a license file if required.
