# Data

This directory holds the input datasets. The raw files are **not redistributed**
in this repository (to respect each source's terms); download them from the
sources below and place them here with the exact filenames shown. All three are
public credit-scoring benchmarks.

| File to place here | Dataset | Rows | Source |
| --- | --- | --- | --- |
| `cs-training.csv` | Give Me Some Credit (GMSC) | 150,000 | Kaggle: *Give Me Some Credit* competition |
| `german.data` | UCI Statlog German Credit | 1,000 | UCI Machine Learning Repository: *Statlog (German Credit Data)* |
| `heloc.csv` | FICO HELOC | 10,459 | FICO Explainable ML Challenge (`heloc_dataset_v1.csv`, rename to `heloc.csv`) |

## Expected formats

- **`cs-training.csv`** — comma-separated, first (unnamed) column is a row index,
  target column `SeriousDlqin2yrs` (1 = default).
- **`german.data`** — space-separated, no header, 20 attribute columns (coded
  `A11`, `A34`, … for categoricals) plus a final target column (1 = good, 2 = bad).
  The notebook recodes the target to 1 = bad.
- **`heloc.csv`** — comma-separated, header row, target column `RiskPerformance`
  with values `Good`/`Bad`. Special codes `-7`, `-8`, `-9` are informative
  missing indicators (handled as their own WoE bins in notebook 06).

## Derived files (created automatically)

Running `notebooks/01_eda_preprocess.ipynb` writes the shared WoE assets used by
the downstream GMSC notebooks:

- `gmsc_clean.parquet` — cleaned GMSC (sentinels/impossible values set to NaN)
- `gmsc_woe.parquet` — WoE-transformed design matrix + target
- `woe_maps.json` — per-feature bin definitions, WoE values, and selection metadata

These derived files are git-ignored; regenerate them by running notebook 01.
Notebooks 05 and 06 read `german.data` / `heloc.csv` directly and build their
own WoE assets in-memory.
