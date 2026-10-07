# Boosted Stumps as Credit Scorecards Without Logistic Regression

**Coefficient-Level Isomorphism and Uncertainty Quantification**

Replication code for the paper of the same title. This repository is released
**anonymously** for double-blind peer review; author and affiliation details are
omitted by design.

---

## What this repository shows

A logistic-regression WoE scorecard is valued for two things: (1) additive,
per-bin **coefficient interpretation**, and (2) a **standard error / confidence
interval** on each coefficient. This code tests, on public credit data, whether
**boosted decision stumps** — fit *without any logistic-regression step* —
reproduce both pillars.

The central empirical claims, all reproducible here:

1. **Estimation (isomorphism).** On a fixed WoE bin grid, a saturated univariate
   stump-boosting model's cumulative leaf curve `s(b)` equals the bin's WoE, and
   the logistic coefficient `β` equals the slope of `s(·)` against WoE. Measured
   `β_eff ≈ 1.00` across **all three datasets** and across numeric, nominal, and
   informative-missing bins.
2. **Inference (SE alignment).** With a fair shared-coordinate bootstrap, the
   stump bin-contribution SE is statistically indistinguishable from the logistic
   one on isomorphic features; a missing bin loosens the alignment (a consistent
   boundary condition).
3. **Inference (coverage).** Split-conformal prediction sets on the predicted
   default probability reach nominal coverage for the stump model just as for the
   logistic model (marginal and class-conditional / Mondrian).
4. **Performance.** Discrimination parity is dataset-regime dependent: the stump
   model is marginally better on large numeric data (GMSC) and the logistic model
   marginally better on small/nominal (German) and balanced informative-missing
   (HELOC) data — superiority is *not* assumed, it is tested per dataset.

See `results/` for the exact tables and figures behind each claim.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/                     # inputs (not redistributed — see data/README.md)
│   └── README.md
├── notebooks/                # 7 single-cell notebooks, run in order
│   ├── 01_eda_preprocess.ipynb
│   ├── 02_logit_woe_vs_stump.ipynb
│   ├── 03_inference_se_alignment.ipynb
│   ├── 04_performance_benchmark.ipynb
│   ├── 05_german_replication.ipynb
│   ├── 06_heloc_missing.ipynb
│   └── 07_conformal_coverage.ipynb
└── results/
    ├── tables/               # CSV outputs (one or more per notebook)
    └── figures/              # greyscale PNG + PDF, 600 dpi
```

Each notebook is **self-contained in a single cell** whose first line is the
notebook filename, and writes its outputs to `results/`. Figures are greyscale,
600 dpi, saved as both PNG and PDF.

---

## Notebooks

| # | Notebook | Dataset | Role | Key output |
| --- | --- | --- | --- | --- |
| 01 | `01_eda_preprocess` | GMSC | Cleaning, monotonic WoE binning (credit convention `WoE = ln(bad%/good%)`), IV-based feature selection (IV ≥ 0.02) | `gmsc_woe.parquet`, `woe_maps.json`, IV table, WoE curves |
| 02 | `02_logit_woe_vs_stump` | GMSC | **Isomorphism.** `β_eff`, `s(b)` vs WoE in an ideal regime (stumps on the WoE grid) and a reality regime (stumps on raw features) | `02_isomorphism_*`, scatter / curve figures |
| 03 | `03_inference_se_alignment` | GMSC | **SE alignment.** Fair-coordinate bootstrap (N=500); log-odds-limit stump vs real-GB cross-check (N=100); missing-bin sub-analysis | `03_se_alignment_*`, B-vs-A agreement figure |
| 04 | `04_performance_benchmark` | GMSC | **Performance.** 50 repeated splits; AUC / KS / Gini / PSI; DeLong paired test; XGBoost ceiling | `04_perf_*`, `04_delong_per_split` |
| 05 | `05_german_replication` | German | Generalization to a **nominal-heavy** dataset; WoE for unordered categories | `05_german_*`, s-vs-WoE / β figures |
| 06 | `06_heloc_missing` | HELOC | Generalization with **informative missing codes** (-7/-8/-9) as their own WoE bins | `06_heloc_*`, code-bin table |
| 07 | `07_conformal_coverage` | GMSC | **Conformal coverage.** Split conformal (LAC + Mondrian) prediction sets; borderline-stratum coverage; PD interval width | `07_coverage`, `07_conditional_coverage` |

Notebooks 02–04 and 07 depend on the shared assets produced by 01. Notebooks 05
and 06 are standalone (they read their raw dataset directly).

---

## Reproduce

```bash
# 1. environment (Python 3.13)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. place the three raw datasets in data/  (see data/README.md)

# 3. run the notebooks in order (headless)
cd notebooks
for nb in 01_eda_preprocess 02_logit_woe_vs_stump 03_inference_se_alignment \
          04_performance_benchmark 05_german_replication 06_heloc_missing \
          07_conformal_coverage; do
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" \
    --ExecutePreprocessor.timeout=5400
done
```

All randomness is seeded (`SEED = 42`). Approximate run times on a modern
multi-core CPU: notebooks 01, 02, 05, 06 a few minutes each; 04 and 07 roughly
5–10 minutes; 03 around 15–20 minutes (its N=100 real-GB cross-check dominates).

### Key configuration knobs

- `IV_THRESH = 0.02` — weak-predictor drop rule (notebooks 01, 05, 06).
- `N_BOOT = 500` — bootstrap replicates for SE (notebooks 03, 05, 06).
- `N_SPLIT` / `N_REP` — repeated train/test splits for performance and coverage.
- `ALPHA = 0.10` — conformal miscoverage (nominal 90% coverage), notebook 07.

The log-odds-limit stump used in the inference notebooks is the saturated
univariate stump-boosting limit (per-bin log-odds deviation); notebook 03
verifies it matches a real `GradientBoostingClassifier(max_depth=1)` refit to
within a few percent on the SE scale, so it is used as a fast, exact stand-in.

---

## Method notes

- **WoE sign convention.** `WoE(bin) = ln( bad% / good% )`, so higher risk → higher
  WoE and the logistic coefficient on WoE is positive. This matches the paper.
- **Missing / sentinel handling.** GMSC past-due sentinels (96/98) and `age = 0`
  are set to NaN and absorbed as WoE bins; a sparse missing bin is neutralized
  (WoE = 0). HELOC's `-7/-8/-9` are kept as *informative* bins (sparse ones merged
  into a single `code_other` bin).
- **Fair shared coordinate (SE).** Each bootstrap replicate recomputes WoE on a
  fixed bin grid and refits *both* models, so logistic and stump contributions
  absorb the same structural + sampling noise — an apples-to-apples SE comparison.

---

## License

Code is released for anonymous peer review. A permissive open-source license will
be attached in the de-anonymized release. The datasets are the property of their
respective sources (see `data/README.md`) and are not redistributed here.
