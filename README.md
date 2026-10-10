# Boosted Stumps as Credit Scorecards Without Logistic Regression

**Coefficient-Level Correspondence and Uncertainty Quantification**

Replication code for the paper of the same title. This repository is released
**anonymously** for double-blind peer review; author and affiliation details are
omitted by design.

> **Revision note.** The notebooks prefixed `R1`–`R6` implement the revised
> analysis (rank-one correspondence; estimand-matched SE; conformal efficiency;
> Venn–Abers PD intervals; out-of-time validation; all binning and feature
> selection performed **inside each training fold**). The original `01`–`07`
> notebooks are kept for provenance; where the two disagree, the `R*` notebooks
> are authoritative.

---

## What this repository shows

A logistic-regression WoE scorecard is valued for two things: (1) additive,
per-bin **coefficient interpretation**, and (2) a **standard error** on each
coefficient. This code tests, on three public credit datasets, whether **boosted
decision stumps** — fit *without any logistic-regression step* — reproduce both,
and states the result as a testable **rank-one correspondence**.

The central claims, all reproducible here:

1. **A rank-one correspondence, not an identity.** On a fixed WoE bin grid,
   componentwise depth-one boosting is coordinate descent on a dummy
   (bin-indicator) additive logistic model, whose saturated limit is the
   dummy-logistic MLE; logistic-WoE is its rank-one sub-model
   `γ_jb = β_j · WoE_j(b)`. The free per-bin coefficients `γ̂_j` lie close to a
   scalar multiple of `WoE_j`, and that multiple — the **projection slope** —
   reproduces the logistic coefficient `β̂_j` (correlation **0.95–0.99** across
   datasets), **not** the value one that a per-feature (naive-Bayes) saturation
   gives. The rank-one restriction is formally rejected in the large GMSC sample
   but not in the small German sample, so correspondence is read through **effect
   size**, not a null test. Componentwise boosting is shown to converge to the
   dummy-logistic MLE.
2. **Estimand-matched standard errors.** By a shared fair-coordinate bootstrap,
   the free-bin and rank-one SEs are of the **same order** on strong features but
   **not strictly equivalent** (a TOST does not certify equality); the apparent
   exact alignment of an earlier version was partly a `1/β̂` artifact.
3. **Conformal efficiency, not marginal coverage.** Marginal coverage is
   automatic for any model; compared by **class-conditional coverage, set-size
   mix, and calibration** (Brier, ECE), the two models are indistinguishable.
4. **Calibrated PD intervals by Venn–Abers.** Binary conformal cannot produce a
   PD interval; a **Venn–Abers** predictor can, and the logistic-free booster
   matches logistic-WoE in interval width (median 0.0035 vs 0.0041) and
   calibration (Brier 0.051, ECE ≈0.011).
5. **Discrimination parity by equivalence testing.** AUC is equivalent within a
   ±0.01 margin (TOST) on all three datasets; moving binning inside the fold
   removes a leakage artifact that had inflated the German gap (0.022 → 0.007).
6. **Out-of-time robustness (4th dataset).** On the Polish bankruptcy family (5
   cohorts, 64 features, 3.9–6.9% bad), within-cohort parity holds and, under a
   **genuine out-of-time shift** (cross-cohort PSI up to 0.46, vs ≈0 for a random
   split), both models lose the same ≈0.065 AUC — the equivalence is not an
   artifact of random resampling.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/                     # inputs (included; see data/README.md)
├── notebooks/
│   ├── 01_eda_preprocess.ipynb          # original submission (provenance)
│   ├── 02_logit_woe_vs_stump.ipynb      #   ...
│   ├── 03_inference_se_alignment.ipynb
│   ├── 04_performance_benchmark.ipynb
│   ├── 05_german_replication.ipynb
│   ├── 06_heloc_missing.ipynb
│   ├── 07_conformal_coverage.ipynb
│   ├── R1_correspondence.ipynb              # REVISION — core correspondence (GMSC)
│   ├── R1b_correspondence_german_heloc.ipynb#   generalization (German, HELOC)
│   ├── R2_se_alignment.ipynb                #   estimand-matched SE + TOST
│   ├── R3_conformal.ipynb                   #   conformal efficiency + calibration
│   ├── R4_discrimination.ipynb             #   three-dataset AUC + equivalence test
│   ├── R5_venn_abers.ipynb                 #   calibrated PD intervals (Venn-Abers)
│   └── R6_out_of_time.ipynb                #   out-of-time / temporal robustness (Polish)
└── results/
    ├── tables/               # CSV outputs (R*_ for the revision)
    └── figures/              # greyscale PNG + PDF, 600 dpi
```

---

## Revision notebooks

| Notebook | Role | Key output |
| --- | --- | --- |
| `R1_correspondence` | GMSC. Free dummy-logit `γ̂` vs rank-one `β̂·WoE`; projection slope vs `β̂`; LR test; componentwise-boosting convergence to the dummy MLE. **In-fold binning.** | `R1_slope_vs_beta`, `R1_boost_convergence`, `R1_lr_test` |
| `R1b_correspondence_german_heloc` | Generalization to nominal-heavy (German) and informative-missing (HELOC). | `R1b_slope_vs_beta_german_heloc` |
| `R2_se_alignment` | Estimand-matched bootstrap SE (free `γ` vs rank-one `β·WoE`), raw and `β̂`-corrected, TOST `[0.8,1.25]`, ratio CI. | `R2_se_ratio_box` |
| `R3_conformal` | Mondrian class-conditional coverage, set-size mix, top-decile cutoff, Brier/ECE. | `R3_conformal_efficiency`, `R3_conformal_summary` |
| `R4_discrimination` | AUC over 50 in-fold splits on all three datasets; DeLong, Wilcoxon, **TOST ±0.01**, XGBoost ceiling. | `R4_discrimination`, `R4_discrimination_summary` |
| `R5_venn_abers` | Inductive Venn–Abers PD intervals (logit-WoE vs stump) on GMSC: interval width + calibration (Brier/ECE). | `R5_va_width_vs_pd`, `R5_va_summary` |
| `R6_out_of_time` | Polish bankruptcy (5 cohorts): within-cohort AUC parity + cross-cohort out-of-time AUC/PSI. | `R6_out_of_time`, `R6_oot_matrix` |

All `R*` notebooks are single-cell, seeded (`SEED=42`), and perform binning and
IV selection inside each training fold. Figures are greyscale, 600 dpi, PNG+PDF.

---

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
for nb in R1_correspondence R1b_correspondence_german_heloc R2_se_alignment \
          R3_conformal R4_discrimination R5_venn_abers R6_out_of_time; do
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" \
    --ExecutePreprocessor.timeout=5400
done
```

Approximate run times (modern multi-core CPU): `R1` ~1 min, `R1b` ~2 min,
`R2` ~2 min, `R3` ~3 min, `R4` ~10 min (its GMSC DeLong is the bottleneck),
`R5` ~2 min, `R6` ~4 min.

### Key configuration

- `IV_THRESH = 0.02` — weak-predictor drop rule (in every fold).
- `N_BOOT = 200` — bootstrap replicates for SE (`R2`).
- `N_SPLIT` — 20 folds for correspondence (`R1`), 50 splits for discrimination (`R4`).
- Equivalence margins: `[0.8, 1.25]` for the SE ratio, `±0.01` for AUC.

---

## Method notes

- **WoE sign convention.** `WoE(b) = ln( bad% / good% )`, higher risk → higher WoE.
- **Free vs rank-one models.** The free model gives each bin its own coefficient
  `γ_jb`; logistic-WoE is the rank-one restriction `γ_jb = β_j·WoE_j(b)`. The
  projection slope of `γ̂_j(·)` onto `WoE_j(·)` is compared with `β̂_j`.
- **Separation-robust SE.** The information matrix diverges under separation in
  extreme credit bins, so SEs are estimated by a shared fair-coordinate bootstrap
  with a lightly penalized dummy fit.
- **PD intervals via Venn–Abers, not binary conformal.** The probability of
  default is unobserved, so binary conformal gives label sets, not PD intervals;
  `R5` fits an inductive Venn–Abers predictor (isotonic, self-contained) on both
  models and compares interval width and calibration.
- **Out-of-time vs random-split PSI.** Random-split PSI ≈ 0 is uninformative;
  `R6` fits on one Polish cohort and deploys on another, so the score-shift PSI is
  materially non-zero and the AUC drop is a real out-of-time measurement.

---

## License

Code is released for anonymous peer review. A permissive open-source license will
be attached in the de-anonymized release. Datasets belong to their respective
sources (see `data/README.md`).
