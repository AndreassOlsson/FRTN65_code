# Exploration finding: credit-g

Written 2026-10-01 from `make explore` (songtaste's explore functions
with this folder's spec). Figures in `figures/01-*.png`.

- **Balance.** 700 good, 300 bad. The majority-class floor is 0.70
  accuracy, against lab 1's 0.60, so accuracy has less room above it.
- **Numbers on their own say little.** The strongest numeric
  separator is duration (AUC 0.63 towards bad: longer loans default
  more), then age (0.43: younger applicants default more) and the
  amount (0.56). Residence, dependents and existing credits separate
  nothing.
- **The categoricals carry the signal.** Checking account status is
  the single strongest attribute: 49% of applicants with a negative
  balance are bad, 12% of those with no checking account. Credit
  history runs the counter-intuitive way the dataset is known for
  ("critical/other existing credit" is 17% bad, "no credits/all paid"
  63%), savings, property and purpose follow.
- **Rare levels.** Several levels have under 50 rows (`retraining` 9,
  `domestic appliance` 12, `other` 12, `foreign_worker = no` 37). A
  stratified fold of 200 rows may hold one or two of them, so a level
  can be missing from a training part: the one-hot encoder's
  `handle_unknown="ignore"` is what keeps that from failing.
- **Outliers.** `credit_amount` has 24 values beyond 3 IQR (up to
  18424 DM), the same right skew lab 1 saw in duration;
  `num_dependents` is binary in effect (1 or 2), its 155 "far out" rows
  are the 2s. Standard scaling is kept for the same reason as lab 1:
  one shared preprocessor, variants measured as methods.
- **No duplicates**, so the protocol's de-duplication is a no-op here.
- **Cost.** The dataset's documentation (OpenML 31) gives a cost
  matrix: calling a bad applicant good costs 5, calling a good one bad
  costs 1. That shapes the protocol more than anything above.
