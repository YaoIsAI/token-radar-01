# Night-Scope Token Launch Radar — Research Note

Model version: radar-6. Snapshot: 758 protocols, as of 2026-11-14. This note ships as
`RESEARCH.md` beside the dashboard; the validation numbers ship as `validation.json`.
Everything here is computed from the snapshot at runtime; nothing is hand-set except the two
constants the data cannot identify (see §3).

## 1. What this radar is

A ranking of pre-token protocols by **implied launch value**: what a token launch could be
worth, if it happens. Every row is

```
implied = round(P_launch x V_conditional)
```

`P_launch` is the probability the protocol launches a token; `V_conditional` is the market cap
it would carry if it did. The board orders rows by `implied` and nothing else. No protocol name
is special-cased; the 40 watchlist protocols are scored by the same function as everyone else
— guaranteed a row, not a rank.

## 2. The model

Both halves are ridge regressions fit on the snapshot's own live-token protocols — the 129
protocols that already have a token, of which 128 have a positive market cap. The ridge strengths
are chosen by five-fold cross-validation (see §3), not hand-set.

**Value — ridge least squares on log(market cap), two features only:**

```
log V_conditional = b0 + b1 x log(TVL) + b2 x category value prior
                  - 0.05 if growth unreported
```

Fitted at runtime: intercept 7.95, log(TVL) 0.66, value prior 2.13. The fitted value is
exponentiated and clamped to e^±30 so no row explodes. **Growth, chains and the age-unknown
flag are not value features any more.** The bootstrap (B = 200) says only log(TVL) (0.45–0.88)
and the value prior (0.47–3.95) are consistently positive; the other three ranges cross zero.
Dropping them improves every holdout — time-split Spearman 0.24 → 0.25, random 0.41 → 0.42,
expanding 0.29 → 0.29, out-of-fold 0.29 → 0.33 — and the median log error falls on all four.
Age was already out of this half since radar-5, where it blew up to 7.51 on the old-only window
and collapsed the ranking (Spearman 0.04).

**Launch — ridge logistic regression on launched-or-not**, fit on all 758 protocols:

```
P_launch = sigmoid(c0 + c1 x age + c2 x growth + c3 x chains
                 + c4 x category launch prior + c5 if age unknown
                 - 0.05 if growth unreported - delay past 3 years)
```

Fitted at runtime: intercept −6.66, age +2.29, growth −0.90, chains +1.12, launch prior +13.09,
age unknown +2.53. The delay term is structural, not fitted: a protocol listed more than 3
years ago with no token is increasingly unlikely to launch now, capped at 0.15 logit.
**The board's P_launch is cross-fitted**: each row is scored by one of five fold models trained
without that row (fold by a hash of the slug), each fold with its own category launch priors, so
no row's own label enters its own probability. The full fit above is the reference; the five fold
fits are in `validation.json`.

**Features, exactly:** **log(TVL)** — natural log of total value locked, floor $1, scored in
logs, never linearly, because market caps span orders of magnitude. **Category priors** — each
category's launch rate and median log(market cap / TVL) among live protocols, shrunk toward the
global rates with strength 10. **Age** — `1 - exp(-age_days / 365)`, saturating; missing for 113
of 758 protocols including 34 live ones; launch only. **Growth** — 7-day TVL change,
tanh-clamped at ±30% and scaled by 20; launch only since radar-6. 4 protocols have no 7-day
change; all pre-token. **Chains** — `1 - exp(-log1p(chain count) / log 9)`, saturating; launch
only since radar-6.

## 3. How the coefficients were fitted

`main.ts` fits both regressions from the snapshot at runtime, deterministically, in pure
TypeScript — ridge least squares by Gaussian elimination for value, ridge IRLS (max 25
iterations) for launch, both on standardized features. No randomness except the bootstrap, which
uses a fixed seed (20261117). No external packages. The fitted coefficients are written into
`data.js`, so the dashboard recomputes every row and checks the board's figure to the dollar.

**Ridge strengths, chosen by cross-validation.** Five folds by a hash of the slug, grid
0.03–30, scored by pooled Spearman (value) and pooled AUC (launch), with the category priors
recalibrated per fold. The profiles are flat — the value profile moves 0.289 to 0.292 across the
whole grid, far inside the sampling noise on 128 rows — so the argmax would be a coin flip (on
this snapshot it lands on the grid's edge, λ = 30, ahead by 0.003). The rule is therefore the
smallest λ within 0.01 of the best: value λ = 0.03 (CV Spearman 0.33), launch λ = 0.03
(CV AUC 0.67).

**The priors must be recalibrated per fold, or the cross-validation scores the leak.** A
held-out row's own market cap enters its category's value prior in the full calibration, and its
own label enters its launch prior. With full priors the cross-validation reads 0.56 (value) and
0.75 (launch AUC); with per-fold priors it reads 0.29 and 0.67 in radar-5 — the honest numbers.
The board itself never had this leak: its rows are pre-token, and its launch probabilities are
cross-fitted.

**What else was tried in round 6, and why it is not in the model.** Every variant was
measured on the same four honest tests, with per-fold recalibrated priors throughout.
Top-weighted fitting (rows weighted 0.5–1.5x by realized value): time-split Spearman 0.30, but
out-of-fold and expanding flat and the time-split median error up — a small-test gain, not a
model gain. Interaction terms (log TVL × growth, log TVL², growth², log TVL × age): time-split
0.13, out-of-fold 0.21. Huber fitting (δ = 0.5–2) and trimming the worst 10% of residuals: flat
or worse. Launch interactions and launch-prior shrinkage (k = 3–50): launch AUC ±0.01, inside
noise. Ensembles: worse on every test. Category-relative features were already rejected in
radar-5 (time-split 0.01, random 0.31). The shipped ablation is the only change that improves
all four tests: dropping growth alone (out-of-fold 0.30), chains alone (0.31), all three
together (0.33). No variant tried moves the time-split past 0.31 or the out-of-fold past 0.36.

Two constants are **not** fitted, because the data cannot identify them: the **missing-growth
penalty, 0.05** (no live protocol is missing growth) and the **delay cap and scale (0.15 / 0.1)**
for old pre-token protocols (one cross-section cannot tell a protocol that will never launch
from one that has not launched yet). Both are small, fixed, and documented here rather than
pretended to be chosen by the data.

## 4. Holdout results

Four checks, all reported on the dashboard:

| Check | Value Spearman | Value top-10 | Value median \|log error\| | Launch AUC |
| --- | --- | --- | --- | --- |
| Time split (66 / 29, cutoff 654 days) | 0.25 | 4 / 10 | 0.88 (about x2.4) | 0.63 |
| Random split (66 / 29) | 0.42 | 6 / 10 | 1.62 (about x5.0) | 0.58 |
| Expanding window (mean of 4 eras) | 0.29 | — | 1.36 | — |
| Out-of-fold (128 live / 758 protocols) | 0.33 | 3 / 10 | 1.43 | 0.67 |

In-sample fit (128 live protocols): RMSE (log) 1.93, median |log error| 0.99 (about x2.7).

**Read the time split first — it is the hard test and the honest one.** It fits the oldest 70%
and predicts the newest 30%, which is what predicting future launches actually requires. Against
radar-5 it improves on ranking (Spearman 0.24 → 0.25, top-10 3 → 4) and cuts the dollar error
(1.11 → 0.88), because the value model now carries only the two signals the data pins down.
The random split is the easier test — same distribution, new rows — and it is the number to
compare against other radars built on this snapshot: Spearman 0.42, median error about x5.0.
The expanding window repeats the time test four times, era by era (folds 0.53, 0.22, 0.47,
−0.05): the mean is 0.29, but the last fold — the newest 19 protocols — scores near zero for
every variant tried. The youngest protocols are, on this snapshot, unrankable: their market caps
are driven by things the ten fields do not carry. The out-of-fold rows predict every row from a
model trained without it.

**Bootstrap stability (B = 200, fixed seed).** Value: log(TVL) (0.45–0.88) and the value prior
(0.47–3.95) are consistently positive — the two features kept. Launch: age (0.36–4.82), growth
(−2.36–−0.01) and age-unknown (0.55–5.05) do not cross zero; chains (−0.56–2.71) does.

## 5. Farm plans — the cost model

Every farm plan is netted to the dollar. For a plan with capital C, one-time setup S, weekly
cost W and a minimum hold of D days (WEEKS = ceil(D / 7)):

```
net cost to run   = C + S + W x WEEKS
spent (sunk)      = S + W x WEEKS          (capital is returned on exit)
expected gross    = P_launch x V_conditional x (C / TVL)
expected net      = expected gross - sunk cost
```

Capital is sized by TVL band ($50 under $100M TVL, $100 under $500M, $250 above); the detail
view shows the net cost beside the sunk cost, expected gross and expected net.

**The one assumption that matters, stated here and in every plan view:** expected gross assumes
any future token is distributed **pro-rata to TVL share**. The snapshot carries no airdrop
terms, no points program, no eligibility rules, so there is no better assumption available and
none is hidden. At $50–$250 position sizes in $50M+ protocols that share is dust: 193 of this
board's 198 farm plans have a negative expected net. That is the honest answer: most farm plans
cost more to run than their expected token share is worth, and the board ranks protocols by
implied value, not by farm profit.

## 6. Limitations

- **The value model is weak.** In-sample median error is about x2.7; on the time-split holdout
  about x2.4. The board is an ordering with wide error bars, not a price list.
- **The youngest protocols cannot be ranked.** The expanding window's newest era scores near
  zero for every variant tried. The board's tail rows — the youngest pre-token protocols — are
  ordered mostly by TVL and category prior, and should be read that way.
- **One cross-section, no history.** The snapshot is a single day (2026-11-14). Growth is a
  7-day change; there is no trajectory, no seasonality, no bear-market behaviour. A protocol
  that launched yesterday and one that waited three years look the same except through age,
  which enters only the launch half.
- **Category priors are thin.** 53 categories share 129 live protocols; several have one or
  two. Shrinkage keeps them honest, but a prior built on two protocols is still mostly the
  global rate.
- **34 live protocols have no listing date** and cannot be placed in time; they are counted,
  not hidden. 4 protocols have no 7-day growth; all pre-token. **No airdrop terms exist in the
  data:** the farm expected-gross figure is a pro-rata scenario, not a forecast of any
  announced distribution.

## 7. What the numbers are not

- **Not a market-cap promise.** Implied value is the model's estimate of what a launch could be
  worth, with a median error of about x2.4 even on the time-split holdout. A protocol ranked #1
  is not worth $53M; it is the protocol the model most expects to be worth the most.
- **Not investment advice.** Nothing here is a recommendation to buy, farm or hold anything.
  Farming costs real money (see §5) and the expected net is usually negative.
- **Not a launch prediction.** P_launch is a score calibrated on a single cross-section, not a
  forecast with a track record (launch AUC 0.63 on the time split).
- **Not DefiLlama's view.** "No token" means the snapshot records no token; it is DefiLlama's
  record, not proof the protocol has none.
