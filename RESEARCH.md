# Night-Scope Token Launch Radar — Research Note

Model version: radar-5. Snapshot: 758 protocols, as of 2026-11-14. This note ships as
`RESEARCH.md` beside the dashboard, and the validation numbers ship as `validation.json`.
Everything here is computed from the snapshot at runtime; nothing in the model is hand-set
except the two constants the data cannot identify (see §3).

## 1. What this radar is

A ranking of pre-token protocols by **implied launch value**: what a token launch could be
worth, if it happens. Every row is

```
implied = round(P_launch x V_conditional)
```

`P_launch` is the probability the protocol launches a token; `V_conditional` is the market cap
it would carry if it did. The board orders rows by `implied` and nothing else. No protocol name
is special-cased, no row is hand-picked, and the 40 watchlist protocols are scored by the same
function as everyone else — they are guaranteed a row, not a rank.

## 2. The model

Both halves are ridge regressions fit on the snapshot's own live-token protocols — the 129
protocols that already have a token, of which 128 have a positive market cap. The ridge strengths
are chosen by five-fold cross-validation (see §3), not hand-set.

**Value — ridge least squares on log(market cap):**

```
log V_conditional = b0 + b1 x log(TVL) + b2 x category value prior
                  + b3 x growth + b4 x chains + b5 if age unknown - 0.05 if growth unreported
```

Fitted at runtime: intercept 7.53, log(TVL) 0.65, value prior 2.27, growth −1.04, chains 0.96,
age unknown −0.18. The fitted value is exponentiated and clamped to e^±30 so no row explodes.
**Age is not a value feature any more.** In radar-4 it was, and on the time-split holdout its
coefficient blew up to 7.51 on the old-only training window and the ranking collapsed (Spearman
0.04). Dropping it lifts the time-split Spearman to 0.24 and halves the median log error, and
costs nothing on the random split (0.42 → 0.41) or the expanding window (0.29 → 0.29). Age is
still a feature of the launch half, where it is identified across the full age range.

**Launch — ridge logistic regression on launched-or-not**, fit on all 758 protocols:

```
P_launch = sigmoid(c0 + c1 x age + c2 x growth + c3 x chains
                 + c4 x category launch prior + c5 if age unknown
                 - 0.05 if growth unreported - delay past 3 years)
```

Fitted at runtime: intercept −6.66, age +2.29, growth −0.90, chains +1.12, launch prior +13.09,
age unknown +2.53. The delay term is a structural correction, not a fit: a protocol listed more
than 3 years ago with no token is increasingly unlikely to launch now, capped at 0.15 logit.
**The board's P_launch is cross-fitted**: each row is scored by one of five fold models trained
without that row (fold by a hash of the slug), each fold with its own category launch priors, so
no row's own label enters its own probability. The full fit above is the reference; the five fold
fits are in `validation.json`.

**Features, exactly:** **log(TVL)** — natural log of total value locked, floor $1, scored in
logs, never linearly, because market caps span orders of magnitude. **Category priors** — each
category's launch rate and median log(market cap / TVL) among live protocols, shrunk toward the
global rates with strength 10. **Age** — `1 - exp(-age_days / 365)`, saturating; days since
`listed_at`, missing for 113 of 758 protocols including 34 live ones; a launch feature only.
**Growth** — 7-day TVL change, tanh-clamped at ±30% and scaled by 20. 4 protocols have no 7-day
change; all 4 are pre-token. **Chains** — `1 - exp(-log1p(chain count) / log 9)`, saturating.

## 3. How the coefficients were fitted

`main.ts` fits both regressions from the snapshot at runtime, deterministically, in pure
TypeScript — ridge least squares by Gaussian elimination for the value model, ridge iteratively
reweighted least squares (max 25 iterations) for the launch model, both on standardized features
so one ridge strength means the same thing in both halves. No randomness anywhere except the
bootstrap, which uses a fixed seed (20261117). No external packages. The fitted coefficients are
written into `data.js`, so the dashboard page recomputes every row with the same numbers and
checks the board's figure to the dollar.

**Ridge strengths, chosen by cross-validation.** Five folds by a hash of the slug, grid
0.03–30, scored by pooled Spearman (value) and pooled AUC (launch), with the category priors
recalibrated per fold. The profiles are flat — the value profile moves 0.289 to 0.292 across the
whole grid, far inside the sampling noise of a rank correlation on 128 rows — so the argmax
would be a coin flip, and on this snapshot it lands on the grid's edge (λ = 30, ahead by 0.003).
The rule is therefore the smallest λ within 0.01 of the best score: value λ = 0.03 (CV Spearman
0.29), launch λ = 0.03 (CV AUC 0.67).

**The priors must be recalibrated per fold, or the cross-validation scores the leak.** A
held-out row's own market cap enters its category's value prior in the full calibration, and its
own label enters its category's launch prior. With the full priors the cross-validation reads
0.56 (value) and 0.75 (launch AUC); with per-fold priors it reads 0.29 and 0.67 — the honest
numbers, and the ones reported here. The board itself never had this leak: its rows are
pre-token and never enter a value prior, and its launch probabilities are cross-fitted.

**Category-relative features were tried and not shipped.** TVL percentile within category and
growth relative to the category median: time-split Spearman 0.01, random-split 0.31, expanding
window 0.13–0.24 against 0.29 for the model as shipped. They cost ranking quality on every
test, so they are not in the model.

Two constants are **not** fitted, because the data cannot identify them: the **missing-growth
penalty, 0.05** (no live protocol is missing growth, so a fitted flag would separate the 4
missing-growth protocols outright) and the **delay cap and scale (0.15 / 0.1)** for old
pre-token protocols (a single cross-section cannot tell a protocol that will never launch from
one that has not launched yet). Both are small, fixed, and documented here rather than
pretended to be chosen by the data.

## 4. Holdout results

Four checks, all reported on the dashboard:

| Check | Value Spearman | Value top-10 | Value median \|log error\| | Launch AUC |
| --- | --- | --- | --- | --- |
| Time split (66 / 29, cutoff 654 days) | 0.24 | 3 / 10 | 1.11 (about x3.0) | 0.63 |
| Random split (66 / 29) | 0.41 | 6 / 10 | 1.72 (about x5.6) | 0.58 |
| Expanding window (mean of 4 eras) | 0.29 | — | 1.21 | — |
| Out-of-fold (128 live / 758 protocols) | 0.29 | 4 / 10 | 1.52 | 0.67 |

In-sample fit (128 live protocols): RMSE (log) 1.91, median |log error| 1.13 (about x3).

**Read the time split first — it is the hard test and the honest one.** It fits the oldest 70%
and predicts the newest 30%, which is what predicting future launches actually requires. Against
radar-4 it improves sixfold on ranking (Spearman 0.04 → 0.24) and halves the dollar error (2.39
→ 1.11), because age is out of the value model. The random split is the easier test — same
distribution, new rows — and it is the number to compare against other radars built on this
snapshot: Spearman 0.41, median error about x5.6. The expanding window repeats the time test
four times, era by era (folds 0.51, 0.20, 0.46, −0.04): the mean is 0.29, but the last fold —
the newest 19 protocols — scores near zero for every variant tried, including the ones with age
back in. The youngest protocols are, on this snapshot, unrankable: their market caps are driven
by things the ten fields do not carry. The out-of-fold rows predict every row from a model
trained without it, priors included.

**Bootstrap stability (B = 200, fixed seed).** Value: only log(TVL) (0.44–0.86) and the category
value prior (0.81–4.19) are consistently positive across resamples; growth (−3.05–0.66), chains
(−1.24–2.98) and age-unknown (−1.48–1.56) all cross zero. Launch: age (0.36–4.82), growth
(−2.36–−0.01) and age-unknown (0.55–5.05) do not cross zero; chains (−0.56–2.71) does. Treat the
small coefficients as directions, not measurements.

## 5. Farm plans — the cost model

Every farm plan is netted to the dollar. For a plan with capital C, one-time setup S, weekly
cost W and a minimum hold of D days (WEEKS = ceil(D / 7)):

```
net cost to run   = C + S + W x WEEKS
spent (sunk)      = S + W x WEEKS          (capital is returned on exit)
expected gross    = P_launch x V_conditional x (C / TVL)
expected net      = expected gross - sunk cost
```

Capital is sized by TVL band ($50 for under $100M TVL, $100 under $500M, $250 above); the
dashboard detail view shows the net cost to run beside the sunk cost, expected gross and
expected net.

**The one assumption that matters, stated here and in every plan view:** expected gross assumes
any future token is distributed **pro-rata to TVL share** — your share of a future token equals
your capital's share of the protocol's TVL. The snapshot carries no airdrop terms, no points
program, no eligibility rules, so there is no better assumption available and none is hidden.
At $50–$250 position sizes in $50M+ protocols that share is dust: 195 of this board's 198 farm
plans have a negative expected net. That is the honest answer: most farm plans cost more to run
than their expected token share is worth, and the board ranks protocols by implied value, not by
farm profit.

## 6. Limitations

- **The value model is weak.** In-sample median error is about x3; on the time-split holdout it
  is about x3.0, and the ranking is moderate at best. The board is an ordering with wide error
  bars, not a price list.
- **The youngest protocols cannot be ranked.** The expanding window's newest era scores near
  zero for every variant tried. The board's tail rows — the youngest pre-token protocols — are
  ordered mostly by TVL and category prior, and should be read that way.
- **One cross-section, no history.** The snapshot is a single day (2026-11-14). Growth is a
  7-day change; there is no trajectory, no seasonality, no bear-market behaviour. A protocol that
  launched yesterday and one that has waited three years look the same except through age, which
  now enters only the launch half.
- **Category priors are thin.** 53 categories share 129 live protocols; several categories have
  one or two. Shrinkage keeps them honest, but a category prior built on two protocols is still
  mostly the global rate.
- **34 live protocols have no listing date** and cannot be placed in time; they are counted in
  the fit and in this note, not hidden. 4 protocols have no 7-day growth; all are pre-token.
  **No airdrop terms exist in the data.** The farm expected-gross figure is a pro-rata scenario,
  not a forecast of any announced distribution.

## 7. What the numbers are not

- **Not a market-cap promise.** Implied value is the model's estimate of what a launch could be
  worth, with a median error of about x3 even in-sample. A protocol ranked #1 is not worth
  $74M; it is the protocol the model most expects to be worth the most.
- **Not investment advice.** Nothing here is a recommendation to buy, farm or hold anything.
  Farming costs real money (see §5) and the expected net is usually negative.
- **Not a launch prediction.** P_launch is a score calibrated on a single cross-section, not a
  forecast with a track record. The launch AUC of 0.63 on the time split is barely above chance.
- **Not DefiLlama's view.** "No token" means the snapshot records no token; it is DefiLlama's
  record, not proof the protocol has none.
