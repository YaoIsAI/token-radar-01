# Night-Scope Token Launch Radar — Research Note

Model version: radar-4. Snapshot: 758 protocols, as of 2026-11-14. This note is part of the
dashboard: it ships as `RESEARCH.md` beside the page, and the validation numbers ship as
`validation.json`. Everything here is computed from the snapshot at runtime; nothing in the model
is hand-set except the two constants the data cannot identify (see §3).

## 1. What this radar is

A ranking of pre-token protocols by **implied launch value**: what a token launch could be worth,
if it happens. Every row is

```
implied = round(P_launch x V_conditional)
```

`P_launch` is the probability the protocol launches a token; `V_conditional` is the market cap it
would carry if it did. The board orders rows by `implied` and nothing else. No protocol name is
special-cased, no row is hand-picked, and the 40 watchlist protocols are scored by the same
function as everyone else — they are guaranteed a row, not a rank.

## 2. The model

Both halves are regressions fit on the snapshot's own live-token protocols — the 129 protocols
that already have a token, of which 128 have a positive market cap.

**Value — least squares on log(market cap).** Features, all computed from the snapshot:

```
log V_conditional = b0 + b1 x log(TVL) + b2 x category value prior + b3 x age
                  + b4 x growth + b5 x chains + b6 if age unknown - 0.05 if growth unreported
```

Fitted at runtime (see §3): intercept 7.86, log(TVL) 0.64, value prior 2.24, age −0.38, growth
−1.06, chains 1.02, age unknown −0.51. The fitted value is exponentiated and clamped to
e^±30 so no row explodes.

**Launch — logistic regression on launched-or-not**, fit on all 758 protocols:

```
P_launch = sigmoid(c0 + c1 x age + c2 x growth + c3 x chains
                 + c4 x category launch prior + c5 if age unknown
                 - 0.05 if growth unreported - delay past 3 years)
```

Fitted at runtime: intercept −6.66, age +2.30, growth −0.90, chains +1.11, launch prior +13.10,
age unknown +2.54. The delay term is a structural correction, not a fit: a protocol listed more
than 3 years ago with no token is increasingly unlikely to launch now, capped at 0.15 logit.

**Features, exactly:**

- **log(TVL)** — natural log of total value locked, floor $1. TVL is scored in logs, never
  linearly, because market caps span orders of magnitude.
- **Category priors** — each category's launch rate and median log(market cap / TVL) among live
  protocols, shrunk toward the global rates with strength 10 (a category with few live protocols
  barely moves its prior). Categories with no live protocol take the global rate.
- **Age** — `1 - exp(-age_days / 365)`, so age saturates: the difference between 1 and 2 years
  matters, between 5 and 6 it barely does. Age is days since `listed_at`; 113 of 758 protocols
  have no listing date, including 34 live ones.
- **Growth** — 7-day TVL change, tanh-clamped at ±30% and scaled by 20, so one spike cannot rule
  the board. 4 protocols have no 7-day change; all 4 are pre-token.
- **Chains** — `1 - exp(-log1p(chain count) / log 9)`, saturating: the first chains matter, the
  tenth barely does.

## 3. How the coefficients were fitted

`main.ts` fits both regressions from the snapshot at runtime, deterministically, in pure
TypeScript — closed-form least squares by Gaussian elimination for the value model, iteratively
reweighted least squares (max 25 iterations) for the launch model. No randomness anywhere except
the bootstrap, which uses a fixed seed (20261117). No external packages. The fitted coefficients
are written into `data.js`, so the dashboard page recomputes every row with the same numbers and
checks the board's figure to the dollar.

Two constants are **not** fitted, because the data cannot identify them:

- **Missing-growth penalty, 0.05** — applied to both halves when a protocol has no 7-day change.
  No live protocol is missing growth, so a fitted flag would separate the 4 missing-growth
  protocols outright. The penalty is small and fixed, and the note says so rather than pretending
  the data chose it.
- **Delay cap and scale (0.15 / 0.1)** — the penalty for old pre-token protocols. The snapshot is
  a single cross-section: it cannot tell a protocol that will never launch from one that has not
  launched yet. The cap is a judgment call, kept small, and documented here.

**Category priors are never fit on held-out labels.** In validation, priors are recalibrated from
the training protocols only, so the holdout numbers are honest.

## 4. Holdout results

Two splits, both 70/30, both reported on the dashboard:

| Check | Time split | Random split |
| --- | --- | --- |
| Train / test (live protocols) | 66 / 29 (cutoff 654 days) | 66 / 29 |
| Value Spearman (predicted vs actual log mcap) | 0.04 | 0.42 |
| Value top-10 hit rate | 4 / 10 | 6 / 10 |
| Value median \|log error\| | 2.39 (about x11) | 1.73 (about x5.6) |
| Launch AUC | 0.63 | 0.58 |

In-sample fit (128 live protocols): RMSE (log) 1.91, median |log error| 1.13 (about x3).

**Read the time split first — it is the hard test and the honest one.** It fits the oldest 70%
and predicts the newest 30%, which is what predicting future launches actually requires. It
fails badly on ranking (Spearman 0.04): the age coefficient blows up in the old-only training
window (7.51) because the training ages barely vary (age score 0.83–0.99), and that coefficient
then scrambles the young test protocols. Removing age from the time-split fit lifts Spearman to
0.24 — the age term is the extrapolation poison. The feature stays in the fit because the task
specifies it, but this note says plainly that the time-split ranking is weak, and the dashboard
shows the number rather than hiding it.

The random split is the easier test — same distribution, new rows — and it is the number to
compare against other radars built on this snapshot: Spearman 0.42, median error about x5.6.

**Bootstrap stability (B = 200, fixed seed).** Only log(TVL) (0.41–0.86) and the category value
prior (0.80–4.19) are consistently positive across resamples. Age (−5.07–2.01), growth
(−3.31–0.65), chains (−1.21–2.93) and age-unknown (−5.08–1.68) all cross zero: the data does not
pin them down. Treat the small coefficients as directions, not measurements.

## 5. Farm plans — the cost model

Every farm plan is netted to the dollar. For a plan with capital C, one-time setup S, weekly
cost W and a minimum hold of D days (WEEKS = ceil(D / 7)):

```
net cost to run   = C + S + W x WEEKS
spent (sunk)      = S + W x WEEKS          (capital is returned on exit)
expected gross    = P_launch x V_conditional x (C / TVL)
expected net      = expected gross - sunk cost
```

The headline figure in the dashboard detail view is the net cost to run; the sunk cost,
expected gross and expected net sit beside it. Capital is sized by TVL band ($50 for under
$100M TVL, $100 under $500M, $250 above), setup and weekly costs by chain tier.

**The one assumption that matters, stated here and in every plan view:** expected gross assumes
any future token is distributed **pro-rata to TVL share** — your share of a future token equals
your capital's share of the protocol's TVL. The snapshot carries no airdrop terms, no points
program, no eligibility rules, so there is no better assumption available and none is hidden.
At $50–$250 position sizes in $50M+ protocols that share is dust: 196 of this board's 198 farm
plans have a negative expected net. That is the honest answer: most farm plans cost more to run
than their expected token share is worth, and the board ranks protocols by implied value, not by
farm profit.

## 6. Limitations

- **The value model is weak.** In-sample median error is about x3; on the time-split holdout it
  is about x11, and the ranking barely beats chance. The board is an ordering with wide error
  bars, not a price list.
- **One cross-section, no history.** The snapshot is a single day (2026-11-14). Growth is a
  7-day change; there is no trajectory, no seasonality, no bear-market behaviour. A protocol that
  launched yesterday and one that has waited three years look the same except through age.
- **Age is the fragile feature.** Its coefficient is unstable across splits and resamples, and it
  is the term that breaks the time-split ranking. Any future revision should either drop age or
  fit it on a longer history.
- **Category priors are thin.** 53 categories share 129 live protocols; several categories have
  one or two. Shrinkage keeps them honest, but a category prior built on two protocols is still
  mostly the global rate.
- **34 live protocols have no listing date** and cannot be placed in time; they are counted in
  the fit and in this note, not hidden. 4 protocols have no 7-day growth; all are pre-token.
- **No airdrop terms exist in the data.** The farm expected-gross figure is a pro-rata scenario,
  not a forecast of any announced distribution.

## 7. What the numbers are not

- **Not a market-cap promise.** Implied value is the model's estimate of what a launch could be
  worth, with a median error of about x3 even in-sample. A protocol ranked #1 is not worth
  $295M; it is the protocol the model most expects to be worth the most.
- **Not investment advice.** Nothing here is a recommendation to buy, farm or hold anything.
  Farming costs real money (see §5) and the expected net is usually negative.
- **Not a launch prediction.** P_launch is a score calibrated on a single cross-section, not a
  forecast with a track record. The launch AUC of 0.63 on the time split is barely above chance.
- **Not DefiLlama's view.** "No token" means the snapshot records no token; it is DefiLlama's
  record, not proof the protocol has none.
