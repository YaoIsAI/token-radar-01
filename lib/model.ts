// Token Radar's model, radar-5: one snapshot of the protocol universe in, one ranked board out.
//
// The board covers every slug in the snapshot's watchlist (the floor, never a ceiling) and adds
// every other pre-token protocol the house's own rule would keep: a protocol is rankable when it
// records no token (no market cap, no CoinGecko id, no symbol), is not an exchange, is not a
// versioned pool, carries no whole word "wrapped", "staked" or "staking" in its name, does not
// begin with an exchange's name, and its category is not Bridge, Canonical Bridge, Liquid Staking,
// Liquid Restaking, Restaked BTC or CEX. DefiLlama's parentProtocol is not one of the snapshot's
// ten fields, so a protocol the watchlist leaves out for a parent it records is still ranked here
// when it meets the visible rule: the watchlist is a floor, never a ceiling. Protocols whose token
// already trades are kept apart and shown with the market's own values, never mixed into the board.
// A token is taken to trade when the snapshot records a market cap for it (129 protocols, 120
// outside CEX): a bare symbol with no market cap is not proof of trading, and those protocols
// appear in neither view. One live record carries a market cap of -1 (autofinance): it counts
// toward the launch prior, but it cannot enter a log(market cap / TVL) median and is left out
// there.
//
// Implied value is a launch probability times a conditional value, and the same function scores
// every row; no protocol name is ever special-cased. radar-5 fits both halves by ridge regression
// on the snapshot's own 129 live-token protocols, at runtime, from the sealed input — the ridge
// strength is chosen by five-fold cross-validation on the live set, not hand-set, and no
// coefficient is hand-set any more except the two the data cannot identify (see below):
//
//   value:  log(market cap) is fit by ridge least squares on the live protocols, on
//           [1, log(TVL), category value prior, growth score, chain score,
//            missing-age flag]. Age was dropped from this half in radar-5: on the time-split
//           holdout its coefficient blew up to 7.51 on the old-only training window and the
//           ranking collapsed (Spearman 0.04); without it the time-split Spearman is 0.24 and
//           the median log error halves. V_conditional is exp(that prediction).
//   launch: launched-or-not is fit by ridge logistic regression (iteratively reweighted least
//           squares) on all 758 protocols, on [1, age score, growth score, chain score,
//           category launch prior, missing-age flag]. P_launch is the sigmoid of that logit,
//           less a delay penalty for protocols listed more than three years ago with no token.
//           The board's P_launch is cross-fitted: each row is scored by one of five fold models
//           trained without that row (fold by a hash of the slug, five-fold), each fold with its
//           own category launch priors, so no row's own label enters its own probability.
//   implied = max(1, round(P_launch x V_conditional))
//
// The two numbers the data cannot identify stay fixed constants, and the research note says why:
// the missing-growth penalty (no live protocol is missing seven-day growth, so no regression can
// price it; four protocols in the whole snapshot are missing it, all pre-token, so a fitted flag
// would separate them outright) and the delay penalty (a structural correction for the censoring
// in "launched by now": a protocol observed old and tokenless has already survived the usual
// launch window, which a regression on current status cannot see on its own).
//
// The fit is validated at runtime, on the same snapshot, and the numbers are on the dashboard and
// in validation.json: a TIME split (the oldest 70% of the live protocols with a recorded listing
// date fit the model, the newest 30% are held out), a RANDOM 70/30 split of the same set, an
// EXPANDING-WINDOW test (five eras by listing date, each era predicted by a model fit only on
// older eras), five-fold OUT-OF-FOLD predictions for both halves, and a bootstrap (B = 200, fixed
// seed) that refits both regressions on resampled data and reports each coefficient's range. 34
// live protocols have no listing date and cannot be placed in time, so the holdout covers the 95
// with one; the note says so. The band around a row recomputes the same function with TVL at 70%
// and 130% and seven-day growth at 80% and 120%: it is a sensitivity range, not a second model.
// Drivers are the three largest signed contributions to the score; risks name what is missing,
// extreme, concentrated or weakly calibrated. The dashboard states the formula, carries the fitted
// coefficients and the validation, and recomputes every row from the same figures, so the page and
// board.json agree to the dollar.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The model's fixed constants: the two the data cannot identify, and the data cleaning. */
export interface ModelConstants {
  missing_growth_penalty: number;
  delay_threshold_days: number;
  delay_scale: number;
  delay_cap: number;
  shrink_k: number;
  growth_clamp_pct: number;
  growth_scale: number;
  band_tvl_down: number;
  band_tvl_up: number;
  band_growth_down: number;
  band_growth_up: number;
}

export const MODEL: ModelConstants = {
  missing_growth_penalty: 0.05,
  delay_threshold_days: 1095,
  delay_scale: 0.1,
  delay_cap: 0.15,
  shrink_k: 10,
  growth_clamp_pct: 30,
  growth_scale: 20,
  band_tvl_down: 0.7,
  band_tvl_up: 1.3,
  band_growth_down: 0.8,
  band_growth_up: 1.2,
};

/** The model's version label: the dashboard names it and the rules quote it. */
export const MODEL_VERSION = 'radar-5';

/** The fitted value regression: log(market cap) on the features, in this order. Age is not one of
 * them any more: radar-5 dropped it from this half (see the header). */
export interface FittedValue {
  intercept: number;
  log_tvl: number;
  value_prior: number;
  growth: number;
  chain: number;
  missing_age: number;
}

/** The fitted launch regression: the logit of launched-or-not, in this order. */
export interface FittedLaunch {
  intercept: number;
  age: number;
  growth: number;
  chain: number;
  launch_prior: number;
  missing_age: number;
}

/** Both fitted halves, as the dashboard recomputes a row from them. `launch` is the full fit, a
 * reference; the board's P_launch comes from `launch_folds`, one fold model per row, each with its
 * own launch priors, so no row's own label enters its own probability. */
export interface FittedModel {
  value: FittedValue;
  launch: FittedLaunch;
  launch_folds: FittedLaunch[];
  launch_fold_priors: Array<{ global_launch_rate: number; categories: Record<string, number> }>;
  ridge: RidgeChoice;
}

/** A ridge strength, chosen by five-fold cross-validation, and the metric that chose it. */
export interface RidgeChoice {
  value_lambda: number;
  value_cv_spearman: number;
  launch_lambda: number;
  launch_cv_auc: number;
  grid: number[];
  folds: number;
}

/** Five-fold out-of-fold metrics: every row predicted by a model trained without it. */
export interface OofMetrics {
  value_spearman: number;
  value_median_abs_log_error: number;
  value_top10_hit: number;
  value_top10_n: number;
  launch_auc: number;
  launch_n: number;
  launch_live_n: number;
}

/** One era of the expanding-window test: a younger era predicted by a model fit only on older eras. */
export interface ExpandingFold {
  train_n: number;
  test_n: number;
  cutoff_age_days: number;
  value_spearman: number;
  value_median_abs_log_error: number;
}

/** One fitted coefficient with its bootstrap range. */
export interface CoefficientRange {
  name: string;
  fitted: number;
  low: number;
  high: number;
  median: number;
}

/** One holdout split's metrics: the value rank and error, and the launch rank. */
export interface SplitMetrics {
  train_n: number;
  test_n: number;
  cutoff_age_days: number;
  value_spearman: number;
  value_top10_hit: number;
  value_top10_n: number;
  value_median_abs_log_error: number;
  launch_auc: number;
  launch_test_n: number;
  launch_test_live_n: number;
}

/** The runtime validation: both splits, the expanding window, the out-of-fold predictions, the
 * in-sample fit, and the bootstrap ranges. */
export interface Validation {
  live_fit_n: number;
  known_age_live_n: number;
  missing_age_live_n: number;
  insample_value_rmse_log: number;
  insample_value_median_abs_log_error: number;
  time_split: SplitMetrics;
  random_split: SplitMetrics;
  ridge: RidgeChoice;
  oof: OofMetrics;
  expanding_window: {
    mean_spearman: number;
    mean_median_abs_log_error: number;
    folds: ExpandingFold[];
  };
  bootstrap_b: number;
  bootstrap_seed: number;
  value_bootstrap: CoefficientRange[];
  launch_bootstrap: CoefficientRange[];
}

/** One category's calibrated priors: how often its protocols launch, and what they are worth. */
export interface CategoryPrior {
  live: number;
  total: number;
  launch_prior: number;
  value_prior: number;
  value_samples: number;
}

/** Everything the dashboard needs to recompute a row exactly as the model scored it. */
export interface Calibration {
  live_count: number;
  live_non_cex_count: number;
  global_launch_rate: number;
  global_value_log_median: number;
  shrink_k: number;
  categories: Record<string, CategoryPrior>;
  // The ranked set's TVLs, sorted: a descriptive percentile for the driver lines only — the
  // fitted value model reads log(TVL) itself, never this grid. Raw TVLs are integers, so they
  // compare exactly in every engine; log1p itself can differ by a last bit between engines,
  // which once flipped a tied percentile.
  tvl_sorted: number[];
}

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** A farm plan's money and exit, beside the steps the gate reads. */
export interface FarmPlan {
  capital_usd: number;
  setup_cost_usd: number;
  weekly_cost_usd: number;
  hold_days: number;
  weeks: number;
  /** All cash committed: capital + setup + weekly cost x weeks. The capital comes back on exit. */
  net_cost_usd: number;
  /** Cash actually spent: setup + weekly cost x weeks. The capital is returned on exit. */
  sunk_cost_usd: number;
  /** The expected value of the farmer's share of a future token, probability-weighted. */
  expected_gross_usd: number;
  /** Expected profit: expected gross less the sunk cost. Usually negative; see the research note. */
  expected_net_usd: number;
  exit_steps: string[];
  risk_flags: string[];
}

/** One row's model detail: the score, its band, its drivers and its risks. `fold` is the
 * cross-fit fold that scored this row's launch probability (0..4). */
export interface RowModel {
  p_launch: number;
  v_conditional: number;
  fold: number;
  age_score: number;
  growth_score: number;
  chain_score: number;
  tvl_score: number;
  launch_prior: number;
  value_prior: number;
  missing_age: boolean;
  missing_growth: boolean;
  delay_penalty: number;
  implied_low: number;
  implied_high: number;
  drivers: string[];
  risks: string[];
  data_quality: 'complete' | 'partial' | 'thin';
  age_days: number | null;
}

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  model: RowModel;
  farm_steps: string[];
  farm_plan: FarmPlan;
}

/** The dashboard's compact view of one row, with the farm plan it carries in board.json. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  model: RowModel;
  farm_steps: string[];
  farm_plan: FarmPlan;
}

/** A live-token protocol: it already trades, so the board shows the market's own value, not a model's. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
  age_days: number | null;
}

/** Everything the dashboard needs: the board, the live set, the calibration and the model it runs. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: ModelConstants;
  model_version: string;
  fitted: FittedModel;
  validation: Validation;
  watchlist_size: number;
  watchlist_covered: number;
  watchlist: string[];
  board_size: number;
  early_size: number;
  live_size: number;
  missing_age_count: number;
  missing_growth_count: number;
  calibration: Calibration;
  rows: RadarRow[];
  live: LiveRow[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** The median of a list, or 0 when it holds nothing. */
function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;
const round2 = (n: number): number => Math.round(n * 100) / 100;

/** A dollar figure in short form, for a driver line. */
function shortUsd(n: number): string {
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `$${Math.round(n / 1e3)}K`;
  return `$${Math.round(n)}`;
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** The percentile of `value` inside a sorted list: the share below it, ties counted half. */
function percentile(sorted: number[], value: number): number {
  if (sorted.length === 0) return 0;
  let below = 0;
  let equal = 0;
  for (const v of sorted) {
    if (v < value) below++;
    else if (v === value) equal++;
  }
  return (below + 0.5 * equal) / sorted.length;
}

/**
 * The live-token set the calibration reads from: a market cap on record, zero excluded (a market
 * cap of zero is no market), one record's -1 kept for the launch count only. A bare symbol with no
 * market cap is not proof of trading.
 */
export function isLive(p: SnapshotProtocol): boolean {
  return p.mcap_usd !== null && p.mcap_usd !== 0;
}

/** The live protocols a log(market cap) regression can read: a positive market cap and TVL. */
function fitLive(snapshot: Snapshot): SnapshotProtocol[] {
  return snapshot.protocols.filter((p) => isLive(p) && p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
}

/**
 * The calibration: each category's launch and value priors, shrunk toward the global rates with a
 * pseudo-count of `shrink_k`, so a category with one live protocol barely moves off the global.
 * `pool` is the protocol set the priors are read from: the whole snapshot for the shipped model,
 * the training era only for a holdout fit, so a holdout never sees its own test set's labels. The
 * TVL percentile grid is filled in by buildBoard once the ranked set is known.
 */
export function calibrate(snapshot: Snapshot, model: ModelConstants = MODEL, pool: SnapshotProtocol[] | null = null): Calibration {
  const set = pool ?? snapshot.protocols;
  const live = set.filter(isLive);
  const liveNonCex = live.filter((p) => p.category !== 'CEX');
  const globalLaunchRate = set.length > 0 ? live.length / set.length : 0;
  const logRatios = live
    .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0)
    .map((p) => Math.log(p.mcap_usd! / p.tvl_usd));
  const globalValueMedian = median(logRatios);
  const byCat = new Map<string, SnapshotProtocol[]>();
  for (const p of set) {
    const key = p.category ?? 'Other';
    const arr = byCat.get(key);
    if (arr !== undefined) arr.push(p);
    else byCat.set(key, [p]);
  }
  const categories: Record<string, CategoryPrior> = {};
  for (const [key, total] of byCat) {
    const liveInCat = total.filter(isLive);
    const vals = liveInCat
      .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0)
      .map((p) => Math.log(p.mcap_usd! / p.tvl_usd));
    const med = vals.length > 0 ? median(vals) : null;
    categories[key] = {
      live: liveInCat.length,
      total: total.length,
      launch_prior: (liveInCat.length + model.shrink_k * globalLaunchRate) / (total.length + model.shrink_k),
      value_prior: med === null ? globalValueMedian : (vals.length * med + model.shrink_k * globalValueMedian) / (vals.length + model.shrink_k),
      value_samples: vals.length,
    };
  }
  return {
    live_count: live.length,
    live_non_cex_count: liveNonCex.length,
    global_launch_rate: globalLaunchRate,
    global_value_log_median: globalValueMedian,
    shrink_k: model.shrink_k,
    categories,
    tvl_sorted: [],
  };
}

/** A category's priors, falling back to the global rates when the category is unseen. */
function priorOf(cal: Calibration, category: string | null): CategoryPrior {
  return cal.categories[category ?? 'Other'] ?? {
    live: 0,
    total: 0,
    launch_prior: cal.global_launch_rate,
    value_prior: cal.global_value_log_median,
    value_samples: 0,
  };
}

/** The feature transforms the regressions share: age, growth and chain scores, 0 when unreported. */
function ageScore(age: number | null): number {
  return age === null ? 0 : 1 - Math.exp(-age / 365);
}
function growthScore(change7d: number | null, model: ModelConstants): number {
  return change7d === null
    ? 0
    : Math.tanh(Math.min(model.growth_clamp_pct, Math.max(-model.growth_clamp_pct, change7d)) / model.growth_scale);
}
function chainScore(chains: number): number {
  return 1 - Math.exp(-Math.log1p(chains) / Math.log(9));
}

/** The value regression's feature row: [1, log TVL, value prior, growth, chain, missing age].
 * Age is deliberately not here any more: on the time-split holdout it is the extrapolation poison
 * (radar-4's age coefficient blew up to 7.51 on the old-only window and scrambled the young test
 * rows); the launch half keeps age, where it is identified across the full age range. */
function valueFeatures(p: SnapshotProtocol, asOf: string, cal: Calibration, model: ModelConstants, tvlMult = 1, growthMult = 1): number[] {
  const age = ageDays(asOf, p.listed_at);
  const prior = priorOf(cal, p.category);
  return [
    1,
    Math.log(Math.max(p.tvl_usd * tvlMult, 1)),
    prior.value_prior,
    growthScore(p.change_7d === null ? null : p.change_7d * growthMult, model),
    chainScore(p.chains.length),
    age === null ? 1 : 0,
  ];
}

/** The launch regression's feature row: [1, age, growth, chain, launch prior, missing age]. */
function launchFeatures(p: SnapshotProtocol, asOf: string, cal: Calibration, model: ModelConstants): number[] {
  const age = ageDays(asOf, p.listed_at);
  const prior = priorOf(cal, p.category);
  return [
    1,
    ageScore(age),
    growthScore(p.change_7d, model),
    chainScore(p.chains.length),
    prior.launch_prior,
    age === null ? 1 : 0,
  ];
}

/** A deterministic generator: mulberry32, so a bootstrap's resamples are the same every run. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Gaussian elimination with partial pivoting; solves A x = b. Every fit in this file goes through
 * it, so a fit is a pure function of its rows: same rows, same coefficients, every run.
 */
function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const m = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r]![c]!) > Math.abs(m[piv]![c]!)) piv = r;
    if (Math.abs(m[piv]![c]!) < 1e-12) throw new Error('singular system');
    const tmp = m[c]!;
    m[c] = m[piv]!;
    m[piv] = tmp;
    for (let r = c + 1; r < n; r++) {
      const f = m[r]![c]! / m[c]![c]!;
      for (let k = c; k <= n; k++) m[r]![k]! -= f * m[c]![k]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r]![n]!;
    for (let k = r + 1; k < n; k++) s -= m[r]![k]! * x[k]!;
    x[r] = s / m[r]![r]!;
  }
  return x;
}

/**
 * The ridge machinery: every fit standardizes its non-intercept columns (mean 0, sd 1, a constant
 * column's sd guarded to 1 so it standardizes to all zeros), adds `lambda` to the standardized
 * non-intercept diagonal, and converts the coefficients back to the raw feature scale. The
 * intercept is never penalized. Standardizing first is what makes one lambda meaningful across
 * features on wildly different scales (log TVL near 20, a prior near 0.1). The conversion back
 * divides by sds[j - 1]: means/sds are indexed by the sliced column, j - 1 for raw feature j.
 */
function standardize(X: number[][]): { Xs: number[][]; means: number[]; sds: number[] } {
  const k = X[0]!.length;
  const means: number[] = [];
  const sds: number[] = [];
  for (let j = 1; j < k; j++) {
    const col = X.map((r) => r[j]!);
    const m = col.reduce((s, v) => s + v, 0) / col.length;
    const sd = Math.sqrt(col.reduce((s, v) => s + (v - m) ** 2, 0) / col.length) || 1;
    means.push(m);
    sds.push(sd);
  }
  const Xs = X.map((r) => [1, ...r.slice(1).map((v, j) => (v - means[j]!) / sds[j]!)]);
  return { Xs, means, sds };
}

function unstandardize(betaStd: number[], means: number[], sds: number[]): number[] {
  const k = betaStd.length;
  const beta = new Array<number>(k).fill(0);
  beta[0] = betaStd[0]! - betaStd.slice(1).reduce((s, v, j) => s + v * means[j]! / sds[j]!, 0);
  for (let j = 1; j < k; j++) beta[j] = betaStd[j]! / sds[j - 1]!;
  return beta;
}

/** Ridge least squares by the normal equations, on standardized features. */
function ridgeOls(X: number[][], y: number[], lambda: number): number[] {
  const { Xs, means, sds } = standardize(X);
  const k = Xs[0]!.length;
  const A: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const b = new Array<number>(k).fill(0);
  for (let i = 0; i < Xs.length; i++) {
    for (let a = 0; a < k; a++) {
      b[a]! += Xs[i]![a]! * y[i]!;
      for (let c = 0; c < k; c++) A[a]![c]! += Xs[i]![a]! * Xs[i]![c]!;
    }
  }
  for (let a = 1; a < k; a++) A[a]![a]! += lambda;
  A[0]![0]! += 1e-9;
  return unstandardize(solve(A, b), means, sds);
}

/**
 * Ridge logistic regression by iteratively reweighted least squares: the logit is fit, the weights
 * are the fitted probabilities' variance, and the loop stops when no coefficient moves by more
 * than 1e-9, or after 25 rounds. The ridge is applied on standardized features, like the value
 * half, so one lambda means the same thing in both halves.
 */
function ridgeLogistic(X: number[][], y: number[], lambda: number): number[] {
  const { Xs, means, sds } = standardize(X);
  const k = Xs[0]!.length;
  let beta = new Array<number>(k).fill(0);
  for (let it = 0; it < 25; it++) {
    const A: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
    const b = new Array<number>(k).fill(0);
    for (let i = 0; i < Xs.length; i++) {
      const z = Math.max(-30, Math.min(30, Xs[i]!.reduce((s, v, j) => s + v * beta[j]!, 0)));
      const p = 1 / (1 + Math.exp(-z));
      const w = Math.max(p * (1 - p), 1e-6);
      for (let a = 0; a < k; a++) {
        b[a]! += Xs[i]![a]! * ((y[i]! - p) + w * z);
        for (let c = 0; c < k; c++) A[a]![c]! += w * Xs[i]![a]! * Xs[i]![c]!;
      }
    }
    for (let a = 1; a < k; a++) A[a]![a]! += lambda;
    A[0]![0]! += 1e-8;
    const next = solve(A, b);
    let moved = 0;
    for (let j = 0; j < k; j++) moved = Math.max(moved, Math.abs(next[j]! - beta[j]!));
    beta = next;
    if (moved < 1e-9) break;
  }
  return unstandardize(beta, means, sds);
}

/** A protocol's cross-fit fold, 0..4, from an FNV-1a hash of its slug: deterministic, no seed. */
export function foldOfSlug(slug: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < slug.length; i++) {
    h ^= slug.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0) % 5;
}

/** The ridge grid the cross-validation searches. */
const LAMBDA_GRID = [0.03, 0.1, 0.3, 1, 3, 10, 30];

/**
 * The cross-validation tolerance: the smallest lambda within this much of the best score wins.
 * The profiles are flat to a few thousandths across the whole grid — far inside the sampling
 * noise of a rank correlation on 128 rows — so the argmax would be a coin flip, and on this
 * snapshot it lands on the grid's edge (lambda 30, ahead by 0.003). The smallest lambda inside
 * the band is the stable choice: the least shrinkage that the data cannot distinguish from the best.
 */
const LAMBDA_TOLERANCE = 0.01;

/** The smallest lambda whose cross-validated score is within the tolerance of the best. */
function pickLambda(scores: Array<{ lambda: number; score: number }>): { lambda: number; cv: number } {
  const best = Math.max(...scores.map((s) => s.score));
  const chosen = scores.find((s) => s.score >= best - LAMBDA_TOLERANCE)!;
  return { lambda: chosen.lambda, cv: round4(chosen.score) };
}

/**
 * The ridge strength for the value half, by five-fold cross-validation on the live set: for each
 * lambda, every fold is predicted by a model fit on the other four, and the pooled Spearman is
 * the score. The category priors are recalibrated per fold from the training protocols only: a
 * held-out row's own market cap enters its category's value prior in the full calibration (a
 * small category's median is that row), and a cross-validation that lets it in scores the leak,
 * not the model — the full-prior version reads 0.56, the honest per-fold version 0.29.
 */
function chooseValueLambda(snapshot: Snapshot, model: ModelConstants): { lambda: number; cv: number } {
  const live = fitLive(snapshot);
  const y = live.map((p) => Math.log(p.mcap_usd!));
  const scores: Array<{ lambda: number; score: number }> = [];
  for (const lambda of LAMBDA_GRID) {
    const pred = new Array<number>(live.length).fill(0);
    for (let f = 0; f < 5; f++) {
      const trainPool = snapshot.protocols.filter((p) => foldOfSlug(p.slug) !== f);
      const calFold = calibrate(snapshot, model, trainPool);
      const tr = live.map((_, i) => i).filter((i) => foldOfSlug(live[i]!.slug) !== f);
      const te = live.map((_, i) => i).filter((i) => foldOfSlug(live[i]!.slug) === f);
      if (te.length === 0) continue;
      const beta = ridgeOls(
        tr.map((i) => valueFeatures(live[i]!, snapshot.as_of, calFold, model)),
        tr.map((i) => y[i]!),
        lambda,
      );
      for (const i of te) {
        pred[i] = valueFeatures(live[i]!, snapshot.as_of, calFold, model).reduce((s, v, j) => s + v * beta[j]!, 0);
      }
    }
    scores.push({ lambda, score: spearman(pred, y) });
  }
  return pickLambda(scores);
}

/**
 * The ridge strength for the launch half, by the same five-fold cross-validation, scored by AUC,
 * with the priors recalibrated per fold for the same reason: a held-out row's own label enters its
 * category's launch prior in the full calibration, and a single-live category's prior is mostly
 * that row.
 */
function chooseLaunchLambda(snapshot: Snapshot, model: ModelConstants): { lambda: number; cv: number } {
  const y = snapshot.protocols.map((p) => (isLive(p) ? 1 : 0));
  const scores: Array<{ lambda: number; score: number }> = [];
  for (const lambda of LAMBDA_GRID) {
    const pred = new Array<number>(y.length).fill(0);
    for (let f = 0; f < 5; f++) {
      const trainPool = snapshot.protocols.filter((p) => foldOfSlug(p.slug) !== f);
      const calFold = calibrate(snapshot, model, trainPool);
      const tr = snapshot.protocols.map((_, i) => i).filter((i) => foldOfSlug(snapshot.protocols[i]!.slug) !== f);
      const te = snapshot.protocols.map((_, i) => i).filter((i) => foldOfSlug(snapshot.protocols[i]!.slug) === f);
      if (te.length === 0) continue;
      const beta = ridgeLogistic(
        tr.map((i) => launchFeatures(snapshot.protocols[i]!, snapshot.as_of, calFold, model)),
        tr.map((i) => y[i]!),
        lambda,
      );
      for (const i of te) {
        const z = Math.max(-30, Math.min(30, launchFeatures(snapshot.protocols[i]!, snapshot.as_of, calFold, model).reduce((s, v, j) => s + v * beta[j]!, 0)));
        pred[i] = 1 / (1 + Math.exp(-z));
      }
    }
    scores.push({ lambda, score: auc(pred, y) });
  }
  return pickLambda(scores);
}

/** Spearman's rank correlation, ties averaged: how well a prediction orders the actual values. */
function spearman(pred: number[], actual: number[]): number {
  const ranks = (arr: number[]): number[] => {
    const idx = arr.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
    const r = new Array<number>(arr.length);
    let i = 0;
    while (i < idx.length) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1]![0] === idx[i]![0]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) r[idx[k]![1]!] = avg;
      i = j + 1;
    }
    return r;
  };
  const rp = ranks(pred);
  const ra = ranks(actual);
  const n = pred.length;
  const mp = rp.reduce((s, v) => s + v, 0) / n;
  const ma = ra.reduce((s, v) => s + v, 0) / n;
  let num = 0;
  let dp = 0;
  let da = 0;
  for (let i = 0; i < n; i++) {
    num += (rp[i]! - mp) * (ra[i]! - ma);
    dp += (rp[i]! - mp) ** 2;
    da += (ra[i]! - ma) ** 2;
  }
  if (dp === 0 || da === 0) return 0;
  return num / Math.sqrt(dp * da);
}

/** The area under the ROC curve, ties counted half: how well a score ranks the launched ones. */
function auc(scores: number[], labels: number[]): number {
  const pos = scores.filter((_, i) => labels[i] === 1);
  const neg = scores.filter((_, i) => labels[i] === 0);
  if (pos.length === 0 || neg.length === 0) return 0;
  let wins = 0;
  for (const p of pos) for (const n of neg) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}

/**
 * The fitted halves, from the snapshot's own live protocols and the whole protocol set. The ridge
 * strengths come from five-fold cross-validation on the same rows (value: pooled Spearman, launch:
 * pooled AUC), with the category priors recalibrated per fold, and the smallest lambda within 0.01
 * of the best score wins, so no lambda is hand-set and a flat profile cannot pick a grid edge.
 * The board's launch probabilities are cross-fitted: five
 * fold models, each trained on the four fifths that exclude its fold, each with its own category
 * launch priors recalibrated from its training protocols only, so no row's own label enters its
 * own probability. The value half needs no cross-fit for the board: it is fit on live-token
 * protocols only, and every board row is pre-token, so a board row was never a fit row.
 */
export function fitModel(snapshot: Snapshot, cal: Calibration, model: ModelConstants = MODEL): FittedModel {
  const live = fitLive(snapshot);
  const valueChoice = chooseValueLambda(snapshot, model);
  const bv = ridgeOls(
    live.map((p) => valueFeatures(p, snapshot.as_of, cal, model)),
    live.map((p) => Math.log(p.mcap_usd!)),
    valueChoice.lambda,
  );
  const launchChoice = chooseLaunchLambda(snapshot, model);
  const bl = ridgeLogistic(
    snapshot.protocols.map((p) => launchFeatures(p, snapshot.as_of, cal, model)),
    snapshot.protocols.map((p) => (isLive(p) ? 1 : 0)),
    launchChoice.lambda,
  );
  const launchFolds: FittedLaunch[] = [];
  const launchFoldPriors: Array<{ global_launch_rate: number; categories: Record<string, number> }> = [];
  for (let f = 0; f < 5; f++) {
    const train = snapshot.protocols.filter((p) => foldOfSlug(p.slug) !== f);
    const calFold = calibrate(snapshot, model, train);
    const bf = ridgeLogistic(
      train.map((p) => launchFeatures(p, snapshot.as_of, calFold, model)),
      train.map((p) => (isLive(p) ? 1 : 0)),
      launchChoice.lambda,
    );
    launchFolds.push({
      intercept: bf[0]!,
      age: bf[1]!,
      growth: bf[2]!,
      chain: bf[3]!,
      launch_prior: bf[4]!,
      missing_age: bf[5]!,
    });
    const priors: Record<string, number> = {};
    for (const [key, prior] of Object.entries(calFold.categories)) priors[key] = prior.launch_prior;
    launchFoldPriors.push({ global_launch_rate: calFold.global_launch_rate, categories: priors });
  }
  return {
    value: {
      intercept: bv[0]!,
      log_tvl: bv[1]!,
      value_prior: bv[2]!,
      growth: bv[3]!,
      chain: bv[4]!,
      missing_age: bv[5]!,
    },
    launch: {
      intercept: bl[0]!,
      age: bl[1]!,
      growth: bl[2]!,
      chain: bl[3]!,
      launch_prior: bl[4]!,
      missing_age: bl[5]!,
    },
    launch_folds: launchFolds,
    launch_fold_priors: launchFoldPriors,
    ridge: {
      value_lambda: valueChoice.lambda,
      value_cv_spearman: valueChoice.cv,
      launch_lambda: launchChoice.lambda,
      launch_cv_auc: launchChoice.cv,
      grid: [...LAMBDA_GRID],
      folds: 5,
    },
  };
}

/** One split's metrics, from a fitted pair applied to its held-out set. */
function splitMetrics(
  valueBeta: number[],
  launchBeta: number[],
  testLive: SnapshotProtocol[],
  testAll: SnapshotProtocol[],
  snapshot: Snapshot,
  cal: Calibration,
  model: ModelConstants,
  cutoffAge: number,
): SplitMetrics {
  const pred = testLive.map((p) => valueFeatures(p, snapshot.as_of, cal, model).reduce((s, v, j) => s + v * valueBeta[j]!, 0));
  const actual = testLive.map((p) => Math.log(p.mcap_usd!));
  const topActual = new Set(
    actual.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]).slice(0, 10).map(([, i]) => i),
  );
  const topPred = new Set(
    pred.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]).slice(0, 10).map(([, i]) => i),
  );
  let hit = 0;
  for (const i of topActual) if (topPred.has(i)) hit++;
  const scores = testAll.map((p) => {
    const z = launchFeatures(p, snapshot.as_of, cal, model).reduce((s, v, j) => s + v * launchBeta[j]!, 0);
    return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
  });
  return {
    train_n: 0,
    test_n: testLive.length,
    cutoff_age_days: cutoffAge,
    value_spearman: round4(spearman(pred, actual)),
    value_top10_hit: hit,
    value_top10_n: 10,
    value_median_abs_log_error: round4(median(actual.map((y, i) => Math.abs(y - pred[i]!)))),
    launch_auc: round4(auc(scores, testAll.map((p) => (isLive(p) ? 1 : 0)))),
    launch_test_n: testAll.length,
    launch_test_live_n: testAll.filter(isLive).length,
  };
}

/**
 * The runtime validation: a time split, a random split, an expanding-window test, five-fold
 * out-of-fold predictions for both halves, and a bootstrap, all from the sealed snapshot. The
 * splits cover only the live protocols with a recorded listing date — 34 live protocols have
 * none and cannot be placed in time, so they are counted here, not hidden. The random split uses
 * the same fixed seed as the bootstrap, and the bootstrap refits both regressions on resampled
 * data with the priors held fixed, so its ranges measure the regressions' own sampling wobble,
 * not the priors'. The split fits use the ridge strengths the cross-validation chose on the full
 * set (a single hyperparameter from a flat profile; the note says so), and the out-of-fold
 * predictions recalibrate the category priors per fold, so a held-out row never enters its own
 * prior either.
 */
export function validateModel(snapshot: Snapshot, cal: Calibration, model: ModelConstants = MODEL): Validation {
  const live = fitLive(snapshot);
  const knownLive = live.filter((p) => ageDays(snapshot.as_of, p.listed_at) !== null);
  const missingAgeLive = live.length - knownLive.length;
  const seed = 20261117;
  const rnd = mulberry32(seed);

  // In-sample fit quality, on the same rows the shipped coefficients are fit from.
  const shipped = fitModel(snapshot, cal, model);
  const shippedValueBeta = [
    shipped.value.intercept, shipped.value.log_tvl, shipped.value.value_prior,
    shipped.value.growth, shipped.value.chain, shipped.value.missing_age,
  ];
  const inResid = live.map((p) => {
    const pred = valueFeatures(p, snapshot.as_of, cal, model).reduce((s, v, j) => s + v * shippedValueBeta[j]!, 0);
    return Math.log(p.mcap_usd!) - pred;
  });
  const insampleRmse = Math.sqrt(inResid.reduce((s, r) => s + r * r, 0) / inResid.length);
  const insampleMed = median(inResid.map(Math.abs));

  // The time split: the oldest 70% of the known-age live protocols fit, the newest 30% are held
  // out. The training era is the protocols listed at or before the cutoff, so the training priors
  // never see a held-out label.
  const byAge = [...knownLive].sort((a, b) => ageDays(snapshot.as_of, b.listed_at)! - ageDays(snapshot.as_of, a.listed_at)!);
  const cut = Math.floor(byAge.length * 0.7);
  const trainLive = byAge.slice(0, cut);
  const testLive = byAge.slice(cut);
  const cutoffAge = ageDays(snapshot.as_of, byAge[cut - 1]!.listed_at)!;
  const trainEra = snapshot.protocols.filter((p) => {
    const a = ageDays(snapshot.as_of, p.listed_at);
    return a !== null && a >= cutoffAge;
  });
  const testEra = snapshot.protocols.filter((p) => {
    const a = ageDays(snapshot.as_of, p.listed_at);
    return a !== null && a < cutoffAge;
  });
  const calTrain = calibrate(snapshot, model, trainEra);
  const bvTrain = ridgeOls(trainLive.map((p) => valueFeatures(p, snapshot.as_of, calTrain, model)), trainLive.map((p) => Math.log(p.mcap_usd!)), shipped.ridge.value_lambda);
  const blTrain = ridgeLogistic(trainEra.map((p) => launchFeatures(p, snapshot.as_of, calTrain, model)), trainEra.map((p) => (isLive(p) ? 1 : 0)), shipped.ridge.launch_lambda);
  const time = splitMetrics(bvTrain, blTrain, testLive, testEra, snapshot, calTrain, model, cutoffAge);
  time.train_n = trainLive.length;

  // The random split: the same known-age live set shuffled by the fixed seed, 70/30.
  const order = knownLive.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const t = order[i]!;
    order[i] = order[j]!;
    order[j] = t;
  }
  const rTrainLive = order.slice(0, Math.floor(order.length * 0.7)).map((i) => knownLive[i]!);
  const rTestLive = order.slice(Math.floor(order.length * 0.7)).map((i) => knownLive[i]!);
  const knownAll = snapshot.protocols.filter((p) => ageDays(snapshot.as_of, p.listed_at) !== null);
  const rOrder = knownAll.map((_, i) => i);
  for (let i = rOrder.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const t = rOrder[i]!;
    rOrder[i] = rOrder[j]!;
    rOrder[j] = t;
  }
  const rTrainAll = rOrder.slice(0, Math.floor(rOrder.length * 0.7)).map((i) => knownAll[i]!);
  const rTestAll = rOrder.slice(Math.floor(rOrder.length * 0.7)).map((i) => knownAll[i]!);
  const calRandom = calibrate(snapshot, model, rTrainAll);
  const bvRandom = ridgeOls(rTrainLive.map((p) => valueFeatures(p, snapshot.as_of, calRandom, model)), rTrainLive.map((p) => Math.log(p.mcap_usd!)), shipped.ridge.value_lambda);
  const blRandom = ridgeLogistic(rTrainAll.map((p) => launchFeatures(p, snapshot.as_of, calRandom, model)), rTrainAll.map((p) => (isLive(p) ? 1 : 0)), shipped.ridge.launch_lambda);
  const random = splitMetrics(bvRandom, blRandom, rTestLive, rTestAll, snapshot, calRandom, model, cutoffAge);
  random.train_n = rTrainLive.length;

  // The out-of-fold predictions: every row predicted by a model trained without it, with the
  // category priors recalibrated per fold, so a held-out row never enters its own prior either.
  // The value half is out-of-fold on the 128 live rows; the launch half on all 758 protocols.
  // The board's value rows are pre-token and were never fit rows, so the board's V_conditional is
  // already out-of-sample; the board's P_launch is the same fold machinery, one fold per row.
  const oofValuePred = new Array<number>(live.length).fill(0);
  const oofLaunchPred = new Array<number>(snapshot.protocols.length).fill(0);
  for (let f = 0; f < 5; f++) {
    const trainPool = snapshot.protocols.filter((p) => foldOfSlug(p.slug) !== f);
    const calFold = calibrate(snapshot, model, trainPool);
    const trainLiveFold = live.filter((p) => foldOfSlug(p.slug) !== f);
    const bvFold = ridgeOls(
      trainLiveFold.map((p) => valueFeatures(p, snapshot.as_of, calFold, model)),
      trainLiveFold.map((p) => Math.log(p.mcap_usd!)),
      shipped.ridge.value_lambda,
    );
    live.forEach((p, i) => {
      if (foldOfSlug(p.slug) !== f) return;
      oofValuePred[i] = valueFeatures(p, snapshot.as_of, calFold, model).reduce((s, v, j) => s + v * bvFold[j]!, 0);
    });
    const blFold = ridgeLogistic(
      trainPool.map((p) => launchFeatures(p, snapshot.as_of, calFold, model)),
      trainPool.map((p) => (isLive(p) ? 1 : 0)),
      shipped.ridge.launch_lambda,
    );
    snapshot.protocols.forEach((p, i) => {
      if (foldOfSlug(p.slug) !== f) return;
      const z = Math.max(-30, Math.min(30, launchFeatures(p, snapshot.as_of, calFold, model).reduce((s, v, j) => s + v * blFold[j]!, 0)));
      oofLaunchPred[i] = 1 / (1 + Math.exp(-z));
    });
  }
  const oofActual = live.map((p) => Math.log(p.mcap_usd!));
  const oofTopActual = new Set(
    oofActual.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]).slice(0, 10).map(([, i]) => i),
  );
  const oofTopPred = new Set(
    oofValuePred.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]).slice(0, 10).map(([, i]) => i),
  );
  let oofHit = 0;
  for (const i of oofTopActual) if (oofTopPred.has(i)) oofHit++;
  const oof: OofMetrics = {
    value_spearman: round4(spearman(oofValuePred, oofActual)),
    value_median_abs_log_error: round4(median(oofActual.map((y, i) => Math.abs(y - oofValuePred[i]!)))),
    value_top10_hit: oofHit,
    value_top10_n: 10,
    launch_auc: round4(auc(oofLaunchPred, snapshot.protocols.map((p) => (isLive(p) ? 1 : 0)))),
    launch_n: snapshot.protocols.length,
    launch_live_n: snapshot.protocols.filter(isLive).length,
  };

  // The expanding window: five eras by listing date, oldest first; each era is predicted by a
  // model fit only on older eras, with priors recalibrated from the older protocols only. This is
  // the "predict the future" test repeated four times, not once: the single time split is its
  // most extreme case (train on the oldest 70%, test on the newest 30%).
  const byAgeDesc = [...knownLive].sort((a, b) => ageDays(snapshot.as_of, b.listed_at)! - ageDays(snapshot.as_of, a.listed_at)!);
  const eraSize = Math.floor(byAgeDesc.length / 5);
  const expandingFolds: ExpandingFold[] = [];
  for (let f = 1; f < 5; f++) {
    const trainRows = byAgeDesc.slice(0, f * eraSize);
    const testRows = byAgeDesc.slice(f * eraSize, (f + 1) * eraSize);
    if (testRows.length === 0) continue;
    const cutoff = ageDays(snapshot.as_of, trainRows[trainRows.length - 1]!.listed_at)!;
    const trainPool = snapshot.protocols.filter((p) => {
      const a = ageDays(snapshot.as_of, p.listed_at);
      return a !== null && a >= cutoff;
    });
    const calEra = calibrate(snapshot, model, trainPool);
    const bvEra = ridgeOls(
      trainRows.map((p) => valueFeatures(p, snapshot.as_of, calEra, model)),
      trainRows.map((p) => Math.log(p.mcap_usd!)),
      shipped.ridge.value_lambda,
    );
    const predEra = testRows.map((p) => valueFeatures(p, snapshot.as_of, calEra, model).reduce((s, v, j) => s + v * bvEra[j]!, 0));
    const actualEra = testRows.map((p) => Math.log(p.mcap_usd!));
    expandingFolds.push({
      train_n: trainRows.length,
      test_n: testRows.length,
      cutoff_age_days: cutoff,
      value_spearman: round4(spearman(predEra, actualEra)),
      value_median_abs_log_error: round4(median(actualEra.map((y, i) => Math.abs(y - predEra[i]!)))),
    });
  }
  const expanding = {
    mean_spearman: round4(expandingFolds.reduce((s, x) => s + x.value_spearman, 0) / expandingFolds.length),
    mean_median_abs_log_error: round4(expandingFolds.reduce((s, x) => s + x.value_median_abs_log_error, 0) / expandingFolds.length),
    folds: expandingFolds,
  };

  // The bootstrap: B resamples of the regression rows, priors held fixed, both halves refit at the
  // shipped ridge strengths.
  const b = 200;
  const valueNames = ['intercept', 'log_tvl', 'value_prior', 'growth', 'chain', 'missing_age'];
  const launchNames = ['intercept', 'age', 'growth', 'chain', 'launch_prior', 'missing_age'];
  const valueBoots: number[][] = [];
  const launchBoots: number[][] = [];
  const Xv = live.map((p) => valueFeatures(p, snapshot.as_of, cal, model));
  const yv = live.map((p) => Math.log(p.mcap_usd!));
  const Xl = snapshot.protocols.map((p) => launchFeatures(p, snapshot.as_of, cal, model));
  const yl = snapshot.protocols.map((p) => (isLive(p) ? 1 : 0));
  for (let k = 0; k < b; k++) {
    const idxV = Array.from({ length: live.length }, () => Math.floor(rnd() * live.length));
    try {
      valueBoots.push(ridgeOls(idxV.map((i) => Xv[i]!), idxV.map((i) => yv[i]!), shipped.ridge.value_lambda));
    } catch {
      // A resample with no variation in one feature is singular; it carries no information.
    }
    const idxL = Array.from({ length: snapshot.protocols.length }, () => Math.floor(rnd() * snapshot.protocols.length));
    try {
      launchBoots.push(ridgeLogistic(idxL.map((i) => Xl[i]!), idxL.map((i) => yl[i]!), shipped.ridge.launch_lambda));
    } catch {
      // Same: a singular resample is skipped, not patched.
    }
  }
  const ranges = (names: string[], boots: number[][], fitted: number[]): CoefficientRange[] =>
    names.map((name, j) => {
      const col = boots.map((row) => row[j]!).sort((x, y) => x - y);
      return {
        name,
        fitted: round4(fitted[j]!),
        low: round4(col[0]!),
        high: round4(col[col.length - 1]!),
        median: round4(col[Math.floor(col.length / 2)]!),
      };
    });
  return {
    live_fit_n: live.length,
    known_age_live_n: knownLive.length,
    missing_age_live_n: missingAgeLive,
    insample_value_rmse_log: round4(insampleRmse),
    insample_value_median_abs_log_error: round4(insampleMed),
    time_split: time,
    random_split: random,
    ridge: shipped.ridge,
    oof,
    expanding_window: expanding,
    bootstrap_b: b,
    bootstrap_seed: seed,
    value_bootstrap: ranges(valueNames, valueBoots, shippedValueBeta),
    launch_bootstrap: ranges(launchNames, launchBoots, [
      shipped.launch.intercept, shipped.launch.age, shipped.launch.growth, shipped.launch.chain,
      shipped.launch.launch_prior, shipped.launch.missing_age,
    ]),
  };
}

/** The numeric core of one protocol's score, before the drivers and risks are written. `fold` is
 * the cross-fit fold that scored this row's launch probability. */
interface ScoreCore {
  implied: number;
  pLaunch: number;
  vConditional: number;
  fold: number;
  ageScore: number;
  growthScore: number;
  chainScore: number;
  tvlScore: number;
  launchPrior: number;
  valuePrior: number;
  missingAge: boolean;
  missingGrowth: boolean;
  delay: number;
  age: number | null;
}

/**
 * One protocol's score under the fitted model: a launch probability times a conditional value.
 * The multipliers let the confidence band recompute the same function with TVL and growth moved;
 * the base call passes none. The fitted value half is exp of the value regression's prediction
 * (age is not a value feature any more); the launch half is the sigmoid of the logit from the
 * fold model trained without this row, with that fold's own category launch prior, less the delay
 * penalty for a protocol listed more than three years ago with no token, and less the fixed
 * missing-growth penalty when seven-day growth is unreported.
 */
export function scoreCore(
  p: SnapshotProtocol,
  asOf: string,
  cal: Calibration,
  fitted: FittedModel,
  model: ModelConstants,
  tvlMult = 1,
  growthMult = 1,
): ScoreCore {
  const age = ageDays(asOf, p.listed_at);
  const missingAge = age === null;
  const missingGrowth = p.change_7d === null;
  const tvl = p.tvl_usd * tvlMult;
  const rawGrowth = p.change_7d === null ? null : p.change_7d * growthMult;
  const ageS = ageScore(age);
  const growthS = growthScore(rawGrowth, model);
  const chainS = chainScore(p.chains.length);
  const tvlS = percentile(cal.tvl_sorted, tvl);
  const delay = age !== null && age > model.delay_threshold_days
    ? Math.min(model.delay_cap, model.delay_scale * Math.log(age / model.delay_threshold_days))
    : 0;
  const prior = priorOf(cal, p.category);
  const vFeatures = valueFeatures(p, asOf, cal, model, tvlMult, growthMult);
  const vBeta = [
    fitted.value.intercept, fitted.value.log_tvl, fitted.value.value_prior,
    fitted.value.growth, fitted.value.chain, fitted.value.missing_age,
  ];
  const logV = vFeatures.reduce((s, v, j) => s + v * vBeta[j]!, 0) - (missingGrowth ? model.missing_growth_penalty : 0);
  const vConditional = Math.exp(Math.max(-30, Math.min(30, logV)));
  const fold = foldOfSlug(p.slug);
  const foldLaunch = fitted.launch_folds[fold]!;
  const foldPrior = fitted.launch_fold_priors[fold]!;
  const launchPrior = foldPrior.categories[p.category ?? 'Other'] ?? foldPrior.global_launch_rate;
  const logit = foldLaunch.intercept
    + foldLaunch.age * ageS
    + foldLaunch.growth * growthS
    + foldLaunch.chain * chainS
    + foldLaunch.launch_prior * launchPrior
    + foldLaunch.missing_age * (missingAge ? 1 : 0)
    - (missingGrowth ? model.missing_growth_penalty : 0)
    - delay;
  const pLaunch = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, logit))));
  return {
    implied: Math.max(1, Math.round(pLaunch * vConditional)),
    pLaunch,
    vConditional,
    fold,
    ageScore: ageS,
    growthScore: growthS,
    chainScore: chainS,
    tvlScore: tvlS,
    launchPrior,
    valuePrior: prior.value_prior,
    missingAge,
    missingGrowth,
    delay,
    age,
  };
}

/** The three largest signed contributions to a row's score, in plain words. */
function driversFor(p: SnapshotProtocol, core: ScoreCore, cal: Calibration): string[] {
  const cat = p.category ?? 'Other';
  const prior = priorOf(cal, p.category);
  const candidates: Array<{ weight: number; order: number; text: string }> = [
    {
      weight: core.tvlScore,
      order: 0,
      text: `TVL ${shortUsd(p.tvl_usd)} — larger than ${Math.round(core.tvlScore * 100)}% of the board`,
    },
    {
      weight: core.launchPrior,
      order: 1,
      text: `${cat} launch prior ${core.launchPrior.toFixed(2)} — ${prior.live} of ${prior.total} in this category have a token`,
    },
    {
      weight: Math.abs(core.valuePrior - cal.global_value_log_median),
      order: 2,
      text: `${cat} value prior ${core.valuePrior.toFixed(2)} — median log(market cap / TVL) for the category`,
    },
    {
      weight: Math.abs(core.growthScore),
      order: 3,
      text: p.change_7d === null ? '' : `7d growth ${p.change_7d >= 0 ? '+' : ''}${p.change_7d.toFixed(2)}%`,
    },
    {
      weight: core.ageScore,
      order: 4,
      text: core.age === null ? '' : `${core.age} days with no token`,
    },
    {
      weight: core.chainScore,
      order: 5,
      text: `Deployed on ${p.chains.length} chain${p.chains.length === 1 ? '' : 's'}`,
    },
  ].filter((c) => c.text !== '');
  candidates.sort((a, b) => b.weight - a.weight || a.order - b.order);
  return candidates.slice(0, 3).map((c) => c.text);
}

/** What is missing, extreme, concentrated or weakly calibrated about a row, at most three. */
function risksFor(p: SnapshotProtocol, core: ScoreCore, cal: Calibration): string[] {
  const cat = p.category ?? 'Other';
  const prior = priorOf(cal, p.category);
  const risks: string[] = [];
  if (core.missingAge) risks.push('Age unknown — the model gives no maturity credit and applies a penalty.');
  if (core.missingGrowth) risks.push('Seven-day growth is not reported — treated as neutral, with a penalty.');
  if (core.delay > 0) risks.push(`Listed ${core.age} days ago with no token — a launch-delay penalty applies.`);
  if (p.change_7d !== null && Math.abs(p.change_7d) > 30) {
    risks.push(`Seven-day change of ${p.change_7d >= 0 ? '+' : ''}${p.change_7d.toFixed(1)}% is extreme and was clamped — it may be a data spike, not real growth.`);
  }
  if (core.tvlScore > 0.9) risks.push('Value is concentrated in one very large TVL — a single large deposit or withdrawal moves it.');
  if (p.chains.length <= 1) risks.push('Deployed on one chain only.');
  if (prior.live < 5) risks.push(`Only ${prior.live} protocol${prior.live === 1 ? '' : 's'} in this category has a token on record — the category prior is weakly calibrated.`);
  if (core.launchPrior < 0.1) risks.push(`Few protocols in this category have launched a token (prior ${core.launchPrior.toFixed(2)}).`);
  return risks.slice(0, 3);
}

/** One row's full model detail, with its confidence band from the same function. */
function rowModel(p: SnapshotProtocol, core: ScoreCore, low: number, high: number, cal: Calibration): RowModel {
  return {
    p_launch: round4(core.pLaunch),
    v_conditional: Math.round(core.vConditional),
    fold: core.fold,
    age_score: round4(core.ageScore),
    growth_score: round4(core.growthScore),
    chain_score: round4(core.chainScore),
    tvl_score: round4(core.tvlScore),
    launch_prior: round4(core.launchPrior),
    value_prior: round4(core.valuePrior),
    missing_age: core.missingAge,
    missing_growth: core.missingGrowth,
    delay_penalty: round4(core.delay),
    implied_low: low,
    implied_high: high,
    drivers: driversFor(p, core, cal),
    risks: risksFor(p, core, cal),
    data_quality: core.missingAge && core.missingGrowth ? 'thin' : core.missingAge || core.missingGrowth ? 'partial' : 'complete',
    age_days: core.age,
  };
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
/** A whole word wrapped, staked or staking in a name: a product of a token that already trades. */
const WRAPPED = /\b(wrapped|staked|staking)\b/i;
/** Categories the house's watchlist rule keeps out, read from input/README.md. */
const EXCLUDED_CATEGORIES = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);
/** An exchange's name at the start of a protocol's name. */
const EXCHANGE_PREFIX = /^(Binance|Coinbase|Bybit|Kraken|Bitget|OKX|Bitfinex|Gemini)\b/;

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/**
 * Whether a protocol is one the board may rank: pre-token and not a wrapped, staked or bridged
 * product of a token that already trades, by the house's visible rule. Every watchlist slug meets
 * it; the watchlist is the floor, and this rule is the same floor applied to the rest.
 */
export function isRankable(p: SnapshotProtocol): boolean {
  if (!isPreToken(p)) return false;
  if (p.category !== null && EXCLUDED_CATEGORIES.has(p.category)) return false;
  if (WRAPPED.test(p.name)) return false;
  if (EXCHANGE_PREFIX.test(p.name)) return false;
  if (VERSIONED.test(p.name)) return false;
  return true;
}

/** The snapshot's protocol by slug. */
function bySlug(protocols: SnapshotProtocol[]): Map<string, SnapshotProtocol> {
  return new Map(protocols.map((p) => [p.slug, p]));
}

/** The protocols the board ranks, strongest first, with ranks 1..n. */
export function buildBoard(snapshot: Snapshot, cal: Calibration, fitted: FittedModel, model: ModelConstants = MODEL): BoardRow[] {
  const index = bySlug(snapshot.protocols);
  const chosen = new Map<string, SnapshotProtocol>();
  // The watchlist is the floor: every slug, even one the snapshot no longer carries (it cannot, but
  // the board's coverage is the gate's row and is never gambled on a missing record).
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  // The early radar: every other rankable protocol, whatever its TVL, so the board is the whole
  // pre-token universe the snapshot can see, not a TVL top-N. A chosen protocol stays chosen.
  for (const p of snapshot.protocols) {
    if (!chosen.has(p.slug) && isRankable(p)) chosen.set(p.slug, p);
  }
  // The TVL percentile grid is the ranked set itself, fixed before any row is scored.
  cal.tvl_sorted = [...chosen.values()].map((p) => p.tvl_usd).sort((a, b) => a - b);

  const scored = [...chosen.values()].map((p) => {
    const core = scoreCore(p, snapshot.as_of, cal, fitted, model);
    const low = scoreCore(p, snapshot.as_of, cal, fitted, model, model.band_tvl_down, model.band_growth_down);
    const high = scoreCore(p, snapshot.as_of, cal, fitted, model, model.band_tvl_up, model.band_growth_up);
    return { p, core, low: low.implied, high: high.implied };
  });
  scored.sort((a, b) => b.core.implied - a.core.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  return scored.map((s, i) => {
    const { steps, plan } = farmPlan(s.p, s.core.age, s.core.pLaunch, s.core.vConditional);
    return {
      rank: i + 1,
      slug: s.p.slug,
      name: s.p.name,
      category: s.p.category,
      chains: s.p.chains,
      tvl_usd: s.p.tvl_usd,
      change_7d: s.p.change_7d,
      mcap_usd: s.p.mcap_usd,
      listed_at: s.p.listed_at,
      gecko_id: s.p.gecko_id,
      symbol: s.p.symbol,
      token_status: 'pre-token',
      age_days: s.core.age,
      implied_value_usd: s.core.implied,
      model: rowModel(s.p, s.core, s.low, s.high, cal),
      farm_steps: steps.map(stepText),
      farm_plan: plan,
    };
  });
}

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** The setup gas estimate: approve plus deposit, more when the position is larger. */
function setupCostFor(capital: number): number {
  return capital >= 150 ? 8 : 4;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/**
 * The category's one action, in the words a farmer can follow. Nothing here is asserted as
 * verified: the action runs only if the protocol's own app confirms the route, and the asset is
 * the one the app accepts, never a named token the snapshot cannot prove.
 */
function actionFor(p: SnapshotProtocol, capital: number): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `If the app confirms lending, supply $${capital} of the asset the app accepts to ${name} and keep the position open.`;
  if (c.includes('dex') || c.includes('yield aggregator') || c.includes('leveraged farm')) return `If the app confirms a pool, provide $${capital} of liquidity to the pool the app shows on ${name} and keep the position open.`;
  if (c.includes('yield') || c.includes('farm')) return `If the app confirms a vault, deposit $${capital} of the asset the app accepts into the vault it shows on ${name} and keep the shares.`;
  if (c.includes('bridge')) return `If the app confirms the route, bridge $${capital} of the asset the app accepts through ${name} to the destination the app quotes, after checking the quoted fee.`;
  if (c.includes('staking') || c.includes('restak')) return `If the app confirms staking, stake $${capital} of the staking asset the app names through ${name} and keep the receipt.`;
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('basis')) return `If the app confirms margin deposits, deposit $${capital} as margin on ${name} and open one small hedge — never borrow to farm.`;
  if (c.includes('prediction')) return `If the app confirms markets, fund $${capital} and place one market position on ${name}.`;
  if (c.includes('curator') || c.includes('allocator') || c.includes('onchain capital')) return `If the app confirms a vault, deposit $${capital} into the vault it shows on ${name} and keep the shares.`;
  if (c.includes('stable')) return `If the app confirms minting, mint or deposit $${capital} of the stable asset it shows through ${name}.`;
  if (c.includes('rwa')) return `If the app confirms the offering, mint $${capital} of the yield asset it shows on ${name} and hold it.`;
  if (c.includes('insurance')) return `If the app confirms cover, buy $${capital} of the cover it shows from ${name} and keep the policy.`;
  return `If the app confirms deposits, deposit $${capital} of the asset the app accepts into ${name} and keep the position open.`;
}

/** The weekly qualifying action, matched to the same category as the main action. No borrowing. */
function weeklyFor(p: SnapshotProtocol): string {
  const c = (p.category ?? '').toLowerCase();
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return 'Check the position, harvest any yield, and re-deposit it.';
  if (c.includes('dex') || c.includes('yield aggregator') || c.includes('leveraged farm')) return 'Swap a small amount through the pool, or add a little more liquidity.';
  if (c.includes('yield') || c.includes('farm')) return 'Harvest the yield, then re-deposit it so the position keeps growing.';
  if (c.includes('bridge')) return 'Bridge a small amount back, so the wallet shows activity in both directions.';
  if (c.includes('staking') || c.includes('restak')) return 'Restake the staking rewards instead of withdrawing them.';
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('basis')) return 'Close the hedge and open a fresh one at a small size.';
  if (c.includes('prediction')) return 'Settle one market and place a position in another.';
  if (c.includes('curator') || c.includes('allocator') || c.includes('onchain capital')) return 'Top the vault up by a small amount, or move between its vaults once.';
  if (c.includes('stable')) return 'Swap between the protocol’s stable pairs for a small amount.';
  if (c.includes('rwa')) return 'Claim the yield the position earned and hold it in the same wallet.';
  if (c.includes('insurance')) return 'Check the policy’s status and extend it if it is close to expiry.';
  return 'Make one qualifying action, such as a deposit, a swap or a vote.';
}

/**
 * A protocol's farm plan: six steps, each an action with a cost and a time, plus the plan's money
 * and exit. The plan is derived from the protocol's own record (category, chains, TVL, age), never
 * from a hand-written list of names. The first step verifies the product entry in the protocol's
 * own app; nothing after it assumes a deposit route, an asset, a vault or a bridge that the
 * snapshot cannot prove.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null): FarmStep[] {
  const capital = capitalFor(p);
  const setup = setupCostFor(capital);
  const hold = holdDaysFor(age);
  const action = actionFor(p, capital);
  const weekly = weeklyFor(p);
  return [
    {
      text: `Open the protocol's own app from the link the protocol itself publishes and confirm which chain and asset it accepts for deposits, before moving any funds, and never trust a deposit address from a social post.`,
      cost: '$0 to check',
      time: '10 minutes',
    },
    {
      text: action,
      cost: `$${capital} of capital and about $${setup} of gas`,
      time: '15 minutes to set up',
    },
    {
      text: `Do one qualifying action each week: ${weekly}`,
      cost: 'about $2 of gas a week',
      time: '10 minutes a week',
    },
    {
      text: `Hold the position for at least ${hold} days so the activity is counted in more than one snapshot.`,
      cost: `the $${capital} stays in the position`,
      time: `${hold} days`,
    },
    {
      text: `Exit when the minimum holding period is complete, TVL falls by 25%, the protocol pauses deposits, or the weekly cost exceeds the expected reward — withdraw through the app's own route.`,
      cost: 'the withdrawal fee and gas the app quotes',
      time: '10 minutes',
    },
    {
      text: `Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.`,
      cost: 'nothing but the log',
      time: '5 minutes',
    },
  ];
}

/** What can go wrong with this farm, in the order a farmer should weigh it. */
function riskFlagsFor(p: SnapshotProtocol, age: number | null): string[] {
  const c = (p.category ?? '').toLowerCase();
  const flags = ['Deposit route unverified — confirm it in the protocol’s own app before moving funds.'];
  if (p.chains.length <= 1) flags.push('Single-chain deployment — no fallback chain if that chain congests.');
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('basis') || c.includes('leveraged')) {
    flags.push('Leverage can liquidate the position — keep the size small and never borrow to farm.');
  }
  if (c.includes('rwa')) flags.push('Real-world-asset redemption terms and exit fees vary — read them before depositing.');
  if (c.includes('bridge')) flags.push('Bridging adds the destination chain’s fees and delays — check the quoted fee first.');
  if (c.includes('insurance')) flags.push('Cover terms and claim rules are set by the protocol — read them before buying.');
  if (age === null) flags.push('Age unknown — the holding window is an estimate, not a record.');
  if (p.change_7d === null) flags.push('No seven-day growth on record — momentum is unknown.');
  return flags.slice(0, 4);
}

/**
 * A protocol's farm plan: the steps plus its money, its hold and its way out, all netted to the
 * dollar. The net cost is every dollar committed — capital, setup gas and the weekly gas over the
 * hold — and the capital comes back on exit, so only the setup and weekly gas are actually spent.
 * The expected gross is the model's launch probability times the model's conditional value times
 * the farmer's share of TVL (capital / TVL): it assumes any future token is distributed pro-rata
 * to TVL share, because the snapshot carries no airdrop terms at all, and it is a scenario, not a
 * promise — the research note states the assumption in full. The expected net is that gross less
 * the sunk cost. At the position sizes this board uses, the expected net is usually negative: a
 * $50–$250 position in a $50M+ protocol is a dust share of any future token.
 */
export function farmPlan(p: SnapshotProtocol, age: number | null, pLaunch: number, vConditional: number): { steps: FarmStep[]; plan: FarmPlan } {
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const setup = setupCostFor(capital);
  const weeks = Math.max(1, Math.ceil(hold / 7));
  const weekly = 2;
  const sunk = setup + weekly * weeks;
  const netCost = capital + sunk;
  const share = p.tvl_usd > 0 ? capital / p.tvl_usd : 0;
  const gross = pLaunch * vConditional * share;
  return {
    steps: farmSteps(p, age),
    plan: {
      capital_usd: capital,
      setup_cost_usd: setup,
      weekly_cost_usd: weekly,
      hold_days: hold,
      weeks,
      net_cost_usd: round2(netCost),
      sunk_cost_usd: round2(sunk),
      expected_gross_usd: round2(gross),
      expected_net_usd: round2(gross - sunk),
      exit_steps: [
        'Withdraw through the app’s own route once the minimum holding period is complete.',
        'Stop and exit early if TVL falls 25%, deposits pause, or the weekly cost exceeds the expected reward.',
        'Expect the withdrawal fee and gas the app quotes; never pay a third party to claim.',
      ],
      risk_flags: riskFlagsFor(p, age),
    },
  };
}

/**
 * The live-token protocols, every one of them, by market cap: a market cap on record is the
 * market's own value, and a bare symbol with no market cap is not proof of trading. CEX rows are
 * exchanges, not protocols to farm, and stay out.
 */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => isLive(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
      age_days: ageDays(snapshot.as_of, p.listed_at),
    }));
}

/** A board row's compact form for data.js, carrying the farm plan board.json carries. */
export function radarRow(row: BoardRow): RadarRow {
  return {
    rank: row.rank,
    slug: row.slug,
    name: row.name,
    category: row.category,
    chains: row.chains,
    tvl_usd: row.tvl_usd,
    change_7d: row.change_7d,
    mcap_usd: row.mcap_usd,
    listed_at: row.listed_at,
    age_days: row.age_days,
    implied_value_usd: row.implied_value_usd,
    model: row.model,
    farm_steps: row.farm_steps,
    farm_plan: row.farm_plan,
  };
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot, model: ModelConstants = MODEL): { board: BoardRow[]; radar: RadarData } {
  const cal = calibrate(snapshot, model);
  const fitted = fitModel(snapshot, cal, model);
  const validation = validateModel(snapshot, cal, model);
  const board = buildBoard(snapshot, cal, fitted, model);
  const live = liveRows(snapshot);
  const early = board.filter((r) => !snapshot.watchlist.includes(r.slug)).length;
  const covered = snapshot.watchlist.filter((s) => board.some((r) => r.slug === s)).length;
  const missingAge = board.filter((r) => r.age_days === null).length;
  const missingGrowth = board.filter((r) => r.change_7d === null).length;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model,
    model_version: MODEL_VERSION,
    fitted,
    validation,
    watchlist_size: snapshot.watchlist.length,
    watchlist_covered: covered,
    watchlist: [...snapshot.watchlist].sort(),
    board_size: board.length,
    early_size: early,
    live_size: live.length,
    missing_age_count: missingAge,
    missing_growth_count: missingGrowth,
    calibration: cal,
    rows: board.map(radarRow),
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared implied value, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.trim().length < 20 || s.trim().length > 600) return `${r.slug} has a farm step of ${s.trim().length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
