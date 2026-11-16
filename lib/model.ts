// Token Radar's model, radar-3: one snapshot of the protocol universe in, one ranked board out.
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
// Implied value is a launch probability times a conditional value, both calibrated from the
// snapshot's own 129 live-token protocols, and the same function scores every row; no protocol
// name is ever special-cased:
//
//   age_score    = 1 - exp(-age_days / 365); 0 when the age is unknown, plus a penalty below
//   growth_score = tanh(clamp(change_7d, -30, 30) / 20); 0 when growth is unreported, plus a penalty
//   chain_score  = 1 - exp(-ln(1 + chains) / ln(9))
//   tvl_score    = the percentile of log1p(tvl_usd) inside the ranked set (never a linear TVL)
//   delay        = min(0.15, 0.10 x ln(age_days / 1095)) once a protocol has run three years with
//                  no token; 0 before that, and 0 when the age is unknown
//   launch_prior = (live in the category + 10 x global live rate) / (protocols in the category + 10)
//   value_prior  = the same shrinkage over the category's median log(market cap / TVL)
//   P_launch     = sigmoid(-0.60 + 0.30 x age + 0.20 x growth + 0.10 x chains + 0.40 x launch_prior
//                          - 0.10 when the age is unknown - 0.05 when growth is unreported - delay)
//   V_conditional = tvl_usd x exp(value_prior - 0.45 + 0.20 x tvl + 0.20 x growth + 0.15 x age
//                          + 0.10 x chains - 0.10 when the age is unknown - 0.05 when growth is
//                          unreported)
//   implied      = max(1, round(P_launch x V_conditional))
//
// The band around a row recomputes the same function with TVL at 70% and 130% and seven-day growth
// at 80% and 120%: it is a sensitivity range, not a second model. Drivers are the three largest
// signed contributions to the score; risks name what is missing, extreme, concentrated or weakly
// calibrated. The dashboard states the formula, carries the calibration, and recomputes every row
// from the same figures, so the page and board.json agree.

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

/** The model's constants, as the dashboard states them. */
export interface ModelConstants {
  intercept: number;
  age_weight: number;
  growth_weight: number;
  chain_weight: number;
  launch_prior_weight: number;
  missing_age_penalty: number;
  missing_growth_penalty: number;
  delay_threshold_days: number;
  delay_scale: number;
  delay_cap: number;
  value_offset: number;
  value_tvl_weight: number;
  value_growth_weight: number;
  value_age_weight: number;
  value_chain_weight: number;
  shrink_k: number;
  growth_clamp_pct: number;
  growth_scale: number;
  band_tvl_down: number;
  band_tvl_up: number;
  band_growth_down: number;
  band_growth_up: number;
}

export const MODEL: ModelConstants = {
  intercept: -0.6,
  age_weight: 0.3,
  growth_weight: 0.2,
  chain_weight: 0.1,
  launch_prior_weight: 0.4,
  missing_age_penalty: 0.1,
  missing_growth_penalty: 0.05,
  delay_threshold_days: 1095,
  delay_scale: 0.1,
  delay_cap: 0.15,
  value_offset: -0.45,
  value_tvl_weight: 0.2,
  value_growth_weight: 0.2,
  value_age_weight: 0.15,
  value_chain_weight: 0.1,
  shrink_k: 10,
  growth_clamp_pct: 30,
  growth_scale: 20,
  band_tvl_down: 0.7,
  band_tvl_up: 1.3,
  band_growth_down: 0.8,
  band_growth_up: 1.2,
};

/** The model's version label: the dashboard names it and the rules quote it. */
export const MODEL_VERSION = 'radar-3';

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
  // The ranked set's TVLs, sorted: the percentile of log1p(TVL) is the percentile of TVL itself
  // (log1p is monotone), and raw TVLs are integers, so they compare exactly in every engine —
  // log1p itself can differ by a last bit between engines, which once flipped a tied percentile.
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
  exit_steps: string[];
  risk_flags: string[];
}

/** One row's model detail: the score, its band, its drivers and its risks. */
export interface RowModel {
  p_launch: number;
  v_conditional: number;
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

/**
 * The calibration: each category's launch and value priors, shrunk toward the global rates with a
 * pseudo-count of `shrink_k`, so a category with one live protocol barely moves off the global.
 * The TVL percentile grid is filled in by buildBoard once the ranked set is known.
 */
export function calibrate(snapshot: Snapshot, model: ModelConstants = MODEL): Calibration {
  const live = snapshot.protocols.filter(isLive);
  const liveNonCex = live.filter((p) => p.category !== 'CEX');
  const globalLaunchRate = snapshot.protocols.length > 0 ? live.length / snapshot.protocols.length : 0;
  const logRatios = live
    .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0)
    .map((p) => Math.log(p.mcap_usd! / p.tvl_usd));
  const globalValueMedian = median(logRatios);
  const byCat = new Map<string, SnapshotProtocol[]>();
  for (const p of snapshot.protocols) {
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

/** The numeric core of one protocol's score, before the drivers and risks are written. */
interface ScoreCore {
  implied: number;
  pLaunch: number;
  vConditional: number;
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
 * One protocol's score under `model`: a launch probability times a conditional value. The
 * multipliers let the confidence band recompute the same function with TVL and growth moved;
 * the base call passes none.
 */
export function scoreCore(
  p: SnapshotProtocol,
  asOf: string,
  cal: Calibration,
  model: ModelConstants,
  tvlMult = 1,
  growthMult = 1,
): ScoreCore {
  const age = ageDays(asOf, p.listed_at);
  const missingAge = age === null;
  const missingGrowth = p.change_7d === null;
  const tvl = p.tvl_usd * tvlMult;
  const rawGrowth = p.change_7d === null ? null : p.change_7d * growthMult;
  const ageScore = age === null ? 0 : 1 - Math.exp(-age / 365);
  const growthScore = rawGrowth === null
    ? 0
    : Math.tanh(Math.min(model.growth_clamp_pct, Math.max(-model.growth_clamp_pct, rawGrowth)) / model.growth_scale);
  const chainScore = 1 - Math.exp(-Math.log1p(p.chains.length) / Math.log(9));
  const tvlScore = percentile(cal.tvl_sorted, tvl);
  const delay = age !== null && age > model.delay_threshold_days
    ? Math.min(model.delay_cap, model.delay_scale * Math.log(age / model.delay_threshold_days))
    : 0;
  const prior = cal.categories[p.category ?? 'Other'] ?? {
    live: 0,
    total: 0,
    launch_prior: cal.global_launch_rate,
    value_prior: cal.global_value_log_median,
    value_samples: 0,
  };
  const logit = model.intercept
    + model.age_weight * ageScore
    + model.growth_weight * growthScore
    + model.chain_weight * chainScore
    + model.launch_prior_weight * prior.launch_prior
    - (missingAge ? model.missing_age_penalty : 0)
    - (missingGrowth ? model.missing_growth_penalty : 0)
    - delay;
  const pLaunch = 1 / (1 + Math.exp(-logit));
  const vConditional = tvl * Math.exp(
    prior.value_prior + model.value_offset
      + model.value_tvl_weight * tvlScore
      + model.value_growth_weight * growthScore
      + model.value_age_weight * ageScore
      + model.value_chain_weight * chainScore
      - (missingAge ? model.missing_age_penalty : 0)
      - (missingGrowth ? model.missing_growth_penalty : 0),
  );
  return {
    implied: Math.max(1, Math.round(pLaunch * vConditional)),
    pLaunch,
    vConditional,
    ageScore,
    growthScore,
    chainScore,
    tvlScore,
    launchPrior: prior.launch_prior,
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
  const prior = cal.categories[cat] ?? { live: 0, total: 0, launch_prior: cal.global_launch_rate, value_prior: cal.global_value_log_median, value_samples: 0 };
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
  const prior = cal.categories[cat] ?? { live: 0, total: 0, launch_prior: cal.global_launch_rate, value_prior: cal.global_value_log_median, value_samples: 0 };
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
export function buildBoard(snapshot: Snapshot, cal: Calibration, model: ModelConstants = MODEL): BoardRow[] {
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
    const core = scoreCore(p, snapshot.as_of, cal, model);
    const low = scoreCore(p, snapshot.as_of, cal, model, model.band_tvl_down, model.band_growth_down);
    const high = scoreCore(p, snapshot.as_of, cal, model, model.band_tvl_up, model.band_growth_up);
    return { p, core, low: low.implied, high: high.implied };
  });
  scored.sort((a, b) => b.core.implied - a.core.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  return scored.map((s, i) => {
    const { steps, plan } = farmPlan(s.p, s.core.age);
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

/** A protocol's farm plan: the steps plus its money, its hold and its way out. */
export function farmPlan(p: SnapshotProtocol, age: number | null): { steps: FarmStep[]; plan: FarmPlan } {
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  return {
    steps: farmSteps(p, age),
    plan: {
      capital_usd: capital,
      setup_cost_usd: setupCostFor(capital),
      weekly_cost_usd: 2,
      hold_days: hold,
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
  const board = buildBoard(snapshot, cal, model);
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
