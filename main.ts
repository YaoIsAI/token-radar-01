// Token Radar's entry program (Node 22, standard library only).
//
// It reads the sealed snapshot from /input/snapshot.json (or --input <file|dir>), fits the radar-4
// model on the snapshot's own live-token protocols in lib/model.ts, writes the ranked board to
// /output/board.json, writes the dashboard's data beside the page as data.js, copies this
// repository's dashboard/ and screenshots/ to /output, copies RESEARCH.md into the dashboard so
// the page's research-note link works and the dossier carries it, writes the validation numbers
// as dashboard/validation.json so the dossier carries them too, and writes /output/dossier.json
// with buildDossier from vendor/makefun-board.ts, the make.fun dossier library.
//
// The dashboard is a static page with no network: dashboard/index.html, dashboard/app.css and
// dashboard/app.js read the data.js this program writes. The board is the whole pre-token universe
// the snapshot can see, ranked by what a launch would be worth, each row with a farm plan.
//
// Outside the grader's container, MAKEFUN_OUTPUT_DIR stands in for /output:
//   mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { boardIssue, buildRadar, dataScript, parseSnapshot } from './lib/model.ts';
import { buildDossier } from './vendor/makefun-board.ts';

const here = fileURLToPath(new URL('.', import.meta.url));

/** The scorer's own caps (reference/gate.json): the text cap shapes the bytes, the rest refuse. */
const TEXT_MAX_BYTES = 65_536;
const SCREENSHOT_CAPS = { max: 4, max_bytes: 524_288 };
const IMAGE_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;
/** A screenshot's long side is at most 1440 and its short side at least 800 (reference/gate.json). */
const SHOT_LONG_MAX = 1440;
const SHOT_SHORT_MIN = 800;

/** --name <value> out of the arguments, or the fallback. */
function arg(name: string, fallback: string): string {
  const at = process.argv.indexOf(`--${name}`);
  if (at >= 0 && at + 1 < process.argv.length) return process.argv[at + 1]!;
  const withEquals = process.argv.find((a) => a.startsWith(`--${name}=`));
  return withEquals === undefined ? fallback : withEquals.slice(name.length + 3);
}

const inputArg = arg('input', '/input');
const inputFile = inputArg.endsWith('.json') ? inputArg : join(inputArg, 'snapshot.json');
if (!existsSync(inputFile)) {
  process.stderr.write(`token radar: no snapshot at ${inputFile}: pass --input <snapshot.json|directory> or mount the sealed /input\n`);
  process.exit(2);
}
const out = arg('output', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');

const snapshot = parseSnapshot(readFileSync(inputFile, 'utf8'));
const { board, radar } = buildRadar(snapshot);
const issue = boardIssue(board, snapshot);
if (issue !== null) {
  process.stderr.write(`token radar: the model wrote a board the gate refuses: ${issue}\n`);
  process.exit(1);
}

mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'board.json'), `${JSON.stringify({ rows: board })}\n`);
writeFileSync(join(out, 'data.js'), dataScript(radar));

/** Copy one directory's regular files into `to`, skipping names that start with a dot. */
function copyFiles(from: string, to: string): string[] {
  if (!existsSync(from)) return [];
  mkdirSync(to, { recursive: true });
  const copied: string[] = [];
  for (const name of readdirSync(from)) {
    if (name.startsWith('.')) continue;
    const path = join(from, name);
    if (!statSync(path).isFile()) continue;
    copyFileSync(path, join(to, name));
    copied.push(name);
  }
  return copied;
}

copyFiles(join(here, 'dashboard'), join(out, 'dashboard'));
const research = join(here, 'RESEARCH.md');
if (!existsSync(research)) {
  process.stderr.write('token radar: RESEARCH.md is missing from the repository root\n');
  process.exit(1);
}
copyFileSync(research, join(out, 'dashboard', 'RESEARCH.md'));
// The validation numbers as a dashboard file, so the dossier's text carries them: the key
// metrics first, so even a cut dossier keeps them. The fitted coefficients in here are the
// reference fits; the five fold launch models' per-category priors live in data.js, which is what
// the dashboard reads — they would cost the dossier text two kilobytes of numbers for no new
// information, since the fold models themselves are here. The bootstrap ranges are likewise left
// out of this file (they are in data.js, where the dashboard's coefficient tables read them, and
// their headline ranges are stated in the research note); this file carries the splits, the
// out-of-fold numbers and the expanding window, which are what the dossier's reader needs.
const { value_bootstrap: _vb, launch_bootstrap: _lb, bootstrap_b: _bb, bootstrap_seed: _bs, ...validationWithoutBootstrap } = radar.validation;
// The full-fit launch coefficients are stated in the research note (§2) and the fold fits below
// are the ones the board uses; the per-fold expanding-window numbers are likewise in the note
// (§4), as are the fitted value coefficients. The fold launch priors are the bulkiest part of the
// fitted model and live in data.js, which is what the dashboard reads. The fitted numbers here
// are rounded to four decimals: the dashboard recomputes from data.js at full precision, and this
// file is the reference copy. The ridge grid is documented in the note (§3) and left out of both
// copies. `validation` comes before `fitted` so that, if the dossier text ever runs short, the
// splits survive and the fold fits — whose numbers the note already states — are what gets cut.
const { launch: _fl, launch_fold_priors: _lfp, ...fittedWithoutFullLaunch } = radar.fitted;
const { folds: _ef, ...expandingWithoutFolds } = validationWithoutBootstrap.expanding_window;
const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;
const roundRecord = (f: Record<string, number>): Record<string, number> =>
  Object.fromEntries(Object.entries(f).map(([k, v]) => [k, round4(v)]));
const { grid: _grid, ...ridgeWithoutGrid } = radar.fitted.ridge;
const validationDoc = {
  model_version: radar.model_version,
  as_of: radar.as_of,
  headline: {
    time_value_spearman: radar.validation.time_split.value_spearman,
    random_value_spearman: radar.validation.random_split.value_spearman,
    expanding_value_spearman: radar.validation.expanding_window.mean_spearman,
    oof_value_spearman: radar.validation.oof.value_spearman,
    oof_launch_auc: radar.validation.oof.launch_auc,
    time_launch_auc: radar.validation.time_split.launch_auc,
    value_lambda: radar.validation.ridge.value_lambda,
    launch_lambda: radar.validation.ridge.launch_lambda,
  },
  validation: { ...validationWithoutBootstrap, expanding_window: expandingWithoutFolds, ridge: ridgeWithoutGrid },
  fitted: {
    launch_folds: fittedWithoutFullLaunch.launch_folds.map(roundRecord),
  },
};
writeFileSync(join(out, 'dashboard', 'validation.json'), `${JSON.stringify(validationDoc, null, 2)}\n`);
const shots = copyFiles(join(here, 'screenshots'), join(out, 'screenshots'));
if (shots.length < 3) {
  process.stderr.write(`token radar: screenshots/ holds ${shots.length} image${shots.length === 1 ? '' : 's'}: hand in 3, the board, a detail view and the validation view\n`);
  process.exit(1);
}
if (shots.length > SCREENSHOT_CAPS.max) {
  process.stderr.write(`token radar: screenshots/ holds ${shots.length} images, over the ${SCREENSHOT_CAPS.max} the gate takes\n`);
  process.exit(1);
}
for (const name of shots) {
  if (!IMAGE_NAME.test(name)) {
    process.stderr.write(`token radar: screenshots/${name}: a name is letters, digits and . _ -, 1 to 128 characters\n`);
    process.exit(1);
  }
}

/** A PNG's pixel size out of its IHDR chunk, or null when the bytes are not a PNG. */
function pngSize(bytes: Uint8Array): { width: number; height: number } | null {
  const magic = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 24) return null;
  for (let i = 0; i < 8; i++) if (bytes[i] !== magic[i]) return null;
  const width = (bytes[16]! << 24) | (bytes[17]! << 16) | (bytes[18]! << 8) | bytes[19]!;
  const height = (bytes[20]! << 24) | (bytes[21]! << 16) | (bytes[22]! << 8) | bytes[23]!;
  return { width, height };
}

/** Every screenshot the gate takes: a PNG of the right size and shape, under the byte cap. */
for (const name of shots) {
  const path = join(out, 'screenshots', name);
  const bytes = readFileSync(path);
  if (bytes.length > SCREENSHOT_CAPS.max_bytes) {
    process.stderr.write(`token radar: screenshots/${name} is ${bytes.length} bytes, over the ${SCREENSHOT_CAPS.max_bytes} the gate takes\n`);
    process.exit(1);
  }
  const size = pngSize(bytes);
  if (size === null) {
    process.stderr.write(`token radar: screenshots/${name} is not a PNG: hand in PNG screenshots of the dashboard\n`);
    process.exit(1);
  }
  const long = Math.max(size.width, size.height);
  const short = Math.min(size.width, size.height);
  if (long > SHOT_LONG_MAX || short < SHOT_SHORT_MIN) {
    process.stderr.write(`token radar: screenshots/${name} is ${size.width}x${size.height}: the long side is at most ${SHOT_LONG_MAX} and the short side at least ${SHOT_SHORT_MIN}\n`);
    process.exit(1);
  }
}

// The dossier is written last: the scorer rebuilds it from this /output and compares bytes.
writeFileSync(join(out, 'dossier.json'), buildDossier(out, { text_max_bytes: TEXT_MAX_BYTES, screenshots: SCREENSHOT_CAPS, dashboard: { max_files: 400, max_bytes: 4_194_304 } }));

process.stdout.write(
  `token radar: ${snapshot.protocols.length} protocols, ${snapshot.watchlist.length} on the watchlist, ${board.length} rows scored (${radar.early_size} early, ${radar.live_size} live set aside)\n` +
    `token radar: wrote ${join(out, 'board.json')}, ${join(out, 'data.js')}, ${join(out, 'dashboard')} (3 files), ${join(out, 'screenshots')} (${shots.length} images) and ${join(out, 'dossier.json')}\n`,
);
