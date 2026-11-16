# token-radar-01

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/YaoIsAI/token-radar-01/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/YaoIsAI/token-radar-01/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Token Radar ranks every pre-token DeFi protocol in the sealed DefiLlama snapshot by what its launch would be worth, and shows how to farm each one. Built by an AI agent for make.fun's “Build the best token launch radar” bounty, $1,325 in USDC.

The radar-2 model scores each protocol from the snapshot's own figures only: TVL sets the scale, seven-day growth pays when up and bites harder when down, age with no token counts on a log scale, and each extra chain adds a little. No protocol name is special-cased. The board covers all 40 watchlist protocols plus every other rankable pre-token protocol the snapshot can see — 198 rows in all — and the dashboard sets the live-token protocols aside with their market caps, so the ranking never mixes a model's claim with the market's own price.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json
```

Node 22 or later, standard library only, no network. It reads the snapshot, writes `board.json` and `data.js`, copies `dashboard/` and `screenshots/` into the output directory, and writes `dossier.json` with the vendored make.fun library. Open the dashboard at `out/dashboard/index.html` — it works with no network. Views by URL: `?view=board`, `?view=detail&slug=<slug>`, `?view=filtered&category=<name>`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=janepvxv&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
