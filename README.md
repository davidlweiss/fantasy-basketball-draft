# Fantasy basketball draft toolkit 2026-27 (Yahoo 9-cat, 10 teams, 13 rounds, snake)

Everything is rebuilt from `data/fntsy_2026-2027.xlsx` + `config.json`. Needs Python 3 with `numpy` and `openpyxl`.

## Run order
| Step | Command | Output |
|---|---|---|
| 1. Read the spreadsheet | `python src/extract_data.py` | `data/players.json` |
| 2. Category scarcity by round | `python src/analysis_scarcity.py` | `results/scarcity.json` |
| 3. Strategy per slot + round plans | `python src/build_plans.py` (~30 min; `--quick` to test) | `results/plans.json` |
| 4. Best first-four combos | `python src/build_foursomes.py` (all slots ~1 hr) or `... 8 9` | `results/foursomes.json` |
| 5. Build the pages | `python src/build_pages.py` | `dist/draft-board.html`, `dist/draft-plans.html` |
| Live during the draft | `python src/next_pick.py --roster "A,B" --pick 24 --gone "C,D"` | ranked Fit list |

Results already in `results/` were produced by these scripts; `dist/` is the built pages (open in any browser).

## Settings (`config.json`)
- `teams`, `rounds`, `pool` (z-scores use the top-N players by projected rank), `reach` (max picks past ADP you'll take a player).
- `first_four_exclude`: names you will NOT draft in the first 4 rounds (exact names from `data/players.json`). Used by `build_foursomes.py`; re-run it, then `build_pages.py`.
- The sheet's own "Do not draft" column is always excluded everywhere.

## Method in one paragraph
z = (stat - mean) / sd over the top-130 by projected rank (FG%/FT% as makes above an average shooter on the same attempts; TO flipped). Team z = average of your players' z. Par(n) = average z of the top 10n players by ADP (what an average team holds after n picks). Fit = sum over categories of player z x weight, weight = clip(1 - 0.5 x (your avg z - par), 0.4, 1.8): categories where you're behind par count more. Skip players whose ADP is > `reach` picks after your pick.
First four: enumerate every reachable 4-player set assuming others take best ADP, shortlist, simulate the remaining 9 rounds (opponents pick ADP + noise; you follow Fit) and score expected categories won out of 9. Absolute numbers are inflated; compare only. Gaps under ~0.1 are ties.

## Files
- `src/draftlib.py` model + simulator; `web/*_template.html` page sources; `data/`, `results/`, `dist/` as above.
- Note: published board data now labels the five R1-2 targets (Jokic, Tatum, Edwards, Mitchell, Haliburton) as "1-2" (previously "null").

## 6-pick planner (tree + top combos)
- `python src/build_tree.py` -> `results/tree_10team.json`: top-10-ADP anchors (slot k = k-th by ADP), 5 ranked children per node for rounds 2-6 (3,125 paths/anchor).
- `python src/build_top.py` -> `results/top_10team.json`: best 10 combos per slot, re-scored on fresh draws.
- `python src/build_pages.py` also writes `dist/draft-tree.html` (tree + top-combos tabs).
- Ranking uses ONLY the picks made so far (`play_prefix`: your k players vs each opponent's k players, win % per category). Later rounds are not assumed. "Emphasize later" = categories where your players sit furthest behind par; "typical fill" is guidance from simulating the Fit rule and does not affect ranking.
- `do_not_draft` (config) = players you never pick; opponents still take them.
- `results/plan6.json` / `build_plan6.py` is an earlier beam-search version (ranked with an assumed fill).

## Season model (calibrated to the league's real standings)
- `data/games_played.csv` -> projected games/minutes appended to each `players.json` row (cols 16-17). `ZS` in `draftlib.py` = z-scores on season totals (per-game x games).
- `play(..., season=make_season_world(seed, sig))` simulates season totals with per-team "volume" multipliers (games played / active roster / streaming) and noisy weekly category results. Parameters (`SEASON` in `draftlib.py`) are fitted by `python src/calibrate_season.py` to the 10-team standings (spreads and correlations match; draft-survival `gamma=0.5` is an assumption the data cannot pin down).
- Default `play()` behaviour (no `season=`) is unchanged, so older results still reproduce.
- `make_season_world(seed, sig, fixed=True)`: rosters stay as drafted all season (no volume factor, no waiver fade); totals = per-game x projected games; only weekly matchup noise remains (fixed absolute size = real spread x fitted k). This is the mode to use for draft strategy.
