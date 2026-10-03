# Football Player/Game analysis
The project attempts to 
- analyze football players using fbref data and machine learning (fbref_test.ipynb).
- Use whoscored/fotmob data to visualize/analyze football game results and stats (model template.ipynb)

The former currently analyzes a specifc player statistically, and then uses ML to find 20 players that are the "most similar" statistically.
The latter uses whoscored data to analyze game shotmaps, pass networks which shows the trend of each team's game flow, and uses fotmob data to visualize general game stats like xG, possession, etc which
indicates general facts about each team's style.

# サッカー選手/試合分析
このプロジェクトは、以下のことを目指します。
- fbrefデータと機械学習を用いてサッカー選手を分析する (fbref_test.ipynb)。
- whoscored/fotmobデータを用いてサッカーの試合結果と統計を視覚化・分析する (model template.ipynb)。

前者は現在、特定の選手を統計的に分析し、その後、機械学習を用いて統計的に「最も類似」する20人の選手を特定します。
後者は、whoscoredデータを用いて試合のシュートマップやパスネットワークを分析し、各チームの試合の流れの傾向を示します。また、fotmobデータを用いて、xGやポゼッションといった各チームのスタイルに関する一般的な事実を示す一般的な試合統計を視覚化します。

## Desktop app (`footy_desktop_app.py`)

A Qt (PySide6) desktop UI for player profiles and similar-player search. It
runs the Sofascore + touch/action-map pipeline in `sofascore_similarity.py`
(WhoScored maps for clustering, Sofascore league stats for quantitative
similarity). The code is pure Python/Qt with no OS-specific paths, so the
same script runs unmodified on Linux, macOS, and Windows — only the
setup/launch commands differ.

### Required libraries

Listed in [`requirements-desktop.txt`](requirements-desktop.txt), same on
every platform:

- PySide6 (Qt bindings for the UI)
- matplotlib, mplsoccer, pyfonts (pitch maps / plots rendered to images)
- numpy, pandas, polars, scipy, scikit-learn (data processing and similarity search)
- soccerdata, lxml (data readers used by the analysis pipeline modules)

Python 3.12 is required: the pinned `numpy` and `lxml` versions publish no
wheels for 3.13+, so pip would fall back to compiling them from source.

### Setup

**Linux / macOS** (bash/zsh terminal)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-desktop.txt
```

**Windows** (PowerShell or Command Prompt)
```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-desktop.txt
```

### Running

**Linux / macOS**
```bash
./run_footy_desktop.sh
```

**Windows**
```bat
run_footy_desktop.bat
```

Both scripts `cd` into the repo root and launch `footy_desktop_app.py` using
the interpreter from `.venv`, so they can be run (or double-clicked, on
Windows) from anywhere.

### Search filters

The controls above both tabs narrow the search:

| Filter | Stage it affects |
| --- | --- |
| Player / Team / League | resolves which player is the target |
| Season / Target season | which season the target player is taken from |
| Search seasons (Similar tab) | which seasons matches may come from; none ticked = every season |
| Hide target's other seasons | keeps the target's own other seasons out of the results |
| Min touches / Clusters / Use action features | the KMeans style clustering itself |
| Min 90s | drops low-minute players from the comparison pool |
| Age from / Age to | drops out-of-range players from the returned matches |
| Include unknown age | keeps or drops players with no known birth date |
| Top N | how many matches to return |
| Physicality | shows each player's physicality profile (see below) |
| Match physicality (Similar tab) | adds closeness on the physicality axes as one more similarity ranker |
| Min phys pct (Similar tab) | only returns players at or above this physicality percentile in their role |

### Seasons

Every profile and every match is a *player-season* (`season_similarity.py`),
so the same player can appear once per season, and the results list says which
season each row is. For example, target Mesut Özil in 15/16 with only 25/26
ticked to find his closest current equivalents, or target Arda Güler in 25/26
with nothing ticked to search every season since 13/14.

The target's season and the searched seasons are clustered together on
touch/action shape, which the WhoScored event data records the same way in
every season. The stats ranking then uses only the Sofascore stats that
*every* season in the pool records: Sofascore's older seasons are thinner
(13/14 and 14/15 have little beyond goals, assists and cards; expected goals
and assists start in 22/23), and a stat a season lacks would otherwise read as
zero. The profile's "Ranked on" line lists what was left out. When fewer than
15 stats are shared, as with anything involving 13/14 or 14/15, matches are
ranked by touch/action-shape distance instead. So that one thin season does
not reduce every all-seasons search to shape only, 13/14 and 14/15 are left
out of a search whose target is from a richer season (the profile says so);
they are searched, on shape, when the target is from one of them or they are
the only seasons ticked. Ages are the player's age in that season.

Each season's shape vectors are built once, taking a few minutes per season,
and are then cached. Run `python season_similarity.py --warm` to build them all
up front. The same search is available on the command line:

```bash
python season_similarity.py --player "Ozil" --team Arsenal --season 2015 --scope 2025
python season_similarity.py --player "Arda Guler" --season 2025          # every season
```

`Age from` / `Age to` both read "Any" at 0, so either end can be left open.
Unlike `Min 90s`, the age range is applied to the ranked results rather than to
the comparison pool — percentiles, role scores and RRF similarity values are
identical to an unfiltered run, and `Top N` counts only age-eligible players.
Age is never a clustering or similarity input; style clusters stay age-blind.

### Player ages

Age comes from each player's Sofascore birth date, stored in the
`date_of_birth` column of `sofascore_player_stats.csv` and converted to an age
on load, so a cached CSV never reports stale ages. A fresh `--scrape` includes
it. To add birth dates to a CSV scraped before this column existed:

```bash
python sofascore_similarity.py --scrape-ages
```

That makes one request per unique player and caches results in
`sofascore_player_dob.json`, so an interrupted run resumes where it stopped.
Until it has been run, ages show as `-` and the age filter has nothing to match
on. Scraping needs `ScraperFC` installed (see `requirements-desktop.txt`);
Sofascore rejects plain HTTP clients, so the request goes through ScraperFC's
browser-backed getters.

The same filters exist on the CLI:

```bash
python sofascore_similarity.py --player "Bruno Fernandes" --with-maps \
    --min-age 21 --max-age 25
```
### Tight Space tab

The third tab runs the tight-space model in `tight_space.py`: a league-wide
congestion field (defensive actions per touch, per pitch cell), a situational
difficulty model for P(lose the ball), and per-axis scores for how a player
actually performs once the situation is congested — all z-scored against their
own role group, so a full-back is compared with full-backs.

Press **Load / Build scores** to score a season. The first build for a given set
of settings reads every cached match csv for the selected leagues and fits the
models, which takes several minutes and a few GB of RAM; the resulting table is
written to `.desktop_cache/tight_space/` and every later run with the same
settings loads it instantly. Tick **Force rebuild** to refit anyway, or set
**Max matches** for a quick trial run on a slice of the season.

| Control | What it changes |
| --- | --- |
| Leagues / Season | which cached events are pooled into the field and the peer set |
| Tight quantile | the difficulty percentile above which a situation counts as tight |
| Min tight actions | sample a player needs before they are scored at all |
| Max matches | caps matches loaded, for a fast trial |
| Rank by | ranks on one axis, or on the summary blend, exposure or sample |
| Role | `DF`/`MF`/`FW`, a line and side (`MF-C`, `DF-W`), or a side alone (`C`/`W`) |
| Blend weights | re-fuses the axes into the summary score — instant, nothing is refitted |

Selecting a row shows that player's card: exposure (how many of their on-ball
actions happen in congested situations, and where that ranks in their role), the
three axes with their percentile in role, and the underlying components with
their sample sizes. **Diagnostics** reports the split-half reliability of every
axis and component, the axis correlations, and the score's spread across role
groups.

Read an axis before the summary blend. The axes are near-independent — retention
against beating the man correlates at r = -0.13 — so the blend across them is a
stated preference rather than a measured trait, which is why its weights are
adjustable. Scores are execution given what the player chose to attempt, so they
credit nothing for volume; `tight share` carries how often a player is in there
at all.

Once a table is loaded, the Player Profile and Similar Players tabs also show a
compact tight-space panel for any player who cleared the minimum sample.

The same model on the CLI:

```bash
python tight_space.py --leagues "ENG-Premier League" --top 25
python tight_space.py --sort-by axis_beating_man --role MF-C --top 20
python tight_space.py --player "Bruno Fernandes" --validate
```

## Physicality (`physicality.py`)

A per-position physicality profile, built the same way as the tight-space model:
a few near-independent axes, each z-scored within the player's role group
(formation line plus central/wide, so a centre-back is compared with
centre-backs), with a summary blend on top.

There is no height, weight or tracking data, so "physical" is read from contested
balls. WhoScored logs every duel as an adjacent pair of events that names both
players and the winner (`Aerial`/`Aerial`, `Tackle` against `TakeOn` or
`Dispossessed`, `Challenge` against a successful `TakeOn`). Duel strength is
therefore a **Bradley-Terry rating** fitted over every head-to-head in the
season: P(i beats j) = sigmoid(s_i − s_j + context). That credits a striker for
beating centre-backs in the air rather than penalising them for losing to them.
Within role groups the rating replicates better across odd/even matches than a
raw aerial win rate does (r 0.54 against 0.45 on 25/26).

| Axis | What it measures |
| --- | --- |
| Aerial duels | opponent-adjusted aerial rating, with pitch location as context |
| Ground duels | opponent-adjusted rating as the *tackler* in ground duels |
| Taking contact | fouls drawn above what the player's touch locations predict |
| Engagement | volume: aerials, tackling duels and fouls committed per 90 |
| Carrying | ground covered with the ball: progressive carry metres and 15m+ carries, possession-adjusted |
| Work rate | defensive range (spread of defensive actions, in metres) and defensive actions in the opposition half, possession-adjusted |

**Running data.** None of the sources here has any: no distance covered,
sprints or speeds. Carrying and work rate stand in for it. A carry is
reconstructed from consecutive events, running from where a player got the ball
(a team-mate's pass end, or their own recovery or touch) to where their next
action starts. Both are possession-adjusted to 50% ("PAdj"), so a full-back at a
dominant side does not out-carry everyone just because the team has the ball. Two
other proxies were tested and dropped. **Carry speed** (distance over the gap
between timestamps) fails because WhoScored clocks are whole seconds. It
replicated at ~0.4, and its "fastest" players were Füllkrug and Mühren.
**Late-game output** as a stamina proxy replicated at ~0.3. Speed, sprint counts
and total distance need tracking data, e.g. the league or club physical reports.

The carrier's side of ground duels is left out on purpose: it is mostly take-ons,
and its leaderboard (Yamal, Doku, Mbappé, Vinícius) is dribbling skill, which the
tight-space "beating the man" axis already covers.

The summary blend weights the axes **by position**, e.g. aerial 0.35 and ground
0.25 for a centre-back, aerial 0.40 and contact 0.25 for a central striker,
carrying 0.20 for full-backs and wingers, and ground 0.30 with work rate 0.20 for
a central midfielder (`ROLE_AXIS_WEIGHTS`). Those weights are a stated
preference, not a measurement. The axes themselves are measured the same way for
everyone. Z-scores are capped at ±3 before blending, so a single extreme axis
cannot decide the summary on its own. Alexander-Arnold's 40% tackling-duel win
rate in 23/24 is −5.8 SD among full-backs, for example.

Measured on 25/26, across all seven leagues:

| | split-half r (within role) | Spearman-Brown |
| --- | --- | --- |
| Aerial duels | 0.43 | 0.61 |
| Ground duels | 0.24 | 0.38 |
| Taking contact | 0.72 | 0.84 |
| Engagement | 0.71 | 0.83 |
| Carrying | 0.76 | 0.86 |
| Work rate | 0.47 | 0.64 |
| Summary blend | 0.55 | 0.71 |

The axes are close to independent. The largest correlation is work rate with
engagement (r = 0.33), which is expected, since both count defensive actions. Against independent Sofascore
stats, within role, each axis tracks the stat it should: aerial vs aerial duels
won % r = 0.67, contact vs fouled per 90 r = 0.70, engagement vs duels per 90
r = 0.61, ground vs ground duels won % r = 0.35. Ground duels is the weakest axis.
Most players have only ~70 tackling duels a season, so treat its percentile as
indicative.

The model measures who wins contests and how often a player seeks them, not
temperament. A player whose reputation for being physical comes from aggression
rather than winning duels can score lower than expected.

Scores are per season, cached under `.desktop_cache/physicality/` (1–2
minutes per season to build). `python season_similarity.py --warm` builds them
along with the shape vectors, or `python physicality.py --warm` builds just these.
Players under 600 minutes in a season are not scored.

In the app, the Player Profile and Similar Players tabs show the profile for any
scored player, and the results list has a **Phys** column (percentile in role).
**Match physicality** adds closeness on the six axes as one more ranker in the
RRF fusion. **Min phys pct** is a filter like the age range: similarity scores are
unchanged, only who is returned.

```bash
python physicality.py --season 2025 --top 25
python physicality.py --role DF-C --sort-by axis_aerial --top 20
python physicality.py --player "Virgil van Dijk"
python physicality.py --validate --external-check
python season_similarity.py --player "Diego Costa" --season 2015 --match-physicality --min-phys-pct 60
```

## Low blocks, and what breaks them (`low_block.py`)

Finds the phases where a team is sitting in a low block, then asks which
attacking patterns actually produce chances against one. Runs on the cached
WhoScored events under `league_games/` — 2,323 games across seven leagues.

```bash
python low_block.py --all-leagues --plot --out low_block_sequences.csv
python low_block.py --league "ENG-Premier League" --clusters 6 --plot
```

### How a low block is identified

There is no tracking data here, so defensive shape is inferred from where the
defending team's *ball events* happen. Ranked over a season the estimate puts
West Ham, Wolves and Burnley deepest and Arsenal, City and Brighton highest, so
it is measuring something real — but it can never see an off-ball player, and
"block" throughout means "where this team is engaging the ball", not a literal
defensive line.

Three conditions have to hold at the moment an attack enters the final third:

- **deep** — the defence's last 10 defensive actions average in the bottom third
  of the league's own distribution;
- **passive** — at least median opponent passes allowed per defensive action. A
  deep block that is *also* passive is a side choosing to sit. A deep block with
  a low count is a side being besieged, which is a different phase;
- **settled** — the attack managed 3+ final-third passes or 8+ seconds there, so
  a counter that arrived before the defence was set does not count.

That leaves 12,195 settled attacks against a low block.

### Shot xG

`match_xg/` holds Understat match totals, which are one number per team per game.
Judging whether a block was *broken* needs a value per chance, so `fit_xg` fits a
shot-level model on the event data itself, out-of-fold by match. It predicts 6,518
xG against 6,505 actual goals, and calibrates to within a point across every
distance band. `BigChance` and `OneOnOne` are excluded from the features — they
are judgements made after the fact — and kept only as descriptors.

### What it finds

The analysis is deliberately split in two, because a single table ranking
features by xG is close to circular: "played a pass into the box" wins it, and
that is most of the way to a chance by definition.

**Stage 1 — getting into the box**, scored on reach rate, using only build-up
features. Switching play is far and away the strongest: one switch takes an
attack from a 28% to a 58% chance of reaching the box, two or more from 34% to
65%. Take-ons (0.95×) and lay-offs (0.76×) are *below* baseline — dribbling at a
low block does not work.

**Stage 2 — what the delivery is worth**, conditioned on having reached the box,
so deliveries are ranked against each other rather than against failure. Through
balls are worth most by a distance (0.151 xG per attack) and are blocked only 8%
of the time; cutbacks 0.085 but 39% blocked; crosses 0.068; carrying it in is
worst at 0.029 and 55% blocked.

The clusters make the same point: "patient, switching, cutback, box-occupying"
returns 0.083 xG per attack over 1,697 of them, while plain patience without the
box occupation returns 0.015 despite spending nearly twice as long in the final
third. Time on the ball is not the variable; what you do with the width is.

One negative result worth keeping: a low block does **not** degrade chance
quality. xG per shot is 0.095 against a low block and 0.095 against a high one.
What changes is the route in, not the value of what arrives.

### Consecutive-pass patterns

`pass_chains` keeps each attack's passes in order, mirrored so every attack
enters the final third on the same ("near") side — left- and right-sided versions
of one move then pool into a single pattern, while the relative movement that
matters survives intact.

Ranking whole three-pass motifs is partly circular: a motif ending in the middle
of the box has already got the ball into the box, and that is most of what
produces xG. `route_comparison` holds the destination fixed and varies only the
route, over the 955 low-block attacks whose last pass lands in the middle of the
box:

| penultimate pass played from | xG per attack |
|---|---|
| inside the box, near half-space | 0.238 – 0.283 |
| inside the box, near wing / byline | 0.121 – 0.143 |
| outside the box, wide | 0.068 – 0.111 |

Same final ball, same destination, and the half-space origin is worth roughly
twice the byline one and three times a wide ball from outside the box. That is
the single most actionable number in this analysis: against a low block it is not
the cutback that creates the chance, it is *where the cutback is played from*.

### Figures

`--plot` writes seven figures to `low_block_plots/`:

| file | what it shows |
|---|---|
| `block_shape.png` | where each kind of block engages the ball |
| `build_up.png` | stages 1 and 2 side by side, with standard errors |
| `delivery.png` | shot-origin density by how the ball entered the box |
| `archetypes.png` | the clusters as pitch maps, ranked by xG produced |
| `motifs.png` | the highest-value three-pass patterns, each drawn as a path |
| `chain_examples.png` | six real moves, numbered pass by pass |
| `teams.png` | who breaks low blocks, and who sits in one that holds |
