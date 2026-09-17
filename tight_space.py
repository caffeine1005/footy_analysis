"""Rate players on how well they operate in tight spaces — congested areas with
opponents close enough to contest the ball.

The honest constraint this module is designed around: WhoScored/Opta event data
has no freeze frames, no tracking, and no `under_pressure` flag, so the actual
distance to the nearest opponent is *not* recoverable. Trying to infer it from
nearby opponent event coordinates fails badly — only ~9% of on-ball events have
a non-duel opponent event within 2s, the implied median distance comes out at an
implausible ~12m (you only ever see defenders who did something recordable), and
worst of all the feature is endogenous, because a `Clearance`/`Interception`/
`BlockedPass` next to your touch usually *is* the touch failing. Pass completion
across distance buckets is essentially flat (18/26/25/28/28%) while all the
signal sits in whether an opponent intervened at all, i.e. in the outcome.

So instead of measuring distance, this measures **congestion** and grades
performance against it:

  1. A league-wide congestion field: defensive actions per on-ball touch, per
     pitch cell. Structural, exogenous, and steep — the central penalty box runs
     ~2.7 defensive actions per touch against ~0.12 in deep wide areas.
  2. Situational tightness on top of location: how long the possession has been
     running (settled play against a set block is tighter than transition) and
     how deep/high the opponent's defensive block is currently engaging.
  3. A `difficulty` model — P(lose the ball) from *situation only*, no player
     identity and no action choice. Its calibrated output is the tightness score
     each event is bucketed on.
  4. Sub-skill models that add the player's *intent* (action type, pass length
     and direction, body part). Residuals against these measure execution given
     what the player chose to attempt, which is what removes the role confound:
     a centre-back's safe sideways ball and a striker's forward flick in the box
     no longer share one baseline.

The output is a profile across three axes, not one rating, because the data does
not support one rating. Measured over 1,870 players the axes are mutually
independent to slightly negative — retention against beating the man is r =
-0.13, retention against winning contact r = -0.28 — and the first principal
component of the underlying components explains only 36% of the variance. An
earlier single composite scored *worse* on split-half reliability (r = 0.27) than
its own best component (0.55), which is what averaging near-independent signals
does. "Good in tight spaces" is several separable abilities that trade off, so
`--sort-by` an axis is usually more meaningful than the summary blend.

Capability and exposure are reported separately and deliberately. Conditioning on
intent means a player gets no credit here for *choosing* hard actions, only for
executing them — so `tight_share` (how much of their game happens in congested
situations) carries that half of the picture. This is also the salvageable part
of "how close do opponents get": how *often* a player works in the league's most
contested areas is measurable even though how close is not.

Roles come from the event data itself, not a name join. `FormationSet` /
`FormationChange` events carry parallel `InvolvedPlayers` and `PlayerPosition`
arrays where the position is a line index (1=GK, 2=DF, 3=MF, 4=FW, 5=bench),
giving essentially complete coverage, then each line is split central/wide so
wingers are not benchmarked against holding midfielders. A name join against
`sofascore_player_stats.csv` only hits ~58% and is used, optionally, for
validation only.

Everything is validated rather than asserted: `--validate` reports the difficulty
model's AUC and calibration, split-half reliability per axis and component, the
axis correlations, and the score's flatness across role groups.

CLI:

    python tight_space.py --leagues "ENG-Premier League" --top 25
    python tight_space.py --sort-by axis_beating_man --role MF --top 20
    python tight_space.py --player "Bruno Fernandes"
    python tight_space.py --validate --external-check --out tight_space_scores.csv
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from touchmap_similarity import DEFAULT_LEAGUES, DEFAULT_SEASON, load_all_league_events

BINS = (12, 8)  # x_bins, y_bins across the opta 0-100 pitch, matching touchmap_similarity

# Periods that are actual football, with a cumulative-minute clock. WhoScored's
# `minute` already runs continuously across halves (SecondHalf spans 45-96), so
# minute*60 + second is a valid match-wide timestamp without per-period offsets.
PLAY_PERIODS = ("FirstHalf", "SecondHalf", "FirstPeriodOfExtraTime", "SecondPeriodOfExtraTime")

# On-ball actions where a player is trying to keep or use the ball. Shots are
# excluded: a blocked shot in a crowded box is not a failure to handle tight
# space, and including them punishes strikers for doing their job.
ATTACKING_ON_BALL = ("Pass", "TakeOn", "BallTouch", "Dispossessed")

# Actions that mark a defender actively contesting. These are what the
# congestion field counts, and what the block-height estimate is built from.
DEF_ACTION_TYPES = (
    "Tackle", "Challenge", "Interception", "Clearance", "BlockedPass", "Aerial", "Foul",
)

# Restarts are not open play; a throw-in taken unopposed tells us nothing about
# playing under congestion, and set pieces would badly distort the box cells.
SET_PIECE_QUALIFIERS = (
    "ThrowIn", "FreekickTaken", "IndirectFreekickTaken", "DirectFreekick",
    "CornerTaken", "GoalKick", "KeeperThrow", "Penalty", "ThrowinSetPiece", "SetPiece",
)

LINE_TO_ROLE = {1: "GK", 2: "DF", 3: "MF", 4: "FW"}

# Each formation line is split into a central and a wide half at the line's own
# median lateral distance from the centre, rather than at a fixed cut. The lines
# alone are too coarse to compare like with like: a 4-3-3 winger is recorded on
# line 3, so without a split Salah is benchmarked against central midfielders,
# whose take-on volume and congestion profile are nothing like his.
#
# A within-line median is used because a fixed threshold does not survive contact
# with the data. A left centre-back in a back four sits at |y-50| ~ 18-20 and a
# full-back at ~32-38, so any constant lands on top of the centre-backs and split
# them arbitrarily: a cut at 21 put only 26 of 111 defenders in the central group.
# The median is also structurally justified — a back four is two central and two
# wide players by construction — and self-calibrates to each league's formation mix
# while guaranteeing both peer groups are large enough to z-score against.
MIN_ROLE_GROUP = 20

# Rolling window (in defensive actions) used to estimate where a team is
# currently defending. Short enough to track a shift in block height within a
# half, long enough not to swing on a single clearance.
BLOCK_WINDOW = 12

SITUATION_FEATURES = [
    "x", "y", "congestion", "poss_t", "poss_idx", "block_height", "time_on_ball", "att_third_t",
]


def _has_qualifier(series: pd.Series, names) -> pd.Series:
    """Substring-test the raw qualifier string rather than literal_eval'ing 3.4M rows.

    Qualifiers are stored as stringified Python lists of dicts, so parsing them
    properly is ~100x slower than a regex over the raw text and buys nothing for
    presence checks.
    """
    if isinstance(names, str):
        names = (names,)
    pattern = "|".join(f"'displayName': '{re.escape(n)}'" for n in names)
    return series.fillna("").str.contains(pattern, regex=True)


def _qualifier_value(series: pd.Series, name: str) -> pd.Series:
    """Pull a single qualifier's numeric value out of the raw string."""
    pattern = rf"'displayName': '{re.escape(name)}'[^}}]*}}, 'value': '([-\d.]+)'"
    return pd.to_numeric(series.fillna("").str.extract(pattern, expand=False), errors="coerce")


# ---------------------------------------------------------------------------
# roles, straight out of the formation events
# ---------------------------------------------------------------------------


def extract_roles(events: pd.DataFrame) -> pd.DataFrame:
    """Map each player_id to a role via `PlayerPosition` on formation events.

    `InvolvedPlayers` and `PlayerPosition` are comma-separated parallel arrays on
    every `FormationSet`/`FormationChange` event. Bench entries (line 5) are
    dropped so a substitute is typed by the line they actually played in, and a
    player's role is the modal line across every snapshot they appear on-pitch in.
    """
    fm = events[events["type"].isin(("FormationSet", "FormationChange"))]
    ids = fm["qualifiers"].str.extract(
        r"'displayName': 'InvolvedPlayers'[^}]*}, 'value': '([\d,]+)'", expand=False
    )
    pos = fm["qualifiers"].str.extract(
        r"'displayName': 'PlayerPosition'[^}]*}, 'value': '([\d, ]+)'", expand=False
    )
    ok = ids.notna() & pos.notna()

    pairs = []
    for id_str, pos_str in zip(ids[ok], pos[ok]):
        pid = [int(v) for v in id_str.split(",") if v.strip()]
        lines = [int(v) for v in pos_str.split(",") if v.strip()]
        pairs.extend(zip(pid, lines))
    if not pairs:
        return pd.DataFrame(columns=["player_id", "role"])

    fr = pd.DataFrame(pairs, columns=["player_id", "line"])
    fr = fr[fr["line"].isin(LINE_TO_ROLE)]
    role = (
        fr.groupby("player_id")["line"]
        .agg(lambda s: s.mode().iloc[0])
        .map(LINE_TO_ROLE)
        .rename("role")
        .reset_index()
    )
    return role


# ---------------------------------------------------------------------------
# congestion field
# ---------------------------------------------------------------------------


@dataclass
class CongestionField:
    """League-wide defensive-action intensity per pitch cell.

    `per_touch` is defensive actions per on-ball touch in that cell — a measure
    of how heavily contested ball activity there is, independent of how much
    activity happens. Defensive actions are mirrored into the attacking team's
    frame first (Opta stores every team attacking toward x=100, so an opponent's
    event at (x, y) is physically at (100-x, 100-y) from the other side).
    """

    per_touch: np.ndarray
    def_counts: np.ndarray
    touch_counts: np.ndarray
    bins: tuple[int, int]

    def lookup(self, x: pd.Series, y: pd.Series) -> np.ndarray:
        gx, gy = cell_index(x, y, self.bins)
        return self.per_touch[gx, gy]

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_touch).round(2)


def cell_index(x, y, bins: tuple[int, int] = BINS) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = bins
    gx = np.clip((np.asarray(x, dtype=float) / 100.0 * nx).astype(int), 0, nx - 1)
    gy = np.clip((np.asarray(y, dtype=float) / 100.0 * ny).astype(int), 0, ny - 1)
    return gx, gy


def build_congestion_field(events: pd.DataFrame, bins: tuple[int, int] = BINS) -> CongestionField:
    """Count defensive actions per on-ball touch in each cell, pooled over all matches."""
    nx, ny = bins
    on_ball = events[events["type"].isin(ATTACKING_ON_BALL) & events["x"].notna()]
    defn = events[events["type"].isin(DEF_ACTION_TYPES) & events["x"].notna()]

    tgx, tgy = cell_index(on_ball["x"], on_ball["y"], bins)
    dgx, dgy = cell_index(100.0 - defn["x"], 100.0 - defn["y"], bins)

    touch_counts = np.zeros((nx, ny))
    def_counts = np.zeros((nx, ny))
    np.add.at(touch_counts, (tgx, tgy), 1)
    np.add.at(def_counts, (dgx, dgy), 1)

    per_touch = np.divide(
        def_counts, touch_counts, out=np.zeros_like(def_counts), where=touch_counts > 0
    )
    return CongestionField(per_touch, def_counts, touch_counts, bins)


# ---------------------------------------------------------------------------
# event annotation
# ---------------------------------------------------------------------------


def _annotate_match(g: pd.DataFrame) -> pd.DataFrame:
    """Add possession structure, block height, and retention outcome for one match."""
    g = g.sort_values(["minute", "second"], kind="stable").reset_index(drop=True)
    g["t"] = g["minute"] * 60 + g["second"].fillna(0)

    # Possession spells: a new one starts whenever the team in control changes.
    team = g["team_id"]
    g["poss"] = (team != team.shift()).cumsum()
    g["poss_t"] = g["t"] - g.groupby("poss")["t"].transform("min")
    g["poss_idx"] = g.groupby("poss").cumcount()

    # Retention: does the acting team still have the ball two events later? Two
    # rather than one, because the very next event is often the opponent's failed
    # contest (a lost Challenge) while possession never actually changed hands.
    nxt1 = team.shift(-1)
    nxt2 = team.shift(-2)
    kept_possession = (nxt1 == team) | (nxt2 == team)
    action_worked = g["outcome_type"].eq("Successful") & g["type"].ne("Dispossessed")
    g["lost"] = (~(action_worked & kept_possession)).astype(int)

    # Time the player had before acting, as a proxy for being closed down. Only
    # 1s granularity so it is noisy per event but informative in aggregate.
    g["time_on_ball"] = (g["t"] - g["t"].shift()).clip(0, 15).fillna(0)

    # How long the ball has been in the final third this possession — sustained
    # attacking-third play means a set, compact block rather than a fast break.
    in_att_third = g["x"] >= 66.7
    g["att_third_t"] = (
        g.assign(_a=np.where(in_att_third, g["t"], np.nan))
        .groupby("poss")["_a"]
        .transform(lambda s: s.ffill().pipe(lambda v: v - v.min() if v.notna().any() else v))
        .fillna(0)
        .clip(0, 120)
    )

    # Block height: where each team is currently engaging, expressed in the
    # *opponent's* attacking frame so it reads as "how high up the pitch, from
    # the attacker's point of view, is the defence winning the ball". Rolling
    # over prior defensive actions only, and shifted, so it never sees the
    # current event's outcome.
    defn = g[g["type"].isin(DEF_ACTION_TYPES) & g["x"].notna()]
    g["block_height"] = np.nan
    if not defn.empty:
        for def_team, d in defn.groupby("team_id"):
            height = (100.0 - d["x"]).rolling(BLOCK_WINDOW, min_periods=3).mean().shift()
            ref = pd.DataFrame({"t": d["t"].to_numpy(), "block_height": height.to_numpy()}).dropna()
            if ref.empty:
                continue
            targets = g.index[g["team_id"] != def_team]
            if len(targets) == 0:
                continue
            merged = pd.merge_asof(
                pd.DataFrame({"t": g.loc[targets, "t"].to_numpy()}, index=targets).sort_values("t"),
                ref.sort_values("t"),
                on="t",
                direction="backward",
            )
            g.loc[merged.index, "block_height"] = merged["block_height"].to_numpy()
    return g


def annotate_events(events: pd.DataFrame, field: CongestionField) -> pd.DataFrame:
    """Filter to open-play attacking on-ball actions and attach every situational feature."""
    ev = events[events["period"].isin(PLAY_PERIODS)].copy()
    ev = ev[ev["team_id"].notna() & ev["x"].notna()]

    parts = [_annotate_match(g) for _, g in ev.groupby("game_id", sort=False)]
    ev = pd.concat(parts, ignore_index=True)

    ev["is_set_piece"] = _has_qualifier(ev["qualifiers"], SET_PIECE_QUALIFIERS)
    ev["congestion"] = field.lookup(ev["x"], ev["y"])
    # A neutral fill (halfway line) for the opening minutes before a block can
    # be estimated, so early events are not silently dropped by the models.
    ev["block_height"] = ev["block_height"].fillna(50.0)
    return ev


def open_play_actions(ev: pd.DataFrame) -> pd.DataFrame:
    """The population every score is computed over: open-play attacking on-ball actions."""
    return ev[
        ev["type"].isin(ATTACKING_ON_BALL)
        & ~ev["is_set_piece"]
        & ev["player_id"].notna()
    ].copy()


def refine_roles(roles: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Split each formation line into central and wide using where the player touches the ball.

    Adds a `role_group` (e.g. `DF-W` for a full-back, `MF-W` for a winger) used as
    the peer set for every z-score and percentile, while `role` is kept for display.
    """
    lateral = (
        actions.assign(_dev=(actions["y"] - 50.0).abs())
        .groupby("player_id")["_dev"]
        .median()
        .rename("lateral_dev")
        .reset_index()
    )
    out = roles.merge(lateral, on="player_id", how="outer")
    out["role"] = out["role"].fillna("UNK")

    # Only players with enough of a sample to place get split; the rest keep the
    # bare line so they are never assigned to a peer group on a handful of touches.
    med = out.groupby("role")["lateral_dev"].transform("median")
    side = np.where(
        out["lateral_dev"].isna() | med.isna(), "",
        np.where(out["lateral_dev"] >= med, "-W", "-C"),
    )
    out["role_group"] = out["role"] + side
    return out


# ---------------------------------------------------------------------------
# difficulty and expected-value models
# ---------------------------------------------------------------------------


def _fit_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int = 42) -> np.ndarray:
    """Out-of-fold P(y=1), grouped by match.

    Predictions must be out-of-fold: an in-sample baseline partly memorises the
    very players it is meant to be a neutral yardstick for, which shrinks exactly
    the residuals this module reports. Grouping by match keeps both halves of a
    contest on the same side of the split.
    """
    oof = np.zeros(len(X))
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        model = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
        model.fit(X, y)
        return model.predict_proba(X)[:, 1]

    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = HistGradientBoostingClassifier(
            random_state=seed, max_iter=200, learning_rate=0.08, min_samples_leaf=50
        )
        model.fit(X.iloc[train], y[train])
        oof[test] = model.predict_proba(X.iloc[test])[:, 1]
    return oof


def add_difficulty(actions: pd.DataFrame) -> pd.DataFrame:
    """Score each action's situational difficulty: P(possession lost | situation).

    Deliberately blind to who the player is and to what they chose to do, so the
    output measures the situation rather than the response to it. That makes it
    usable both as the tightness axis events are bucketed on and as an exposure
    measure.
    """
    X = actions[SITUATION_FEATURES]
    actions["difficulty"] = _fit_oof(X, actions["lost"].to_numpy(), actions["game_id"].to_numpy())
    return actions


def difficulty_diagnostics(actions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Check the tightness axis is real: does predicted difficulty track actual loss rate?

    If this came back flat, every score in the module would be bucketing events on
    noise, so it is worth reporting rather than assuming.
    """
    from sklearn.metrics import roc_auc_score

    d = actions[["difficulty", "lost", "congestion", "block_height", "poss_t"]].dropna()
    d = d.assign(bucket=pd.qcut(d["difficulty"], n_bins, labels=False, duplicates="drop"))
    tab = (
        d.groupby("bucket")
        .agg(n=("lost", "size"), predicted=("difficulty", "mean"), actual=("lost", "mean"),
             mean_congestion=("congestion", "mean"), mean_block_height=("block_height", "mean"))
        .reset_index()
    )
    tab.attrs["auc"] = roc_auc_score(d["lost"], d["difficulty"])
    return tab


def _pass_features(passes: pd.DataFrame) -> pd.DataFrame:
    """Situation plus the pass the player actually attempted."""
    f = passes[SITUATION_FEATURES].copy()
    f["length"] = _qualifier_value(passes["qualifiers"], "Length")
    f["angle"] = _qualifier_value(passes["qualifiers"], "Angle")
    f["dx"] = passes["end_x"] - passes["x"]
    f["dy"] = passes["end_y"] - passes["y"]
    f["cross"] = _has_qualifier(passes["qualifiers"], "Cross").astype(int)
    f["longball"] = _has_qualifier(passes["qualifiers"], "Longball").astype(int)
    f["headpass"] = _has_qualifier(passes["qualifiers"], "HeadPass").astype(int)
    f["chipped"] = _has_qualifier(passes["qualifiers"], "Chipped").astype(int)
    f["first_touch"] = _has_qualifier(passes["qualifiers"], "FirstTouch").astype(int)
    f["layoff"] = _has_qualifier(passes["qualifiers"], "LayOff").astype(int)
    f["throughball"] = _has_qualifier(passes["qualifiers"], "Throughball").astype(int)
    return f


def build_expectations(actions: pd.DataFrame, ev: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit one expected-value model per sub-skill and return the scored event subsets.

    Each model conditions on situation *and* intent, so its residual isolates
    execution: whether this player completed the pass / beat the man / won the
    duel more often than a league-average player attempting the same thing from
    the same place under the same congestion.
    """
    out: dict[str, pd.DataFrame] = {}

    passes = actions[actions["type"] == "Pass"].copy()
    passes["exp_fail"] = _fit_oof(
        _pass_features(passes),
        passes["outcome_type"].ne("Successful").to_numpy().astype(int),
        passes["game_id"].to_numpy(),
    )
    passes["fail"] = passes["outcome_type"].ne("Successful").astype(int)
    out["pass"] = passes

    takeons = actions[actions["type"] == "TakeOn"].copy()
    if len(takeons) > 200:
        takeons["exp_fail"] = _fit_oof(
            takeons[SITUATION_FEATURES],
            takeons["outcome_type"].ne("Successful").to_numpy().astype(int),
            takeons["game_id"].to_numpy(),
        )
        takeons["fail"] = takeons["outcome_type"].ne("Successful").astype(int)
    out["takeon"] = takeons

    # Aerials sit outside `open_play_actions` (they are not is_touch events) so
    # they are pulled from the annotated frame directly.
    aerials = ev[ev["type"].eq("Aerial") & ev["player_id"].notna() & ~ev["is_set_piece"]].copy()
    if len(aerials) > 200:
        aerials["exp_fail"] = _fit_oof(
            aerials[SITUATION_FEATURES],
            aerials["outcome_type"].ne("Successful").to_numpy().astype(int),
            aerials["game_id"].to_numpy(),
        )
        aerials["fail"] = aerials["outcome_type"].ne("Successful").astype(int)
    out["aerial"] = aerials

    # `Foul` events pair up mirrored, with outcome Successful marking the player
    # who *won* the foul. Drawing one in a congested area is direct evidence of
    # beating or absorbing a defender, and unlike inferred proximity it is not
    # contaminated by the outcome of a pass.
    fouls_won = ev[
        ev["type"].eq("Foul") & ev["outcome_type"].eq("Successful") & ev["player_id"].notna()
    ].copy()
    out["foul_won"] = fouls_won

    out["retain"] = actions
    return out


# ---------------------------------------------------------------------------
# player scoring
# ---------------------------------------------------------------------------


def _shrink(raw: pd.Series, n: pd.Series, k: float) -> pd.Series:
    """Pull small-sample rates toward the population mean.

    `k` is the sample size at which a player's own record and the prior carry
    equal weight, so a player with 30 tight take-ons is not ranked above one with
    300 on the strength of noise.
    """
    return raw * (n / (n + k))


# Per sub-skill: which fitted subset it reads, whether the congestion filter
# applies, the shrinkage half-weight, and the minimum sample below which a player
# simply does not get a score for it.
#
# The `tight_only` flag encodes a distinction that matters more than it looks.
# Passes and general ball retention need the filter, because most of them happen
# in space and only the congested ones are evidence about tight areas. Take-ons,
# aerials and fouls won do not: a take-on *is* an attempt to beat a defender who
# is already right in front of you, and an aerial *is* a contested ball. Filtering
# those by pitch congestion discards most of the sample to establish something
# the event type already guarantees, which is what left `dribble_score` and
# `aerial_score` at split-half r ~ 0.06-0.10 in the first cut. Situation still
# enters through each expectation model, so a box take-on is graded differently
# from a halfway-line one.
# `min_n` is deliberately low, with shrinkage rather than exclusion doing the work
# of handling thin samples. A hard cutoff turned out to be actively harmful: it
# flipped component availability between the two halves of the reliability test,
# so a player was scored on a different weighted mix in each half, which alone
# held the composite at r ~ 0.33 while its own best component sat at 0.55. A
# shrunk score of ~0 expresses "no information" and keeps the weighting stable,
# where a NaN silently re-weights everything else.
COMPONENT_SPEC = {
    "pass": {"source": "pass", "tight_only": True, "k": 120.0, "min_n": 15},
    "dribble": {"source": "takeon", "tight_only": False, "k": 25.0, "min_n": 4},
    "aerial": {"source": "aerial", "tight_only": False, "k": 25.0, "min_n": 4},
}


def score_players(
    expectations: dict[str, pd.DataFrame],
    roles: pd.DataFrame,
    tight_quantile: float = 0.75,
    min_tight_actions: int = 120,
    weights: dict[str, float] | None = None,
    axis_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-player exposure and above-expectation execution in the tightest situations.

    `tight_quantile` picks the difficulty cut defining "tight" — the default 0.75
    keeps the top quarter of situations by predicted ball-loss risk. `weights`
    sets each component's pull inside its axis and is derived from measured
    reliability by `build` rather than assumed; `axis_weights` blends the axes into
    the single summary score and is a stated preference, since the axes are
    near-independent.
    """
    actions = expectations["retain"]
    cut = actions["difficulty"].quantile(tight_quantile)
    key = ["player_id", "player", "team"]
    t_actions = actions[actions["difficulty"] >= cut]

    base = (
        actions.groupby(key)
        .agg(actions_total=("lost", "size"), mean_difficulty=("difficulty", "mean"))
        .reset_index()
    )
    tight_counts = (
        t_actions.groupby(key)
        .agg(tight_actions=("lost", "size"), tight_congestion=("congestion", "mean"))
        .reset_index()
    )
    out = base.merge(tight_counts, on=key, how="left")
    out["tight_actions"] = out["tight_actions"].fillna(0).astype(int)
    out["tight_share"] = out["tight_actions"] / out["actions_total"]

    # Retention above expectation, per 100 tight actions.
    ret = (
        t_actions.groupby(key)
        .agg(_a=("lost", "mean"), _e=("difficulty", "mean"))
        .assign(retain_raw=lambda d: (d["_e"] - d["_a"]) * 100)
        .reset_index()[key + ["retain_raw"]]
    )
    out = out.merge(ret, on=key, how="left")
    out["retain_score"] = _shrink(out["retain_raw"], out["tight_actions"], 150.0)

    for label, spec in COMPONENT_SPEC.items():
        df = expectations[spec["source"]]
        if "exp_fail" not in df.columns or df.empty:
            out[f"{label}_score"] = np.nan
            out[f"{label}_n"] = 0
            continue
        sub = df[df["difficulty"] >= cut] if spec["tight_only"] else df
        agg = (
            sub.groupby(key)
            .agg(n=("fail", "size"), a=("fail", "mean"), e=("exp_fail", "mean"))
            .reset_index()
        )
        agg[f"{label}_score"] = _shrink((agg["e"] - agg["a"]) * 100, agg["n"], spec["k"])
        agg.loc[agg["n"] < spec["min_n"], f"{label}_score"] = np.nan
        agg = agg.rename(columns={"n": f"{label}_n"})
        out = out.merge(agg[key + [f"{label}_n", f"{label}_score"]], on=key, how="left")
        out[f"{label}_n"] = out[f"{label}_n"].fillna(0).astype(int)

    fouls = expectations["foul_won"].groupby(key).size().rename("fouls_won").reset_index()
    out = out.merge(fouls, on=key, how="left")
    out["fouls_won"] = out["fouls_won"].fillna(0).astype(int)
    out["fouls_won_p100"] = 100 * out["fouls_won"] / out["actions_total"].replace(0, np.nan)

    out = out.merge(roles, on="player_id", how="left")
    out["role"] = out["role"].fillna("UNK")
    if "role_group" not in out.columns:
        out["role_group"] = out["role"]
    out["role_group"] = out["role_group"].fillna(out["role"])
    out = out[(out["tight_actions"] >= min_tight_actions) & (out["role"] != "GK")].copy()
    if out.empty:
        raise ValueError(f"no outfield player reached {min_tight_actions} tight actions")
    # A peer set needs enough members for a z-score to mean anything; rare groups
    # fall back to their bare formation line.
    counts = out["role_group"].value_counts()
    out.loc[out["role_group"].map(counts).lt(MIN_ROLE_GROUP), "role_group"] = out["role"]

    weights = weights or dict(COMPONENT_PRIOR)
    axis_weights = axis_weights or DEFAULT_AXIS_WEIGHTS

    def _z(frame: pd.DataFrame, cols: list[str]) -> None:
        """Z-score within peer group: a centre-back and a winger face different problems."""
        for col in cols:
            frame[f"z_{col}"] = frame.groupby("role_group")[col].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0
            )

    _z(out, COMPONENT_COLS)
    out["tight_share_pct"] = out.groupby("role_group")["tight_share"].rank(pct=True) * 100

    def _blend(frame: pd.DataFrame, spec: dict[str, float], prefix: str) -> np.ndarray:
        """Weighted mean over present members, renormalised so absence is not a penalty."""
        w = np.array([spec[c] for c in spec])
        Z = frame[[f"{prefix}{c}" for c in spec]].to_numpy()
        wsum = ((~np.isnan(Z)) * w).sum(axis=1)
        return np.where(
            wsum > 0, np.nansum(np.nan_to_num(Z) * w, axis=1) / np.where(wsum > 0, wsum, 1), np.nan
        )

    for axis, spec in AXES.items():
        # Reliability-discounted weights within the axis, so a weak member
        # (aerials replicate at only r ~ 0.07) cannot dilute its axis.
        spec = {c: weights.get(c, w) for c, w in spec.items()}
        spec = {c: w for c, w in spec.items() if w > 0} or dict(AXES[axis])
        out[f"axis_{axis}"] = _blend(out, spec, "z_")
        out[f"pct_{axis}"] = out.groupby("role_group")[f"axis_{axis}"].rank(pct=True) * 100

    # Re-standardise each axis within role so the axes are on a common scale
    # before any cross-axis blending.
    _z(out, AXIS_COLS)
    out["tight_space_score"] = _blend(
        out, {f"axis_{a}": w for a, w in axis_weights.items()}, "z_"
    )
    out["tight_pct"] = out.groupby("role_group")["tight_space_score"].rank(pct=True) * 100

    present = ~out[COMPONENT_COLS].isna().to_numpy()
    out["components_used"] = present.sum(axis=1)
    return out.sort_values("tight_space_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


# Playing well in tight spaces is not one skill, and the data says so clearly.
# Measured across 1,870 players, the components split into near-independent
# groups: retention and passing move together (r = 0.65), dribbling is unrelated
# to both (r = -0.12 and -0.06), and drawing fouls runs *against* retention
# (r = -0.31). The first principal component explains only 36% of the variance.
#
# That negative correlation is interpretable rather than a defect: a player who
# draws fouls per touch is one who carries the ball into contact and gets stopped
# illegally, which is a different way of surviving congestion than never being
# caught in it. Two styles, not two readings of one trait.
#
# So the components are grouped into axes that each hold together internally, and
# reported as a profile. Averaging near-independent axes into one number cancels
# variance instead of reinforcing it, which is why an earlier single composite
# scored *worse* (split-half r = 0.27) than its own best component (0.55).
AXES = {
    # Keeping and moving the ball when squeezed. The reliable core.
    "retention": {"retain_score": 0.60, "pass_score": 0.40},
    # Beating the man. Independent of retention, so genuinely separate information.
    "beating_man": {"dribble_score": 1.00},
    # Out-muscling: winning contact and drawing fouls rather than avoiding them.
    "winning_contact": {"fouls_won_p100": 0.80, "aerial_score": 0.20},
}
COMPONENT_PRIOR = {c: w for spec in AXES.values() for c, w in spec.items()}
COMPONENT_COLS = list(COMPONENT_PRIOR)
AXIS_COLS = [f"axis_{a}" for a in AXES]

# Default blend across axes when a single ranking number is wanted. Equal by
# default precisely because the axes are near-independent: there is no
# data-driven basis for preferring one, so this is a stated preference rather
# than a measurement, and `--axis-weights` exists to override it per use case.
DEFAULT_AXIS_WEIGHTS = {a: 1.0 / len(AXES) for a in AXES}


def split_half_reliability(
    expectations: dict[str, pd.DataFrame],
    roles: pd.DataFrame,
    tight_quantile: float = 0.75,
    min_tight_actions: int = 120,
    weights: dict[str, float] | None = None,
    axis_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Re-score on odd vs even matches and correlate, to separate skill from noise.

    A metric that does not reproduce itself across disjoint halves of the same
    season is measuring variance, however plausible its leaderboard looks.

    `split_half_r` compares two half-size samples, so it understates what the
    full-season metric is worth; `reliability` is the Spearman-Brown correction
    2r/(1+r), the estimate for the whole sample. Minimums are relaxed to a third
    of the headline value, since each half holds half the matches.
    """
    halves = {}
    for h in (0, 1):
        sub = {}
        for name, df in expectations.items():
            if df.empty:
                sub[name] = df
                continue
            ids = pd.factorize(df["game_id"])[0]
            sub[name] = df[ids % 2 == h]
        try:
            scored = score_players(
                sub, roles, tight_quantile,
                min_tight_actions=max(40, min_tight_actions // 3),
                weights=weights, axis_weights=axis_weights,
            )
        except ValueError:
            return pd.DataFrame()
        # Scores are per (player, team), so a mid-season transfer yields two rows
        # for one player. Keep the larger stint so player_id is a unique key to
        # align the two halves on.
        halves[h] = (
            scored.sort_values("tight_actions", ascending=False)
            .drop_duplicates("player_id")
            .set_index("player_id")
        )

    common = halves[0].index.intersection(halves[1].index)
    rows = []
    metrics = [("axis", c) for c in AXIS_COLS] + [("component", c) for c in COMPONENT_COLS]
    metrics += [("exposure", "tight_share"), ("summary", "tight_space_score")]
    for kind, c in metrics:
        a, b = halves[0].loc[common, c], halves[1].loc[common, c]
        ok = a.notna() & b.notna()
        r = a[ok].corr(b[ok]) if ok.sum() > 5 else np.nan
        rows.append({
            "kind": kind,
            "metric": c,
            "n_players": int(ok.sum()),
            "split_half_r": r,
            "reliability": (2 * r / (1 + r)) if pd.notna(r) and r > -1 else np.nan,
        })
    return pd.DataFrame(rows)


def external_check(scores: pd.DataFrame, stats_path: str | Path = "sofascore_player_stats.csv") -> pd.DataFrame:
    """Correlate the score against independent Sofascore season aggregates.

    A face-validity check on a wholly separate data source: the score should track
    dribble success and duel win rates, and move against being dispossessed. The
    name join only resolves ~58% of players, which is far too lossy to build on but
    fine for a directional check on a thousand-odd of them.
    """
    from name_utils import normalize_series

    stats = pd.read_csv(stats_path)
    stats["_k"] = normalize_series(stats["player"])
    sc = scores.copy()
    sc["_k"] = normalize_series(sc["player"])
    merged = sc.merge(stats.drop_duplicates("_k"), on="_k", suffixes=("", "_ss"))

    expected = {
        "successfulDribblesPercentage": "+",
        "groundDuelsWonPercentage": "+",
        "totalDuelsWonPercentage": "+",
        "wasFouled": "+",
        "accuratePassesPercentage": "+",
        "dispossessed": "-",
        "possessionLost": "-",
    }
    rows = []
    for col, sign in expected.items():
        if col not in merged.columns:
            continue
        per90 = col in ("wasFouled", "dispossessed", "possessionLost")
        v = merged[col]
        if per90:
            v = v / merged["minutesPlayed"].replace(0, np.nan) * 90
        ok = v.notna() & merged["tight_space_score"].notna()
        rows.append({
            "sofascore_metric": col,
            "expected_sign": sign,
            "n": int(ok.sum()),
            "r": merged.loc[ok, "tight_space_score"].corr(v[ok]),
        })
    out = pd.DataFrame(rows)
    out["as_expected"] = np.where(
        out["expected_sign"].eq("+"), out["r"] > 0, out["r"] < 0
    )
    return out


# A component is dropped from its axis if it replicates at less than this fraction
# of the axis's best member. Ordinary reliability weighting is not enough here,
# because members of an axis are built from overlapping events — passes are most
# of the action population, so `pass_score` and `retain_score` share their errors
# rather than averaging them out. Blending `pass_score` (r ~ 0.19) into
# `retain_score` (r ~ 0.56) at even 18% weight pushed the axis down to r ~ 0.41.
# Excluded components are still computed and reported, just not fused.
AXIS_RELATIVE_FLOOR = 0.5


def reliability_weights(rel: pd.DataFrame, floor: float = 0.05) -> dict[str, float]:
    """Discount each component's within-axis weight by how much of it replicates.

    Applied *within* an axis rather than across the whole score, which is the
    safe place for it. Weighting the whole metric by reliability was its own trap:
    fouls won replicates far better than anything else (r ~ 0.79) because drawing
    fouls is a stable stylistic trait, so it claimed the largest share and began
    pulling players who are poor at everything else into the top 20. Reliability
    tells you what is signal, not what is on target. Confined inside an axis it
    only settles the balance between members that already measure the same thing.
    """
    r = rel.set_index("metric")["split_half_r"]
    out: dict[str, float] = {}
    for spec in AXES.values():
        scored = {c: r.get(c, np.nan) for c in spec}
        usable = {c: v for c, v in scored.items() if pd.notna(v) and v > floor}
        if not usable:
            out.update({c: w / sum(spec.values()) for c, w in spec.items()})
            continue
        best = max(usable.values())
        kept = {c: spec[c] * v for c, v in usable.items() if v >= AXIS_RELATIVE_FLOOR * best}
        total = sum(kept.values())
        out.update({c: v / total for c, v in kept.items()})
        for c in spec:
            out.setdefault(c, 0.0)
    return out


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def build(
    leagues: list[str] | None = None,
    season: int | str = DEFAULT_SEASON,
    events_root: str | Path = "league_games",
    max_matches: int | None = None,
    tight_quantile: float = 0.75,
    min_tight_actions: int = 120,
    axis_weights: dict[str, float] | None = None,
) -> dict:
    """Load events, build the congestion field, fit every model, and score players."""
    events = load_all_league_events(leagues=leagues, season=season, events_root=events_root)
    if max_matches is not None:
        keep = pd.unique(events["game_id"])[:max_matches]
        events = events[events["game_id"].isin(keep)]
    print(f"events: {len(events):,} rows across {events['game_id'].nunique():,} matches", flush=True)

    roles = extract_roles(events)
    print(f"roles resolved for {len(roles):,} players", flush=True)

    field = build_congestion_field(events)
    ev = annotate_events(events, field)
    actions = open_play_actions(ev)
    print(f"open-play on-ball actions: {len(actions):,}", flush=True)

    roles = refine_roles(roles, actions)

    actions = add_difficulty(actions)
    expectations = build_expectations(actions, ev)

    # Two passes: score once to have something to measure, measure each
    # component's split-half reliability, then re-fuse weighting by that. The
    # within-axis weights are therefore a property of the data rather than an
    # assumption, and they adapt as more leagues are pooled in.
    rel = split_half_reliability(expectations, roles, tight_quantile, min_tight_actions)
    weights = reliability_weights(rel) if not rel.empty else dict(COMPONENT_PRIOR)
    print("within-axis component weights from measured reliability: "
          + ", ".join(f"{c.replace('_score', '')}={w:.2f}" for c, w in weights.items()), flush=True)

    scores = score_players(
        expectations, roles, tight_quantile, min_tight_actions,
        weights=weights, axis_weights=axis_weights,
    )
    rel_final = split_half_reliability(
        expectations, roles, tight_quantile, min_tight_actions,
        weights=weights, axis_weights=axis_weights,
    )
    return {
        "field": field,
        "events": ev,
        "actions": actions,
        "expectations": expectations,
        "roles": roles,
        "scores": scores,
        "weights": weights,
        "axis_weights": axis_weights or DEFAULT_AXIS_WEIGHTS,
        "reliability": rel_final,
    }


def axis_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    """Correlation between the axes, to show they carry independent information.

    If these came back strongly positive the profile could fairly be collapsed to
    one number. They do not, which is the whole reason it is reported as a profile.
    """
    z = scores[[f"z_axis_{a}" for a in AXES]].rename(columns=lambda c: c[7:])
    return z.corr()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rate players on how well they play in congested, tightly-contested areas."
    )
    parser.add_argument("--leagues", nargs="*", default=None,
                        help=f"Leagues to pool. Default: all {len(DEFAULT_LEAGUES)} collected.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--max-matches", type=int, default=None, help="Cap matches, for quick runs")
    parser.add_argument("--tight-quantile", type=float, default=0.75,
                        help="Difficulty percentile above which a situation counts as tight")
    parser.add_argument("--min-tight-actions", type=int, default=120)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--player", default=None, help="Show one player's card instead of the leaderboard")
    parser.add_argument("--role", default=None, help="Restrict the leaderboard to DF/MF/FW")
    parser.add_argument("--sort-by", default=None,
                        help="Rank on one axis instead of the blend, e.g. axis_retention, "
                             "axis_beating_man, axis_winning_contact, tight_share")
    parser.add_argument("--axis-weights", default=None,
                        help="Blend weights for the summary score, e.g. "
                             "'retention=0.6,beating_man=0.3,winning_contact=0.1'")
    parser.add_argument("--validate", action="store_true", help="Report split-half reliability")
    parser.add_argument("--external-check", action="store_true",
                        help="Correlate against independent Sofascore season aggregates")
    parser.add_argument("--show-field", action="store_true", help="Print the congestion field")
    parser.add_argument("--out", default=None, help="Write the full score table to CSV")
    args = parser.parse_args()

    parsed_axis_weights = None
    if args.axis_weights:
        parsed = {}
        for part in args.axis_weights.split(","):
            name, _, val = part.partition("=")
            name = name.strip()
            if name not in AXES:
                raise SystemExit(f"unknown axis {name!r}; expected one of {list(AXES)}")
            parsed[name] = float(val)
        total = sum(parsed.values()) or 1.0
        parsed_axis_weights = {a: parsed.get(a, 0.0) / total for a in AXES}

    result = build(
        leagues=args.leagues,
        season=args.season,
        max_matches=args.max_matches,
        tight_quantile=args.tight_quantile,
        min_tight_actions=args.min_tight_actions,
        axis_weights=parsed_axis_weights,
    )
    scores = result["scores"]

    if args.show_field:
        print("\ncongestion field — defensive actions per on-ball touch")
        print("rows: own goal -> opponent goal, cols: touchline -> touchline")
        print(result["field"].describe().to_string())

    display = [
        "player", "team", "role_group", "tight_actions", "tight_share",
        "pct_retention", "pct_beating_man", "pct_winning_contact",
        "tight_space_score", "tight_pct",
    ]

    if args.player:
        from name_utils import normalize_name, normalize_series

        hit = scores[normalize_series(scores["player"]).str.contains(
            normalize_name(args.player), regex=False)]
        if hit.empty:
            raise SystemExit(f"no player matching {args.player!r} cleared the minimums")
        for _, row in hit.iterrows():
            peers = scores[scores["role_group"] == row["role_group"]]
            print(f"\n{row['player']} ({row['team']}) — {row['role_group']}, "
                  f"vs {len(peers) - 1} peers in the same role group")
            print(f"  exposure: {row['tight_actions']:,} tight actions, "
                  f"{row['tight_share']:.0%} of all their on-ball actions "
                  f"({row['tight_share_pct']:.0f}th pct in role), "
                  f"mean congestion {row['tight_congestion']:.2f}")
            card = []
            for axis, spec in AXES.items():
                card.append({"axis": axis, "z": row[f"axis_{axis}"],
                             "pct_in_role": row[f"pct_{axis}"], "component": "", "value": np.nan})
                for c in spec:
                    card.append({"axis": "", "z": np.nan, "pct_in_role": np.nan,
                                 "component": c.replace("_score", ""), "value": row[c]})
            print(pd.DataFrame(card).round(2).to_string(index=False, na_rep=""))
            print(f"  summary blend {row['tight_space_score']:+.2f} "
                  f"({row['tight_pct']:.0f}th pct) — a preference-weighted mix of "
                  "near-independent axes, not a single measured trait")
    else:
        board = scores if args.role is None else scores[scores["role"] == args.role.upper()]
        if args.sort_by:
            # Drop unscored rows, or the bottom of the table fills with players who
            # simply had too few of that action to be graded on it.
            board = board.dropna(subset=[args.sort_by]).sort_values(
                args.sort_by, ascending=False
            )
        label = args.sort_by or "tight_space_score"
        print(f"\ntop {args.top} by {label} — percentiles are within role")
        print(board.head(args.top)[display].round(1).to_string(index=False))
        print(f"\nbottom 10 of {len(board)}")
        print(board.tail(10)[display].round(1).to_string(index=False))

    if args.validate:
        diag = difficulty_diagnostics(result["actions"])
        print(f"\ndifficulty model — AUC {diag.attrs['auc']:.3f}; "
              "predicted vs actual ball-loss rate by decile")
        print(diag.round(3).to_string(index=False))

        print("\nsplit-half reliability (odd vs even matches); "
              "reliability = Spearman-Brown estimate for the full sample")
        print(result["reliability"].round(3).to_string(index=False))

        print("\naxis correlations — near-zero is why this is a profile, not one number")
        print(axis_correlations(scores).round(3).to_string())

        print("\nscore by role group — a flat spread means the role confound is controlled")
        print(
            scores.groupby("role_group")
            .agg(players=("tight_space_score", "size"),
                 mean_score=("tight_space_score", "mean"),
                 mean_tight_share=("tight_share", "mean"),
                 mean_tight_congestion=("tight_congestion", "mean"),
                 mean_lateral_dev=("lateral_dev", "mean"))
            .round(3).to_string()
        )

    if args.external_check:
        print("\nface validity vs independent Sofascore season stats")
        print(external_check(scores).round(3).to_string(index=False))

    if args.out:
        scores.to_csv(args.out, index=False)
        print(f"\nscores written to {args.out}")
