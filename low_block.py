"""Find low blocks in event data, then find what breaks them.

The question this answers: when a team drops into a low block, which attacking
patterns actually produce good chances against it, and which only look busy?

There is no tracking data here, so defensive shape is inferred from where the
defending team's *ball events* happen. That is the standard public-data proxy and
it behaves: ranked over a season it puts West Ham, Wolves and Burnley at the
bottom and Arsenal, City and Brighton at the top. What it cannot see is an
off-ball player, so "block" here always means "where this team is engaging the
ball right now", not a literal back four's y-coordinate.

Pipeline
--------
1. `fit_xg`            shot-level xG, fitted out-of-fold on the event data itself
2. `segment_possessions`  possession spells robust to blocks and duels
3. `block_state`       rolling per-team estimate of how deep/compact/passive a defence is
4. `build_sequences`   one row per attacking possession, with shape-of-attack features
5. `label_low_block`   which of those were settled attacks against a low block
6. `find_patterns`     cluster those sequences and rank the clusters by xG produced

Usage
-----
    python low_block.py --league "ENG-Premier League" --plot
    python low_block.py --all-leagues --out low_block_sequences.csv
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tight_space import (
    PLAY_PERIODS,
    SET_PIECE_QUALIFIERS,
    _has_qualifier,
    _qualifier_value,
    defensive_action_mask,
)
from touchmap_similarity import DEFAULT_LEAGUES, DEFAULT_SEASON, load_all_league_events

# Opta 0-100 coordinates map onto a 105x68m pitch; distances and angles are only
# meaningful once both axes are back in metres.
PITCH_X, PITCH_Y = 105.0, 68.0
GOAL_WIDTH = 7.32

# Thirds and box edges in Opta units. The box is x>=83, |y-50|<=21.1.
FINAL_THIRD = 66.7
BOX_X, BOX_HALF_Y = 83.0, 21.1
SIX_YARD_X = 94.2

# Defensive actions to roll the block estimate over. Wider than tight_space's set:
# a recovery or a keeper claim is also evidence of where a team is defending.
BLOCK_ACTION_TYPES = (
    "Tackle", "Challenge", "Interception", "Clearance", "BlockedPass", "Aerial",
    "Foul", "BallRecovery", "Save", "KeeperPickup", "Punch", "Smother",
)

# A gap this long between consecutive events means play stopped, so the spell
# ends there rather than carrying a dead clock across the break.
STOPPAGE_GAP = 30.0

# How many recent defensive actions define "where this team is defending now".
# 10 is roughly two to three minutes of a settled defensive phase: long enough
# that one clearance cannot swing it, short enough to catch a side dropping off
# after going ahead.
BLOCK_WINDOW = 10

# Events that are not a team doing something with the ball, and must not start,
# end, or extend a possession.
NON_POSSESSION_TYPES = (
    "Start", "End", "SubstitutionOn", "SubstitutionOff", "FormationSet",
    "FormationChange", "Card", "TeamSetUp", "CornerAwarded", "OffsideGiven",
)

SHOT_TYPES = ("Goal", "SavedShot", "MissedShots", "ShotOnPost")

# A penalty is not a chance the attack created, so it is priced at the league
# conversion rate rather than modelled, and excluded from the model's training.
PENALTY_XG = 0.79


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def shot_geometry(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Distance to goal centre and the angle the goal subtends, both in metres."""
    dx = (100.0 - x) / 100.0 * PITCH_X
    dy = (y - 50.0) / 100.0 * PITCH_Y
    dist = np.hypot(dx, dy)
    # Angle between the vectors to each post; collapses to ~0 from the byline.
    a = np.hypot(dx, dy - GOAL_WIDTH / 2)
    b = np.hypot(dx, dy + GOAL_WIDTH / 2)
    cos = np.clip((a**2 + b**2 - GOAL_WIDTH**2) / (2 * a * b), -1.0, 1.0)
    return pd.DataFrame(
        {"dist": dist, "angle": np.degrees(np.arccos(cos)), "dx": dx, "dy": np.abs(dy)},
        index=x.index,
    )


# ---------------------------------------------------------------------------
# shot xG, fitted on the event data itself
# ---------------------------------------------------------------------------
#
# Understat match totals are already cached under match_xg/, but they are one
# number per team per game. Judging whether a low block was *broken* needs a value
# per chance, in the same coordinate frame as everything else, so the model is
# fitted here on WhoScored shots directly.
#
# `BigChance` and `OneOnOne` are deliberately left out of the features. They are
# analyst judgements applied after the fact and correlate with the shot going in
# for reasons the geometry does not contain, so a model that leans on them scores
# its own hindsight. They are kept on the output frame as descriptors only.


def _preceding_pass(ev: pd.DataFrame, window_s: float = 6.0) -> pd.DataFrame:
    """For every event, the last pass by the same team within `window_s` seconds.

    Used to type the assist. `related_event_id` would be exact, but the cached
    csvs carry no `event_id` column to join it back to, and the immediately
    preceding same-team pass is the assist in the overwhelming majority of cases.
    """
    p = ev["type"].eq("Pass")
    cols = {
        "prev_pass_t": ev["t"].where(p),
        "prev_pass_x": ev["x"].where(p),
        "prev_pass_y": ev["y"].where(p),
        "prev_pass_end_x": ev["end_x"].where(p),
        "prev_pass_end_y": ev["end_y"].where(p),
        "prev_pass_cross": _has_qualifier(ev["qualifiers"], "Cross").where(p),
        "prev_pass_through": _has_qualifier(ev["qualifiers"], "Throughball").where(p),
        "prev_pass_long": _has_qualifier(ev["qualifiers"], "Longball").where(p),
        "prev_pass_layoff": _has_qualifier(ev["qualifiers"], "LayOff").where(p),
    }
    out = pd.DataFrame(cols, index=ev.index).groupby(
        [ev["game_id"], ev["team_id"]], sort=False
    ).ffill()
    stale = (ev["t"] - out["prev_pass_t"]) > window_s
    return out.mask(stale.fillna(True), other=np.nan)


def shot_frame(ev: pd.DataFrame) -> pd.DataFrame:
    """Every shot with the features the xG model needs."""
    sh = ev[ev["type"].isin(SHOT_TYPES)].copy()
    q = sh["qualifiers"]
    geo = shot_geometry(sh["x"], sh["y"])
    prev = _preceding_pass(ev).loc[sh.index]

    # A cutback is a pass from near the byline played backwards into the box; it
    # is the single most productive low-block pattern in the literature, so it is
    # typed explicitly rather than lumped in with crosses.
    cutback = (
        prev["prev_pass_x"].ge(BOX_X)
        & prev["prev_pass_end_x"].lt(prev["prev_pass_x"])
        & prev["prev_pass_y"].sub(50).abs().ge(15)
        & prev["prev_pass_end_y"].sub(50).abs().le(20)
    )

    out = pd.DataFrame(
        {
            "dist": geo["dist"],
            "angle": geo["angle"],
            "x": sh["x"],
            "y_dev": (sh["y"] - 50).abs(),
            "header": _has_qualifier(q, "Head").astype(int),
            "other_body": _has_qualifier(q, "OtherBodyPart").astype(int),
            "volley": _has_qualifier(q, ("Volley", "OverHeadKick")).astype(int),
            "first_touch": _has_qualifier(q, "FirstTouch").astype(int),
            "from_corner": _has_qualifier(q, "FromCorner").astype(int),
            "fast_break": _has_qualifier(q, "FastBreak").astype(int),
            "set_piece": _has_qualifier(q, ("SetPiece", "ThrowinSetPiece")).astype(int),
            "direct_fk": _has_qualifier(q, "DirectFreekick").astype(int),
            "individual": _has_qualifier(q, "IndividualPlay").astype(int),
            "assisted": _has_qualifier(q, "Assisted").astype(int),
            "asst_cross": prev["prev_pass_cross"].fillna(False).astype(int),
            "asst_through": prev["prev_pass_through"].fillna(False).astype(int),
            "asst_long": prev["prev_pass_long"].fillna(False).astype(int),
            "asst_layoff": prev["prev_pass_layoff"].fillna(False).astype(int),
            "asst_cutback": cutback.fillna(False).astype(int),
        },
        index=sh.index,
    )
    out["is_pen"] = _has_qualifier(q, "Penalty").astype(int)
    out["is_goal"] = sh["is_goal"].fillna(False).astype(bool).astype(int)
    out["big_chance"] = _has_qualifier(q, "BigChance").astype(int)
    out["one_on_one"] = _has_qualifier(q, "OneOnOne").astype(int)
    # Blocked shots convert at exactly 0.000 and are ~30% of all shots, but
    # blockedness is an outcome, not a property of the chance, so it stays out of
    # the model and is carried as a descriptor. It matters here in its own right:
    # smothering shots is a large part of how a low block actually defends.
    out["blocked"] = _has_qualifier(q, "Blocked").astype(int)
    for c in ("game_id", "team_id", "team", "player", "player_id", "minute", "second", "t", "y"):
        if c in sh.columns:
            out[c] = sh[c]
    return out


XG_FEATURES = [
    "dist", "angle", "x", "y_dev", "header", "other_body", "volley", "first_touch",
    "from_corner", "fast_break", "set_piece", "direct_fk", "individual", "assisted",
    "asst_cross", "asst_through", "asst_long", "asst_layoff", "asst_cutback",
]


def fit_xg(shots: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Out-of-fold xG for every shot, grouped by match so no game trains on itself."""
    xg = pd.Series(np.nan, index=shots.index, dtype=float)
    xg[shots["is_pen"].eq(1)] = PENALTY_XG

    fit = shots[shots["is_pen"].eq(0)]
    X, y = fit[XG_FEATURES], fit["is_goal"].to_numpy()
    groups = fit["game_id"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    oof = np.full(len(fit), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = HistGradientBoostingClassifier(
            max_depth=4, max_iter=300, learning_rate=0.06,
            min_samples_leaf=60, l2_regularization=1.0, random_state=seed,
        )
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    xg.loc[fit.index] = oof
    return xg


def xg_calibration(shots: pd.DataFrame, by: str = "xg", n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual conversion by bin — the model's honesty check.

    Bin by `dist` to read calibration cleanly. By `xg` the bottom half looks flat,
    because a third of all shots are blocked and convert at zero while the model,
    correctly, cannot see that coming; that noise is concentrated in the low bins.
    """
    s = shots[shots["is_pen"].eq(0)].dropna(subset=["xg"])
    bins = pd.qcut(s[by], n_bins, duplicates="drop")
    return (
        s.groupby(bins, observed=True)
        .agg(shots=("xg", "size"), pred=("xg", "mean"), actual=("is_goal", "mean"))
        .rename_axis(by)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# event preparation
# ---------------------------------------------------------------------------


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Playable events in match order, with a match clock and set-piece flag.

    WhoScored's `minute` already runs continuously across halves, so
    minute*60+second is a valid match-wide clock without per-period offsets.
    """
    ev = events[events["period"].isin(PLAY_PERIODS)].copy()
    ev = ev[ev["team_id"].notna() & ev["x"].notna()]
    ev = ev.sort_values(["game_id", "minute", "second"], kind="stable").reset_index(drop=True)
    ev["t"] = ev["minute"] * 60 + ev["second"].fillna(0)
    ev["is_set_piece"] = _has_qualifier(ev["qualifiers"], SET_PIECE_QUALIFIERS)
    ev["is_shot"] = ev["type"].isin(SHOT_TYPES)
    return ev


def load_events(
    leagues: list[str] | None = None,
    season: int | str = DEFAULT_SEASON,
    events_root: str | Path = "league_games",
) -> pd.DataFrame:
    """Load cached league events and prepare them."""
    return prepare_events(load_all_league_events(leagues, season, events_root))


# ---------------------------------------------------------------------------
# possession spells
# ---------------------------------------------------------------------------


def segment_possessions(ev: pd.DataFrame) -> pd.Series:
    """Possession id per event, robust to blocks, duels and failed contests.

    Splitting on every change of `team_id` — the cheap version — shreds exactly
    the phases this analysis is about. A settled attack against a low block is a
    continuous stream of opponent interventions: blocked passes, half-clearances,
    lost aerials, tackles that rebound straight back. Naive segmentation turns one
    three-minute siege into forty possessions of two events each.

    So a change of team only ends the spell once the new team has *actually* got
    the ball: either two consecutive events of their own, or one successful pass.
    A single clearance that comes straight back does not count.
    """
    team = ev["team_id"]
    real = ~ev["type"].isin(NON_POSSESSION_TYPES)

    # Length of the run of same-team events this event belongs to.
    grp = (team != team.shift()) | (ev["game_id"] != ev["game_id"].shift())
    run = grp.cumsum()
    run_len = run.map(run[real].value_counts())

    settled_pass = ev["type"].eq("Pass") & ev["outcome_type"].eq("Successful")
    first_of_run = grp & real
    took_control = first_of_run & (run_len.fillna(0).ge(2) | settled_pass)

    # A goal, and the kickoff that follows, always end the spell.
    after_goal = ev["type"].eq("Goal").shift(fill_value=False)
    new_game = ev["game_id"] != ev["game_id"].shift()
    new_period = (ev["period"] != ev["period"].shift()) | new_game
    # A long gap is the ball out of play, an injury or a VAR check. Without this
    # the clock keeps running and a handful of spells stretch to seven minutes,
    # poisoning every duration feature built on top of them.
    stoppage = ev["t"].diff().gt(STOPPAGE_GAP) & ~new_game

    return (took_control | after_goal | new_period | stoppage).cumsum().rename("poss")


# ---------------------------------------------------------------------------
# block state: how deep, how compact, how passive
# ---------------------------------------------------------------------------


def block_state(ev: pd.DataFrame, window: int = BLOCK_WINDOW) -> pd.DataFrame:
    """Rolling estimate, per team, of where and how they are currently defending.

    Every column is in the *defending* team's own frame, where x=0 is their own
    goal, so a low number means deep. All four are rolled over the team's last
    `window` defensive actions and then shifted, so a value attached to an event
    never contains that event's own outcome.

    - `blk_line`      mean x of recent defensive actions. The block's height.
    - `blk_depth`     spread of x. A compact block engages in a narrow band.
    - `blk_width`     spread of y. Low means the defence is not being stretched.
    - `blk_passivity` opponent passes allowed per defensive action, PPDA-style.
                      A deep block that is *also* passive is the real low block;
                      a deep block with low passivity is a team under siege and
                      scrapping, which is a different phase.
    """
    d = ev[defensive_action_mask(ev) | ev["type"].isin(BLOCK_ACTION_TYPES)]
    d = d[d["x"].notna()]
    passes = ev["type"].eq("Pass")
    pass_no = passes.groupby(ev["game_id"]).cumsum()

    out = []
    for (gid, tid), g in d.groupby(["game_id", "team_id"], sort=False):
        r = g["x"].rolling(window, min_periods=3)
        # Opponent passes between one defensive action and the next.
        allowed = pass_no.reindex(g.index).diff()
        out.append(
            pd.DataFrame(
                {
                    "game_id": gid,
                    "def_team_id": tid,
                    "t": g["t"].to_numpy(),
                    "blk_line": r.mean().shift().to_numpy(),
                    "blk_depth": r.std().shift().to_numpy(),
                    "blk_width": g["y"].rolling(window, min_periods=3).std().shift().to_numpy(),
                    "blk_passivity": allowed.rolling(window, min_periods=3).mean().shift().to_numpy(),
                }
            )
        )
    return pd.concat(out, ignore_index=True).dropna(subset=["blk_line"])


def defensive_actions(ev: pd.DataFrame, window: int = BLOCK_WINDOW) -> pd.DataFrame:
    """Defensive actions, each tagged with the acting team's own block height.

    `block_state` measures the same quantity but keyed to time for joining onto
    the *opponent's* events. This keeps it on the action itself, which is what a
    map of "where does a low block actually defend" needs.
    """
    d = ev[defensive_action_mask(ev) | ev["type"].isin(BLOCK_ACTION_TYPES)]
    d = d[d["x"].notna()].copy()
    d["own_blk_line"] = (
        d.groupby(["game_id", "team_id"], sort=False)["x"]
        .transform(lambda s: s.rolling(window, min_periods=3).mean().shift())
    )
    return d


def attach_block_state(ev: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Join each event the block state of the team it is being played *against*."""
    ev = ev.copy()
    # Opponent per event: the other team_id present in that match.
    pair = ev.groupby("game_id")["team_id"].agg(lambda s: list(pd.unique(s))[:2])
    opp_map = {}
    for gid, ts in pair.items():
        if len(ts) == 2:
            opp_map[(gid, ts[0])] = ts[1]
            opp_map[(gid, ts[1])] = ts[0]
    ev["opp_team_id"] = [opp_map.get(k) for k in zip(ev["game_id"], ev["team_id"])]

    cols = ["blk_line", "blk_depth", "blk_width", "blk_passivity"]
    left = ev[["game_id", "opp_team_id", "t"]].reset_index().rename(
        columns={"opp_team_id": "def_team_id"}
    )
    merged = pd.merge_asof(
        left.sort_values("t"),
        state.sort_values("t"),
        on="t",
        by=["game_id", "def_team_id"],
        direction="backward",
    ).set_index("index")
    for c in cols:
        ev[c] = merged[c]
    return ev


# ---------------------------------------------------------------------------
# attacking sequences
# ---------------------------------------------------------------------------
#
# Lanes follow the five-lane model rather than left/right. Mirroring the pitch
# doubles the sample behind every pattern, and "attacked down the halfspace" is
# the tactically meaningful statement — which halfspace is not.

LANE_EDGES = [0, 20, 40, 60, 80, 100]
LANE_NAMES = ["wide", "halfspace", "central", "halfspace", "wide"]


def _lane(y: pd.Series) -> pd.Series:
    d = (y - 50).abs()
    return pd.Series(
        np.where(d <= 10, "central", np.where(d <= 30, "halfspace", "wide")),
        index=y.index,
    )


def build_sequences(ev: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """One row per attacking possession: how the attack was built, and what it produced.

    Features are computed only over events by the team in control, and the
    final-third block is measured at the moment of entry — what the attack was
    looking at when it arrived, not the average of a defence that had already
    been pulled apart.
    """
    ev = ev.copy()
    if "poss" not in ev:
        ev["poss"] = segment_possessions(ev)

    # The team in control of the spell: whoever has the most events in it.
    att = ev.groupby("poss")["team_id"].agg(lambda s: s.value_counts().idxmax())
    ev["att_team_id"] = ev["poss"].map(att)
    a = ev[ev["team_id"] == ev["att_team_id"]].copy()

    q = a["qualifiers"]
    a["is_pass"] = a["type"].eq("Pass")
    a["ok"] = a["outcome_type"].eq("Successful")
    a["in_f3"] = a["x"] >= FINAL_THIRD
    a["in_box"] = (a["x"] >= BOX_X) & (a["y"] - 50).abs().le(BOX_HALF_Y)
    a["cross"] = a["is_pass"] & _has_qualifier(q, "Cross")
    a["through"] = a["is_pass"] & _has_qualifier(q, "Throughball")
    a["layoff"] = a["is_pass"] & _has_qualifier(q, "LayOff")
    a["takeon"] = a["type"].eq("TakeOn")
    a["takeon_won"] = a["takeon"] & a["ok"]
    a["dy"] = (a["end_y"] - a["y"]).abs()
    a["dx"] = a["end_x"] - a["x"]
    # A switch is a long lateral ball that changes the side of attack.
    a["switch"] = a["is_pass"] & a["ok"] & a["dy"].ge(30)
    # A cutback: from near the byline, played backwards, into the middle.
    a["cutback"] = (
        a["is_pass"] & a["ok"]
        & a["x"].ge(BOX_X) & a["dx"].lt(0)
        & (a["y"] - 50).abs().ge(15) & (a["end_y"] - 50).abs().le(20)
    )
    # Carrying into the box under your own steam, rather than being passed in.
    a["carry_in"] = a["takeon"] & a["ok"] & a["x"].ge(70)
    a["pass_into_box"] = (
        a["is_pass"] & a["ok"] & ~a["in_box"]
        & a["end_x"].ge(BOX_X) & (a["end_y"] - 50).abs().le(BOX_HALF_Y)
    )

    f3 = a[a["in_f3"]]
    g, gf = a.groupby("poss", sort=False), f3.groupby("poss", sort=False)

    seq = pd.DataFrame(
        {
            "game_id": g["game_id"].first(),
            "league": g["league"].first() if "league" in a else "",
            "team_id": g["att_team_id"].first(),
            "team": g["team"].first(),
            "opp_team_id": g["opp_team_id"].first(),
            "t_start": g["t"].min(),
            "duration": g["t"].max() - g["t"].min(),
            "n_events": g.size(),
            "n_passes": g["is_pass"].sum(),
            "start_x": g["x"].first(),
            "max_x": g["x"].max(),
            "n_players": g["player_id"].nunique(),
            "from_set_piece": g["is_set_piece"].max(),
            # final-third work
            "n_passes_f3": gf["is_pass"].sum(),
            "t_in_f3": gf["t"].max() - gf["t"].min(),
            "n_switch": gf["switch"].sum(),
            "n_cross": gf["cross"].sum(),
            "n_cutback": gf["cutback"].sum(),
            "n_through": gf["through"].sum(),
            "n_layoff": gf["layoff"].sum(),
            "n_takeon": gf["takeon"].sum(),
            "n_takeon_won": gf["takeon_won"].sum(),
            "n_box_touch": gf["in_box"].sum(),
            "n_carry_in": gf["carry_in"].sum(),
            "n_pass_into_box": gf["pass_into_box"].sum(),
            "width_used": gf["y"].max() - gf["y"].min(),
            "n_players_f3": gf["player_id"].nunique(),
        }
    )
    seq[[c for c in seq.columns if c.startswith("n_") or c.startswith("t_in")]] = (
        seq[[c for c in seq.columns if c.startswith("n_") or c.startswith("t_in")]].fillna(0)
    )

    # The moment the attack arrived in the final third, and the block it met there.
    entry = f3.sort_values("t").groupby("poss", sort=False).first()
    seq["entry_t"] = entry["t"]
    seq["entry_y"] = entry["y"]
    seq["entry_lane"] = _lane(entry["y"])
    seq["blk_line"] = entry["blk_line"]
    seq["blk_depth"] = entry["blk_depth"]
    seq["blk_width"] = entry["blk_width"]
    seq["blk_passivity"] = entry["blk_passivity"]
    seq["build_up_t"] = seq["entry_t"] - seq["t_start"]

    # How the ball first got into the box. The delivery is typed, not the receipt:
    # a cross is played from *outside* the box, so keying on the first event
    # already inside it classifies almost every cross as whatever touch followed.
    a["enters_box"] = a["in_box"] | a["pass_into_box"] | (a["cross"] & a["ok"])
    box = a[a["enters_box"]].sort_values("t").groupby("poss", sort=False).first()
    box = box.reindex(seq.index)
    seq["box_entry_type"] = np.select(
        [box["cutback"].fillna(False), box["cross"].fillna(False),
         box["through"].fillna(False), box["takeon"].fillna(False),
         box["type"].eq("Pass")],
        ["cutback", "cross", "through", "carry", "pass"], default="none",
    )
    seq.loc[box["x"].isna(), "box_entry_type"] = "none"
    seq["reached_box"] = box["x"].notna().astype(int)

    # Outcome.
    s = shots.copy()
    s["poss"] = ev["poss"].reindex(s.index)
    sg = s.groupby("poss")
    seq["xg"] = sg["xg"].sum()
    seq["max_shot_xg"] = sg["xg"].max()
    seq["n_shots"] = sg.size()
    seq["n_blocked"] = sg["blocked"].sum()
    seq["goals"] = sg["is_goal"].sum()
    seq["big_chance"] = sg["big_chance"].max()
    for c in ("xg", "max_shot_xg", "n_shots", "n_blocked", "goals", "big_chance"):
        seq[c] = seq[c].fillna(0)

    name = ev.groupby("team_id")["team"].first()
    seq["opp_team"] = seq["opp_team_id"].map(name)
    return seq.reset_index()


# ---------------------------------------------------------------------------
# which attacks actually faced a low block
# ---------------------------------------------------------------------------
#
# A low block is not just a deep one. Two different things put a defence deep:
# choosing to sit, and being pinned there while scrambling. The first is passive —
# the defence concedes passes and waits — and is what this analysis is about. The
# second is a siege, where the defending team is making constant interventions.
# `blk_passivity` (opponent passes per defensive action) separates them.
#
# "Settled" then rules out the other confound: a counter-attack arriving before
# the defence is set is not breaking down a block, it is beating one that was
# never there.

LOW_BLOCK_Q = 0.33     # deepest third of blocks faced, measured at final-third entry
PASSIVE_Q = 0.50       # and at least median passivity, to exclude sieges
SETTLED_PASSES = 3     # final-third passes before an attack counts as settled
SETTLED_SECONDS = 8.0


def label_low_block(
    seq: pd.DataFrame,
    low_q: float = LOW_BLOCK_Q,
    passive_q: float = PASSIVE_Q,
) -> pd.DataFrame:
    """Flag each attack as reaching the final third, settled, and facing a low block.

    Thresholds are percentiles of the league's own distribution rather than fixed
    coordinates, so the same code gives a sensible split in a league that defends
    higher or deeper on average.
    """
    out = seq.copy()
    out["reached_f3"] = out["max_x"].ge(FINAL_THIRD)
    att = out[out["reached_f3"]]

    line_cut = att["blk_line"].quantile(low_q)
    pass_cut = att["blk_passivity"].quantile(passive_q)

    out["settled"] = out["reached_f3"] & (
        out["n_passes_f3"].ge(SETTLED_PASSES) | out["t_in_f3"].ge(SETTLED_SECONDS)
    )
    out["deep_block"] = out["blk_line"].le(line_cut)
    out["passive_block"] = out["blk_passivity"].ge(pass_cut)
    out["vs_low_block"] = out["settled"] & out["deep_block"] & out["passive_block"]
    out["vs_high_block"] = out["settled"] & out["blk_line"].ge(att["blk_line"].quantile(1 - low_q))
    out.attrs["line_cut"] = line_cut
    out.attrs["pass_cut"] = pass_cut
    return out


# Features the pattern search is allowed to look at. Outcome columns are excluded
# by construction: the point is to explain xG, not to rediscover it.
PATTERN_FEATURES = [
    "n_passes_f3", "t_in_f3", "n_switch", "n_cross", "n_cutback", "n_through",
    "n_layoff", "n_takeon", "n_takeon_won", "n_box_touch", "n_carry_in",
    "n_pass_into_box", "width_used", "n_players_f3", "build_up_t", "duration",
]


def lift_table(seq: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    """xG produced per attack, for each candidate pattern, against the baseline.

    Every row is "attacks against a low block that contained at least one X".
    `lift` is xG per attack relative to all low-block attacks, and `se` is the
    standard error on that mean, because a pattern that appears 120 times with
    huge variance is not a finding.
    """
    base = seq["xg"].mean()
    conds = {
        "cutback": seq["n_cutback"].ge(1),
        "cross": seq["n_cross"].ge(1),
        "switch of play": seq["n_switch"].ge(1),
        "2+ switches": seq["n_switch"].ge(2),
        "through ball": seq["n_through"].ge(1),
        "lay-off / third man": seq["n_layoff"].ge(1),
        "take-on attempted": seq["n_takeon"].ge(1),
        "take-on beaten": seq["n_takeon_won"].ge(1),
        "carry into box": seq["n_carry_in"].ge(1),
        "pass into box": seq["n_pass_into_box"].ge(1),
        "3+ box touches": seq["n_box_touch"].ge(3),
        "wide entry": seq["entry_lane"].eq("wide"),
        "halfspace entry": seq["entry_lane"].eq("halfspace"),
        "central entry": seq["entry_lane"].eq("central"),
        "quick (<6s in f3)": seq["t_in_f3"].lt(6),
        "patient (>20s in f3)": seq["t_in_f3"].gt(20),
        "6+ f3 passes": seq["n_passes_f3"].ge(6),
        "full width used (>45)": seq["width_used"].gt(45),
        "5+ players in f3": seq["n_players_f3"].ge(5),
        "from set piece": seq["from_set_piece"].astype(bool),
    }
    rows = []
    for name, m in conds.items():
        s = seq.loc[m, "xg"]
        if len(s) < min_n:
            continue
        rows.append(
            {
                "pattern": name,
                "n": len(s),
                "share": len(s) / len(seq),
                "xg_per_att": s.mean(),
                "lift": s.mean() / base if base else np.nan,
                "se": s.std(ddof=1) / np.sqrt(len(s)),
                "shot_rate": seq.loc[m, "n_shots"].gt(0).mean(),
                "block_rate": seq.loc[m, "n_blocked"].sum() / max(seq.loc[m, "n_shots"].sum(), 1),
            }
        )
    out = pd.DataFrame(rows).sort_values("xg_per_att", ascending=False)
    out.attrs["baseline"] = base
    return out.reset_index(drop=True)


def find_patterns(seq: pd.DataFrame, k: int = 6, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster low-block attacks into archetypes and rank them by what they produce.

    Clustering is on the shape of the attack only. xG is never a feature, so the
    spread in xG across clusters is a result rather than a restatement of the
    input.
    """
    X = seq[PATTERN_FEATURES].fillna(0)
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    out = seq.copy()
    out["cluster"] = km.labels_

    profile = out.groupby("cluster").agg(
        n=("xg", "size"),
        xg_per_att=("xg", "mean"),
        shot_rate=("n_shots", lambda s: (s > 0).mean()),
        goals=("goals", "sum"),
        **{f: (f, "mean") for f in PATTERN_FEATURES},
    )
    profile["share"] = profile["n"] / len(out)
    profile["se"] = out.groupby("cluster")["xg"].std(ddof=1) / np.sqrt(profile["n"])
    profile["label"] = [_name_cluster(r) for _, r in profile.iterrows()]
    return out, profile.sort_values("xg_per_att", ascending=False)


def _name_cluster(r: pd.Series) -> str:
    """A readable handle for an archetype, from whatever dominates its profile."""
    bits = []
    if r["t_in_f3"] < 6:
        bits.append("quick")
    elif r["t_in_f3"] > 20:
        bits.append("patient")
    if r["n_switch"] >= 0.5:
        bits.append("switching")
    if r["n_cutback"] >= 0.25:
        bits.append("cutback")
    elif r["n_cross"] >= 0.8:
        bits.append("crossing")
    if r["n_takeon"] >= 0.7:
        bits.append("dribble-led")
    if r["n_through"] >= 0.2:
        bits.append("through-ball")
    if r["n_box_touch"] >= 1.5:
        bits.append("box-occupying")
    if not bits:
        bits.append("probing")
    return " ".join(bits)


# ---------------------------------------------------------------------------
# two-stage pattern analysis
# ---------------------------------------------------------------------------
#
# A single table ranking every feature by xG is close to circular: "played a pass
# into the box" tops it, but that is most of the way to a chance by definition, so
# the answer restates the question. Breaking the low block is two separate
# problems, and they have different answers:
#
#   stage 1  getting into the box at all  — a build-up question
#   stage 2  what to do on arrival        — a delivery question
#
# Stage 1 is scored on P(reach box), using only features of the build-up. Stage 2
# is conditioned on having reached the box, so every row is comparable and the
# delivery types are ranked against each other rather than against failure.

BUILD_UP_FEATURES = {
    "switch of play": ("n_switch", 1),
    "2+ switches": ("n_switch", 2),
    "take-on attempted": ("n_takeon", 1),
    "take-on beaten": ("n_takeon_won", 1),
    "lay-off / third man": ("n_layoff", 1),
    "6+ final-third passes": ("n_passes_f3", 6),
    "5+ players involved": ("n_players_f3", 5),
    "full width used (>45)": ("width_used", 45),
    "patient (>20s in f3)": ("t_in_f3", 20),
}


def box_entry_rates(seq: pd.DataFrame, min_n: int = 80) -> pd.DataFrame:
    """Stage 1 — which build-up patterns actually get the ball into the box.

    Scored on reach rate rather than xG so it cannot be won by features that are
    themselves nearly a chance. Every condition here is a property of the build-up,
    observable before the ball arrives.
    """
    base = seq["reached_box"].mean()
    rows = []
    for name, (col, cut) in BUILD_UP_FEATURES.items():
        m = seq[col].ge(cut)
        if m.sum() < min_n or (~m).sum() < min_n:
            continue
        hit, miss = seq.loc[m, "reached_box"], seq.loc[~m, "reached_box"]
        rows.append(
            {
                "build_up": name,
                "n": int(m.sum()),
                "share": m.mean(),
                "box_rate": hit.mean(),
                "box_rate_without": miss.mean(),
                "lift": hit.mean() / base if base else np.nan,
                "se": hit.std(ddof=1) / np.sqrt(len(hit)),
                "xg_per_att": seq.loc[m, "xg"].mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values("box_rate", ascending=False)
    out.attrs["baseline"] = base
    return out.reset_index(drop=True)


def delivery_value(seq: pd.DataFrame, min_n: int = 40) -> pd.DataFrame:
    """Stage 2 — conditional on reaching the box, what each delivery is worth.

    `blocked_share` is the point of the exercise as much as xG is: against a packed
    box the difference between deliveries is partly how many of the shots they
    create ever reach the goal.
    """
    got = seq[seq["reached_box"].eq(1)]
    out = (
        got.groupby("box_entry_type")
        .agg(
            n=("xg", "size"),
            xg_per_att=("xg", "mean"),
            shot_rate=("n_shots", lambda s: (s > 0).mean()),
            shots=("n_shots", "sum"),
            blocked=("n_blocked", "sum"),
            goals=("goals", "sum"),
            xg=("xg", "sum"),
        )
        .query("n >= @min_n")
    )
    out["xg_per_shot"] = out["xg"] / out["shots"].clip(lower=1)
    out["blocked_share"] = out["blocked"] / out["shots"].clip(lower=1)
    out["se"] = got.groupby("box_entry_type")["xg"].std(ddof=1) / np.sqrt(out["n"])
    return out.sort_values("xg_per_att", ascending=False)


def compare_blocks(seq: pd.DataFrame) -> pd.DataFrame:
    """Headline contrast: what a low block actually changes about an attack."""
    rows = {}
    for name, m in [
        ("vs low block", seq["vs_low_block"]),
        ("vs high block", seq["vs_high_block"]),
        ("all settled", seq["settled"]),
    ]:
        d = seq[m]
        shots = max(d["n_shots"].sum(), 1)
        rows[name] = {
            "attacks": len(d),
            "xg_per_attack": d["xg"].mean(),
            "box_rate": d["reached_box"].mean(),
            "shot_rate": d["n_shots"].gt(0).mean(),
            "xg_per_shot": d["xg"].sum() / shots,
            "blocked_share": d["n_blocked"].sum() / shots,
            "goals_per_100": 100 * d["goals"].sum() / len(d),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
#
# House style is the project's dark pitch (`#0C0D0E`). The three categorical hues
# are validated against that surface (worst adjacent CVD dE 9.4, normal-vision
# 26.5, all >= 3:1 contrast), so series stay separable for colourblind readers.
# Nominal categories get one hue, never a value-ramp: bar length already encodes
# the value, and ramping it too would burn the only free channel to say it twice.

BG_COLOR = "#0C0D0E"
LINE_COLOR = "#BBBBBB"
TEXT_1, TEXT_2 = "#FFFFFF", "#9A9A94"
SERIES_1, SERIES_2, SERIES_3 = "#3987e5", "#d95926", "#199e70"
GRID_COLOR = "#26282B"


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("lb_seq", [BG_COLOR, "#1d3f68", SERIES_1, "#9fc5f2"])


def _div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("lb_div", [SERIES_1, "#5a5c60", SERIES_2])


def _style_axes(ax, xlabel: str = "", title: str = "") -> None:
    ax.set_facecolor(BG_COLOR)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.xaxis.grid(True, color=GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    if title:
        ax.set_title(title, color=TEXT_1, fontsize=12, loc="left", pad=12)


def plot_block_shape(ev: pd.DataFrame, seq: pd.DataFrame, path: str | Path = "lb_block_shape.png"):
    """Where each kind of block engages the ball.

    Actions are split by the acting team's *own* rolling block height, not by
    which possession they fell in. Selecting by possession looks sensible and is
    useless: every settled attack ends up defending its own box, so both panels
    come out identical and the figure argues for something the data never said.

    The vertical separation between the panels is partly by construction — the
    split is on mean height, so the heights differ. What is not by construction,
    and is the thing to read, is the lateral shape: how far the engagement spreads
    across the pitch, and how much of it concentrates in front of goal.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import VerticalPitch

    d = defensive_actions(ev).dropna(subset=["own_blk_line"])
    lo_cut = seq.attrs.get("line_cut", d["own_blk_line"].quantile(LOW_BLOCK_Q))
    hi_cut = d["own_blk_line"].quantile(1 - LOW_BLOCK_Q)
    panels = [
        (f"Low block  (own actions averaging x <= {lo_cut:.0f})", d[d["own_blk_line"].le(lo_cut)]),
        (f"High block  (x >= {hi_cut:.0f})", d[d["own_blk_line"].ge(hi_cut)]),
    ]

    pitch = VerticalPitch(pitch_type="opta", half=True, pitch_color=BG_COLOR,
                          line_color=LINE_COLOR, linewidth=1, line_zorder=3)
    fig, axs = pitch.draw(nrows=1, ncols=2, figsize=(11, 7.5))
    fig.patch.set_facecolor(BG_COLOR)

    stats = []
    for _, g in panels:
        # Flipped into the attacker's frame so both panels face the same way.
        st = pitch.bin_statistic(100 - g["x"], 100 - g["y"], statistic="count", bins=(14, 10))
        # Per-mille of each group's own actions, so the panels compare *shape*
        # rather than how many actions happened to fall into each group.
        st["statistic"] = st["statistic"] / max(st["statistic"].sum(), 1) * 1000
        stats.append(st)
    vmax = max(np.percentile(st["statistic"], 99) for st in stats)
    for ax, (name, g), st in zip(np.atleast_1d(axs), panels, stats):
        pitch.heatmap(st, ax=ax, cmap=_seq_cmap(), vmin=0, vmax=vmax, zorder=1)
        ax.set_title(f"{name}\n{len(g):,} defensive actions", color=TEXT_1, fontsize=11)
    fig.suptitle("Where a defence engages the ball, by how deep it is sitting",
                 color=TEXT_1, fontsize=14, y=0.97)
    fig.text(0.5, 0.05, "Attacking direction upward · shared colour scale · "
             "per mille of each group's own actions",
             color=TEXT_2, fontsize=9, ha="center")
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_delivery_maps(shots: pd.DataFrame, lo: pd.DataFrame, path: str | Path = "lb_delivery.png"):
    """Where the shots come from, by how the ball entered the box.

    This replaces an earlier "low block vs high block, xG per shot" difference map.
    That map was worth dropping: across all seven leagues the two are 0.095 and
    0.095, so the panel was a field of neutral grey dressed up as a finding. The
    delivery split is where the spatial difference actually lives.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import VerticalPitch

    entry = lo.set_index("poss")["box_entry_type"]
    s = shots[shots["poss"].isin(entry.index)].copy()
    s["entry"] = s["poss"].map(entry)

    order = (
        s[s["entry"].ne("none")].groupby("entry")
        .agg(shots=("xg", "size"), xg=("xg", "sum"))
        # A density map needs enough shots to have a shape. Carry-ins clear the
        # bar for the stage-2 *table* but not for a map, so they drop out here.
        .query("shots >= 50")
        .assign(xg_per_shot=lambda d: d["xg"] / d["shots"])
        .sort_values("xg_per_shot", ascending=False)
    )

    pitch = VerticalPitch(pitch_type="opta", half=True, pitch_color=BG_COLOR,
                          line_color=LINE_COLOR, linewidth=1, line_zorder=3)
    k = len(order)
    fig, axs = pitch.draw(nrows=1, ncols=k, figsize=(3.7 * k, 5.4))
    fig.patch.set_facecolor(BG_COLOR)
    axs = np.atleast_1d(axs).ravel()

    stats = {}
    for name in order.index:
        g = s[s["entry"].eq(name)]
        st = pitch.bin_statistic(g["x"], g["y"], bins=(9, 7))
        st["statistic"] = st["statistic"] / max(st["statistic"].sum(), 1) * 1000
        stats[name] = st
    vmax = max(np.percentile(st["statistic"], 97) for st in stats.values())

    for ax, (name, r) in zip(axs, order.iterrows()):
        pitch.heatmap(stats[name], ax=ax, cmap=_seq_cmap(), vmin=0, vmax=vmax, zorder=1)
        ax.set_title(f"{name}\n{r['xg_per_shot']:.3f} xG per shot\n{int(r['shots']):,} shots",
                     color=TEXT_1, fontsize=10)
    fig.suptitle("A low block is beaten by where the ball arrives, not how often",
                 color=TEXT_1, fontsize=14, y=1.02)
    fig.text(0.5, -0.02, "Shot-origin density, per mille of each delivery's own shots · "
             "shared colour scale · settled attacks against a low block only",
             color=TEXT_2, fontsize=9, ha="center")
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_build_up(lo: pd.DataFrame, path: str | Path = "lb_build_up.png"):
    """Stage 1 and stage 2 side by side: getting into the box, then what to do there.

    Two panels rather than two y-axes on one plot. The measures are different
    quantities (a rate and an xG total) and overlaying them would invent a
    relationship the data does not contain.
    """
    import matplotlib.pyplot as plt

    b = box_entry_rates(lo).sort_values("box_rate")
    d = delivery_value(lo).sort_values("xg_per_att")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), facecolor=BG_COLOR)

    ax = axs[0]
    ax.barh(b["build_up"], b["box_rate"], xerr=b["se"], color=SERIES_1, height=0.62,
            error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 3})
    ax.axvline(b.attrs["baseline"], color=TEXT_2, ls="--", lw=1)
    ax.text(b.attrs["baseline"], len(b) - 0.35,
            f" baseline {b.attrs['baseline']:.0%}", color=TEXT_2, fontsize=8, va="center")
    pad = max(b["box_rate"]) * 0.03
    for y, (v, e, n) in enumerate(zip(b["box_rate"], b["se"], b["n"])):
        ax.text(v + e + pad, y, f"{v:.0%}  (n={n:,})", color=TEXT_1, fontsize=8, va="center")
    ax.set_xlim(0, max(b["box_rate"] + b["se"]) * 1.42)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style_axes(ax, "share of attacks that reach the box",
                "1 · Which build-up gets you into the box")

    ax = axs[1]
    ax.barh(d.index, d["xg_per_att"], xerr=d["se"], color=SERIES_2, height=0.62,
            error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 3})
    pad = d["xg_per_att"].max() * 0.03
    for y, (v, e, n, bl) in enumerate(zip(d["xg_per_att"], d["se"], d["n"], d["blocked_share"])):
        ax.text(v + e + pad, y, f"{v:.3f}   n={int(n):,}   {bl:.0%} blocked",
                color=TEXT_1, fontsize=8, va="center")
    ax.set_xlim(0, (d["xg_per_att"] + d["se"]).max() * 1.75)
    _style_axes(ax, "xG per attack (attacks that reached the box only)",
                "2 · What the delivery is worth once you are there")

    fig.suptitle("Breaking a low block is two problems, and they have different answers",
                 color=TEXT_1, fontsize=15, x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.005, "Bars show the mean; whiskers are one standard error.",
             color=TEXT_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    return path


def plot_archetypes(
    ev: pd.DataFrame,
    lo: pd.DataFrame,
    profile: pd.DataFrame,
    path: str | Path = "lb_archetypes.png",
):
    """Each attacking archetype as a pitch, ordered by the xG it produces.

    The map is where that archetype actually puts the ball in the final third,
    as a share of its own touches, so a rare archetype is not simply a fainter
    version of a common one.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import VerticalPitch

    cluster = lo.set_index("poss")["cluster"]
    f3 = ev[ev["x"].ge(FINAL_THIRD) & ev["team_id"].eq(ev["att_team_id"])].copy()
    f3["cluster"] = f3["poss"].map(cluster)
    f3 = f3.dropna(subset=["cluster"])

    k = len(profile)
    pitch = VerticalPitch(pitch_type="opta", half=True, pitch_color=BG_COLOR,
                          line_color=LINE_COLOR, linewidth=1, line_zorder=3)
    ncols = min(k, 3)
    nrows = int(np.ceil(k / ncols))
    fig, axs = pitch.draw(nrows=nrows, ncols=ncols, figsize=(4.4 * ncols, 5.6 * nrows))
    fig.patch.set_facecolor(BG_COLOR)
    axs = np.atleast_1d(axs).ravel()

    stats = {}
    for cid in profile.index:
        g = f3[f3["cluster"].eq(cid)]
        st = pitch.bin_statistic(g["x"], g["y"], bins=(10, 8))
        st["statistic"] = st["statistic"] / max(st["statistic"].sum(), 1) * 1000
        stats[cid] = st
    vmax = max(np.percentile(st["statistic"], 98) for st in stats.values())

    for ax, (cid, r) in zip(axs, profile.iterrows()):
        pitch.heatmap(stats[cid], ax=ax, cmap=_seq_cmap(), vmin=0, vmax=vmax, zorder=1)
        ax.set_title(
            f"{r['label']}\n{r['xg_per_att']:.3f} xG per attack"
            f"   ·   {int(r['n']):,} attacks ({r['share']:.0%})",
            color=TEXT_1, fontsize=10,
        )
    for ax in axs[k:]:
        ax.set_visible(False)
    fig.suptitle("Low-block attacking archetypes, ranked by what they produce",
                 color=TEXT_1, fontsize=15, y=0.99)
    fig.text(0.5, 0.02, "Final-third touch density, per mille of each archetype's own touches · "
             "shared colour scale · clustering never sees xG",
             color=TEXT_2, fontsize=9, ha="center")
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_teams(seq: pd.DataFrame, min_attacks: int = 60, path: str | Path = "lb_teams.png"):
    """Who breaks low blocks, and who sits in one that holds.

    Attacking quality is xG per settled attack against a low block; defensive
    quality is the same number conceded. One axis each, no ranking colour: the
    position on the plot is the statement.
    """
    import matplotlib.pyplot as plt

    lo = seq[seq["vs_low_block"]]
    atk = lo.groupby("team").agg(att=("xg", "size"), xg=("xg", "mean"))
    opp = lo.groupby("opp_team").agg(dfn=("xg", "size"), conceded=("xg", "mean"))
    t = atk.join(opp, how="inner")
    t = t[(t["att"] >= min_attacks) & (t["dfn"] >= min_attacks)]

    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor=BG_COLOR)
    ax.scatter(t["xg"], t["conceded"], s=46, color=SERIES_1, zorder=3,
               edgecolor=BG_COLOR, linewidth=1.5)
    for name, r in t.iterrows():
        ax.annotate(name, (r["xg"], r["conceded"]), fontsize=7.5, color=TEXT_2,
                    xytext=(5, 3), textcoords="offset points")
    ax.axvline(t["xg"].mean(), color=GRID_COLOR, lw=1)
    ax.axhline(t["conceded"].mean(), color=GRID_COLOR, lw=1)
    _style_axes(ax, "xG per settled attack AGAINST a low block  →  better at breaking one",
                "Breaking low blocks vs holding one")
    ax.set_ylabel("xG per attack CONCEDED while in a low block  →  worse at holding one",
                  color=TEXT_2, fontsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# pipeline + cli
# ---------------------------------------------------------------------------


def build(
    leagues: list[str] | None = None,
    season: int | str = DEFAULT_SEASON,
    events_root: str | Path = "league_games",
    k: int = 6,
) -> dict:
    """Run the whole pipeline and return every frame the reports and plots need."""
    ev = load_events(leagues, season, events_root)
    print(f"{len(ev):,} events, {ev['game_id'].nunique():,} games", flush=True)

    ev["poss"] = segment_possessions(ev)
    shots = shot_frame(ev)
    shots["xg"] = fit_xg(shots)
    shots["poss"] = ev["poss"].reindex(shots.index)
    print(f"{len(shots):,} shots, {int(shots['is_goal'].sum()):,} goals, "
          f"{shots['xg'].sum():.0f} xG", flush=True)

    ev = attach_block_state(ev, block_state(ev))
    seq = label_low_block(build_sequences(ev, shots))
    ev["att_team_id"] = ev["poss"].map(seq.set_index("poss")["team_id"])

    lo = seq[seq["vs_low_block"]].copy()
    lo, profile = find_patterns(lo, k=k)
    chains = pass_chains(ev, lo)
    motifs = chain_motifs(chains, lo, k=3, min_n=25, top=18)
    print(f"{len(seq):,} possessions, {int(seq['reached_f3'].sum()):,} reaching the final "
          f"third, {len(lo):,} settled attacks against a low block", flush=True)
    return {"events": ev, "shots": shots, "sequences": seq, "low_block": lo,
            "profile": profile, "chains": chains, "motifs": motifs}


def report(res: dict) -> None:
    """Print every table, in the order the argument is made."""
    seq, lo = res["sequences"], res["low_block"]
    pd.set_option("display.width", 200)

    print("\n=== xG model calibration, by shot distance ===")
    print(xg_calibration(res["shots"], by="dist", n_bins=8).round(4).to_string(index=False))

    print("\n=== what a low block actually changes ===")
    print(compare_blocks(seq).round(4).to_string())

    print("\n=== stage 1: which build-up reaches the box ===")
    b = box_entry_rates(lo)
    print(f"baseline: {b.attrs['baseline']:.1%} of low-block attacks reach the box")
    print(b.round(4).to_string(index=False))

    print("\n=== stage 2: what the delivery is worth, given the box was reached ===")
    print(delivery_value(lo).round(4).to_string())

    print("\n=== archetypes ===")
    cols = ["label", "n", "share", "xg_per_att", "se", "shot_rate", "t_in_f3",
            "n_switch", "n_cross", "n_cutback", "n_takeon", "n_box_touch"]
    print(res["profile"][cols].round(3).to_string())

    print("\n=== consecutive-pass patterns: the 3-pass motifs attacks end on ===")
    m = res["motifs"]
    print(f"baseline: {m.attrs['baseline']:.4f} xG per attack, over "
          f"{m.attrs['n_attacks']:,} attacks with a 3-pass tail")
    print(m.round(4).to_string())

    print("\n=== same destination, different route: attacks whose last pass "
          "lands in the middle of the box ===")
    rc = route_comparison(res["chains"], lo)
    print(f"baseline: {rc.attrs['baseline']:.4f} xG over {rc.attrs['n_attacks']:,} "
          f"attacks ending in {rc.attrs['ends']}")
    print(rc.round(4).to_string())

    print("\n=== single zone-to-zone passes, by the xG of the attacks containing them ===")
    print(zone_transitions(res["chains"], lo).head(15).round(4).to_string())

    print("\n=== every pattern, unconditional (read stages 1 and 2 first) ===")
    lt = lift_table(lo)
    print(f"baseline: {lt.attrs['baseline']:.4f} xG per low-block attack")
    print(lt.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--league", action="append", dest="leagues",
                    help="repeatable; defaults to every collected league")
    ap.add_argument("--all-leagues", action="store_true")
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--events-root", default="league_games")
    ap.add_argument("--clusters", type=int, default=6)
    ap.add_argument("--plot", action="store_true", help="write the figures")
    ap.add_argument("--plot-dir", default="low_block_plots")
    ap.add_argument("--out", help="write the sequence table to this csv")
    args = ap.parse_args()

    leagues = None if args.all_leagues else args.leagues
    res = build(leagues, args.season, args.events_root, k=args.clusters)
    report(res)

    if args.out:
        # Only attacks that reached the final third: the rest are midfield spells
        # that no part of this analysis looks at, and they quadruple the file.
        keep = res["sequences"][res["sequences"]["reached_f3"]]
        keep.to_csv(args.out, index=False)
        print(f"\nwrote {args.out} ({len(keep):,} attacks reaching the final third)")

    if args.plot:
        d = Path(args.plot_dir)
        d.mkdir(parents=True, exist_ok=True)
        for p in (
            plot_block_shape(res["events"], res["sequences"], d / "block_shape.png"),
            plot_delivery_maps(res["shots"], res["low_block"], d / "delivery.png"),
            plot_build_up(res["low_block"], d / "build_up.png"),
            plot_archetypes(res["events"], res["low_block"], res["profile"], d / "archetypes.png"),
            plot_motifs(res["motifs"], d / "motifs.png"),
            plot_chain_examples(res["chains"], res["low_block"], res["shots"],
                                d / "chain_examples.png"),
            plot_teams(res["sequences"], path=d / "teams.png"),
        ):
            print(f"wrote {p}")




# ---------------------------------------------------------------------------
# pass chains: the actual sequences that broke the block
# ---------------------------------------------------------------------------
#
# Everything above counts *whether* a pattern appeared. This part keeps the
# order, because against a low block the order is most of the point: a cutback
# after a switch is a different move from a switch after a cutback, and the
# counting version cannot tell them apart.
#
# Each chain is mirrored so the attack's final-third entry is always on the same
# side of the pitch. Left- and right-sided versions of one move then pool into a
# single motif — doubling the sample behind every pattern — while the *relative*
# movement that matters is preserved exactly: a switch still reads as near side
# to far side.

# Final-third bands, in Opta x. Below `FINAL_THIRD` is build-up.
BANDS = [(FINAL_THIRD, BOX_X, "edge"), (BOX_X, SIX_YARD_X, "box"), (SIX_YARD_X, 101.0, "six")]

# Lanes across the mirrored pitch. "near" is the side the attack entered on.
LANES = [(0, 20, "far wing"), (20, 40, "far half"), (40, 60, "middle"),
         (60, 80, "near half"), (80, 101, "near wing")]


def _zone(x: pd.Series, y: pd.Series, coarse: bool = False) -> pd.Series:
    """Compact zone label, e.g. `box/near half`. Build-up play collapses to one zone.

    `coarse` merges the six-yard band into the box. Five lanes times three bands
    is 16 zones and so 4,096 possible three-pass motifs, which spreads 12,000
    attacks far too thin to rank. The lane axis is the tactically interesting one
    — wing against halfspace against middle — so depth is what gets merged.
    """
    bands = [(FINAL_THIRD, BOX_X, "edge"), (BOX_X, 101.0, "box")] if coarse else BANDS
    band = pd.Series("build", index=x.index, dtype=object)
    for lo_x, hi_x, name in bands:
        band[(x >= lo_x) & (x < hi_x)] = name
    lane = pd.Series("middle", index=y.index, dtype=object)
    for lo_y, hi_y, name in LANES:
        lane[(y >= lo_y) & (y < hi_y)] = name
    return np.where(band.eq("build"), "build-up", band + "/" + lane)


def pass_chains(ev: pd.DataFrame, seq: pd.DataFrame, before_entry: float = 12.0) -> pd.DataFrame:
    """Every pass in each attack, in order, mirrored and zone-coded.

    Keeps `before_entry` seconds of build-up either side of the final-third entry,
    because the move that breaks a block often starts behind it — a switch is
    played from deep and only looks like a pattern if the deep pass is in the
    chain.
    """
    keep = seq.set_index("poss")
    p = ev[
        ev["type"].eq("Pass")
        & ev["outcome_type"].eq("Successful")
        & ev["team_id"].eq(ev["att_team_id"])
        & ev["poss"].isin(keep.index)
    ].copy()

    p["entry_t"] = p["poss"].map(keep["entry_t"])
    p = p[p["t"] >= p["entry_t"] - before_entry]

    # Mirror so every attack enters the final third on the high-y ("near") side.
    flip = p["poss"].map(keep["entry_y"]).lt(50)
    for col in ("y", "end_y"):
        p[col] = np.where(flip, 100.0 - p[col], p[col])

    p = p.sort_values(["poss", "t"], kind="stable")
    p["step"] = p.groupby("poss").cumcount()
    p["zone_from"] = _zone(p["x"], p["y"])
    p["zone_to"] = _zone(p["end_x"], p["end_y"])
    p["coarse_from"] = _zone(p["x"], p["y"], coarse=True)
    p["coarse_to"] = _zone(p["end_x"], p["end_y"], coarse=True)

    q = p["qualifiers"]
    p["dy"] = p["end_y"] - p["y"]
    p["is_switch"] = p["dy"].abs().ge(30)
    p["is_cross"] = _has_qualifier(q, "Cross")
    p["is_through"] = _has_qualifier(q, "Throughball")
    p["is_cutback"] = (
        p["x"].ge(BOX_X) & p["end_x"].lt(p["x"])
        & p["y"].sub(50).abs().ge(15) & p["end_y"].sub(50).abs().le(20)
    )
    p["xg"] = p["poss"].map(keep["xg"])
    return p


def chain_motifs(
    chains: pd.DataFrame,
    seq: pd.DataFrame,
    k: int = 3,
    min_n: int = 25,
    top: int = 18,
    max_build_up: int = 1,
) -> pd.DataFrame:
    """The `k`-pass zone motifs an attack ends on, ranked by the xG they produce.

    Ranked on xG per attack rather than on how often the motif appears: the common
    ones are common because they are easy, not because they work.

    `max_build_up` drops motifs that are mostly passes from behind the final third.
    Those are real attacks, but they describe how the ball arrived rather than any
    pattern worked against the block, and left in they crowd out the thing being
    looked for.
    """
    tail = chains.groupby("poss").tail(k)
    motif = (
        tail.groupby("poss")
        .agg(motif=("coarse_to", lambda z: " → ".join(z)), passes=("coarse_to", "size"))
        .query("passes == @k")
    )
    s = seq.set_index("poss").loc[motif.index]
    motif["xg"] = s["xg"].to_numpy()
    motif["shot"] = s["n_shots"].gt(0).to_numpy()
    motif["reached_box"] = s["reached_box"].to_numpy()

    motif = motif[motif["motif"].str.count("build-up") <= max_build_up]
    out = (
        motif.groupby("motif")
        .agg(n=("xg", "size"), xg_per_att=("xg", "mean"), shot_rate=("shot", "mean"),
             box_rate=("reached_box", "mean"), se=("xg", lambda v: v.std(ddof=1) / np.sqrt(len(v))))
        .query("n >= @min_n")
        .sort_values("xg_per_att", ascending=False)
    )
    out.attrs["baseline"] = seq["xg"].mean()
    out.attrs["n_attacks"] = len(motif)
    return out.head(top)


def zone_transitions(chains: pd.DataFrame, seq: pd.DataFrame, min_n: int = 40) -> pd.DataFrame:
    """Single zone-to-zone passes, ranked by the xG of the attacks containing them.

    The motif table answers "what shape of move works"; this answers the smaller
    question of which individual ball is worth playing, and has far more sample
    behind each row.
    """
    x = chains.dropna(subset=["xg"])
    out = (
        x.groupby(["zone_from", "zone_to"])
        .agg(n=("xg", "size"), xg_per_att=("xg", "mean"),
             se=("xg", lambda v: v.std(ddof=1) / np.sqrt(len(v))))
        .query("n >= @min_n")
        .sort_values("xg_per_att", ascending=False)
    )
    out.attrs["baseline"] = seq["xg"].mean()
    return out


# Where each zone label sits on the pitch, for drawing a motif as a path.
# Build-up has no lane (it is everything behind the final third), so it is
# anchored on the centre spot just short of the line.
def _zone_centroid(label: str) -> tuple[float, float]:
    if label == "build-up":
        return 58.0, 50.0
    band, lane = label.split("/")
    x = 74.8 if band == "edge" else 92.0
    y = {name: (lo + hi) / 2 for lo, hi, name in LANES}[lane]
    return x, min(y, 96.0)


def plot_motifs(
    motifs: pd.DataFrame,
    path: str | Path = "lb_motifs.png",
    k: int = 6,
    baseline: float | None = None,
):
    """The highest-value consecutive-pass patterns, each drawn as a path.

    This replaces a flow field of every pass, which at this volume was a mat of
    overlapping arrows that said nothing. A motif is an ordered sequence, and the
    honest way to draw an ordered sequence is as a path with the steps numbered.

    Positions are zone centroids, not real coordinates: the claim is "wing, then
    the near half-space inside the box, then the middle", not any particular pass.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import VerticalPitch

    top = motifs.head(k)
    base = baseline if baseline is not None else motifs.attrs.get("baseline", np.nan)

    pitch = VerticalPitch(pitch_type="opta", half=True, pitch_color=BG_COLOR,
                          line_color=LINE_COLOR, linewidth=1, line_zorder=1)
    ncols = min(len(top), 3)
    nrows = int(np.ceil(len(top) / ncols))
    fig, axs = pitch.draw(nrows=nrows, ncols=ncols, figsize=(4.3 * ncols, 5.6 * nrows))
    fig.patch.set_facecolor(BG_COLOR)
    axs = np.atleast_1d(axs).ravel()

    for ax, (label, r) in zip(axs, top.iterrows()):
        pts = [_zone_centroid(z) for z in label.split(" → ")]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pitch.arrows(xs[:-1], ys[:-1], xs[1:], ys[1:], ax=ax, color=SERIES_1,
                     width=2.6, headwidth=4.5, headlength=4.5, zorder=3)
        for i, (px, py) in enumerate(pts, 1):
            pitch.annotate(str(i), (px, py), ax=ax, color=TEXT_1, fontsize=8,
                           ha="center", va="center", zorder=5,
                           bbox={"boxstyle": "circle,pad=0.24", "fc": BG_COLOR,
                                 "ec": SERIES_1, "lw": 1.0})
        lift = f"{r['xg_per_att'] / base:.1f}x" if base == base else ""
        ax.set_title(
            label.replace(" → ", "  →  ")
            + f"\n{r['xg_per_att']:.3f} xG per attack  ({lift} baseline)"
            + f"\nn={int(r['n'])}  ·  {r['shot_rate']:.0%} end in a shot",
            color=TEXT_1, fontsize=9,
        )
        pitch.text(70, 90, "near", ax=ax, color=TEXT_2, fontsize=7.5, ha="center")
        pitch.text(70, 10, "far", ax=ax, color=TEXT_2, fontsize=7.5, ha="center")
    for ax in axs[len(top):]:
        ax.set_visible(False)

    fig.suptitle("The consecutive-pass patterns that break a low block",
                 color=TEXT_1, fontsize=15, y=1.0)
    fig.text(0.5, 0.005, "Positions are zone centroids, not real coordinates · every attack "
             "mirrored so it enters the final third on the near side · "
             "ranked by xG per attack, not by how often the pattern appears",
             color=TEXT_2, fontsize=9, ha="center")
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_chain_examples(
    chains: pd.DataFrame,
    seq: pd.DataFrame,
    shots: pd.DataFrame,
    path: str | Path = "lb_chain_examples.png",
    n: int = 6,
    last: int = 7,
    min_passes_f3: int = 4,
):
    """The highest-xG chains themselves, one pitch each, passes numbered in order.

    One example per team, so the panel shows six ways of breaking a low block
    rather than six of whichever side did it best that season. These are drawn
    unmirrored: they are real moves, and a real move has a real side.

    Ranking purely on xG fills the panel with one cross and a tap-in — the highest
    single values belong to moves with almost no passing in them, which is the
    opposite of what a figure about pass sequences should show. So a minimum of
    `min_passes_f3` final-third passes applies, and possessions containing a
    penalty are dropped: a penalty's 0.79 swamps the sort and says nothing about
    how the block was opened.
    """
    import matplotlib.pyplot as plt
    from mplsoccer import VerticalPitch

    s = seq.set_index("poss")
    pens = set(shots.loc[shots["is_pen"].eq(1), "poss"].dropna())
    best = (
        s[s["n_shots"].gt(0)
          & s["n_passes_f3"].ge(min_passes_f3)
          & ~s.index.isin(pens)]
        .sort_values("xg", ascending=False)
        .groupby("team", sort=False)
        .head(1)
        .head(n)
    )

    pitch = VerticalPitch(pitch_type="opta", half=True, pitch_color=BG_COLOR,
                          line_color=LINE_COLOR, linewidth=1, line_zorder=1)
    ncols = min(n, 3)
    nrows = int(np.ceil(len(best) / ncols))
    fig, axs = pitch.draw(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 5.9 * nrows))
    fig.patch.set_facecolor(BG_COLOR)
    axs = np.atleast_1d(axs).ravel()

    for ax, (poss, r) in zip(axs, best.iterrows()):
        c = chains[chains["poss"].eq(poss)].tail(last)
        if c.empty:
            continue
        # Unmirror: pass_chains flipped these to pool motifs, and an example move
        # should be drawn on the side it actually happened.
        y, ey = (100 - c["y"], 100 - c["end_y"]) if r["entry_y"] < 50 else (c["y"], c["end_y"])
        fade = np.linspace(0.45, 1.0, len(c))
        # Only successful passes are drawn, so consecutive arrows do not always
        # meet: the gap is a carry, a duel, or a touch that kept the move alive.
        # A dotted link makes the move read as one continuous action instead of
        # a scatter of disconnected arrows.
        ex, ey_, sx, sy_ = c["end_x"].to_numpy()[:-1], ey.to_numpy()[:-1],             c["x"].to_numpy()[1:], y.to_numpy()[1:]
        gap = np.hypot(sx - ex, sy_ - ey_) > 2.0
        if gap.any():
            pitch.lines(ex[gap], ey_[gap], sx[gap], sy_[gap], ax=ax, color=TEXT_2,
                        lw=1.0, linestyle=(0, (2, 2)), zorder=2, comet=False)
        pitch.arrows(c["x"], y, c["end_x"], ey, ax=ax, color=SERIES_1, width=2.2,
                     headwidth=4.5, headlength=4.5, alpha=fade, zorder=3)
        for i, (px, py) in enumerate(zip(c["x"], y), 1):
            pitch.annotate(str(i), (px, py), ax=ax, color=TEXT_1, fontsize=7.5,
                           ha="center", va="center", zorder=5,
                           bbox={"boxstyle": "circle,pad=0.22", "fc": BG_COLOR,
                                 "ec": SERIES_1, "lw": 0.9})
        sh = shots[shots["poss"].eq(poss)].sort_values("xg", ascending=False).head(1)
        if not sh.empty:
            sy = 100 - sh["y"] if r["entry_y"] < 50 else sh["y"]
            pitch.scatter(sh["x"], sy, ax=ax, s=210, marker="*", color=SERIES_2,
                          edgecolor=BG_COLOR, linewidth=1, zorder=6)
        who = c["player"].dropna().iloc[-1] if c["player"].notna().any() else ""
        shot_n = int(r["n_shots"])
        ax.set_title(f"{r['team']} vs {r['opp_team']}\n{r['xg']:.2f} xG from "
                     f"{shot_n} shot{'s' if shot_n != 1 else ''}  ·  "
                     f"{int(r['n_passes_f3'])} final-third passes\nlast touch: {who}",
                     color=TEXT_1, fontsize=9.5)
    for ax in axs[len(best):]:
        ax.set_visible(False)
    fig.suptitle("Six low blocks being broken, pass by pass", color=TEXT_1, fontsize=15, y=1.0)
    fig.text(0.5, 0.01, "Arrows numbered in order, fading in from the start of the move · "
             "dotted links are carries · star marks the best chance in the move, which "
             "in a multi-shot possession need not be the last event · one example per team",
             color=TEXT_2, fontsize=9, ha="center")
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    return path




def route_comparison(
    chains: pd.DataFrame,
    seq: pd.DataFrame,
    ends: str = "box/middle",
    k: int = 3,
    min_n: int = 20,
) -> pd.DataFrame:
    """Among attacks whose last pass lands in the same zone, which route was worth more.

    `chain_motifs` ranks whole motifs, and that ranking is partly circular: an
    attack whose last pass lands in the middle of the box has already reached the
    box, and reaching the box is most of what produces xG. Every row here ends in
    the same zone, so that advantage is held constant and what is left is the
    route — which is the part a coach can actually choose.
    """
    tail = chains.groupby("poss").tail(k)
    m = (
        tail.groupby("poss")
        .agg(route=("coarse_to", lambda z: " → ".join(list(z)[:-1])),
             final=("coarse_to", "last"), passes=("coarse_to", "size"))
        .query("passes == @k and final == @ends")
    )
    s = seq.set_index("poss").loc[m.index]
    m["xg"], m["shot"] = s["xg"].to_numpy(), s["n_shots"].gt(0).to_numpy()

    out = (
        m.groupby("route")
        .agg(n=("xg", "size"), xg_per_att=("xg", "mean"), shot_rate=("shot", "mean"),
             se=("xg", lambda v: v.std(ddof=1) / np.sqrt(len(v))))
        .query("n >= @min_n")
        .sort_values("xg_per_att", ascending=False)
    )
    out.attrs["ends"] = ends
    out.attrs["baseline"] = m["xg"].mean()
    out.attrs["n_attacks"] = len(m)
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
