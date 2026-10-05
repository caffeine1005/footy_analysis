"""Rate how physical a player is, relative to others in the same position.

There is no height, weight, speed or tracking data here, so "physical" has to be
read off what WhoScored/Opta events do record: contested balls. Every duel is
logged as an adjacent pair of events, one per player — `Aerial` against
`Aerial`, `Tackle` against the `TakeOn` or `Dispossessed` it stopped, and
`Challenge` against the `TakeOn` that beat it — so each duel names both players
and who won it.

That pairing is what makes this more than a win-rate table. A raw aerial win
rate mostly measures who a player goes up against: a striker contests headers
with centre-backs all season, a full-back with wingers. So duel strength is a
Bradley-Terry rating, P(i beats j) = sigmoid(s_i - s_j + context), fitted over
every head-to-head in the pooled leagues, where the context term carries pitch
location (for aerials) or which side of a ground duel the player was on. The
L2 penalty on the ratings is the shrinkage: a player with ten duels sits near the
population mean, one with two hundred earns their own number. Within role
groups this replicates better across odd/even matches than raw win rate does
(aerials r ~ 0.54 against 0.45 on 25/26). Raw ground-duel win rate looks *more*
reliable, but only because it bakes in a player's mix of tackling versus carrying
duels — the tackler wins roughly four times as often — which is a stable role
trait rather than strength.

Ground duels are rated from the tackler's side only. The carrier's side of the
same model is dominated by take-ons, and its leaderboard (Yamal, Doku, Mbappé,
Vinícius) is dribbling skill, which `tight_space`'s beating-the-man axis already
measures. Taking it out keeps this about contact.

The output is a profile across six axes, as in `tight_space`:

  * aerial      — opponent-adjusted aerial duel rating;
  * ground      — opponent-adjusted rating in ground duels, as the tackler;
  * contact     — fouls drawn above what the player's touch locations predict,
                  i.e. taking contact on the ball rather than avoiding it;
  * engagement  — volume: aerials, tackling duels and fouls committed per 90.
                  How often a player gets into physical contests at all, which
                  the three execution axes deliberately give no credit for;
  * carrying    — ground covered with the ball: progressive carry metres and
                  long (15m+) carries, possession-adjusted;
  * work_rate   — defensive coverage: how widely spread a player's defensive
                  actions are, and how many happen in the opponent's half,
                  possession-adjusted.

The last two stand in for running data, which none of the sources here carry.
A carry is reconstructed from consecutive events: where a player received the
ball (a team-mate's pass end, or their own recovery/touch) to where their next
action starts. Only *volume* survives this. Carry speed was tried — distance over
the gap between timestamps — and failed: WhoScored clocks are whole seconds,
split-half reliability was ~0.4, and the "fastest" players were Füllkrug and
Mühren. Late-game output as a stamina proxy failed too (reliability ~0.3). Both
are left out rather than dressed up.

Possession adjustment ("PAdj") rescales a count to what it would be at 50%
possession: carries against the player's own team's share of the ball, pressing
actions against the opponent's. Otherwise a PSG full-back out-carries everyone
because PSG have the ball, and a side that is always defending looks tireless.

Every axis is z-scored within the player's role group (formation line plus a
central/wide split, from `tight_space.refine_roles`), so a centre-back is rated
against centre-backs and a winger against wingers. The summary blend then uses
*position-specific* weights — aerial ability is most of what "physical" means for
a centre-back or a target striker, ground duels for a holding midfielder — which
are a stated preference, overridable, and not a measurement.

CLI:

    python physicality.py --season 2025 --top 25
    python physicality.py --role DF-C --sort-by axis_aerial --top 20
    python physicality.py --player "Virgil van Dijk"
    python physicality.py --validate --external-check
    python physicality.py --warm               # build and cache every season
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

import tight_space as ts
from touchmap_similarity import DEFAULT_LEAGUES, DEFAULT_SEASON, estimate_minutes, load_all_league_events

# Bumped whenever the scoring changes, so cached tables from an older model are
# rebuilt rather than silently mixed with new ones.
MODEL_VERSION = 3

CACHE_DIR = Path(".desktop_cache") / "physicality"

# Two events count as one duel when they are adjacent in the match stream, by
# opposite teams, and at most this many seconds apart. In practice both halves
# of a pair share a timestamp; one second of slack covers rounding.
PAIR_MAX_GAP_S = 1

# Inverse L2 strength for the Bradley-Terry fits (sklearn's C). Chosen on
# split-half reliability within role groups over 25/26: stronger shrinkage than
# this starts flattening real differences, weaker lets small samples through.
BT_C = 0.02

# Fouls are rare, so a player's own rate takes a few hundred actions to outweigh
# the league's (same constant, same reasoning as `tight_space.FOULS_WON_K`).
CONTACT_K = 300.0

DEFAULT_MIN_MINUTES = 600

# Z-scores are capped here before any blending (see `_z_within`).
Z_CLIP = 3.0

# Opta's 0-100 pitch in metres, for carry distances.
PITCH_M = (105.0, 68.0)

# A carry of at least this many metres counts as a long carry.
LONG_CARRY_M = 15.0

# Defensive actions a player needs before the spread of them is read as range.
MIN_RANGE_ACTIONS = 15

# Events that can start a carry by the same player, and events a carry can end in.
CARRY_OWN_START = ("BallRecovery", "Interception", "TakeOn", "BallTouch", "Tackle")
CARRY_END = ("Pass", "TakeOn", "Dispossessed", "BallTouch", "MissedShots", "SavedShot", "Goal", "ShotOnPost")

AXES = {
    "aerial": {"aerial_score": 1.0},
    "ground": {"ground_score": 1.0},
    "contact": {"contact_score": 1.0},
    "engagement": {"aerials_p90": 0.4, "ground_duels_p90": 0.4, "fouls_committed_p90": 0.2},
    "carrying": {"prog_carry_m_padj": 0.6, "long_carries_padj": 0.4},
    "work_rate": {"def_range_m": 0.5, "high_def_padj": 0.5},
}
AXIS_LABELS = {
    "aerial": "Aerial duels",
    "ground": "Ground duels",
    "contact": "Taking contact",
    "engagement": "Engagement",
    "carrying": "Carrying",
    "work_rate": "Work rate",
}
COMPONENT_COLS = [c for spec in AXES.values() for c in spec]
AXIS_COLS = [f"axis_{a}" for a in AXES]

# What "physical" means depends on the position, so the summary blend does too.
# These are a stated preference, not a fit: the axes are measured the same way
# for everyone, only how they are fused into one number differs.
#
# The running proxies (carrying, work_rate) weigh most where covering ground is
# the job — full-backs, wingers, box-to-box midfielders — and least for
# centre-backs and target strikers, whose physicality is mostly in the air.
ROLE_AXIS_WEIGHTS = {
    "DF-C": {"aerial": 0.35, "ground": 0.25, "contact": 0.05, "engagement": 0.20, "carrying": 0.05, "work_rate": 0.10},
    "DF-W": {"aerial": 0.15, "ground": 0.30, "contact": 0.05, "engagement": 0.15, "carrying": 0.20, "work_rate": 0.15},
    "MF-C": {"aerial": 0.15, "ground": 0.30, "contact": 0.10, "engagement": 0.15, "carrying": 0.10, "work_rate": 0.20},
    "MF-W": {"aerial": 0.05, "ground": 0.25, "contact": 0.20, "engagement": 0.15, "carrying": 0.20, "work_rate": 0.15},
    "FW-C": {"aerial": 0.40, "ground": 0.05, "contact": 0.25, "engagement": 0.10, "carrying": 0.05, "work_rate": 0.15},
    "FW-W": {"aerial": 0.10, "ground": 0.15, "contact": 0.25, "engagement": 0.15, "carrying": 0.20, "work_rate": 0.15},
}


def _line_weights(line: str) -> dict[str, float]:
    """A player whose side could not be placed gets the mean of their line's two."""
    c, w = ROLE_AXIS_WEIGHTS.get(f"{line}-C"), ROLE_AXIS_WEIGHTS.get(f"{line}-W")
    if c is None or w is None:
        return {a: 1.0 / len(AXES) for a in AXES}
    return {a: (c[a] + w[a]) / 2 for a in AXES}


def axis_weights_for(role_group: str) -> dict[str, float]:
    if role_group in ROLE_AXIS_WEIGHTS:
        return ROLE_AXIS_WEIGHTS[role_group]
    return _line_weights(str(role_group).split("-")[0])


# ---------------------------------------------------------------------------
# duels, straight out of the event pairs
# ---------------------------------------------------------------------------


def _adjacent_pairs(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Each event beside the one after it, and whether the two form a contest.

    Requires `events` in within-match stream order, which is how the match csvs
    are stored and concatenated.
    """
    cur = events.reset_index(drop=True)
    nxt = cur.shift(-1)
    t_cur = cur["minute"] * 60 + cur["second"].fillna(0)
    t_nxt = nxt["minute"] * 60 + nxt["second"].fillna(0)
    contest = (
        nxt["game_id"].eq(cur["game_id"])
        & nxt["team_id"].ne(cur["team_id"])
        & (t_nxt - t_cur).abs().le(PAIR_MAX_GAP_S)
        & cur["player_id"].notna()
        & nxt["player_id"].notna()
    ).to_numpy()
    return cur, nxt, contest


def pair_duels(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Head-to-head aerial and ground duels, one row per contest.

    `aerial`: winner_id, loser_id and winner_x (in the winner's attacking frame).
    `ground`: def_id, carry_id, def_x (tackler's frame) and def_won.
    """
    cur, nxt, contest = _adjacent_pairs(events)
    a, b = cur["type"], nxt["type"]
    ao, bo = cur["outcome_type"], nxt["outcome_type"]

    aer = contest & a.eq("Aerial").to_numpy() & b.eq("Aerial").to_numpy() & ao.ne(bo).to_numpy()
    first_won = ao.eq("Successful").to_numpy()
    idx = np.flatnonzero(aer)
    fw = first_won[idx]
    aerial = pd.DataFrame({
        "game_id": cur["game_id"].to_numpy()[idx],
        "winner_id": np.where(fw, cur["player_id"].to_numpy()[idx], nxt["player_id"].to_numpy()[idx]),
        "loser_id": np.where(fw, nxt["player_id"].to_numpy()[idx], cur["player_id"].to_numpy()[idx]),
        "winner_x": np.where(fw, cur["x"].to_numpy()[idx], nxt["x"].to_numpy()[idx]),
    })

    # (first type, first outcome or None, second type, second outcome or None,
    #  whether the tackler is listed first, whether the tackler won)
    ground_specs = [
        ("TakeOn", "Unsuccessful", "Tackle", None, False, True),
        ("Dispossessed", None, "Tackle", None, False, True),
        ("Tackle", None, "Dispossessed", None, True, True),
        ("Tackle", None, "TakeOn", "Unsuccessful", True, True),
        ("Challenge", None, "TakeOn", "Successful", True, False),
        ("TakeOn", "Successful", "Challenge", None, False, False),
    ]
    parts = []
    for t1, o1, t2, o2, def_first, def_won in ground_specs:
        m = contest & a.eq(t1).to_numpy() & b.eq(t2).to_numpy()
        if o1:
            m &= ao.eq(o1).to_numpy()
        if o2:
            m &= bo.eq(o2).to_numpy()
        i = np.flatnonzero(m)
        d_src, c_src = (cur, nxt) if def_first else (nxt, cur)
        parts.append(pd.DataFrame({
            "game_id": cur["game_id"].to_numpy()[i],
            "def_id": d_src["player_id"].to_numpy()[i],
            "carry_id": c_src["player_id"].to_numpy()[i],
            "def_x": d_src["x"].to_numpy()[i],
            "def_won": np.full(len(i), int(def_won)),
        }))
    ground = pd.concat(parts, ignore_index=True)
    return {"aerial": aerial, "ground": ground}


def _bt_design(
    first: np.ndarray, second: np.ndarray, context: np.ndarray, ids: pd.Index | None = None
) -> tuple[sp.csr_matrix, pd.Index]:
    """Sparse +1/-1 player columns for a two-player contest, plus context columns."""
    if ids is None:
        ids = pd.Index(pd.unique(np.concatenate([first, second])))
    n = len(first)
    rows = np.arange(n)
    players = sp.csr_matrix(
        (
            np.concatenate([np.ones(n), -np.ones(n)]),
            (np.concatenate([rows, rows]), np.concatenate([ids.get_indexer(first), ids.get_indexer(second)])),
        ),
        shape=(n, len(ids)),
    )
    return sp.hstack([players, sp.csr_matrix(context)]).tocsr(), ids


def fit_aerial_strength(aerial: pd.DataFrame, C: float = BT_C, seed: int = 42) -> pd.Series:
    """Opponent-adjusted aerial rating per player_id (log-odds scale, mean ~0).

    The pairs are stored winner-first, which would make every label 1, so each
    one is oriented at random (seeded, so a rebuild is reproducible). Location
    enters as how far up the pitch player one is: the side defending its own box
    is expected to win more, and a striker should not be marked down for
    contesting headers at the wrong end.
    """
    if aerial.empty:
        return pd.Series(dtype=float)
    flip = np.random.default_rng(seed).random(len(aerial)) < 0.5
    w, lo = aerial["winner_id"].to_numpy(), aerial["loser_id"].to_numpy()
    first = np.where(flip, lo, w)
    second = np.where(flip, w, lo)
    x_first = np.where(flip, 100.0 - aerial["winner_x"].to_numpy(), aerial["winner_x"].to_numpy())
    y = (~flip).astype(int)
    X, ids = _bt_design(first, second, ((x_first - 50.0) / 50.0).reshape(-1, 1))
    model = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(X, y)
    return pd.Series(model.coef_[0][: len(ids)], index=ids, name="aerial_score")


def fit_ground_strength(ground: pd.DataFrame, C: float = BT_C) -> pd.Series:
    """Opponent-adjusted rating as the tackler in ground duels, per player_id.

    Tacklers and carriers get separate parameters, since beating a dribbler and
    beating a tackler are different abilities; only the tackler side is kept
    (see module doc). The intercept absorbs the tackler's structural advantage,
    and location how far up the pitch the duel happened.
    """
    if ground.empty:
        return pd.Series(dtype=float)
    d_ids = pd.Index(pd.unique(ground["def_id"]))
    c_ids = pd.Index(pd.unique(ground["carry_id"]))
    n = len(ground)
    rows = np.arange(n)
    X = sp.hstack([
        sp.csr_matrix((np.ones(n), (rows, d_ids.get_indexer(ground["def_id"]))), shape=(n, len(d_ids))),
        sp.csr_matrix((-np.ones(n), (rows, c_ids.get_indexer(ground["carry_id"]))), shape=(n, len(c_ids))),
        sp.csr_matrix(((ground["def_x"].to_numpy() - 50.0) / 50.0).reshape(-1, 1)),
    ]).tocsr()
    model = LogisticRegression(C=C, max_iter=3000).fit(X, ground["def_won"].to_numpy())
    return pd.Series(model.coef_[0][: len(d_ids)], index=d_ids, name="ground_score")


# ---------------------------------------------------------------------------
# preparation: everything a score needs, reduced from the raw events once
# ---------------------------------------------------------------------------


@dataclass
class Prepared:
    """The slices of a season's events the scorer reads, kept small.

    Split-half reliability re-scores on subsets of matches, so everything here
    carries `game_id` and can be filtered without touching the raw events again.
    """

    duels: dict[str, pd.DataFrame]
    fouls: pd.DataFrame  # game_id, player_id, won (1 = drew it, 0 = committed it), x, y
    actions: pd.DataFrame  # open-play on-ball actions: game_id, player_id, x, y
    minutes: pd.DataFrame  # game_id, player_id, minutes
    names: pd.DataFrame  # player_id, player, team, league
    roles: pd.DataFrame  # player_id, role, role_group, lateral_dev
    carries: pd.DataFrame  # game_id, player_id, dist_m, prog_m
    defensive: pd.DataFrame  # defensive actions: game_id, player_id, x, y
    possession: pd.DataFrame  # game_id, player_id, own_share (team's share of touches)

    def subset(self, games) -> "Prepared":
        games = set(games)

        def f(df: pd.DataFrame) -> pd.DataFrame:
            return df[df["game_id"].isin(games)]

        return Prepared(
            {k: f(v) for k, v in self.duels.items()},
            f(self.fouls), f(self.actions), f(self.minutes), self.names, self.roles,
            f(self.carries), f(self.defensive), f(self.possession),
        )


def extract_carries(events: pd.DataFrame) -> pd.DataFrame:
    """Ball carries reconstructed from consecutive events, one row per carry.

    A carry runs from where a player got the ball to where their next action
    starts, when the next event is by the same team:
      * after a team-mate's completed pass, from its end point (the receiver is
        whoever acts next);
      * after the player's own recovery, interception, take-on, touch or tackle,
        from that event's location.
    Distances are in metres on a 105 x 68 pitch; `prog_m` is the part towards
    the opponent's goal. Requires `events` in within-match stream order.
    """
    ev = events[events["period"].isin(ts.PLAY_PERIODS)].reset_index(drop=True)
    nxt = ev.shift(-1)
    follow = (
        nxt["game_id"].eq(ev["game_id"])
        & nxt["team_id"].eq(ev["team_id"])
        & nxt["type"].isin(CARRY_END)
        & nxt["player_id"].notna()
        & nxt["x"].notna()
    )
    received = (
        follow & ev["type"].eq("Pass") & ev["outcome_type"].eq("Successful")
        & nxt["player_id"].ne(ev["player_id"])
    )
    own = (
        follow & ev["type"].isin(CARRY_OWN_START) & ev["outcome_type"].eq("Successful")
        & nxt["player_id"].eq(ev["player_id"])
    )
    keep = (received | own).to_numpy()
    sx = np.where(received, ev["end_x"], ev["x"])[keep]
    sy = np.where(received, ev["end_y"], ev["y"])[keep]
    dx = (nxt["x"].to_numpy()[keep] - sx) * PITCH_M[0] / 100.0
    dy = (nxt["y"].to_numpy()[keep] - sy) * PITCH_M[1] / 100.0
    out = pd.DataFrame({
        "game_id": ev["game_id"].to_numpy()[keep],
        "player_id": nxt["player_id"].to_numpy()[keep],
        "dist_m": np.hypot(dx, dy),
        "prog_m": np.clip(dx, 0, None),
    })
    # A "carry" longer than most of the pitch is a missing event, not a run.
    return out[out["dist_m"].between(0, 80)].reset_index(drop=True)


def possession_shares(events: pd.DataFrame) -> pd.DataFrame:
    """Each player's team's share of the ball in each match, by share of touches."""
    touches = events[events["is_touch"].fillna(False).astype(bool) & events["team_id"].notna()]
    per_team = touches.groupby(["game_id", "team_id"]).size()
    share = (per_team / per_team.groupby(level="game_id").transform("sum")).rename("own_share")
    players = events.loc[events["player_id"].notna(), ["game_id", "team_id", "player_id"]].drop_duplicates(
        ["game_id", "player_id"]
    )
    return players.merge(share.reset_index(), on=["game_id", "team_id"], how="left")[
        ["game_id", "player_id", "own_share"]
    ].fillna({"own_share": 0.5})


def _minutes_by_game(events: pd.DataFrame) -> pd.DataFrame:
    """Minutes per (game, player), via `estimate_minutes` keyed on a game copy."""
    ev = events[["game_id", "period", "expanded_minute", "player", "player_id", "type", "card_type"]]
    ev = ev.assign(_game=ev["game_id"])
    per = estimate_minutes(ev, ["_game", "player_id"])
    return per.rename_axis(["game_id", "player_id"]).rename("minutes").reset_index()


def prepare(events: pd.DataFrame) -> Prepared:
    duels = pair_duels(events)

    fouls = events[events["type"].eq("Foul") & events["player_id"].notna()]
    fouls = pd.DataFrame({
        "game_id": fouls["game_id"].to_numpy(),
        "player_id": fouls["player_id"].to_numpy(),
        # Foul events come in mirrored pairs: Successful is the player fouled.
        "won": fouls["outcome_type"].eq("Successful").to_numpy().astype(int),
        "x": fouls["x"].to_numpy(),
        "y": fouls["y"].to_numpy(),
    })

    on_ball = events[
        events["type"].isin(ts.ATTACKING_ON_BALL)
        & events["period"].isin(ts.PLAY_PERIODS)
        & events["player_id"].notna()
        & events["x"].notna()
    ]
    on_ball = on_ball[~ts._has_qualifier(on_ball["qualifiers"], ts.SET_PIECE_QUALIFIERS)]
    actions = on_ball[["game_id", "player_id", "x", "y"]].reset_index(drop=True)

    named = events[events["player_id"].notna()]
    cols = ["player_id", "player", "team"] + (["league"] if "league" in named.columns else [])
    names = named[cols].drop_duplicates(["player_id", "player", "team"]).reset_index(drop=True)
    if "league" not in names.columns:
        names["league"] = None

    roles = ts.refine_roles(ts.extract_roles(events), actions)

    defn = events[ts.defensive_action_mask(events) & events["player_id"].notna()]
    defensive = defn[["game_id", "player_id", "x", "y"]].reset_index(drop=True)

    return Prepared(
        duels, fouls, actions, _minutes_by_game(events), names, roles,
        extract_carries(events), defensive, possession_shares(events),
    )


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _z_within(frame: pd.DataFrame, cols: list[str]) -> None:
    """Z-score within role group, capped at +/- Z_CLIP.

    The cap is there for the blends. About 1% of players sit beyond 3 SD on some
    component — Alexander-Arnold's 40% tackling-duel win rate in 23/24 is -5.8
    among full-backs — and uncapped, one such axis outweighs the other five in
    the summary. Percentiles are rank-based and unaffected.
    """
    for col in cols:
        frame[f"z_{col}"] = frame.groupby("role_group")[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0
        ).clip(-Z_CLIP, Z_CLIP)


def apply_axis_weights(
    scores: pd.DataFrame, weights: dict[str, dict[str, float]] | None = None
) -> pd.DataFrame:
    """Re-fuse the axes into the summary under per-role weights; nothing refits.

    `weights` maps role_group -> {axis: weight}; roles it does not name fall back
    to `ROLE_AXIS_WEIGHTS`.
    """
    weights = weights or {}
    out = scores.copy()
    blend = np.full(len(out), np.nan)
    for group, idx in out.groupby("role_group").groups.items():
        spec = weights.get(group) or axis_weights_for(group)
        rows = out.index.get_indexer(idx)
        blend[rows] = ts._blend(out.loc[idx], {f"axis_{a}": w for a, w in spec.items()}, "z_")
    out["physicality_score"] = blend
    out["phys_pct"] = out.groupby("role_group")["physicality_score"].rank(pct=True) * 100
    return out.sort_values("physicality_score", ascending=False).reset_index(drop=True)


def score_players(
    prep: Prepared,
    min_minutes: float = DEFAULT_MIN_MINUTES,
    axis_weights: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """One row per (player_id, player, team) stint, scored on the whole season.

    Ratings and rates are properties of the player, not the stint, so a
    mid-season transfer shows the same numbers on both clubs' rows.
    """
    minutes = prep.minutes.groupby("player_id")["minutes"].sum()
    out = pd.DataFrame({"minutes": minutes})

    aer, gr = prep.duels["aerial"], prep.duels["ground"]
    out["aerial_score"] = fit_aerial_strength(aer)
    out["ground_score"] = fit_ground_strength(gr)
    out["aerial_n"] = pd.concat([aer["winner_id"], aer["loser_id"]]).value_counts()
    out["aerial_won_pct"] = aer["winner_id"].value_counts() / out["aerial_n"] * 100
    out["ground_n"] = gr["def_id"].value_counts()
    out["ground_won_pct"] = gr.loc[gr["def_won"].eq(1), "def_id"].value_counts() / out["ground_n"] * 100

    fouls = prep.fouls
    out["fouls_won"] = fouls.loc[fouls["won"].eq(1), "player_id"].value_counts()
    out["fouls_committed"] = fouls.loc[fouls["won"].eq(0), "player_id"].value_counts()
    for col in ("aerial_n", "ground_n", "fouls_won", "fouls_committed"):
        out[col] = out[col].fillna(0).astype(int)

    per90 = 90.0 / out["minutes"].replace(0, np.nan)
    out["aerials_p90"] = out["aerial_n"] * per90
    out["ground_duels_p90"] = out["ground_n"] * per90
    out["fouls_committed_p90"] = out["fouls_committed"] * per90

    # Fouls drawn above what the player's touch locations would draw on average,
    # per 100 open-play actions — the same per-cell baseline `tight_space` uses,
    # so a striker is not credited just for receiving the ball in the box.
    acts = prep.actions
    won = fouls[fouls["won"].eq(1) & fouls["x"].notna()]
    foul_counts = np.zeros(ts.BINS)
    action_counts = np.zeros(ts.BINS)
    np.add.at(foul_counts, ts.cell_index(won["x"], won["y"]), 1)
    agx, agy = ts.cell_index(acts["x"], acts["y"])
    np.add.at(action_counts, (agx, agy), 1)
    rate = np.divide(foul_counts, action_counts, out=np.zeros_like(foul_counts), where=action_counts > 0)
    exp = pd.Series(rate[agx, agy], index=acts.index).groupby(acts["player_id"]).sum()
    n_actions = acts.groupby("player_id").size()
    out["actions_total"] = n_actions.reindex(out.index).fillna(0).astype(int)
    fouls_in_play = won.groupby("player_id").size().reindex(out.index).fillna(0)
    raw = 100 * (fouls_in_play - exp.reindex(out.index).fillna(0)) / out["actions_total"].replace(0, np.nan)
    prior = 100 * (len(won) - rate[agx, agy].sum()) / max(len(acts), 1)
    out["contact_score"] = ts._shrink(raw, out["actions_total"], CONTACT_K, prior)

    # Running proxies, possession-adjusted: each match's minutes are split into
    # minutes with and without the ball by the team's share of touches, and a
    # count is scaled to 45 of each — i.e. to what it would be at 50% possession.
    poss = prep.minutes.merge(prep.possession, on=["game_id", "player_id"], how="left")
    poss["own_share"] = poss["own_share"].fillna(0.5)
    own_min = (poss["minutes"] * poss["own_share"]).groupby(poss["player_id"]).sum()
    opp_min = (poss["minutes"] * (1 - poss["own_share"])).groupby(poss["player_id"]).sum()
    own_pad = 45.0 / own_min.reindex(out.index).replace(0, np.nan)
    opp_pad = 45.0 / opp_min.reindex(out.index).replace(0, np.nan)

    car = prep.carries
    out["carry_m_p90"] = car.groupby("player_id")["dist_m"].sum().reindex(out.index).fillna(0) * per90
    out["prog_carry_m_padj"] = car.groupby("player_id")["prog_m"].sum().reindex(out.index).fillna(0) * own_pad
    long_n = car[car["dist_m"] >= LONG_CARRY_M].groupby("player_id").size()
    out["long_carries_padj"] = long_n.reindex(out.index).fillna(0) * own_pad

    dfn = prep.defensive
    g = dfn.assign(
        xm=dfn["x"] * PITCH_M[0] / 100.0, ym=dfn["y"] * PITCH_M[1] / 100.0
    ).groupby("player_id")
    # Spread of a player's defensive actions around their own centre, in metres:
    # how much of the pitch they defend over, rather than how often.
    spread = np.sqrt(g["xm"].var() + g["ym"].var())
    out["def_range_m"] = spread.where(g.size() >= MIN_RANGE_ACTIONS).reindex(out.index)
    high = dfn[dfn["x"] >= 50].groupby("player_id").size()
    out["high_def_padj"] = high.reindex(out.index).fillna(0) * opp_pad

    # Unrated for a duel type means no duels of that type at all; a shrunk
    # rating near the mean would say "average", which is not what we know.
    out.loc[out["aerial_n"].eq(0), "aerial_score"] = np.nan
    out.loc[out["ground_n"].eq(0), "ground_score"] = np.nan

    out = out[out["minutes"] >= min_minutes].rename_axis("player_id").reset_index()
    out = out.merge(prep.roles[["player_id", "role", "role_group", "lateral_dev"]], on="player_id", how="left")
    out["role"] = out["role"].fillna("UNK")
    out["role_group"] = out["role_group"].fillna(out["role"])
    out = out[~out["role"].isin(("GK", "UNK"))].copy()
    if out.empty:
        raise ValueError(f"no outfield player reached {min_minutes:g} minutes")
    counts = out["role_group"].value_counts()
    out.loc[out["role_group"].map(counts).lt(ts.MIN_ROLE_GROUP), "role_group"] = out["role"]

    _z_within(out, COMPONENT_COLS)
    for axis, spec in AXES.items():
        out[f"axis_{axis}"] = ts._blend(out, spec, "z_")
        out[f"pct_{axis}"] = out.groupby("role_group")[f"axis_{axis}"].rank(pct=True) * 100
    _z_within(out, AXIS_COLS)

    out = out.merge(prep.names, on="player_id", how="left")
    return apply_axis_weights(out, axis_weights)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def split_half_reliability(prep: Prepared, min_minutes: float = DEFAULT_MIN_MINUTES) -> pd.DataFrame:
    """Re-score on odd vs even matches and correlate, as in `tight_space`.

    `reliability` is the Spearman-Brown estimate for the full season.
    """
    games = pd.unique(prep.minutes["game_id"])
    halves = {}
    for h in (0, 1):
        sub = prep.subset(games[h::2])
        try:
            scored = score_players(sub, min_minutes=min_minutes / 2)
        except ValueError:
            return pd.DataFrame()
        halves[h] = scored.drop_duplicates("player_id").set_index("player_id")
    common = halves[0].index.intersection(halves[1].index)
    rows = []
    metrics = [("axis", c) for c in AXIS_COLS] + [("component", c) for c in COMPONENT_COLS]
    metrics += [("summary", "physicality_score")]
    for kind, col in metrics:
        a, b = halves[0].loc[common, col], halves[1].loc[common, col]
        # Correlate within role, as the scores are read: otherwise the fact that
        # centre-backs head more balls than wingers passes for reliability.
        g = halves[0].loc[common, "role_group"]
        ok = a.notna() & b.notna()
        a = a[ok] - a[ok].groupby(g[ok]).transform("mean")
        b = b[ok] - b[ok].groupby(g[ok]).transform("mean")
        r = a.corr(b) if ok.sum() > 5 else np.nan
        rows.append({
            "kind": kind, "metric": col, "n_players": int(ok.sum()), "split_half_r": r,
            "reliability": (2 * r / (1 + r)) if pd.notna(r) and r > -1 else np.nan,
        })
    return pd.DataFrame(rows)


def axis_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    return scores[[f"z_axis_{a}" for a in AXES]].rename(columns=lambda c: c[7:]).corr()


def external_check(scores: pd.DataFrame, stats_path: str | Path = "sofascore_player_stats.csv") -> pd.DataFrame:
    """Correlate each axis with the Sofascore stat it should track, by name join.

    Within role group, for the same reason as the reliability check. Only a
    face-validity check: the join resolves a bit over half the players.
    """
    from name_utils import normalize_series

    stats = pd.read_csv(stats_path)
    stats["_k"] = normalize_series(stats["player"])
    sc = scores.copy()
    sc["_k"] = normalize_series(sc["player"])
    m = sc.merge(stats.drop_duplicates("_k"), on="_k", suffixes=("", "_ss"))
    mins = m["minutesPlayed"].replace(0, np.nan)
    checks = [
        ("axis_aerial", "aerialDuelsWonPercentage", m["aerialDuelsWonPercentage"]),
        ("axis_ground", "groundDuelsWonPercentage", m["groundDuelsWonPercentage"]),
        ("axis_contact", "wasFouled per 90", m["wasFouled"] / mins * 90),
        ("axis_engagement", "duels per 90", (m["totalDuelsWon"] + m["duelLost"]) / mins * 90),
    ]
    rows = []
    for axis, label, v in checks:
        ok = v.notna() & m[axis].notna()
        g = m.loc[ok, "role_group"]
        a = m.loc[ok, axis] - m.loc[ok, axis].groupby(g).transform("mean")
        b = v[ok] - v[ok].groupby(g).transform("mean")
        rows.append({"axis": axis, "sofascore": label, "n": int(ok.sum()), "r_within_role": a.corr(b)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# end to end, and the per-season cache the similarity search reads
# ---------------------------------------------------------------------------


def build(
    leagues: list[str] | None = None,
    season: int | str = DEFAULT_SEASON,
    events_root: str | Path = "league_games",
    min_minutes: float = DEFAULT_MIN_MINUTES,
    validate: bool = True,
    events: pd.DataFrame | None = None,
) -> dict:
    if events is None:
        events = load_all_league_events(
            leagues=leagues, season=season, events_root=events_root, persist=False
        )
    prep = prepare(events)
    del events
    scores = score_players(prep, min_minutes=min_minutes)
    rel = split_half_reliability(prep, min_minutes=min_minutes) if validate else pd.DataFrame()
    return {"prepared": prep, "scores": scores, "reliability": rel}


_lock = threading.RLock()
_memo: dict[str, pd.DataFrame] = {}


def _cache_key(leagues: list[str], season: int, events_root: str, min_minutes: float) -> str | None:
    import analysis_cache

    events_key = analysis_cache.league_events_key(leagues, season, events_root)
    if events_key is None:
        return None
    payload = {"events": events_key, "min_minutes": float(min_minutes), "v": MODEL_VERSION}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cached_season(
    season: int,
    leagues: list[str] | None = None,
    events_root: str = "league_games",
    min_minutes: float = DEFAULT_MIN_MINUTES,
) -> pd.DataFrame | None:
    """The season's score table if it is already built, else None. Never builds."""
    leagues = list(leagues or DEFAULT_LEAGUES)
    key = _cache_key(leagues, int(season), events_root, min_minutes)
    if key is None:
        return None
    with _lock:
        if key in _memo:
            return _memo[key]
        path = CACHE_DIR / f"{key}_scores.csv"
        if not path.exists():
            return None
        try:
            scores = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError):
            return None
        _memo[key] = scores
        return scores


def season_scores(
    season: int,
    leagues: list[str] | None = None,
    events_root: str = "league_games",
    min_minutes: float = DEFAULT_MIN_MINUTES,
    progress=None,
) -> pd.DataFrame:
    """Score table for one season, built once and cached under `.desktop_cache/physicality/`.

    The key hashes the match csvs (via `analysis_cache`), so re-scraping a
    league rebuilds on its own.
    """
    season = int(season)
    leagues = list(leagues or DEFAULT_LEAGUES)
    hit = cached_season(season, leagues, events_root, min_minutes)
    if hit is not None:
        return hit
    key = _cache_key(leagues, season, events_root, min_minutes)
    if key is None:
        raise ValueError(f"no event data for season {season} under {events_root}/")
    with _lock:
        if progress:
            progress(f"Physicality {season % 100:02d}/{(season + 1) % 100:02d}: "
                     "reading events and fitting duel ratings (first time only)...")
        result = build(leagues, season, events_root, min_minutes=min_minutes)
        scores = result["scores"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        scores.to_csv(CACHE_DIR / f"{key}_scores.csv", index=False)
        (CACHE_DIR / f"{key}_meta.json").write_text(
            json.dumps({
                "season": season,
                "leagues": leagues,
                "min_minutes": min_minutes,
                "model_version": MODEL_VERSION,
                "n_players": int(len(scores)),
                "reliability": result["reliability"].to_dict(orient="records"),
            }, indent=2),
            encoding="utf-8",
        )
        _memo[key] = scores
        return scores


def find_player(scores: pd.DataFrame | None, player: str, team: str | None = None) -> pd.Series | None:
    """Exact (player, team) row if there is one, else an accent-insensitive name match."""
    from name_utils import normalize_name, normalize_series

    if scores is None or scores.empty or not player:
        return None
    exact = scores[(scores["player"] == player) & ((scores["team"] == team) if team else True)]
    if not exact.empty:
        return exact.iloc[0]
    keys = normalize_series(scores["player"])
    target = normalize_name(player)
    hits = scores[keys == target]
    if hits.empty:
        hits = scores[keys.str.contains(target, regex=False)]
    if hits.empty:
        return None
    if team:
        same = hits[normalize_series(hits["team"]) == normalize_name(team)]
        if not same.empty:
            hits = same
    return hits.sort_values("minutes", ascending=False).iloc[0]


def profile_dict(row: pd.Series | None) -> dict | None:
    """The compact form the similarity search attaches to each profile."""
    if row is None:
        return None

    def num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    return {
        "role_group": row.get("role_group"),
        "score": num(row.get("physicality_score")),
        "pct": num(row.get("phys_pct")),
        "axes": {
            a: {"z": num(row.get(f"axis_{a}")), "pct": num(row.get(f"pct_{a}"))} for a in AXES
        },
        "z_axes": [num(row.get(f"z_axis_{a}")) for a in AXES],
        "aerial_n": int(row.get("aerial_n") or 0),
        "aerial_won_pct": num(row.get("aerial_won_pct")),
        "ground_n": int(row.get("ground_n") or 0),
        "ground_won_pct": num(row.get("ground_won_pct")),
        "aerials_p90": num(row.get("aerials_p90")),
        "ground_duels_p90": num(row.get("ground_duels_p90")),
        "fouls_won": int(row.get("fouls_won") or 0),
        "fouls_committed_p90": num(row.get("fouls_committed_p90")),
        "carry_m_p90": num(row.get("carry_m_p90")),
        "prog_carry_m_padj": num(row.get("prog_carry_m_padj")),
        "long_carries_padj": num(row.get("long_carries_padj")),
        "def_range_m": num(row.get("def_range_m")),
        "high_def_padj": num(row.get("high_def_padj")),
        "minutes": num(row.get("minutes")),
    }


def warm(seasons=None, leagues=None) -> None:
    import season_similarity as ss

    for year in seasons or ss.available_seasons():
        print(f"[{ss.season_label(year)}] physicality ...", flush=True)
        season_scores(year, leagues=leagues, progress=lambda msg: print(f"  {msg}", flush=True))
    print("done", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rate players on physicality within their position.")
    parser.add_argument("--leagues", nargs="*", default=None)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--min-minutes", type=float, default=DEFAULT_MIN_MINUTES)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--player", default=None)
    parser.add_argument("--role", default=None,
                        help="DF/MF/FW, a line and side (DF-C, FW-W) or a side alone (C/W)")
    parser.add_argument("--sort-by", default=None,
                        help="axis_aerial, axis_ground, axis_contact, axis_engagement, "
                             "axis_carrying, axis_work_rate, ...")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--external-check", action="store_true")
    parser.add_argument("--warm", action="store_true", help="build and cache every season, then exit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.warm:
        warm(leagues=args.leagues)
        raise SystemExit(0)

    result = build(args.leagues, args.season, min_minutes=args.min_minutes, validate=args.validate)
    scores = result["scores"]
    display = ["player", "team", "role_group", "minutes", "aerial_n", "ground_n",
               "pct_aerial", "pct_ground", "pct_contact", "pct_engagement",
               "pct_carrying", "pct_work_rate", "physicality_score", "phys_pct"]

    if args.player:
        from name_utils import normalize_name, normalize_series

        hit = scores[normalize_series(scores["player"]).str.contains(normalize_name(args.player), regex=False)]
        if hit.empty:
            raise SystemExit(f"no player matching {args.player!r} reached {args.min_minutes:g} minutes")
        for _, row in hit.iterrows():
            print(f"\n{row['player']} ({row['team']}) — {row['role_group']}, {row['minutes']:.0f} min")
            for axis in AXES:
                print(f"  {AXIS_LABELS[axis]:<16} z {row[f'axis_{axis}']:+.2f}   "
                      f"{row[f'pct_{axis}']:.0f}th pct in role")
            print(f"  aerials {row['aerial_n']} ({row['aerial_won_pct']:.0f}% won, "
                  f"{row['aerials_p90']:.1f}/90), tackling duels {row['ground_n']} "
                  f"({row['ground_won_pct']:.0f}% won), fouls drawn {row['fouls_won']}, "
                  f"committed {row['fouls_committed_p90']:.2f}/90")
            print(f"  carries {row['carry_m_p90']:.0f} m/90 ({row['prog_carry_m_padj']:.0f} m progressive, "
                  f"{row['long_carries_padj']:.1f} long, PAdj), defensive range "
                  f"{row['def_range_m']:.1f} m, {row['high_def_padj']:.1f} defensive actions "
                  "in the opposition half (PAdj)")
            print(f"  summary {row['physicality_score']:+.2f} ({row['phys_pct']:.0f}th pct in role), "
                  f"weights {axis_weights_for(row['role_group'])}")
    else:
        board = ts.filter_by_role(scores, args.role)
        if args.sort_by:
            board = board.dropna(subset=[args.sort_by]).sort_values(args.sort_by, ascending=False)
        print(f"\ntop {args.top} by {args.sort_by or 'physicality_score'} — percentiles within role")
        print(board.head(args.top)[display].round(2).to_string(index=False))

    if args.validate:
        print("\nsplit-half reliability within role (odd vs even matches)")
        print(result["reliability"].round(3).to_string(index=False))
        print("\naxis correlations")
        print(axis_correlations(scores).round(3).to_string())
    if args.external_check:
        print("\nface validity vs Sofascore, within role")
        print(external_check(scores).round(3).to_string(index=False))
    if args.out:
        scores.to_csv(args.out, index=False)
