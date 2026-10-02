"""Rate chance creators who do their creating in tight spaces — the Isco archetype.

Built on top of `tight_space.py` and reusing its situational `difficulty` model,
which is the tightness axis here too. Where that module asks "does this player
survive congestion", this one asks "does this player *create* from it": receive
in the crowded zones in front of a set block and still find the pass that ends in
a shot.

The pieces:

  1. A shot xG model on the same WhoScored events, from shot location, angle and
     the Opta situation tags (header, big chance, fast break, set piece, ...).
     Out-of-fold, so no shot's value comes from a model that saw its outcome.
  2. Key passes linked to the shot they created. Passes carry `KeyPass` /
     `ShotAssist` qualifiers and shots carry `related_player_id` (the assister),
     so each assisted shot is joined to that player's most recent key pass in the
     same match. The pass's xA is the shot's xG.
  3. Creation baselines from *situation only* — P(key pass) and E[xA] given where
     the ball is, how congested that is, how settled the possession is and how
     high the block sits. Deliberately blind to action choice: picking the killer
     pass instead of recycling is the skill being measured, so conditioning on it
     would erase the signal. Conditioning on location does the opposite job, so
     a player is not credited just for getting the ball at the edge of the box.
  4. Everything is scored on *tight attacking* actions only: open-play on-ball
     actions in the opponent's half whose situational difficulty is in the top
     half of that half's actions. The top quartile was tried first; it roughly
     halved each player's sample, left creation at split-half r ~ 0.17, and
     scored 589 players instead of ~1,600 at the same minimum.

Measured over 7 leagues (2,323 matches, 1,603 players in the reliability split):
creation r = 0.30 split-half (0.46 full-season), security 0.59 (0.74), exposure
0.83 (0.91), summary 0.35 (0.52). Within creation, box entries replicate best
(0.36), key passes next (0.20) and xA worst (0.12, dropped by the reliability
floor) — xA inherits all of a teammate's finishing-position noise. Tight and open
creation correlate at only r = 0.13-0.31 per component, so this is not just a
"good creator" ranking under another name. Creation is the noisiest axis: read
it as a season-long signal, and do not over-read small gaps between players.

Reported as a profile, like its parent module:

  - creation: key passes and xA above expected per 100 tight attacking actions,
    blended within the axis by measured split-half reliability;
  - security: ball retention above expected on the same actions, which is what
    separates a press-resistant 10 from a player who forces a through ball every
    time he is squeezed;
  - exposure: `tight_share`, how much of a player's whole game is played in those
    situations. Not a skill, but it *is* the archetype — a creator who only
    creates in space is a different player.

`--validate` also scores the same residual on the *non-tight* attacking actions and
correlates the two, which is the check that this measures something beyond "good
creator": if tight and open creation were the same trait, the tight filter would
only be adding noise.

CLI:

    python tight_creator.py --role MF-C,FW-C --top 25
    python tight_creator.py --player "Isco"
    python tight_creator.py --sort-by axis_creation --role MF
    python tight_creator.py --validate --out tight_creator_scores.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from tight_space import (
    MIN_ROLE_GROUP,
    SITUATION_FEATURES,
    _fit_oof,
    _has_qualifier,
    _shrink,
    add_difficulty,
    annotate_events,
    build_congestion_field,
    extract_roles,
    filter_by_role,
    open_play_actions,
    refine_roles,
)
from touchmap_similarity import DEFAULT_LEAGUES, DEFAULT_SEASON, load_all_league_events

SHOT_TYPES = ("Goal", "SavedShot", "MissedShots", "ShotOnPost")

# Opta situation tags that change a shot's value before it is struck. Outcome tags
# (Blocked, GoalMouthY/Z, the Miss*/High*/Low* placements) are left out, since they
# describe what happened to the shot rather than the chance.
XG_FLAGS = (
    "Head", "OtherBodyPart", "BigChance", "FastBreak", "FromCorner", "SetPiece",
    "ThrowinSetPiece", "DirectFreekick", "Penalty", "Volley", "FirstTouch", "OneOnOne",
    "IndividualPlay",
)

KEY_PASS_QUALIFIERS = ("KeyPass", "ShotAssist", "IntentionalGoalAssist")

# Crosses are not creation *from* tight space: they are what a player in a wide,
# uncontested lane does to reach the crowded box. Left in, they put crossing
# wing-backs (Udol, Dimarco, Daniel Muñoz) on top of the leaderboard, because the
# difficulty model rightly rates wide final-third balls as likely to be lost.
EXCLUDED_CREATION_QUALIFIERS = ("Cross",)

# The penalty area in opta units (0-100 each way on a 105x68 pitch).
BOX_X = 83.0
BOX_Y = (21.1, 78.9)

# A key pass and its shot are a pass and a finish, not a move. Anything longer
# than this between them is a stale join to an earlier pass by the same player.
MAX_ASSIST_GAP_S = 20

# Actions from here forward (opta x, 0-100) count as attacking. Being squeezed
# in your own third is a retention problem, not a creative one.
ATTACKING_X = 50.0

# Shrinkage half-weights, in tight attacking actions. Key passes run at a few per
# hundred such actions and xA is noisier still (it inherits every shot's xG), so
# both need a few hundred actions before a player's own rate outweighs the league's.
CREATION_K = {"penetration_score": 150.0, "chance_score": 200.0, "xa_score": 300.0}
SECURITY_K = 150.0

AXES = {
    "creation": {"penetration_score": 0.4, "chance_score": 0.4, "xa_score": 0.2},
    "security": {"security_score": 1.0},
    "exposure": {"tight_share": 1.0},
}
COMPONENT_COLS = [c for spec in AXES.values() for c in spec]
AXIS_COLS = [f"axis_{a}" for a in AXES]

# Stated preference, not a measurement: the archetype is a creator first, whose
# creating happens under pressure, and who does not give the ball away doing it.
DEFAULT_AXIS_WEIGHTS = {"creation": 0.6, "security": 0.2, "exposure": 0.2}

# Same rule as `tight_space.AXIS_RELATIVE_FLOOR`: key passes and xA are built from
# the same events and share their errors, so a much less reliable member is dropped
# from the axis rather than blended in.
AXIS_RELATIVE_FLOOR = 0.5


# ---------------------------------------------------------------------------
# shot value
# ---------------------------------------------------------------------------


def _shot_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Geometry in metres on a 105x68 pitch plus the pre-shot situation flags."""
    dx = (100.0 - shots["x"]) * 1.05
    dy = (shots["y"] - 50.0) * 0.68
    f = pd.DataFrame(index=shots.index)
    f["x"] = shots["x"]
    f["y"] = shots["y"]
    f["distance"] = np.hypot(dx, dy)
    # Angle subtended by the 7.32m goal mouth: the standard single best xG feature.
    f["goal_angle"] = np.arctan2(7.32 * dx, dx**2 + dy**2 - 3.66**2) % np.pi
    for flag in XG_FLAGS:
        f[flag.lower()] = _has_qualifier(shots["qualifiers"], flag).astype(int)
    return f


def add_shot_xg(ev: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold xG for every shot in the annotated frame; own goals excluded."""
    shots = ev[
        ev["type"].isin(SHOT_TYPES)
        & ~_has_qualifier(ev["qualifiers"], "OwnGoal")
        & ev["player_id"].notna()
    ].copy()
    shots["goal"] = shots["type"].eq("Goal").astype(int)
    shots["xg"] = _fit_oof(_shot_features(shots), shots["goal"].to_numpy(), shots["game_id"].to_numpy())
    return shots


def link_key_passes(ev: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Join each assisted shot to the assister's latest key pass before it.

    Returns one row per linked pass (`seq`, `xa`). A pass is linked to at most one
    shot — Opta credits a key pass with the single shot it set up.
    """
    passes = ev[
        ev["type"].eq("Pass") & _has_qualifier(ev["qualifiers"], KEY_PASS_QUALIFIERS)
    ][["seq", "game_id", "team_id", "player_id", "t"]]
    assisted = shots[shots["related_player_id"].notna()][
        ["seq", "game_id", "team_id", "related_player_id", "t", "xg"]
    ]
    if passes.empty or assisted.empty:
        return pd.DataFrame(columns=["seq", "xa"])

    linked = pd.merge_asof(
        assisted.sort_values("seq").rename(columns={"seq": "shot_seq", "t": "shot_t"}),
        passes.sort_values("seq").rename(columns={"player_id": "related_player_id", "team_id": "pass_team"}),
        left_on="shot_seq",
        right_on="seq",
        by=["game_id", "related_player_id"],
        direction="backward",
    )
    ok = (
        linked["seq"].notna()
        & linked["pass_team"].eq(linked["team_id"])
        & (linked["shot_t"] - linked["t"]).between(0, MAX_ASSIST_GAP_S)
    )
    out = linked[ok].drop_duplicates("seq")[["seq", "xg"]].rename(columns={"xg": "xa"})
    out["seq"] = out["seq"].astype(int)
    out.attrs["link_rate"] = ok.mean()
    return out


# ---------------------------------------------------------------------------
# creation baselines
# ---------------------------------------------------------------------------


def _fit_oof_poisson(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int = 42) -> np.ndarray:
    """Out-of-fold E[y] for a non-negative, mostly-zero target, grouped by match."""
    oof = np.zeros(len(X))
    n_splits = min(5, len(np.unique(groups)))
    params = dict(loss="poisson", random_state=seed, max_iter=200, learning_rate=0.08, min_samples_leaf=50)
    if n_splits < 2:
        return HistGradientBoostingRegressor(**params).fit(X, y).predict(X)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = HistGradientBoostingRegressor(**params)
        model.fit(X.iloc[train], y[train])
        oof[test] = model.predict(X.iloc[test])
    return oof


def _in_box(x: pd.Series, y: pd.Series) -> pd.Series:
    return (x >= BOX_X) & y.between(*BOX_Y)


def add_creation_expectations(actions: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Attach actual and situation-expected creation outcomes to attacking actions.

    Three outcomes, from densest to most valuable:
      - `penetration`: a completed, non-cross pass from outside the box into it.
        About as frequent as a key pass but not gated on a teammate choosing to
        shoot, which makes it the most reliable of the three. End coordinates are
        used only in the *target*, and only for completed passes, so the leak
        described in `tight_space._pass_features` does not apply.
      - `key_pass`: the pass led directly to a shot.
      - `xa`: the xG of that shot.
    """
    att = actions[actions["x"] >= ATTACKING_X].copy()
    creative = att["type"].eq("Pass") & ~_has_qualifier(att["qualifiers"], EXCLUDED_CREATION_QUALIFIERS)
    att["key_pass"] = (creative & _has_qualifier(att["qualifiers"], KEY_PASS_QUALIFIERS)).astype(int)
    att["xa"] = att["seq"].map(links.set_index("seq")["xa"]).fillna(0.0).where(att["key_pass"].eq(1), 0.0)
    att["penetration"] = (
        creative
        & att["outcome_type"].eq("Successful")
        & _in_box(att["end_x"], att["end_y"])
        & ~_in_box(att["x"], att["y"])
    ).astype(int)

    X = att[SITUATION_FEATURES]
    groups = att["game_id"].to_numpy()
    att["exp_penetration"] = _fit_oof(X, att["penetration"].to_numpy(), groups)
    att["exp_key_pass"] = _fit_oof(X, att["key_pass"].to_numpy(), groups)
    att["exp_xa"] = _fit_oof_poisson(X, att["xa"].to_numpy(), groups)
    return att


# ---------------------------------------------------------------------------
# player scoring
# ---------------------------------------------------------------------------


def _residual_table(sub: pd.DataFrame, key: list[str], prefix: str) -> pd.DataFrame:
    """Per-player counts and above-expectation rates per 100 actions for one slice."""
    agg = (
        sub.groupby(key)
        .agg(n=("lost", "size"), pen=("penetration", "sum"), exp_pen=("exp_penetration", "sum"),
             kp=("key_pass", "sum"), exp_kp=("exp_key_pass", "sum"),
             xa=("xa", "sum"), exp_xa=("exp_xa", "sum"),
             lost=("lost", "mean"), exp_lost=("difficulty", "mean"))
        .reset_index()
    )
    priors = {
        "penetration_score": 100 * (sub["penetration"].sum() - sub["exp_penetration"].sum()) / len(sub),
        "chance_score": 100 * (sub["key_pass"].sum() - sub["exp_key_pass"].sum()) / len(sub),
        "xa_score": 100 * (sub["xa"].sum() - sub["exp_xa"].sum()) / len(sub),
        "security_score": 100 * (sub["difficulty"].mean() - sub["lost"].mean()),
    }
    raw = {
        "penetration_score": 100 * (agg["pen"] - agg["exp_pen"]) / agg["n"],
        "chance_score": 100 * (agg["kp"] - agg["exp_kp"]) / agg["n"],
        "xa_score": 100 * (agg["xa"] - agg["exp_xa"]) / agg["n"],
        "security_score": 100 * (agg["exp_lost"] - agg["lost"]),
    }
    ks = {**CREATION_K, "security_score": SECURITY_K}
    out = agg[key].copy()
    out[f"{prefix}actions"] = agg["n"]
    out[f"{prefix}box_entries"] = agg["pen"]
    out[f"{prefix}key_passes"] = agg["kp"]
    out[f"{prefix}xa"] = agg["xa"]
    for c in raw:
        out[f"{prefix}{c}"] = _shrink(raw[c], agg["n"], ks[c], priors[c])
    return out


def score_creators(
    att: pd.DataFrame,
    actions_total: pd.DataFrame,
    roles: pd.DataFrame,
    cut: float,
    min_tight_actions: int = 150,
    weights: dict[str, float] | None = None,
    axis_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Profile every player on creation, security and exposure in tight attacking play.

    `cut` is the difficulty threshold for "tight", fixed by the caller from the full
    population so that both halves of a reliability split use the same definition.
    `actions_total` is each player's count of *all* open-play actions, the
    denominator for exposure.
    """
    key = ["player_id", "player", "team"]
    tight = att[att["difficulty"] >= cut]

    out = _residual_table(tight, key, "")
    out = out.rename(columns={"actions": "tight_actions", "box_entries": "tight_box_entries",
                              "key_passes": "tight_key_passes", "xa": "tight_xa"})
    out = out.merge(actions_total, on=key, how="left")
    out["tight_share"] = out["tight_actions"] / out["actions_total"]
    out["tight_box_p100"] = 100 * out["tight_box_entries"] / out["tight_actions"]
    out["tight_kp_p100"] = 100 * out["tight_key_passes"] / out["tight_actions"]
    out["tight_xa_p100"] = 100 * out["tight_xa"] / out["tight_actions"]

    # The same residuals on attacking actions that were *not* tight, kept only for
    # the check that tight creation is a separate trait from creation in space.
    open_ = _residual_table(att[att["difficulty"] < cut], key, "open_")
    open_cols = ["open_actions", "open_penetration_score", "open_chance_score", "open_xa_score"]
    out = out.merge(open_[key + open_cols], on=key, how="left")

    out = out.merge(roles, on="player_id", how="left")
    out["role"] = out["role"].fillna("UNK")
    if "role_group" not in out.columns:
        out["role_group"] = out["role"]
    out["role_group"] = out["role_group"].fillna(out["role"])
    out = out[(out["tight_actions"] >= min_tight_actions) & (out["role"] != "GK")].copy()
    if out.empty:
        raise ValueError(f"no outfield player reached {min_tight_actions} tight attacking actions")
    counts = out["role_group"].value_counts()
    out.loc[out["role_group"].map(counts).lt(MIN_ROLE_GROUP), "role_group"] = out["role"]

    weights = weights or {c: w for spec in AXES.values() for c, w in spec.items()}
    axis_weights = axis_weights or DEFAULT_AXIS_WEIGHTS

    def _z(frame: pd.DataFrame, cols: list[str]) -> None:
        for col in cols:
            frame[f"z_{col}"] = frame.groupby("role_group")[col].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0
            )

    def _blend(frame: pd.DataFrame, spec: dict[str, float]) -> np.ndarray:
        w = np.array(list(spec.values()))
        Z = frame[[f"z_{c}" for c in spec]].to_numpy()
        wsum = ((~np.isnan(Z)) * w).sum(axis=1)
        return np.where(wsum > 0, np.nansum(np.nan_to_num(Z) * w, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)

    _z(out, COMPONENT_COLS)
    for axis, spec in AXES.items():
        spec = {c: weights.get(c, w) for c, w in spec.items()}
        spec = {c: w for c, w in spec.items() if w > 0} or dict(AXES[axis])
        out[f"axis_{axis}"] = _blend(out, spec)
        out[f"pct_{axis}"] = out.groupby("role_group")[f"axis_{axis}"].rank(pct=True) * 100

    _z(out, AXIS_COLS)
    out["tight_creator_score"] = _blend(out, {f"axis_{a}": w for a, w in axis_weights.items()})
    out["creator_pct"] = out.groupby("role_group")["tight_creator_score"].rank(pct=True) * 100
    return out.sort_values("tight_creator_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def split_half_reliability(
    att: pd.DataFrame,
    all_actions: pd.DataFrame,
    roles: pd.DataFrame,
    cut: float,
    min_tight_actions: int = 150,
    weights: dict[str, float] | None = None,
    axis_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Re-score on odd vs even matches and correlate; see `tight_space.split_half_reliability`."""
    key = ["player_id", "player", "team"]
    parity = _parity(att)
    halves = {}
    for h in (0, 1):
        a = att[att["game_id"].map(parity) == h]
        tot = (
            all_actions[all_actions["game_id"].map(parity) == h]
            .groupby(key).size().rename("actions_total").reset_index()
        )
        try:
            scored = score_creators(a, tot, roles, cut, max(50, min_tight_actions // 3), weights, axis_weights)
        except ValueError:
            return pd.DataFrame()
        halves[h] = (
            scored.sort_values("tight_actions", ascending=False)
            .drop_duplicates("player_id").set_index("player_id")
        )

    common = halves[0].index.intersection(halves[1].index)
    metrics = [("axis", c) for c in AXIS_COLS] + [("component", c) for c in COMPONENT_COLS]
    metrics += [("raw", "tight_box_p100"), ("raw", "tight_kp_p100"), ("raw", "tight_xa_p100"), ("summary", "tight_creator_score")]
    rows = []
    for kind, c in metrics:
        a, b = halves[0].loc[common, c], halves[1].loc[common, c]
        ok = a.notna() & b.notna()
        r = a[ok].corr(b[ok]) if ok.sum() > 5 else np.nan
        rows.append({"kind": kind, "metric": c, "n_players": int(ok.sum()), "split_half_r": r,
                     "reliability": (2 * r / (1 + r)) if pd.notna(r) and r > -1 else np.nan})
    return pd.DataFrame(rows)


def _parity(att: pd.DataFrame) -> dict:
    """game_id -> 0/1, fixed from the attacking frame so every table splits identically."""
    games = pd.unique(att["game_id"])
    return dict(zip(games, np.arange(len(games)) % 2))


def reliability_weights(rel: pd.DataFrame, floor: float = 0.05) -> dict[str, float]:
    """Within-axis weights discounted by split-half r; see `tight_space.reliability_weights`."""
    r = rel.set_index("metric")["split_half_r"]
    out: dict[str, float] = {}
    for spec in AXES.values():
        usable = {c: r.get(c, np.nan) for c in spec}
        usable = {c: v for c, v in usable.items() if pd.notna(v) and v > floor}
        if not usable:
            out.update({c: w / sum(spec.values()) for c, w in spec.items()})
            continue
        best = max(usable.values())
        kept = {c: spec[c] * v for c, v in usable.items() if v >= AXIS_RELATIVE_FLOOR * best}
        total = sum(kept.values())
        out.update({c: kept.get(c, 0.0) / total for c in spec})
    return out


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def build(
    leagues: list[str] | None = None,
    season: int | str = DEFAULT_SEASON,
    events_root: str | Path = "league_games",
    max_matches: int | None = None,
    tight_quantile: float = 0.5,
    min_tight_actions: int = 150,
    axis_weights: dict[str, float] | None = None,
) -> dict:
    events = load_all_league_events(leagues=leagues, season=season, events_root=events_root)
    if max_matches is not None:
        keep = pd.unique(events["game_id"])[:max_matches]
        events = events[events["game_id"].isin(keep)]
    print(f"events: {len(events):,} rows across {events['game_id'].nunique():,} matches", flush=True)

    roles = extract_roles(events)
    field = build_congestion_field(events)
    ev = annotate_events(events, field)
    ev["seq"] = np.arange(len(ev))

    shots = add_shot_xg(ev)
    links = link_key_passes(ev, shots)
    print(f"shots: {len(shots):,}, xG {shots['xg'].sum():,.0f} vs goals {shots['goal'].sum():,}; "
          f"assisted shots linked to a key pass: {links.attrs.get('link_rate', np.nan):.1%}", flush=True)

    actions = open_play_actions(ev)
    roles = refine_roles(roles, actions)
    actions = add_difficulty(actions)
    att = add_creation_expectations(actions, links)
    cut = att["difficulty"].quantile(tight_quantile)
    print(f"attacking open-play actions: {len(att):,}; tight = difficulty >= {cut:.3f}", flush=True)

    key = ["player_id", "player", "team"]
    totals = actions.groupby(key).size().rename("actions_total").reset_index()

    rel = split_half_reliability(att, actions, roles, cut, min_tight_actions)
    weights = reliability_weights(rel) if not rel.empty else None
    if weights:
        print("within-axis weights from measured reliability: "
              + ", ".join(f"{c}={w:.2f}" for c, w in weights.items()), flush=True)

    scores = score_creators(att, totals, roles, cut, min_tight_actions, weights, axis_weights)
    rel_final = split_half_reliability(att, actions, roles, cut, min_tight_actions, weights, axis_weights)
    return {"events": ev, "shots": shots, "actions": actions, "attacking": att, "roles": roles,
            "cut": cut, "scores": scores, "weights": weights, "reliability": rel_final}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rate chance creators who create from tight spaces.")
    parser.add_argument("--leagues", nargs="*", default=None,
                        help=f"Leagues to pool. Default: all {len(DEFAULT_LEAGUES)} collected.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--tight-quantile", type=float, default=0.5,
                        help="Difficulty percentile, within attacking actions, above which play counts as tight")
    parser.add_argument("--min-tight-actions", type=int, default=150)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--player", default=None)
    parser.add_argument("--role", default=None, help="Same syntax as tight_space.py, e.g. 'MF-C,FW-C'")
    parser.add_argument("--sort-by", default=None,
                        help="axis_creation, axis_security, tight_share, tight_xa_p100, ...")
    parser.add_argument("--axis-weights", default=None, help="e.g. 'creation=0.6,security=0.2,exposure=0.2'")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    parsed_axis_weights = None
    if args.axis_weights:
        parsed = {}
        for part in args.axis_weights.split(","):
            name, _, val = part.partition("=")
            if name.strip() not in AXES:
                raise SystemExit(f"unknown axis {name.strip()!r}; expected one of {list(AXES)}")
            parsed[name.strip()] = float(val)
        total = sum(parsed.values()) or 1.0
        parsed_axis_weights = {a: parsed.get(a, 0.0) / total for a in AXES}

    result = build(args.leagues, args.season, max_matches=args.max_matches,
                   tight_quantile=args.tight_quantile, min_tight_actions=args.min_tight_actions,
                   axis_weights=parsed_axis_weights)
    scores = result["scores"]
    display = ["player", "team", "role_group", "tight_actions", "tight_share", "tight_box_p100", "tight_kp_p100",
               "tight_xa_p100", "pct_creation", "pct_security", "pct_exposure", "tight_creator_score", "creator_pct"]

    if args.player:
        from name_utils import normalize_name, normalize_series

        hit = scores[normalize_series(scores["player"]).str.contains(normalize_name(args.player), regex=False)]
        if hit.empty:
            raise SystemExit(f"no player matching {args.player!r} cleared the minimums")
        print(hit[display + ["penetration_score", "chance_score", "xa_score", "security_score"]].round(2).T.to_string())
    else:
        try:
            board = filter_by_role(scores, args.role)
        except ValueError as exc:
            raise SystemExit(str(exc))
        if args.sort_by:
            board = board.dropna(subset=[args.sort_by]).sort_values(args.sort_by, ascending=False)
        scope = f" ({args.role})" if args.role else ""
        print(f"\ntop {args.top}{scope} by {args.sort_by or 'tight_creator_score'} — percentiles within role")
        print(board.head(args.top)[display].round(2).to_string(index=False))

    if args.validate:
        from sklearn.metrics import roc_auc_score

        shots, att = result["shots"], result["attacking"]
        print(f"\nxG model AUC {roc_auc_score(shots['goal'], shots['xg']):.3f}; "
              f"key-pass model AUC {roc_auc_score(att['key_pass'], att['exp_key_pass']):.3f}; "
              f"box-entry model AUC {roc_auc_score(att['penetration'], att['exp_penetration']):.3f}")
        print("\nsplit-half reliability (odd vs even matches)")
        print(result["reliability"].round(3).to_string(index=False))
        print("\naxis correlations")
        print(scores[[f"z_{c}" for c in AXIS_COLS]].corr().round(3).to_string())
        print("\ntight vs open creation — high r means the tight filter adds little beyond 'good creator'")
        for c in ("penetration_score", "chance_score", "xa_score"):
            print(f"  {c}: tight vs open r = {scores[c].corr(scores['open_' + c]):.3f}")
        print("\nby role group")
        print(scores.groupby("role_group")[["tight_creator_score", "tight_kp_p100", "tight_share"]]
              .agg(["size", "mean"]).round(3).to_string())

    if args.out:
        scores.to_csv(args.out, index=False)
        print(f"\nscores written to {args.out}")
