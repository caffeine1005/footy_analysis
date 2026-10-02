"""Tight-space performance tab for the Qt desktop app.

Wraps `tight_space.build()` — the congestion field, the situational difficulty
model and the per-axis execution scores — in a UI that can be browsed rather
than re-run.

Two things shape the design:

  * A full build reads every cached match csv for the selected leagues (~2,300
    matches, 1.4 GB) and fits several out-of-fold models on top. That is minutes,
    not milliseconds, so it runs on a worker thread and the resulting score table
    is cached to disk under `.desktop_cache/tight_space/`. Re-opening the app
    with the same settings loads the csv instead of refitting. The event load
    itself comes from `analysis_cache`, shared with the other two tabs, so a
    rebuild here does not re-read the csvs a profile has already loaded.
  * The axes are near-independent, so the summary blend across them is a stated
    preference rather than a measurement. The axis-weight boxes therefore re-fuse
    the cached table through `tight_space.apply_axis_weights` on the spot — no
    refit — and the leaderboard is sortable by a single axis, which is usually
    the more meaningful read.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import analysis_cache
import tight_space as ts
from name_utils import normalize_name, normalize_series
from touchmap_similarity import DEFAULT_LEAGUES, DEFAULT_SEASON

CACHE_DIR = Path(".desktop_cache") / "tight_space"

AXIS_LABELS = {
    "retention": "Retention",
    "beating_man": "Beating the man",
    "winning_contact": "Winning contact",
}

# What the leaderboard can be ranked on. The axes come first because ranking on
# one axis says something specific, where the summary blend averages signals the
# data says are near-independent.
SORT_OPTIONS = [
    ("Retention", "axis_retention"),
    ("Beating the man", "axis_beating_man"),
    ("Winning contact", "axis_winning_contact"),
    ("Summary blend", "tight_space_score"),
    ("Tight share (exposure)", "tight_share"),
    ("Tight actions (sample)", "tight_actions"),
]


# ---------------------------------------------------------------------------
# options and disk cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TightSpaceOptions:
    """Everything that changes the fitted numbers, and therefore the cache key.

    Axis weights are deliberately absent: they only re-fuse already-computed
    axes, so a table built under one blend serves every other blend too.
    """

    leagues: tuple[str, ...]
    season: int
    tight_quantile: float
    min_tight_actions: int
    max_matches: int | None
    events_root: str = "league_games"

    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def scores_path(self) -> Path:
        return CACHE_DIR / f"{self.key()}_scores.csv"

    def meta_path(self) -> Path:
        return CACHE_DIR / f"{self.key()}_meta.json"

    def describe(self) -> str:
        leagues = "all 7 leagues" if len(self.leagues) == len(DEFAULT_LEAGUES) else ", ".join(
            self.leagues
        )
        bits = [leagues, str(self.season), f"tight>={self.tight_quantile:g}",
                f"min {self.min_tight_actions} actions"]
        if self.max_matches:
            bits.append(f"first {self.max_matches} matches")
        return " · ".join(bits)


@dataclass
class TightSpaceData:
    """A scored table plus the diagnostics needed to read it honestly."""

    options: TightSpaceOptions
    scores: pd.DataFrame
    reliability: pd.DataFrame
    weights: dict[str, float]
    built_at: float
    from_cache: bool = False


def save_cache(data: TightSpaceData) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.scores.to_csv(data.options.scores_path(), index=False)
    data.options.meta_path().write_text(
        json.dumps(
            {
                "options": asdict(data.options),
                "built_at": data.built_at,
                "weights": data.weights,
                "reliability": data.reliability.to_dict(orient="records"),
                "n_players": int(len(data.scores)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_cache(options: TightSpaceOptions) -> TightSpaceData | None:
    scores_path, meta_path = options.scores_path(), options.meta_path()
    if not scores_path.exists() or not meta_path.exists():
        return None
    try:
        scores = pd.read_csv(scores_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, pd.errors.ParserError):
        # A half-written or hand-edited cache should cost a rebuild, not a crash.
        return None
    required = {"tight_space_score", "role_group", *(f"z_axis_{a}" for a in ts.AXES)}
    if not required.issubset(scores.columns):
        # Written by an older version of the scorer: rebuilding is the only option.
        return None
    return TightSpaceData(
        options=options,
        scores=scores,
        reliability=pd.DataFrame(meta.get("reliability") or []),
        weights=meta.get("weights") or {},
        built_at=float(meta.get("built_at") or 0.0),
        from_cache=True,
    )


# ---------------------------------------------------------------------------
# background build
# ---------------------------------------------------------------------------


class BuildSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class BuildWorker(QRunnable):
    """Runs the whole tight-space pipeline off the UI thread."""

    def __init__(self, options: TightSpaceOptions):
        super().__init__()
        self.options = options
        self.signals = BuildSignals()

    @Slot()
    def run(self):
        opts = self.options
        try:
            result = ts.build(
                leagues=list(opts.leagues) or None,
                season=opts.season,
                events_root=opts.events_root,
                max_matches=opts.max_matches,
                tight_quantile=opts.tight_quantile,
                min_tight_actions=opts.min_tight_actions,
            )
            scores = result["scores"].copy()
            events = result.get("events")
            if events is not None and "league" in events.columns:
                # `scores` is keyed on (player, team) only; the league each team
                # plays in is worth carrying into the table when several are pooled.
                team_league = events.groupby("team")["league"].agg(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else None
                )
                scores["league"] = scores["team"].map(team_league)
            data = TightSpaceData(
                options=opts,
                scores=scores,
                reliability=result["reliability"].copy(),
                weights=dict(result["weights"]),
                built_at=time.time(),
            )
            # The build holds every event and every fitted frame — several GB for
            # a full seven-league run. Drop it before the table goes back to the
            # UI so the app is not left sitting on it, and let go of the shared
            # event table too: this tab has its own score cache and will not ask
            # for the events again, where holding them costs ~1.7GB for the rest
            # of the session.
            del result
            analysis_cache.release_events()
            save_cache(data)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(data)


# ---------------------------------------------------------------------------
# small widget helpers
# ---------------------------------------------------------------------------


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("section")
    return label


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


def _card() -> QFrame:
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setObjectName("card")
    return frame


def _count(value) -> str:
    """Format a sample count, tolerating the NaN a missing merge leaves behind."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "0"
    return f"{int(val):,}" if np.isfinite(val) else "0"


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return "-"
    return f"{val:.{digits}f}"


def _ordinal(value) -> str:
    """'58th', '2nd' — percentile ranks read better as ordinals in prose."""
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return "-"
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class _NumericItem(QTableWidgetItem):
    """Table cell that shows formatted text but sorts on the underlying number."""

    def __init__(self, text: str, value: float | None):
        super().__init__(text)
        self.setFlags(self.flags() ^ Qt.ItemIsEditable)
        self.value = float(value) if value is not None and np.isfinite(value) else -np.inf

    def __lt__(self, other):
        if isinstance(other, _NumericItem):
            return self.value < other.value
        return super().__lt__(other)


def _read_only(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
    return item


def _plain_table(headers: list[str], rows: list[list[str]]) -> QTableWidget:
    table = QTableWidget(max(len(rows), 1), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setSelectionMode(QAbstractItemView.NoSelection)
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    for col in range(1, len(headers)):
        table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.setItem(r, c, _read_only(value))
    height = 30 + 30 * max(len(rows), 1)
    table.setMinimumHeight(height)
    table.setMaximumHeight(height + 10)
    return table


# ---------------------------------------------------------------------------
# player card
# ---------------------------------------------------------------------------


def player_card(row: pd.Series, scores: pd.DataFrame, *, title: str | None = None) -> QWidget:
    """The desktop equivalent of `tight_space.py --player`: exposure, axes, components."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)

    peers = scores[scores["role_group"] == row["role_group"]]
    header = _card()
    form = QFormLayout(header)
    name = f"{row['player']} ({row['team']})"
    if row.get("league"):
        name += f" — {row['league']}"
    form.addRow(_section(title or "Tight-Space Profile"), QLabel(name))
    form.addRow("Role group", QLabel(f"{row['role_group']}  ·  vs {max(len(peers) - 1, 0)} peers"))
    form.addRow(
        "Exposure",
        QLabel(
            f"{int(row['tight_actions']):,} tight actions, "
            f"{float(row['tight_share']):.0%} of their on-ball actions "
            f"({_ordinal(row.get('tight_share_pct'))} pct in role), "
            f"mean congestion {_fmt(row.get('tight_congestion'))}"
        ),
    )
    form.addRow(
        "Summary blend",
        QLabel(
            f"{_fmt(row.get('tight_space_score')):>5}  "
            f"({_ordinal(row.get('tight_pct'))} pct in role)"
        ),
    )
    layout.addWidget(header)

    axis_rows = [
        [
            AXIS_LABELS.get(axis, axis),
            _fmt(row.get(f"axis_{axis}")),
            _fmt(row.get(f"pct_{axis}"), 0),
        ]
        for axis in ts.AXES
    ]
    comp_rows = []
    for axis, spec in ts.AXES.items():
        for comp in spec:
            comp_rows.append(
                [
                    f"{comp.replace('_score', '')}  ({AXIS_LABELS.get(axis, axis)})",
                    _fmt(row.get(comp)),
                    _count(row.get(comp.replace("_score", "_n"))),
                ]
            )

    grid = QGridLayout()
    axis_box = QGroupBox("Axes (z within role group)")
    axis_layout = QVBoxLayout(axis_box)
    axis_layout.addWidget(_plain_table(["Axis", "z", "Pct in role"], axis_rows))
    grid.addWidget(axis_box, 0, 0)

    comp_box = QGroupBox("Components (per 100 actions, above expectation)")
    comp_layout = QVBoxLayout(comp_box)
    comp_layout.addWidget(_plain_table(["Component", "Value", "n"], comp_rows))
    grid.addWidget(comp_box, 0, 1)
    holder = QWidget()
    holder.setLayout(grid)
    layout.addWidget(holder)

    layout.addWidget(
        _muted(
            "Axes are near-independent, so the summary blend is a stated preference, not a "
            "measured trait — read an axis before the blend. Components are graded against "
            "what the situation and the player's own action choice predicted, so this is "
            "execution, not volume; `tight share` carries how often they are in there at all."
        )
    )
    return widget


def find_player(
    scores: pd.DataFrame, player: str, team: str | None = None
) -> pd.Series | None:
    """Accent-insensitive lookup, preferring an exact name and the right club.

    A mid-season transfer leaves one row per stint, so ties break on the larger
    sample rather than on row order.
    """
    if scores is None or scores.empty or not player:
        return None
    target = normalize_name(player)
    if not target:
        return None
    keys = normalize_series(scores["player"])
    hits = scores[keys == target]
    if hits.empty:
        hits = scores[keys.str.contains(target, regex=False)]
    if hits.empty:
        return None
    if team:
        same_team = hits[normalize_series(hits["team"]) == normalize_name(team)]
        if not same_team.empty:
            hits = same_team
    return hits.sort_values("tight_actions", ascending=False).iloc[0]


# ---------------------------------------------------------------------------
# the tab
# ---------------------------------------------------------------------------


class TightSpaceTab(QWidget):
    TABLE_HEADERS = [
        "#", "Player", "Team", "League", "Role", "Tight actions", "Tight share",
        "Retention", "Beating man", "Winning contact", "Blend", "Pct",
    ]

    def __init__(self):
        super().__init__()
        self.pool = QThreadPool.globalInstance()
        self.data: TightSpaceData | None = None
        self.view: pd.DataFrame = pd.DataFrame()
        self._elapsed = QTimer(self)
        self._elapsed.setInterval(1000)
        self._elapsed.timeout.connect(self._tick)
        self._started_at = 0.0

        layout = QVBoxLayout(self)
        title = QLabel("Tight-Space Performance")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(
            _muted(
                "Congestion-graded execution: how a player keeps, beats and out-muscles in the "
                "league's most contested situations, z-scored against their own role group."
            )
        )

        layout.addWidget(self._build_controls())
        layout.addWidget(self._build_view_controls())

        splitter = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, len(self.TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(self.TABLE_HEADERS)
        header = self.table.horizontalHeader()
        for col, name in enumerate(self.TABLE_HEADERS):
            mode = QHeaderView.Stretch if name in ("Player", "Team", "League") else QHeaderView.ResizeToContents
            header.setSectionResizeMode(col, mode)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._show_selected)
        splitter.addWidget(self.table)

        self.detail = QWidget()
        self.detail_layout = QVBoxLayout(self.detail)
        self.detail_layout.setAlignment(Qt.AlignTop)
        self.detail_layout.addWidget(_muted("Select a player to see their card."))
        splitter.addWidget(self.detail)
        splitter.setSizes([420, 400])
        layout.addWidget(splitter, stretch=1)

        self.load_cached()

    # -- construction ------------------------------------------------------

    def _build_controls(self) -> QGroupBox:
        box = QGroupBox("Model")
        grid = QGridLayout(box)

        self.leagues = QListWidget()
        self.leagues.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.leagues.addItems(DEFAULT_LEAGUES)
        self.leagues.selectAll()
        self.leagues.setMaximumHeight(96)
        self.leagues.setToolTip(
            "Leagues pooled into one congestion field and one peer set.\n"
            "Ctrl-click to narrow; selecting none is the same as selecting all."
        )
        grid.addWidget(QLabel("Leagues"), 0, 0)
        grid.addWidget(self.leagues, 0, 1, 3, 1)

        self.season = QSpinBox()
        self.season.setRange(2000, 2100)
        self.season.setValue(int(DEFAULT_SEASON))
        grid.addWidget(QLabel("Season"), 0, 2)
        grid.addWidget(self.season, 0, 3)

        self.tight_quantile = QDoubleSpinBox()
        self.tight_quantile.setRange(0.50, 0.95)
        self.tight_quantile.setSingleStep(0.05)
        self.tight_quantile.setDecimals(2)
        self.tight_quantile.setValue(0.75)
        self.tight_quantile.setToolTip(
            "Difficulty percentile above which a situation counts as tight.\n"
            "0.75 keeps the top quarter by predicted ball-loss risk."
        )
        grid.addWidget(QLabel("Tight quantile"), 1, 2)
        grid.addWidget(self.tight_quantile, 1, 3)

        self.min_tight_actions = QSpinBox()
        self.min_tight_actions.setRange(20, 2000)
        self.min_tight_actions.setSingleStep(10)
        self.min_tight_actions.setValue(120)
        self.min_tight_actions.setToolTip("Tight actions a player needs before they are scored")
        grid.addWidget(QLabel("Min tight actions"), 2, 2)
        grid.addWidget(self.min_tight_actions, 2, 3)

        self.max_matches = QSpinBox()
        self.max_matches.setRange(0, 10000)
        self.max_matches.setSingleStep(50)
        self.max_matches.setSpecialValueText("All")
        self.max_matches.setValue(0)
        self.max_matches.setToolTip(
            "Cap the matches loaded, for a quick trial run.\n"
            "A full build reads every cached match and takes minutes."
        )
        grid.addWidget(QLabel("Max matches"), 0, 4)
        grid.addWidget(self.max_matches, 0, 5)

        self.force_rebuild = QCheckBox("Force rebuild")
        self.force_rebuild.setToolTip("Refit even when a cached score table exists for these settings")
        grid.addWidget(self.force_rebuild, 1, 4, 1, 2)

        self.build_button = QPushButton("Load / Build scores")
        self.build_button.clicked.connect(self.run_build)
        grid.addWidget(self.build_button, 2, 4)

        self.diagnostics_button = QPushButton("Diagnostics")
        self.diagnostics_button.setEnabled(False)
        self.diagnostics_button.clicked.connect(self.show_diagnostics)
        grid.addWidget(self.diagnostics_button, 2, 5)

        self.status = QLabel("Idle")
        self.status.setWordWrap(True)
        grid.addWidget(self.status, 3, 0, 1, 6)
        grid.setColumnStretch(1, 1)
        return box

    def _build_view_controls(self) -> QGroupBox:
        box = QGroupBox("View")
        grid = QGridLayout(box)

        self.sort_by = QComboBox()
        for label, col in SORT_OPTIONS:
            self.sort_by.addItem(label, col)
        self.sort_by.currentIndexChanged.connect(self.refresh_view)
        grid.addWidget(QLabel("Rank by"), 0, 0)
        grid.addWidget(self.sort_by, 0, 1)

        self.role_filter = QLineEdit()
        self.role_filter.setPlaceholderText("all positions")
        self.role_filter.setToolTip(
            "A line (DF/MF/FW), a line and side (MF-C, DF-W), or a side alone (C/W).\n"
            "Comma-separate to combine, e.g. 'MF-C,FW-C'."
        )
        self.role_filter.editingFinished.connect(self.refresh_view)
        grid.addWidget(QLabel("Role"), 0, 2)
        grid.addWidget(self.role_filter, 0, 3)

        self.search = QLineEdit()
        self.search.setPlaceholderText("filter by player or team")
        self.search.textChanged.connect(self.refresh_view)
        grid.addWidget(QLabel("Search"), 0, 4)
        grid.addWidget(self.search, 0, 5)

        self.top_n = QSpinBox()
        self.top_n.setRange(5, 2000)
        self.top_n.setSingleStep(25)
        self.top_n.setValue(50)
        self.top_n.valueChanged.connect(self.refresh_view)
        grid.addWidget(QLabel("Show"), 0, 6)
        grid.addWidget(self.top_n, 0, 7)

        self.axis_weights: dict[str, QDoubleSpinBox] = {}
        col = 1
        grid.addWidget(QLabel("Blend weights"), 1, 0)
        for axis in ts.AXES:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(round(ts.DEFAULT_AXIS_WEIGHTS[axis], 2))
            spin.setPrefix(f"{AXIS_LABELS.get(axis, axis)} ")
            spin.setToolTip(
                "Weight this axis carries in the summary blend. Re-fuses the existing "
                "table instantly — nothing is refitted."
            )
            spin.valueChanged.connect(self.reblend)
            self.axis_weights[axis] = spin
            grid.addWidget(spin, 1, col, 1, 2)
            col += 2
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 1)
        return box

    # -- options -----------------------------------------------------------

    def options(self) -> TightSpaceOptions:
        picked = [item.text() for item in self.leagues.selectedItems()]
        leagues = tuple(sorted(picked)) if picked else tuple(sorted(DEFAULT_LEAGUES))
        return TightSpaceOptions(
            leagues=leagues,
            season=int(self.season.value()),
            tight_quantile=round(float(self.tight_quantile.value()), 2),
            min_tight_actions=int(self.min_tight_actions.value()),
            max_matches=int(self.max_matches.value()) or None,
        )

    def current_axis_weights(self) -> dict[str, float]:
        raw = {axis: float(spin.value()) for axis, spin in self.axis_weights.items()}
        total = sum(raw.values())
        if total <= 0:
            return dict(ts.DEFAULT_AXIS_WEIGHTS)
        return {axis: w / total for axis, w in raw.items()}

    # -- building ----------------------------------------------------------

    def load_cached(self) -> bool:
        """Show a previously built table for the current settings, if there is one."""
        data = load_cache(self.options())
        if data is None:
            self.status.setText(
                "No cached scores for these settings — press Load / Build scores. "
                "A full seven-league build reads every match and takes several minutes."
            )
            return False
        self._adopt(data)
        return True

    def run_build(self):
        opts = self.options()
        if not self.force_rebuild.isChecked():
            cached = load_cache(opts)
            if cached is not None:
                self._adopt(cached)
                return
        self.build_button.setEnabled(False)
        self._started_at = time.time()
        self._elapsed.start()
        self.status.setText(f"Building — {opts.describe()} …")
        worker = BuildWorker(opts)
        worker.signals.finished.connect(self.build_finished)
        worker.signals.failed.connect(self.build_failed)
        self.pool.start(worker)

    def _tick(self):
        secs = int(time.time() - self._started_at)
        self.status.setText(
            f"Building — {self.options().describe()} … {secs // 60}m {secs % 60:02d}s "
            "(loading events, fitting difficulty and expectation models)"
        )

    @Slot(object)
    def build_finished(self, data: TightSpaceData):
        self._elapsed.stop()
        self.build_button.setEnabled(True)
        self.force_rebuild.setChecked(False)
        self._adopt(data)

    @Slot(str)
    def build_failed(self, error: str):
        self._elapsed.stop()
        self.build_button.setEnabled(True)
        self.status.setText("Build failed")
        QMessageBox.critical(self, "Tight-space build failed", error)

    def _adopt(self, data: TightSpaceData):
        self.data = data
        self.diagnostics_button.setEnabled(True)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(data.built_at)) if data.built_at else "?"
        source = "cached" if data.from_cache else "built"
        self.status.setText(
            f"{len(data.scores):,} players scored — {data.options.describe()} "
            f"({source} {when})"
        )
        self.reblend()

    # -- viewing -----------------------------------------------------------

    def reblend(self):
        """Re-fuse the axes under the current weights, then redraw."""
        if self.data is None:
            return
        self.data.scores = ts.apply_axis_weights(self.data.scores, self.current_axis_weights())
        self.refresh_view()

    def refresh_view(self):
        if self.data is None:
            return
        scores = self.data.scores
        try:
            board = ts.filter_by_role(scores, self.role_filter.text().strip() or None)
        except ValueError as exc:
            self.status.setText(str(exc))
            return

        needle = normalize_name(self.search.text())
        if needle:
            board = board[
                normalize_series(board["player"]).str.contains(needle, regex=False)
                | normalize_series(board["team"]).str.contains(needle, regex=False)
            ]

        sort_col = self.sort_by.currentData()
        if sort_col in board.columns:
            # Unscored rows are dropped rather than sunk to the bottom: a player
            # with too few take-ons to grade is not a bad dribbler.
            board = board.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False)
        self.view = board.head(int(self.top_n.value()))
        self._populate()

    def _populate(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.view))
        for row_idx, (index, row) in enumerate(self.view.iterrows()):
            rank = _NumericItem(str(row_idx + 1), row_idx + 1)
            # The score-table index travels with the row so selection survives
            # a click-to-sort on any column.
            rank.setData(Qt.UserRole, int(index))
            cells = [
                rank,
                _read_only(str(row.get("player", ""))),
                _read_only(str(row.get("team", ""))),
                _read_only(str(row.get("league", "") or "")),
                _read_only(str(row.get("role_group", ""))),
                _NumericItem(f"{int(row['tight_actions']):,}", row["tight_actions"]),
                _NumericItem(f"{float(row['tight_share']):.0%}", row["tight_share"]),
            ]
            for axis in ts.AXES:
                cells.append(
                    _NumericItem(_fmt(row.get(f"pct_{axis}"), 0), row.get(f"pct_{axis}"))
                )
            cells.append(
                _NumericItem(_fmt(row.get("tight_space_score")), row.get("tight_space_score"))
            )
            cells.append(_NumericItem(_fmt(row.get("tight_pct"), 0), row.get("tight_pct")))
            for col_idx, item in enumerate(cells):
                self.table.setItem(row_idx, col_idx, item)
        self.table.setSortingEnabled(True)
        if len(self.view):
            self.table.selectRow(0)
        else:
            self._clear_detail()
            self.detail_layout.addWidget(_muted("No players match the current filters."))

    def _show_selected(self):
        ranges = self.table.selectedRanges()
        if not ranges or self.data is None:
            return
        item = self.table.item(ranges[0].topRow(), 0)
        if item is None:
            return
        index = item.data(Qt.UserRole)
        if index is None or index not in self.data.scores.index:
            return
        self._render_detail(self.data.scores.loc[index])

    def _clear_detail(self):
        while self.detail_layout.count():
            widget = self.detail_layout.takeAt(0).widget()
            if widget is not None:
                # Unparent before the deferred delete, or the old card keeps
                # painting over the new one until the event loop catches up.
                widget.setParent(None)
                widget.deleteLater()

    def _render_detail(self, row: pd.Series):
        self._clear_detail()
        self.detail_layout.addWidget(player_card(row, self.data.scores))

    # -- integration with the rest of the app ------------------------------

    def summary_box(
        self, player: str, team: str | None = None, season: int | None = None
    ) -> QGroupBox | None:
        """A compact card for `player`, for embedding in the Player Profile tab.

        Returns None when no table is loaded, the loaded table is for a season
        other than `season`, or the player never cleared the minimum sample, so
        the caller can simply skip the section.
        """
        if self.data is None:
            return None
        if season is not None and int(self.data.options.season) != int(season):
            return None
        row = find_player(self.data.scores, player, team)
        if row is None:
            return None
        box = QGroupBox("Tight-Space Performance")
        layout = QFormLayout(box)
        layout.addRow(
            "Exposure",
            QLabel(
                f"{int(row['tight_actions']):,} tight actions "
                f"({float(row['tight_share']):.0%} of on-ball, "
                f"{_ordinal(row.get('tight_share_pct'))} pct in {row['role_group']})"
            ),
        )
        for axis in ts.AXES:
            layout.addRow(
                AXIS_LABELS.get(axis, axis),
                QLabel(
                    f"{_fmt(row.get(f'axis_{axis}')):>5} z   ·   "
                    f"{_ordinal(row.get(f'pct_{axis}'))} pct in role"
                ),
            )
        layout.addRow(
            "Summary blend",
            QLabel(
                f"{_fmt(row.get('tight_space_score'))}  "
                f"({_ordinal(row.get('tight_pct'))} pct)"
            ),
        )
        layout.addRow(_muted(f"From the Tight Space tab · {self.data.options.describe()}"))
        return box

    # -- diagnostics -------------------------------------------------------

    def diagnostics_text(self) -> str:
        """The `--validate` output for the loaded table, as text."""
        if self.data is None:
            return ""
        parts = [f"settings: {self.data.options.describe()}", ""]
        if self.data.weights:
            parts += [
                "within-axis component weights, from measured split-half reliability:",
                "  " + ", ".join(
                    f"{c.replace('_score', '')}={w:.2f}" for c, w in self.data.weights.items()
                ),
                "",
            ]
        if not self.data.reliability.empty:
            parts += [
                "split-half reliability (odd vs even matches); reliability is the",
                "Spearman-Brown estimate for the full sample:",
                self.data.reliability.round(3).to_string(index=False),
                "",
            ]
        try:
            parts += [
                "axis correlations — near-zero is why this is a profile, not one number:",
                ts.axis_correlations(self.data.scores).round(3).to_string(),
                "",
            ]
        except KeyError:
            pass
        parts += [
            "score by role group — a flat spread means the role confound is controlled:",
            self.data.scores.groupby("role_group")
            .agg(
                players=("tight_space_score", "size"),
                mean_score=("tight_space_score", "mean"),
                mean_tight_share=("tight_share", "mean"),
            )
            .round(3)
            .to_string(),
        ]
        return "\n".join(parts)

    def show_diagnostics(self):
        if self.data is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Tight-space diagnostics")
        dialog.resize(760, 620)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QTextEdit.NoWrap)
        text.setFontFamily("monospace")
        text.setPlainText(self.diagnostics_text())
        layout.addWidget(text)
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close)
        layout.addLayout(row)
        dialog.exec()
