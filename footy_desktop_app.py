"""Qt desktop UI for player profiles, similar-player search and tight-space scores.

Run on Arch from the repository root:

    ./run_footy_desktop.sh

The first two tabs use the Sofascore + touch-map pipeline in
`sofascore_similarity.py` (same flow as `player_profile.py`, but stage-2 stats
come from Sofascore instead of FBref), made season-aware by
`season_similarity.py`: a profile is of one player in one season, and a
similar-player search takes a target season plus a set of seasons to search
(none ticked = every season), listing each match as a player-season. The third, in `tight_space_panel.py`,
runs the congestion-graded model in `tight_space.py` and, once its table is
loaded, adds a tight-space panel to every profile rendered here. Name lookup is
accent-insensitive via `name_utils.py`, so plain English input such as
"Arda Guler" can resolve accented names stored in the data.

Everything the three tabs share that does not depend on the player asked about —
the concatenated match events, the per-player shape vectors, the clustering and
the pass-angle features — is cached by `analysis_cache` on disk and in memory,
so only the first run of a given dataset pays for it. Rendered dashboards are
reused the same way. Set `FOOTY_NO_CACHE=1` to recompute everything instead.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QComboBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pass_angle_radar as par
import season_similarity as ss
import sofascore_similarity as sofa
import touchmap_similarity as tms
from name_utils import strip_accents
from tight_space_panel import TightSpaceTab


CACHE_DIR = Path(".desktop_cache") / "maps"


@dataclass
class AnalysisOptions:
    player: str
    team: str | None
    league: str | None
    season: int
    # None = every season. Profiles search their own season only.
    scope_seasons: list[int] | None
    exclude_target_other_seasons: bool
    min_touches: int
    n_clusters: int
    pool_per_season: int
    top_n: int
    min_90s: float | None
    min_age: float | None
    max_age: float | None
    include_unknown_age: bool
    action_features: bool
    flip_flanks: bool
    physicality: bool
    match_physicality: bool
    min_physicality_pct: float | None
    save_maps_dir: Path | None


def optional_text(value: str) -> str | None:
    value = value.strip()
    return value or None


def slug(value: str) -> str:
    folded = strip_accents(value).lower()
    chars = [ch if ch.isalnum() else "_" for ch in folded]
    return "_".join("".join(chars).split("_")).strip("_") or "player"


def fmt_num(value, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "-"
    return f"{val:.{digits}f}"


def stat_rows(rows: list[tuple[str, float, float]]) -> list[tuple[str, str, str]]:
    return [(sofa.pretty_stat(col), f"{value:.2f}", f"p{pct:.0f}") for col, value, pct in rows]


class WorkerSignals(QObject):
    finished = Signal(str, dict)
    failed = Signal(str, str)
    progress = Signal(str, str)


class AnalysisWorker(QRunnable):
    def __init__(self, target: str, options: AnalysisOptions):
        super().__init__()
        self.target = target
        self.options = options
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        opts = self.options
        try:
            result = ss.analyze_seasons(
                opts.player,
                team=opts.team,
                league=opts.league,
                target_season=opts.season,
                scope_seasons=opts.scope_seasons,
                exclude_target_other_seasons=opts.exclude_target_other_seasons,
                min_touches=opts.min_touches,
                n_clusters=opts.n_clusters,
                pool_per_season=opts.pool_per_season,
                top_n=opts.top_n,
                min_90s=opts.min_90s,
                min_age=opts.min_age,
                max_age=opts.max_age,
                include_unknown_age=opts.include_unknown_age,
                action_features=opts.action_features,
                flip_flanks=opts.flip_flanks,
                physicality=opts.physicality,
                match_physicality=opts.match_physicality,
                min_physicality_pct=opts.min_physicality_pct,
                save_maps_dir=opts.save_maps_dir,
                progress=lambda msg: self.signals.progress.emit(self.target, msg),
            )
        except Exception as exc:  # noqa: BLE001 - show analysis failures in UI
            self.signals.failed.emit(self.target, str(exc))
            return
        self.signals.finished.emit(self.target, result)


class SeasonScope(QWidget):
    """Tick the seasons to search; none ticked means every season."""

    def __init__(self, seasons: list[int]):
        super().__init__()
        self.list = QListWidget()
        # One wrapping row of checkboxes rather than a tall column.
        self.list.setFlow(QListView.LeftToRight)
        self.list.setWrapping(True)
        self.list.setResizeMode(QListView.Adjust)
        self.list.setSpacing(2)
        self.list.setFixedHeight(52)
        for year in sorted(seasons, reverse=True):
            item = QListWidgetItem(ss.season_label(year))
            item.setData(Qt.UserRole, year)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list.addItem(item)
        self.list.itemChanged.connect(lambda _item: self._update_summary())

        latest = QPushButton("Latest only")
        latest.clicked.connect(lambda: self.set_checked({max(seasons)} if seasons else set()))
        # 13/14 and 14/15 carry too few Sofascore stats to rank on (see
        # season_similarity), so this is the "everything with full stats" set.
        since = QPushButton("Since 15/16")
        since.setToolTip("Tick every season from 15/16 on, leaving out 13/14 and 14/15")
        since.clicked.connect(lambda: self.set_checked({y for y in seasons if y >= 2015}))
        clear = QPushButton("All seasons")
        clear.clicked.connect(lambda: self.set_checked(set()))
        self.summary = QLabel()
        self.summary.setObjectName("muted")

        buttons = QHBoxLayout()
        buttons.addWidget(latest)
        buttons.addWidget(since)
        buttons.addWidget(clear)
        buttons.addWidget(self.summary, stretch=1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list)
        layout.addLayout(buttons)
        self._update_summary()

    def _items(self):
        return [self.list.item(i) for i in range(self.list.count())]

    def selected(self) -> list[int] | None:
        years = [int(it.data(Qt.UserRole)) for it in self._items() if it.checkState() == Qt.Checked]
        return sorted(years) or None

    def set_checked(self, years: set[int]):
        for it in self._items():
            it.setCheckState(Qt.Checked if int(it.data(Qt.UserRole)) in years else Qt.Unchecked)
        self._update_summary()

    def _update_summary(self):
        years = self.selected()
        self.summary.setText(
            "Searching every season"
            if not years
            else "Searching " + ", ".join(ss.season_label(y) for y in years)
        )


class SearchControls(QWidget):
    def __init__(self, include_top_n: bool, seasons: list[int]):
        super().__init__()
        self.include_top_n = include_top_n

        self.player = QLineEdit()
        self.team = QLineEdit()
        self.league = QLineEdit()
        self.season = QComboBox()
        for year in sorted(seasons, reverse=True):
            self.season.addItem(ss.season_label(year), year)
        self.season.setToolTip("The season the target player is taken from")
        self.scope = None
        self.exclude_self = None
        if include_top_n:
            self.scope = SeasonScope(seasons)
            self.scope.setToolTip(
                "Seasons to find matches in. Leave all unticked to search every season."
            )
            self.exclude_self = QCheckBox("Hide target's other seasons")
            self.exclude_self.setChecked(True)
            self.exclude_self.setToolTip(
                "Keep the target player's own other seasons out of the results."
            )
        self.min_touches = QSpinBox()
        self.min_touches.setRange(1, 100000)
        self.min_touches.setValue(int(tms.DEFAULT_MIN_TOUCHES))
        self.pool_size = QSpinBox()
        self.pool_size.setRange(20, 5000)
        self.pool_size.setSingleStep(50)
        self.pool_size.setValue(ss.DEFAULT_POOL_PER_SEASON)
        self.pool_size.setToolTip(
            "How many of the target's nearest player-seasons (by touch/action shape)\n"
            "go on to the stats ranking, per season searched."
        )
        self.top_n = QSpinBox()
        self.top_n.setRange(1, 50)
        self.top_n.setValue(10)
        self.min_90s = QDoubleSpinBox()
        self.min_90s.setRange(0.0, 10000.0)
        self.min_90s.setDecimals(1)
        self.min_90s.setSpecialValueText("Any")
        self.min_90s.setValue(0.0)
        # 0 reads as "Any" so either end of the range can be left open.
        self.min_age = QSpinBox()
        self.min_age.setRange(0, 50)
        self.min_age.setSpecialValueText("Any")
        self.min_age.setValue(0)
        self.min_age.setToolTip("Only return matches at or above this age")
        self.max_age = QSpinBox()
        self.max_age.setRange(0, 50)
        self.max_age.setSpecialValueText("Any")
        self.max_age.setValue(0)
        self.max_age.setToolTip("Only return matches at or below this age")
        self.include_unknown_age = QCheckBox("Include unknown age")
        self.include_unknown_age.setChecked(True)
        self.include_unknown_age.setToolTip(
            "Sofascore has no birth date for some players. Untick to drop them\n"
            "when an age range is set instead of showing them as age '-'."
        )
        self.action_features = QCheckBox("Use action features")
        self.action_features.setChecked(True)
        self.flip_flanks = QCheckBox("Flip flanks (L↔R)")
        self.flip_flanks.setChecked(False)
        self.flip_flanks.setToolTip(
            "Mirror the target's heatmaps across the pitch before searching.\n"
            "Use this to find the other-footed / opposite-side version of a player\n"
            "(e.g. a right-footed inverted winger matching a flipped Gareth Bale)."
        )
        self.physicality = QCheckBox("Physicality")
        self.physicality.setChecked(True)
        self.physicality.setToolTip(
            "Show each player's physicality profile (physicality.py): aerial and ground\n"
            "duel ratings, taking contact, engagement, carrying and work rate,\n"
            "within their role group.\n"
            "A season not yet scored takes about a minute the first time\n"
            "(or run `python physicality.py --warm` once)."
        )
        self.match_physicality = None
        self.min_phys_pct = None
        if include_top_n:
            self.match_physicality = QCheckBox("Match physicality")
            self.match_physicality.setChecked(False)
            self.match_physicality.setToolTip(
                "Also rank on closeness to the target's physicality profile,\n"
                "as one more ranker alongside the stats-based ones."
            )
            self.min_phys_pct = QSpinBox()
            self.min_phys_pct.setRange(0, 99)
            self.min_phys_pct.setSpecialValueText("Any")
            self.min_phys_pct.setValue(0)
            self.min_phys_pct.setToolTip(
                "Only return players at or above this physicality percentile\n"
                "within their role group. Unscored players are dropped when set."
            )

        layout = QGridLayout(self)
        layout.addWidget(QLabel("Player"), 0, 0)
        layout.addWidget(self.player, 0, 1)
        layout.addWidget(QLabel("Team"), 0, 2)
        layout.addWidget(self.team, 0, 3)
        layout.addWidget(QLabel("League"), 0, 4)
        layout.addWidget(self.league, 0, 5)
        layout.addWidget(QLabel("Target season" if include_top_n else "Season"), 0, 6)
        layout.addWidget(self.season, 0, 7)

        layout.addWidget(self.physicality, 1, 0)
        if self.match_physicality is not None:
            layout.addWidget(self.match_physicality, 1, 1)
        layout.addWidget(QLabel("Min touches"), 1, 2)
        layout.addWidget(self.min_touches, 1, 3)
        layout.addWidget(QLabel("Pool / season"), 1, 4)
        layout.addWidget(self.pool_size, 1, 5)
        if include_top_n:
            layout.addWidget(QLabel("Top N"), 1, 6)
            layout.addWidget(self.top_n, 1, 7)
        layout.addWidget(QLabel("Min 90s"), 2, 0)
        layout.addWidget(self.min_90s, 2, 1)
        layout.addWidget(QLabel("Age from"), 2, 2)
        layout.addWidget(self.min_age, 2, 3)
        layout.addWidget(QLabel("Age to"), 2, 4)
        layout.addWidget(self.max_age, 2, 5)
        if self.min_phys_pct is not None:
            layout.addWidget(QLabel("Min phys pct"), 2, 6)
            layout.addWidget(self.min_phys_pct, 2, 7)

        layout.addWidget(self.action_features, 3, 0, 1, 2)
        layout.addWidget(self.flip_flanks, 3, 2, 1, 2)
        layout.addWidget(self.include_unknown_age, 3, 4, 1, 2)
        if self.exclude_self is not None:
            layout.addWidget(self.exclude_self, 3, 6, 1, 2)
        if self.scope is not None:
            layout.addWidget(QLabel("Search seasons"), 4, 0)
            layout.addWidget(self.scope, 4, 1, 1, 7)

        for col in (1, 3, 5, 7):
            layout.setColumnStretch(col, 1)

    def options(self, *, save_maps: bool, force_top_n: int | None = None) -> AnalysisOptions:
        player = self.player.text().strip()
        if not player:
            raise ValueError("Enter a player name.")

        season = self.season.currentData()
        if season is None:
            raise ValueError("No seasons with data were found.")
        season = int(season)

        save_maps_dir = None
        if save_maps:
            # Stable per player-season rather than per run: `analyze` stamps the
            # rendered dashboards with the events they came from and reuses them
            # when nothing has changed, so re-opening a profile costs no
            # rendering. A timestamp here would defeat that and leave a new
            # directory of pngs behind on every click; one directory per player
            # would have each season's render overwrite the last.
            # Flipped maps live under a sibling folder so they never clobber the
            # unflipped cache (stamp also records flip_flanks).
            suffix = "_flipped" if self.flip_flanks.isChecked() else ""
            save_maps_dir = CACHE_DIR / f"{slug(player)}_{season}{suffix}"

        min_90s = None if self.min_90s.value() <= 0 else float(self.min_90s.value())
        min_age = None if self.min_age.value() <= 0 else float(self.min_age.value())
        max_age = None if self.max_age.value() <= 0 else float(self.max_age.value())
        if min_age is not None and max_age is not None and min_age > max_age:
            raise ValueError(
                f"Age from ({min_age:g}) is above Age to ({max_age:g}) — "
                "no player can match that range."
            )

        return AnalysisOptions(
            player=player,
            team=optional_text(self.team.text()),
            league=optional_text(self.league.text()),
            season=season,
            # A profile only needs its own season: it ranks nothing it shows.
            scope_seasons=self.scope.selected() if self.scope is not None else [season],
            exclude_target_other_seasons=(
                self.exclude_self.isChecked() if self.exclude_self is not None else True
            ),
            min_touches=int(self.min_touches.value()),
            n_clusters=sofa.DEFAULT_N_CLUSTERS,
            pool_per_season=int(self.pool_size.value()),
            top_n=int(force_top_n or self.top_n.value()),
            min_90s=min_90s,
            min_age=min_age,
            max_age=max_age,
            include_unknown_age=self.include_unknown_age.isChecked(),
            action_features=self.action_features.isChecked(),
            flip_flanks=self.flip_flanks.isChecked(),
            physicality=self.physicality.isChecked(),
            match_physicality=(
                self.match_physicality is not None and self.match_physicality.isChecked()
            ),
            min_physicality_pct=(
                float(self.min_phys_pct.value())
                if self.min_phys_pct is not None and self.min_phys_pct.value() > 0
                else None
            ),
            save_maps_dir=save_maps_dir,
        )

    def set_identity(
        self, player: str, team: str | None, league: str | None, season: int | None = None
    ):
        self.player.setText(player or "")
        self.team.setText(team or "")
        self.league.setText(league or "")
        if season is not None:
            idx = self.season.findData(int(season))
            if idx >= 0:
                self.season.setCurrentIndex(idx)


class ScrollPanel(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.body = QWidget()
        self.layout = QVBoxLayout(self.body)
        self.layout.setAlignment(Qt.AlignTop)
        self.setWidget(self.body)

    def clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Footy Analysis (Sofascore)")
        self.resize(1280, 860)
        self.pool = QThreadPool.globalInstance()
        self.similar_profiles: list[dict] = []
        self.similar_result: dict | None = None
        self.seasons = ss.available_seasons()

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        self.profile_tab = QWidget()
        self.similar_tab = QWidget()
        # Built before the other tabs because profiles read its score table when
        # one is loaded, to show the player's tight-space card alongside.
        self.tight_tab = TightSpaceTab()
        tabs.addTab(self.profile_tab, "Player Profile")
        tabs.addTab(self.similar_tab, "Similar Players")
        tabs.addTab(self.tight_tab, "Tight Space")
        self.tabs = tabs

        self._build_profile_tab()
        self._build_similar_tab()

    def _build_profile_tab(self):
        layout = QVBoxLayout(self.profile_tab)
        title = QLabel("Player Profile (Sofascore)")
        title.setObjectName("title")
        layout.addWidget(title)

        top = QHBoxLayout()
        self.profile_controls = SearchControls(include_top_n=False, seasons=self.seasons)
        top.addWidget(self.profile_controls, stretch=1)
        buttons = QVBoxLayout()
        self.profile_button = QPushButton("Run Profile")
        self.profile_button.clicked.connect(self.run_profile)
        self.profile_status = QLabel("Idle")
        buttons.addWidget(self.profile_button)
        buttons.addWidget(self.profile_status)
        buttons.addStretch()
        top.addLayout(buttons)
        layout.addLayout(top)

        self.profile_panel = ScrollPanel()
        layout.addWidget(self.profile_panel, stretch=1)

    def _build_similar_tab(self):
        layout = QVBoxLayout(self.similar_tab)
        title = QLabel("Similar Players (Sofascore)")
        title.setObjectName("title")
        layout.addWidget(title)

        top = QHBoxLayout()
        self.similar_controls = SearchControls(include_top_n=True, seasons=self.seasons)
        top.addWidget(self.similar_controls, stretch=1)
        buttons = QVBoxLayout()
        self.similar_button = QPushButton("Find Similar")
        self.similar_button.clicked.connect(self.run_similar)
        self.open_profile_button = QPushButton("Open Selected Profile")
        self.open_profile_button.setEnabled(False)
        self.open_profile_button.clicked.connect(self.open_selected_profile)
        self.similar_status = QLabel("Idle")
        self.similar_status.setWordWrap(True)
        self.similar_status.setMaximumWidth(220)
        buttons.addWidget(self.similar_button)
        buttons.addWidget(self.open_profile_button)
        buttons.addWidget(self.similar_status)
        buttons.addStretch()
        top.addLayout(buttons)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Vertical)
        self.similar_table = QTableWidget(0, 10)
        self.similar_table.setHorizontalHeaderLabels(
            ["#", "Player", "Season", "Team", "League", "Position", "Age", "90s", "Phys", "Score"]
        )
        self.similar_table.horizontalHeaderItem(8).setToolTip(
            "Physicality percentile within the player's role group that season"
        )
        header = self.similar_table.horizontalHeader()
        for col in (1, 3, 4):
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        for col in (0, 2, 5, 6, 7, 8, 9):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.similar_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.similar_table.setSelectionMode(QTableWidget.SingleSelection)
        self.similar_table.itemSelectionChanged.connect(self.show_selected_similar_profile)
        self.similar_table.itemDoubleClicked.connect(lambda _item: self.open_selected_profile())
        splitter.addWidget(self.similar_table)

        self.similar_panel = ScrollPanel()
        splitter.addWidget(self.similar_panel)
        splitter.setSizes([260, 520])
        layout.addWidget(splitter, stretch=1)

    def run_profile(self):
        try:
            opts = self.profile_controls.options(save_maps=True)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return
        self.profile_panel.clear()
        self.profile_button.setEnabled(False)
        self.profile_status.setText(f"Running analysis ({ss.season_label(opts.season)})...")
        self._start_worker("profile", opts)

    def run_similar(self):
        try:
            opts = self.similar_controls.options(save_maps=False)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return
        self.similar_panel.clear()
        self.similar_table.setRowCount(0)
        self.similar_profiles = []
        self.similar_result = None
        self.open_profile_button.setEnabled(False)
        self.similar_button.setEnabled(False)
        self.similar_status.setText("Finding similar players...")
        self._start_worker("similar", opts)

    def _start_worker(self, target: str, opts: AnalysisOptions):
        worker = AnalysisWorker(target, opts)
        worker.signals.finished.connect(self.analysis_finished)
        worker.signals.failed.connect(self.analysis_failed)
        worker.signals.progress.connect(self.analysis_progress)
        self.pool.start(worker)

    @Slot(str, str)
    def analysis_progress(self, target: str, message: str):
        (self.profile_status if target == "profile" else self.similar_status).setText(message)

    @Slot(str, dict)
    def analysis_finished(self, target: str, result: dict):
        # `analyze` reports a player it found in the event data but could not
        # match in the Sofascore table as a result carrying an error, not as a
        # raised exception — so the status has to read it, or a failed run still
        # says "Loaded" over a panel showing the error.
        stage2_error = (result.get("target_profile") or {}).get("error")

        if target == "profile":
            self.profile_button.setEnabled(True)
            touch = result.get("touch_target", {})
            name = f"{touch.get('player', 'player')} {touch.get('season_label', '')}".strip()
            self.profile_status.setText(
                f"No Sofascore stats for {name}" if stage2_error else f"Loaded {name}"
            )
            self.profile_panel.clear()
            self.render_profile(
                self.profile_panel.layout,
                result.get("target_profile", {}),
                result=result,
                maps=result.get("touch_target", {}).get("maps_saved") or {},
                title="Target Profile",
            )
            return

        self.similar_button.setEnabled(True)
        self.similar_result = result
        self.similar_profiles = result.get("quant_profiles", []) or []
        age_meta = result.get("age_meta") or {}
        filtered = age_meta.get("min_age") is not None or age_meta.get("max_age") is not None
        touch = result.get("touch_target", {})
        target_player = touch.get("player", "target")
        if stage2_error:
            status = f"No Sofascore stats for {target_player}"
        else:
            status = f"Loaded {len(self.similar_profiles)} matches"
            if filtered:
                status += f" (age {sofa.format_age_range(age_meta)})"
        self.similar_status.setText(status)
        self.populate_similar_table()
        self.similar_panel.clear()
        self.render_profile(
            self.similar_panel.layout,
            result.get("target_profile", {}),
            result=result,
            title=f"Target: {target_player} {touch.get('season_label', '')}".strip(),
        )

    @Slot(str, str)
    def analysis_failed(self, target: str, error: str):
        if target == "profile":
            self.profile_button.setEnabled(True)
            self.profile_status.setText("Error")
        else:
            self.similar_button.setEnabled(True)
            self.similar_status.setText("Error")
        QMessageBox.critical(self, "Analysis failed", error)

    def populate_similar_table(self):
        self.similar_table.setRowCount(len(self.similar_profiles))
        for row_idx, profile in enumerate(self.similar_profiles):
            values = [
                str(profile.get("quant_rank", row_idx + 1)),
                profile.get("player", ""),
                profile.get("season_label", ""),
                profile.get("team", ""),
                profile.get("league", ""),
                profile.get("position", ""),
                fmt_num(profile.get("age"), 1),
                fmt_num(profile.get("nineties"), 1),
                fmt_num((profile.get("physicality") or {}).get("pct"), 0),
                fmt_num(profile.get("score", profile.get("rrf_score")), 4),
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.similar_table.setItem(row_idx, col_idx, item)

    def selected_similar_index(self) -> int | None:
        ranges = self.similar_table.selectedRanges()
        if not ranges:
            return None
        row = ranges[0].topRow()
        if row < 0 or row >= len(self.similar_profiles):
            return None
        return row

    def show_selected_similar_profile(self):
        idx = self.selected_similar_index()
        self.open_profile_button.setEnabled(idx is not None)
        if idx is None:
            return
        profile = self.similar_profiles[idx]
        self.similar_panel.clear()
        self.render_profile(
            self.similar_panel.layout,
            profile,
            result=self.similar_result,
            title=f"Similar #{profile.get('quant_rank', idx + 1)}",
        )

    def open_selected_profile(self):
        idx = self.selected_similar_index()
        if idx is None:
            return
        profile = self.similar_profiles[idx]
        self.profile_controls.set_identity(
            profile.get("player", ""),
            profile.get("team"),
            profile.get("league"),
            season=profile.get("season_year"),
        )
        self.tabs.setCurrentWidget(self.profile_tab)
        self.run_profile()

    def render_profile(
        self,
        layout: QVBoxLayout,
        profile: dict,
        *,
        result: dict | None = None,
        maps: dict[str, str] | None = None,
        title: str,
    ):
        if profile.get("error"):
            layout.addWidget(section_label("Analysis error"))
            layout.addWidget(wrapped_label(profile["error"]))
            return

        header = card()
        form = QFormLayout(header)
        name = f"{profile.get('player', '')} ({profile.get('team', '')}, {profile.get('league', '')})"
        form.addRow(section_label(title), QLabel(name))
        if profile.get("season_label"):
            form.addRow("Season", QLabel(profile["season_label"]))
        form.addRow("Position", QLabel(str(profile.get("position", "-"))))
        form.addRow("Primary position", QLabel(str(profile.get("primary_pos", "-"))))
        form.addRow("90s", QLabel(fmt_num(profile.get("nineties"), 1)))
        form.addRow("Age", QLabel(fmt_num(profile.get("age"), 1)))

        feature_meta = (result or {}).get("feature_meta") or {}
        if profile.get("quant_rank"):
            kind = "shape" if feature_meta.get("mode") == "shape" else "RRF"
            form.addRow(
                "Similarity",
                QLabel(
                    f"rank #{profile['quant_rank']}    {kind} "
                    f"{fmt_num(profile.get('score', profile.get('rrf_score')), 4)}"
                ),
            )
        if result:
            pool_meta = result.get("pool_meta", {})
            scope = result.get("scope_seasons") or []
            scope_text = (
                "all seasons"
                if scope and scope == self.seasons
                else ", ".join(ss.season_label(y) for y in scope)
            )
            form.addRow(
                "Shape pool",
                QLabel(
                    f"{result.get('cluster_size', '-')} nearest player-seasons "
                    f"({scope_text}), {pool_meta.get('matched', '-')} matched in Sofascore"
                    f"    role cluster {result.get('cluster_label', '-')}"
                ),
            )
            if result.get("flip_flanks") or (result.get("touch_target") or {}).get("flip_flanks"):
                form.addRow(
                    "Flanks",
                    QLabel("Mirrored left ↔ right (searching the opposite-side shape)"),
                )
            if feature_meta:
                if feature_meta.get("mode") == "shape":
                    basis = (
                        f"touch/action shape — only {feature_meta.get('n_features', 0)} "
                        "stats are recorded in every season here"
                    )
                else:
                    basis = f"{feature_meta.get('n_features', '-')} stats"
                dropped = feature_meta.get("dropped") or []
                if dropped:
                    shown = ", ".join(dropped[:6]) + (" …" if len(dropped) > 6 else "")
                    basis += f"; {len(dropped)} not recorded in every season left out ({shown})"
                skipped = feature_meta.get("skipped_seasons") or []
                if skipped:
                    basis += (
                        f". {', '.join(ss.season_label(y) for y in skipped)} not searched: "
                        "Sofascore has too few stats for those seasons (tick only them, or "
                        "target a player from them, to compare on shape)"
                    )
                form.addRow("Ranked on", wrapped_label(basis))
            age_meta = result.get("age_meta") or {}
            if age_meta.get("min_age") is not None or age_meta.get("max_age") is not None:
                form.addRow("Age filter", QLabel(sofa.format_age_range(age_meta)))
            phys_note = physicality_note(result.get("physicality_meta") or {})
            if phys_note:
                form.addRow("Physicality", wrapped_label(phys_note))
        layout.addWidget(header)

        grid = QGridLayout()
        pct = profile.get("percentiles") or {}
        grid.addWidget(stat_table("Strengths", stat_rows(pct.get("strengths", [])[:8])), 0, 0)
        grid.addWidget(stat_table("Weaknesses", stat_rows(pct.get("weaknesses", [])[:8])), 0, 1)
        grid.addWidget(role_box(profile.get("roles")), 1, 0)
        grid.addWidget(pass_angle_box(profile), 1, 1)
        holder = QWidget()
        holder.setLayout(grid)
        layout.addWidget(holder)

        # Only present once the Tight Space tab has a score table loaded, and only
        # for players who cleared its minimum sample, so it is an extra panel
        # rather than something the profile depends on.
        tight = self.tight_tab.summary_box(
            profile.get("player", ""), profile.get("team"), season=profile.get("season_year")
        )
        if tight is not None:
            layout.addWidget(tight)

        if ((result or {}).get("physicality_meta") or {}).get("enabled"):
            layout.addWidget(physicality_box(profile.get("physicality")))

        if maps:
            layout.addWidget(section_label("Dashboards"))
            layout.addWidget(image_grid(maps))


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("section")
    return label


def wrapped_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def card() -> QFrame:
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setObjectName("card")
    return frame


def stat_table(title: str, rows: list[tuple[str, str, str]]) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    table = QTableWidget(max(len(rows), 1), 3)
    table.setHorizontalHeaderLabels(["Stat", "Value/90", "Percentile"])
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
    table.verticalHeader().setVisible(False)
    table.setSelectionMode(QTableWidget.NoSelection)
    table.setMinimumHeight(260)

    if not rows:
        fallback = "(none above cutoff)" if title == "Strengths" else "(none below cutoff)"
        rows = [(fallback, "", "")]

    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            table.setItem(row_idx, col_idx, item)
    layout.addWidget(table)
    return box


def role_box(roles) -> QGroupBox:
    box = QGroupBox("Role Mix")
    layout = QFormLayout(box)
    if roles is None:
        layout.addRow(QLabel("No role data"))
        return box
    for row in roles.iter_rows(named=True):
        layout.addRow(str(row["role"]), QLabel(f"{float(row['score']):+.2f}"))
    return box


def physicality_note(meta: dict) -> str:
    """One line on how physicality took part in this search, if it did."""
    if not meta.get("enabled"):
        return ""
    bits = [f"{meta.get('n_scored', 0)} of {meta.get('n_pool', 0)} in the pool scored"]
    if meta.get("matched"):
        bits.append("closeness on the physicality axes used as a ranker")
    elif not meta.get("target_scored"):
        bits.append("target not scored, so it could not be matched on")
    if meta.get("min_pct"):
        bits.append(f"only players at or above p{meta['min_pct']:.0f} in role returned")
    if meta.get("failed"):
        bits.append("unavailable for " + "; ".join(meta["failed"]))
    return ", ".join(bits)


def physicality_box(phys: dict | None) -> QGroupBox:
    """The player's physicality profile: its axes, the role-weighted blend, samples."""
    import physicality as phys_model

    if not phys:
        box = QGroupBox("Physicality")
        layout = QVBoxLayout(box)
        layout.addWidget(wrapped_label(
            f"Not scored: under {phys_model.DEFAULT_MIN_MINUTES} minutes that season, "
            "or a goalkeeper."
        ))
        return box

    def ordinal(value) -> str:
        if value is None:
            return "-"
        n = int(round(value))
        suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    role = phys.get("role_group") or "-"
    box = QGroupBox(f"Physicality (vs {role})")
    layout = QFormLayout(box)
    weights = phys_model.axis_weights_for(role)
    for axis, label in phys_model.AXIS_LABELS.items():
        ax = (phys.get("axes") or {}).get(axis) or {}
        layout.addRow(
            label,
            QLabel(
                f"{fmt_num(ax.get('z'), 2):>5} z   ·   {ordinal(ax.get('pct'))} pct in role"
                f"   ·   weight {weights.get(axis, 0):.2f}"
            ),
        )
    layout.addRow(
        "Summary",
        QLabel(f"{fmt_num(phys.get('score'), 2)}  ({ordinal(phys.get('pct'))} pct in role)"),
    )
    layout.addRow(
        "Sample",
        wrapped_label(
            f"{phys.get('aerial_n', 0)} aerials ({fmt_num(phys.get('aerial_won_pct'), 0)}% won, "
            f"{fmt_num(phys.get('aerials_p90'), 1)}/90) · "
            f"{phys.get('ground_n', 0)} tackling duels ({fmt_num(phys.get('ground_won_pct'), 0)}% won) · "
            f"{phys.get('fouls_won', 0)} fouls drawn · "
            f"{fmt_num(phys.get('fouls_committed_p90'), 2)} fouls committed/90 · "
            f"{fmt_num(phys.get('minutes'), 0)} min"
        ),
    )
    layout.addRow(
        "Ground covered",
        wrapped_label(
            f"{fmt_num(phys.get('carry_m_p90'), 0)} m carried/90 "
            f"({fmt_num(phys.get('prog_carry_m_padj'), 0)} m progressive, "
            f"{fmt_num(phys.get('long_carries_padj'), 1)} carries of 15m+, PAdj) · "
            f"defensive range {fmt_num(phys.get('def_range_m'), 1)} m · "
            f"{fmt_num(phys.get('high_def_padj'), 1)} defensive actions/90 in the opposition half (PAdj)"
        ),
    )
    note = wrapped_label(
        "Carrying and work rate are reconstructed from event data as a stand-in for running "
        "data, which no source here has; speed and stamina cannot be recovered from it. "
        "PAdj = scaled to 50% possession. "
        "Duel ratings are opponent-adjusted (who they won against, not just how often). "
        "Axes are z-scored within role group that season; the summary weights them by "
        "position, so it is a stated preference rather than a measured trait."
    )
    note.setObjectName("muted")
    layout.addRow(note)
    return box


def pass_angle_box(profile: dict) -> QGroupBox:
    box = QGroupBox("Pass Angle Tendency")
    layout = QVBoxLayout(box)
    text = QTextEdit()
    text.setReadOnly(True)
    pass_angles = profile.get("pass_angles")
    if pass_angles:
        content = par.format_pass_angle_summary(pass_angles)
        cos = profile.get("pass_angle_cos")
        if cos is not None:
            content = f"Cosine vs target: {cos:+.3f}\n{content}"
    else:
        content = "No directed passes."
    text.setPlainText(content)
    text.setMinimumHeight(180)
    layout.addWidget(text)
    return box


def image_grid(maps: dict[str, str]) -> QWidget:
    widget = QWidget()
    layout = QGridLayout(widget)
    order = ["touch", "pass", "pass_angle", "takeon", "shot", "defensive"]
    labels = {
        "touch": "Touch Map",
        "pass": "Pass Map",
        "pass_angle": "Pass Angle Radar",
        "takeon": "Take-on Map",
        "shot": "Shot Map",
        "defensive": "Defensive Map",
    }
    shown = 0
    for key in order:
        path = maps.get(key)
        if not path or not Path(path).exists():
            continue
        frame = card()
        frame_layout = QVBoxLayout(frame)
        frame_layout.addWidget(section_label(labels.get(key, key)))
        pixmap = QPixmap(path)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setPixmap(
            pixmap.scaled(360, 430, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        frame_layout.addWidget(label)
        path_label = wrapped_label(str(path))
        path_label.setObjectName("muted")
        frame_layout.addWidget(path_label)
        layout.addWidget(frame, shown // 3, shown % 3)
        shown += 1
    if shown == 0:
        layout.addWidget(QLabel("No dashboards were saved for this player."), 0, 0)
    return widget


def apply_styles(app: QApplication):
    app.setStyleSheet(
        """
        QLabel#title {
            font-size: 20px;
            font-weight: 700;
        }
        QLabel#section {
            font-size: 14px;
            font-weight: 700;
        }
        QLabel#muted {
            color: #666;
        }
        QFrame#card {
            border: 1px solid #bbb;
            border-radius: 6px;
            padding: 8px;
            background: palette(base);
        }
        QGroupBox {
            font-weight: 700;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        """
    )


def main() -> int:
    app = QApplication(sys.argv)
    apply_styles(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
