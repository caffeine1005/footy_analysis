"""Qt desktop UI for player profiles and similar-player search.

Run on Arch from the repository root:

    ./run_footy_desktop.sh

The UI uses the existing analysis pipeline in `player_profile.py`. Name lookup
is accent-insensitive via `name_utils.py`, so plain English input such as
"Arda Guler" can resolve accented names stored in the data.
"""

from __future__ import annotations

import math
import sys
import time
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
    QHeaderView,
    QLabel,
    QLineEdit,
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
import player_profile as pp
import touchmap_similarity as tms
from name_utils import strip_accents


CACHE_DIR = Path(".desktop_cache") / "maps"


@dataclass
class AnalysisOptions:
    player: str
    team: str | None
    league: str | None
    events_dir: str | None
    season: int
    min_touches: int
    n_clusters: int
    top_n: int
    min_90s: float | None
    action_features: bool
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
    return [(pp.pretty_stat(col), f"{value:.2f}", f"p{pct:.0f}") for col, value, pct in rows]


class WorkerSignals(QObject):
    finished = Signal(str, dict)
    failed = Signal(str, str)


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
            result = pp.analyze(
                opts.player,
                team=opts.team,
                league=opts.league,
                events_dir=opts.events_dir,
                season=opts.season,
                min_touches=opts.min_touches,
                n_clusters=opts.n_clusters,
                top_n=opts.top_n,
                min_90s=opts.min_90s,
                action_features=opts.action_features,
                save_maps_dir=opts.save_maps_dir,
            )
        except Exception as exc:  # noqa: BLE001 - show analysis failures in UI
            self.signals.failed.emit(self.target, str(exc))
            return
        self.signals.finished.emit(self.target, result)


class SearchControls(QWidget):
    def __init__(self, include_top_n: bool):
        super().__init__()
        self.include_top_n = include_top_n

        self.player = QLineEdit()
        self.team = QLineEdit()
        self.league = QLineEdit()
        self.events_dir = QLineEdit()
        self.season = QSpinBox()
        self.season.setRange(2000, 2100)
        self.season.setValue(int(tms.DEFAULT_SEASON))
        self.min_touches = QSpinBox()
        self.min_touches.setRange(1, 100000)
        self.min_touches.setValue(int(tms.DEFAULT_MIN_TOUCHES))
        self.n_clusters = QSpinBox()
        self.n_clusters.setRange(2, 100)
        self.n_clusters.setValue(14)
        self.top_n = QSpinBox()
        self.top_n.setRange(1, 50)
        self.top_n.setValue(10)
        self.min_90s = QDoubleSpinBox()
        self.min_90s.setRange(0.0, 10000.0)
        self.min_90s.setDecimals(1)
        self.min_90s.setSpecialValueText("Any")
        self.min_90s.setValue(0.0)
        self.action_features = QCheckBox("Use action features")
        self.action_features.setChecked(True)

        layout = QGridLayout(self)
        layout.addWidget(QLabel("Player"), 0, 0)
        layout.addWidget(self.player, 0, 1)
        layout.addWidget(QLabel("Team"), 0, 2)
        layout.addWidget(self.team, 0, 3)
        layout.addWidget(QLabel("League"), 0, 4)
        layout.addWidget(self.league, 0, 5)
        layout.addWidget(QLabel("Events dir"), 0, 6)
        layout.addWidget(self.events_dir, 0, 7)

        layout.addWidget(QLabel("Season"), 1, 0)
        layout.addWidget(self.season, 1, 1)
        layout.addWidget(QLabel("Min touches"), 1, 2)
        layout.addWidget(self.min_touches, 1, 3)
        layout.addWidget(QLabel("Clusters"), 1, 4)
        layout.addWidget(self.n_clusters, 1, 5)
        if include_top_n:
            layout.addWidget(QLabel("Top N"), 1, 6)
            layout.addWidget(self.top_n, 1, 7)
        layout.addWidget(QLabel("Min 90s"), 2, 0)
        layout.addWidget(self.min_90s, 2, 1)
        layout.addWidget(self.action_features, 2, 2, 1, 2)

        for col in (1, 3, 5, 7):
            layout.setColumnStretch(col, 1)

    def options(self, *, save_maps: bool, force_top_n: int | None = None) -> AnalysisOptions:
        player = self.player.text().strip()
        if not player:
            raise ValueError("Enter a player name.")

        save_maps_dir = None
        if save_maps:
            save_maps_dir = CACHE_DIR / f"{slug(player)}_{int(time.time())}"

        min_90s = None if self.min_90s.value() <= 0 else float(self.min_90s.value())
        return AnalysisOptions(
            player=player,
            team=optional_text(self.team.text()),
            league=optional_text(self.league.text()),
            events_dir=optional_text(self.events_dir.text()),
            season=int(self.season.value()),
            min_touches=int(self.min_touches.value()),
            n_clusters=int(self.n_clusters.value()),
            top_n=int(force_top_n or self.top_n.value()),
            min_90s=min_90s,
            action_features=self.action_features.isChecked(),
            save_maps_dir=save_maps_dir,
        )

    def set_identity(self, player: str, team: str | None, league: str | None):
        self.player.setText(player or "")
        self.team.setText(team or "")
        self.league.setText(league or "")


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
        self.setWindowTitle("Footy Analysis")
        self.resize(1280, 860)
        self.pool = QThreadPool.globalInstance()
        self.similar_profiles: list[dict] = []
        self.similar_result: dict | None = None

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        self.profile_tab = QWidget()
        self.similar_tab = QWidget()
        tabs.addTab(self.profile_tab, "Player Profile")
        tabs.addTab(self.similar_tab, "Similar Players")
        self.tabs = tabs

        self._build_profile_tab()
        self._build_similar_tab()

    def _build_profile_tab(self):
        layout = QVBoxLayout(self.profile_tab)
        title = QLabel("Player Profile")
        title.setObjectName("title")
        layout.addWidget(title)

        top = QHBoxLayout()
        self.profile_controls = SearchControls(include_top_n=False)
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
        title = QLabel("Similar Players")
        title.setObjectName("title")
        layout.addWidget(title)

        top = QHBoxLayout()
        self.similar_controls = SearchControls(include_top_n=True)
        top.addWidget(self.similar_controls, stretch=1)
        buttons = QVBoxLayout()
        self.similar_button = QPushButton("Find Similar")
        self.similar_button.clicked.connect(self.run_similar)
        self.open_profile_button = QPushButton("Open Selected Profile")
        self.open_profile_button.setEnabled(False)
        self.open_profile_button.clicked.connect(self.open_selected_profile)
        self.similar_status = QLabel("Idle")
        buttons.addWidget(self.similar_button)
        buttons.addWidget(self.open_profile_button)
        buttons.addWidget(self.similar_status)
        buttons.addStretch()
        top.addLayout(buttons)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Vertical)
        self.similar_table = QTableWidget(0, 7)
        self.similar_table.setHorizontalHeaderLabels(
            ["#", "Player", "Team", "League", "Position", "Age", "RRF"]
        )
        header = self.similar_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
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
        self.profile_status.setText("Running analysis...")
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
        self.pool.start(worker)

    @Slot(str, dict)
    def analysis_finished(self, target: str, result: dict):
        if target == "profile":
            self.profile_button.setEnabled(True)
            name = result.get("touch_target", {}).get("player", "player")
            self.profile_status.setText(f"Loaded {name}")
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
        self.similar_status.setText(f"Loaded {len(self.similar_profiles)} matches")
        self.populate_similar_table()
        self.similar_panel.clear()
        target_name = result.get("touch_target", {}).get("player", "")
        self.render_profile(
            self.similar_panel.layout,
            result.get("target_profile", {}),
            result=result,
            title=f"Target: {target_name}",
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
                profile.get("team", ""),
                profile.get("league", ""),
                profile.get("position", ""),
                fmt_num(profile.get("age"), 1),
                fmt_num(profile.get("rrf_score"), 4),
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
        form.addRow("Position", QLabel(str(profile.get("position", "-"))))
        form.addRow("Primary position", QLabel(str(profile.get("primary_pos", "-"))))
        form.addRow("90s", QLabel(fmt_num(profile.get("nineties"), 1)))
        form.addRow("Age", QLabel(fmt_num(profile.get("age"), 1)))

        if profile.get("quant_rank"):
            form.addRow(
                "Similarity",
                QLabel(
                    f"rank #{profile['quant_rank']}    RRF {fmt_num(profile.get('rrf_score'), 4)}"
                ),
            )
        if result:
            pool_meta = result.get("pool_meta", {})
            form.addRow(
                "Cluster",
                QLabel(
                    f"{result.get('cluster_label', '-')}    "
                    f"{result.get('cluster_size', '-')} with touch data, "
                    f"{pool_meta.get('matched', '-')} matched in FBref"
                ),
            )
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
