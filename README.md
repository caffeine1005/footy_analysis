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

A Qt (PySide6) desktop UI for player profiles and similar-player search. The
code is pure Python/Qt with no OS-specific paths, so the same script runs
unmodified on Linux, macOS, and Windows — only the setup/launch commands
differ.

### Required libraries

Listed in [`requirements-desktop.txt`](requirements-desktop.txt), same on
every platform:

- PySide6 (Qt bindings for the UI)
- matplotlib, mplsoccer, pyfonts (pitch maps / plots rendered to images)
- numpy, pandas, polars, scipy, scikit-learn (data processing and similarity search)
- soccerdata, lxml (data readers used by the analysis pipeline modules)

Python 3.12 is recommended (matches the environment this app was built/tested with).

### Setup

**Linux / macOS** (bash/zsh terminal)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-desktop.txt
```

**Windows** (PowerShell or Command Prompt)
```bat
py -3 -m venv .venv
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