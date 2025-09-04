import sys, os, threading, subprocess, importlib.util, time, json, inspect
from pathlib import Path
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QTextEdit,
    QCheckBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
    QProgressBar,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QFileDialog,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

BASE = Path(__file__).resolve().parent
GAMES_DIR = BASE / "games"
LOG_DIR = BASE / "logs"
LOG_FILE = LOG_DIR / "results.jsonl"
CUR_FILE = LOG_DIR / "current.json"
CONFIGS_DIR = BASE / "configs"

# Ensure directories exist
GAMES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def import_game(path):
    try:
        spec = importlib.util.spec_from_file_location(f"games.{path.stem}", str(path))
        if spec is None or spec.loader is None:
            return None
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # Legacy-friendly: only require GAME_NAME and train(); main() optional
        if not hasattr(m, "GAME_NAME") or not hasattr(m, "train"):
            return None
        return m
    except Exception:
        # keep UI alive if a module fails to import
        return None


def scan_games():
    mods = []
    candidates = []
    # New-style: games/*.py
    candidates += sorted(GAMES_DIR.glob("*.py"))
    # Legacy: root-level game files (skip launcher/logger/init)
    candidates += sorted(BASE.glob("*.py"))
    SKIP = {"launcher.py", "logger.py", "__init__.py"}
    seen = set()
    for p in candidates:
        if p.name.startswith("_") or p.name in SKIP:
            continue
        key = (p.name, p.parent)
        if key in seen:
            continue
        seen.add(key)
        m = import_game(p)
        if m is not None:
            mods.append(m)
    return mods


class ScoreboardWidget(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        self.lbl_cur_title = QLabel("<b>Aktuelles Spiel:</b>")
        self.lbl_cur_game = QLabel("-")
        self.lbl_score_title = QLabel("<b>Score:</b>")
        self.lbl_score = QLabel("--")
        for w in [
            self.lbl_cur_title,
            self.lbl_cur_game,
            self.lbl_score_title,
            self.lbl_score,
        ]:
            w.setAlignment(Qt.AlignmentFlag.AlignLeft)
        v.addWidget(self.lbl_cur_title)
        v.addWidget(self.lbl_cur_game)
        v.addWidget(self.lbl_score_title)
        v.addWidget(self.lbl_score)
        v.addWidget(QLabel("<b>Durchschnitte:</b>"))
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["Spiel", "Avg %", "Anzahl"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl)
        v.addWidget(QLabel("Letzte Runs:"))
        self.list_recent = QListWidget()
        v.addWidget(self.list_recent)
        self.last_cur_m = 0.0
        self.last_log_m = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(400)
        self.refresh(force=True)

    def load_lines(self):
        rows = []
        if LOG_FILE.exists():
            with open(LOG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except:
                            pass
        return rows

    def load_current(self):
        if CUR_FILE.exists():
            try:
                with open(CUR_FILE, "r") as f:
                    return json.load(f)
            except:
                return None
        return None

    def aggregate(self, rows):
        by = {}
        for r in rows:
            g = r.get("game", "unknown")
            p = float(r.get("percent", 0.0))
            if g not in by:
                by[g] = {"sum": 0.0, "n": 0}
            by[g]["sum"] += p
            by[g]["n"] += 1
        for g in list(by.keys()):
            s = by[g]["sum"]
            n = by[g]["n"]
            by[g]["avg"] = s / max(1, n)
        return by

    def refresh(self, force=False):
        m1 = os.path.getmtime(CUR_FILE) if CUR_FILE.exists() else 0.0
        m2 = os.path.getmtime(LOG_FILE) if LOG_FILE.exists() else 0.0
        if not force and m1 == self.last_cur_m and m2 == self.last_log_m:
            return
        cur = self.load_current()
        rows = self.load_lines()
        by = self.aggregate(rows)
        if cur:
            self.lbl_cur_game.setText(str(cur.get("game", "-")))
            self.lbl_score.setText(f"{float(cur.get('percent',0.0)):.2f}%")
        else:
            self.lbl_cur_game.setText("-")
            self.lbl_score.setText("--")
        self.tbl.setRowCount(0)
        for g, s in sorted(by.items(), key=lambda kv: kv[1]["avg"], reverse=True):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem(str(g)))
            self.tbl.setItem(r, 1, QTableWidgetItem(f"{s['avg']:.2f}%"))
            self.tbl.setItem(r, 2, QTableWidgetItem(str(s["n"])))
        self.list_recent.clear()
        take = rows[-12:][::-1]
        for r in take:
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("ts", 0)))
            line = f"{t}  {r.get('game','?')}  Ep {r.get('episode','?')}: {float(r.get('percent',0.0)):.1f}%  ({r.get('steps','?')} steps)"
            self.list_recent.addItem(line)
        self.last_cur_m = m1
        self.last_log_m = m2


class StatsWidget(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        r = QHBoxLayout()
        r.addWidget(QLabel("Spiel"))
        self.cb_game = QComboBox()
        r.addWidget(self.cb_game, 1)
        r.addWidget(QLabel("Zeitraum"))
        self.cb_range = QComboBox()
        self.cb_range.addItems(["7 Tage", "30 Tage", "90 Tage", "365 Tage", "Alle"])
        r.addWidget(self.cb_range)
        self.btn_reload = QPushButton("Aktualisieren")
        r.addWidget(self.btn_reload)
        v.addLayout(r)
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Datum", "Avg %", "Anzahl", "Best %"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl, 2)
        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        v.addWidget(self.canvas, 3)
        self.btn_reload.clicked.connect(self.refresh)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_if_changed)
        self.timer.start(800)
        self.last_m = 0.0
        self.games = []
        self.reload_games()
        self.refresh()

    def reload_games(self):
        rows = self._load_lines()
        names = sorted(list({r.get("game", "unknown") for r in rows}))
        self.games = names
        self.cb_game.clear()
        self.cb_game.addItems(names)

    def _load_lines(self):
        rows = []
        if LOG_FILE.exists():
            with open(LOG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except:
                            pass
        return rows

    def _range_days(self):
        m = self.cb_range.currentText()
        if m.startswith("7"):
            return 7
        if m.startswith("30"):
            return 30
        if m.startswith("90"):
            return 90
        if m.startswith("365"):
            return 365
        return None

    def refresh_if_changed(self):
        m = os.path.getmtime(LOG_FILE) if LOG_FILE.exists() else 0.0
        if m != self.last_m:
            self.refresh()
            self.last_m = m

    def refresh(self):
        rows = self._load_lines()
        if not rows:
            self.tbl.setRowCount(0)
            self.fig.clear()
            self.canvas.draw()
            return
        if self.cb_game.count() == 0:
            self.reload_games()
            if self.cb_game.count() == 0:
                self.tbl.setRowCount(0)
                self.fig.clear()
                self.canvas.draw()
                return
        game = self.cb_game.currentText()
        days = self._range_days()
        now = time.time()
        flt = []
        for r in rows:
            if r.get("game") != game:
                continue
            if days is not None and now - r.get("ts", 0) > days * 86400:
                continue
            flt.append(r)
        daily = {}
        for r in flt:
            d = time.strftime("%Y-%m-%d", time.localtime(r.get("ts", 0)))
            p = float(r.get("percent", 0.0))
            if d not in daily:
                daily[d] = {"sum": 0.0, "n": 0, "best": 0.0}
            daily[d]["sum"] += p
            daily[d]["n"] += 1
            if p > daily[d]["best"]:
                daily[d]["best"] = p
        keys = sorted(daily.keys())
        self.tbl.setRowCount(0)
        ys = []
        for d in keys:
            avg = daily[d]["sum"] / max(1, daily[d]["n"])
            n = daily[d]["n"]
            best = daily[d]["best"]
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem(d))
            self.tbl.setItem(r, 1, QTableWidgetItem(f"{avg:.2f}"))
            self.tbl.setItem(r, 2, QTableWidgetItem(str(n)))
            self.tbl.setItem(r, 3, QTableWidgetItem(f"{best:.2f}"))
            ys.append(avg)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.bar(keys, ys)
        ax.set_ylabel("Avg %")
        ax.set_xlabel("Datum")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=45)
        self.fig.tight_layout()
        self.canvas.draw()


class Launcher(QWidget):
    log_msg = pyqtSignal(str)
    enable_start = pyqtSignal(bool)
    progress_range = pyqtSignal(int, int)
    progress_value = pyqtSignal(int)
    progress_format = pyqtSignal(str)
    status_text = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trainer")
        self.resize(980, 700)
        self.games = scan_games()
        self.tabs = QTabWidget()
        self.tab_train = QWidget()
        self.tab_score = ScoreboardWidget()
        self.tab_stats = StatsWidget()
        vroot = QVBoxLayout(self)
        vroot.addWidget(self.tabs)
        self.tabs.addTab(self.tab_train, "Trainer")
        self.tabs.addTab(self.tab_score, "Scoreboard")
        self.tabs.addTab(self.tab_stats, "Statistiken")
        v = QVBoxLayout(self.tab_train)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Spiel"))
        self.cb = QComboBox()
        self.cb.addItems([g.GAME_NAME for g in self.games])
        r1.addWidget(self.cb, 1)
        self.btn_reload = QPushButton("Reload Spiele")
        self.btn_reload.clicked.connect(self.reload_games)
        r1.addWidget(self.btn_reload)
        self.btn_add_desktop = QPushButton("Desktop-Spiel hinzufügen")
        self.btn_add_desktop.clicked.connect(self.add_desktop_game)
        r1.addWidget(self.btn_add_desktop)
        v.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Timesteps"))
        self.steps = QSpinBox()
        self.steps.setRange(10000, 5_000_000)
        self.steps.setSingleStep(10000)
        self.steps.setValue(50000)
        r2.addWidget(self.steps)
        r2.addWidget(QLabel("Eval Episoden"))
        self.eval_eps = QSpinBox()
        self.eval_eps.setRange(1, 100)
        self.eval_eps.setValue(3)
        r2.addWidget(self.eval_eps)
        r2.addWidget(QLabel("Zyklen"))
        self.cycles = QSpinBox()
        self.cycles.setRange(1, 1000000)
        self.cycles.setValue(1)
        r2.addWidget(self.cycles)
        v.addLayout(r2)
        r3 = QHBoxLayout()
        self.chk_eval_only = QCheckBox("Nur Evaluieren")
        self.chk_endless = QCheckBox("Endlos Evaluieren")
        r3.addWidget(self.chk_eval_only)
        r3.addWidget(self.chk_endless)
        self.chk_silent = QCheckBox("Eval im Hintergrund (kein Fenster)")
        r3.addWidget(self.chk_silent)
        self.chk_minimize = QCheckBox("Eval-Fenster minimieren")
        r3.addWidget(self.chk_minimize)
        v.addLayout(r3)
        r4 = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_run)
        r4.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_run)
        r4.addWidget(self.btn_stop)
        v.addLayout(r4)
        r5 = QHBoxLayout()
        r5.addWidget(QLabel("Status"))
        self.lbl_status = QLabel("Bereit")
        r5.addWidget(self.lbl_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("")
        r5.addWidget(self.progress, 1)
        v.addLayout(r5)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        v.addWidget(self.log, 1)
        self.stop_event = threading.Event()
        self.log_msg.connect(self.log.append)
        self.enable_start.connect(self.btn_start.setEnabled)
        self.progress_range.connect(self.progress.setRange)
        self.progress_value.connect(self.progress.setValue)
        self.progress_format.connect(self.progress.setFormat)
        self.status_text.connect(self.lbl_status.setText)

    def _ensure_desktop_engine(self):
        """Create games/desktop_env.py if it does not yet exist."""
        try:
            eng_path = GAMES_DIR / "desktop_env.py"
            if eng_path.exists():
                return
            code = """
import os, time, json
from pathlib import Path
import numpy as np
try:
    import mss
    import pyautogui
    import cv2
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as e:
    raise RuntimeError(f"Fehlende Abhängigkeiten für Desktop-Engine: {e}")

BASE = Path(__file__).resolve().parent.parent
LOG_DIR = BASE / "logs"
CUR_FILE = LOG_DIR / "current.json"
LOG_FILE = LOG_DIR / "results.jsonl"

class DesktopEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.region = tuple(int(x) for x in self.cfg.get("region", [0,0,400,300]))
        self.fps = int(self.cfg.get("fps", 5))
        self.keys = list(self.cfg.get("actions", {}).keys())
        if not self.keys:
            self.keys = ["noop"]
        self.action_space = spaces.Discrete(len(self.keys))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)
        self._sct = mss.mss()
        self._last = None
        self._t_last = time.time()

    def _grab(self):
        x,y,w,h = self.region
        img = np.array(self._sct.grab({"left":x, "top":y, "width":w, "height":h}))
        img = img[:,:,:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (84,84), interpolation=cv2.INTER_AREA)
        return img[...,None]

    def _send_action(self, idx):
        key = self.keys[idx]
        if key == "noop":
            return
        seq = self.cfg.get("actions", {}).get(key, [])
        for k in seq:
            pyautogui.keyDown(k)
        for k in reversed(seq):
            pyautogui.keyUp(k)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._grab()
        self._last = obs
        return obs, {}

    def step(self, action):
        self._send_action(int(action))
        dt = max(1.0/self.fps - (time.time()-self._t_last), 0)
        if dt > 0:
            time.sleep(dt)
        self._t_last = time.time()
        obs = self._grab()
        if self._last is None:
            rew = 0.0
        else:
            dif = np.abs(obs.astype(np.int32) - self._last.astype(np.int32)).mean()
            rew = float(dif / 255.0)
        self._last = obs
        terminated = False
        truncated = False
        return obs, rew, terminated, truncated, {}


def _load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train(timesteps: int, log, *, progress=None):
    cfg_path = os.environ.get("DESKTOP_CFG")
    if not cfg_path:
        raise RuntimeError("DESKTOP_CFG nicht gesetzt.")
    cfg = _load_config(cfg_path)
    env = DummyVecEnv([lambda: DesktopEnv(cfg)])
    model = PPO("CnnPolicy", env, verbose=0)

    chunks = max(1, timesteps // 1000)
    done = 0
    if progress:
        progress(0)
    for i in range(chunks):
        step = 1000 if i < chunks-1 else timesteps-done
        if step <= 0:
            break
        model.learn(total_timesteps=step, progress_bar=False)
        done += step
        pct = int(min(100, (100*done)//max(1,timesteps)))
        if progress:
            progress(pct)

    env2 = DesktopEnv(cfg)
    tot = 0.0
    n = 0
    obs,_ = env2.reset()
    for _ in range(100):
        a = env2.action_space.sample()
        obs, r, term, trunc, _ = env2.step(a)
        tot += r
        n += 1
    avg = (tot/max(1,n))*100.0
    rec = {"ts": int(time.time()), "game": cfg.get("display_name","Desktop"), "episode": 0, "percent": float(avg), "steps": int(done)}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CUR_FILE, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec)+"\n")
    log(f"Train done: ~{avg:.1f}%")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=int, default=3)
    p.add_argument("--cfg", type=str, required=True)
    args = p.parse_args()
    cfg = _load_config(args.cfg)
    env = DesktopEnv(cfg)
    import random
    for ep in range(args.eval):
        tot = 0.0
        obs,_ = env.reset()
        for t in range(120):
            a = random.randrange(env.action_space.n)
            obs, r, term, trunc, _ = env.step(a)
            tot += r
            if term or trunc:
                break
        avg = tot/ max(1,t+1) * 100.0
        rec = {"ts": int(time.time()), "game": cfg.get("display_name","Desktop"), "episode": ep+1, "percent": float(avg), "steps": int(t+1)}
        with open(CUR_FILE, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec)+"\n")
            print(f"Eval Ep {ep+1}: {avg:.1f}% ({t+1} steps)")
"""
            eng_path.write_text(code.strip() + "\n", encoding="utf-8")
        except Exception as e:
            self.append(f"Fehler beim Anlegen der Desktop-Engine: {e}")

    def add_desktop_game(self):
        try:
            game_id, ok = self._prompt_text(
                "Neues Desktop-Spiel", "Eindeutige ID (z.B. zelda):"
            )
            if not ok or not game_id:
                return
            display, ok = self._prompt_text(
                "Neues Desktop-Spiel", "Anzeigename (z.B. The Legend of Zelda):"
            )
            if not ok or not display:
                return
            # App/Bundle auswählen (optional). macOS-Apps sind ".app"-Pakete
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "App auswählen (optional)",
                "/Applications",
                "Applications (*.app);;Alle Dateien (*)",
            )
            app_path = file_path or ""
            # Wenn Dialog abgebrochen wird, bleibt app_path ggf. leer (ist okay)
            region_txt, ok = self._prompt_text(
                "Bildschirm-Region", "x,y,w,h (z.B. 100,100,800,600):", "0,0,800,600"
            )
            if not ok or not region_txt:
                return
            try:
                rx, ry, rw, rh = [int(s.strip()) for s in region_txt.split(",")]
            except Exception:
                QMessageBox.warning(
                    self, "Eingabe", "Ungültige Region. Erwartet: x,y,w,h"
                )
                return
            fps_txt, ok = self._prompt_text("FPS", "Erfassungs-FPS (z.B. 5):", "5")
            if not ok or not fps_txt:
                return
            try:
                fps = max(1, int(fps_txt))
            except Exception:
                QMessageBox.warning(self, "Eingabe", "Ungültige FPS.")
                return
            actions_txt, ok = self._prompt_text(
                "Tasten-Aktionen",
                "Mapping im Format key=comma_getrennte_tasten; mehrere durch |.\nBeispiel: left=a | right=d | jump=space | attack=j,k",
                "left=a | right=d | up=w | down=s | jump=space",
            )
            if not ok:
                return
            actions = {}
            for part in actions_txt.split("|"):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                seq = [t.strip() for t in v.split(",") if t.strip()]
                actions[k.strip()] = seq

            cfg = {
                "id": game_id,
                "display_name": display,
                "app_path": app_path,
                "region": [rx, ry, rw, rh],
                "fps": fps,
                "actions": actions,
                "engine": "desktop_env",
            }
            cfg_path = CONFIGS_DIR / f"{game_id}.json"
            if cfg_path.exists():
                QMessageBox.warning(
                    self, "Vorhanden", f"Config existiert bereits: {cfg_path.name}"
                )
                return
            cfg_path.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            self._ensure_desktop_engine()

            stub_path = GAMES_DIR / f"desktop_{game_id}.py"
            if stub_path.exists():
                QMessageBox.warning(
                    self, "Vorhanden", f"Stub existiert bereits: {stub_path.name}"
                )
                return
            stub_code = f"""
import os, sys
from pathlib import Path
from importlib import import_module

GAME_NAME = {cfg['display_name']!r}

# We only proxy to the generic desktop engine. You never need to edit this file.

def train(timesteps: int, log, *, progress=None):
    os.environ["DESKTOP_CFG"] = str(Path(__file__).resolve().parent.parent / "configs" / ("{cfg['id']}.json"))
    eng = import_module("games.desktop_env")
    return eng.train(timesteps, log, progress=progress)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=int, default=3)
    args = p.parse_args()
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "{cfg['id']}.json"
    os.environ["DESKTOP_CFG"] = str(cfg_path)
    target = Path(__file__).resolve().parent / "desktop_env.py"
    os.execv(sys.executable, [sys.executable, str(target), "--eval", str(args.eval), "--cfg", str(cfg_path)])
"""
            stub_path.write_text(stub_code.strip() + "\n", encoding="utf-8")

            self.append(f"Desktop-Config angelegt: {cfg_path.name}")
            self.append(f"Stub erstellt: {stub_path.name}")
            self.reload_games()
            QMessageBox.information(self, "Fertig", "Desktop-Spiel wurde hinzugefügt.")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", str(e))

    def append(self, msg):
        self.log_msg.emit(msg)

    def reload_games(self):
        self._ensure_desktop_engine()
        self.games = scan_games()
        self.cb.clear()
        self.cb.addItems([g.GAME_NAME for g in self.games])
        if not self.games:
            self.btn_start.setEnabled(False)
        else:
            self.btn_start.setEnabled(True)
        self.append("Spiele neu geladen.")
        self.tab_stats.reload_games()

    def start_run(self):
        if not self.games:
            self.append("Keine Spiele gefunden.")
            return
        idx = self.cb.currentIndex()
        if idx < 0 or idx >= len(self.games):
            self.append("Bitte zuerst ein gültiges Spiel auswählen.")
            return
        mod = self.games[idx]
        ts = int(self.steps.value())
        eps = int(self.eval_eps.value())
        cycles = int(self.cycles.value())
        eval_only = self.chk_eval_only.isChecked()
        endless = self.chk_endless.isChecked()
        self.stop_event.clear()
        self.btn_start.setEnabled(False)
        self.tabs.setCurrentWidget(self.tab_score)
        t = threading.Thread(
            target=self.run_worker,
            args=(mod, ts, eps, cycles, eval_only, endless),
            daemon=True,
        )
        self.progress_range.emit(0, 100)
        self.progress_value.emit(0)
        self.progress_format.emit("")
        self.status_text.emit("Startet…")
        t.start()

    def stop_run(self):
        self.stop_event.set()
        self.append("Stop angefordert.")

    def run_worker(self, mod, ts, eps, cycles, eval_only, endless):
        try:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            env["PYTHONPATH"] = str(BASE) + os.pathsep + env.get("PYTHONPATH", "")
            if endless:
                while not self.stop_event.is_set():
                    if not eval_only:
                        self.append(f"Training {mod.GAME_NAME}...")
                        # Unterstützt das Modul 'progress'?
                        try:
                            has_prog = (
                                "progress" in inspect.signature(mod.train).parameters
                            )
                        except Exception:
                            has_prog = False

                        if has_prog:
                            self.progress_range.emit(0, 100)
                            self.progress_format.emit("Training %p%")
                            self.status_text.emit("Training…")

                            def _pupdate(pct: int):
                                self.progress_value.emit(int(pct))

                            mod.train(ts, self.append, progress=_pupdate)
                            self.progress_value.emit(100)
                        else:
                            self.progress_range.emit(0, 0)  # indeterminate
                            self.progress_format.emit("Training läuft…")
                            self.status_text.emit("Training…")
                            mod.train(ts, self.append)
                            self.progress_range.emit(0, 100)
                            self.progress_value.emit(100)

                    self.append(f"Evaluierung {mod.GAME_NAME}...")
                    self.progress_range.emit(0, 0)  # indeterminate
                    self.progress_format.emit("Evaluierung…")
                    self.status_text.emit("Evaluierung…")
                    # Only pass --eval for maximum compatibility with legacy scripts
                    cmd = [sys.executable, str(Path(mod.__file__)), "--eval", str(eps)]
                    # Stream child process output into the GUI log
                    p = subprocess.Popen(
                        cmd,
                        env=env,
                        cwd=str(BASE),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    if p.stdout is not None:
                        for line in p.stdout:
                            self.append(line.rstrip())
                    p.wait()
            else:
                for i in range(cycles):
                    if self.stop_event.is_set():
                        break
                    if not eval_only:
                        self.append(
                            f"Zyklus {i+1}/{cycles}: Training {mod.GAME_NAME}..."
                        )
                        try:
                            has_prog = (
                                "progress" in inspect.signature(mod.train).parameters
                            )
                        except Exception:
                            has_prog = False

                        if has_prog:
                            self.progress_range.emit(0, 100)
                            self.progress_format.emit("Training %p%")
                            self.status_text.emit("Training…")

                            def _pupdate(pct: int):
                                self.progress_value.emit(int(pct))

                            mod.train(ts, self.append, progress=_pupdate)
                            self.progress_value.emit(100)
                        else:
                            self.progress_range.emit(0, 0)
                            self.progress_format.emit("Training läuft…")
                            self.status_text.emit("Training…")
                            mod.train(ts, self.append)
                            self.progress_range.emit(0, 100)
                            self.progress_value.emit(100)

                    self.append(
                        f"Zyklus {i+1}/{cycles}: Evaluierung {mod.GAME_NAME}..."
                    )
                    self.progress_range.emit(0, 0)
                    self.progress_format.emit("Evaluierung…")
                    self.status_text.emit("Evaluierung…")
                    # Only pass --eval for maximum compatibility with legacy scripts
                    cmd = [sys.executable, str(Path(mod.__file__)), "--eval", str(eps)]
                    # Stream child process output into the GUI log
                    p = subprocess.Popen(
                        cmd,
                        env=env,
                        cwd=str(BASE),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    if p.stdout is not None:
                        for line in p.stdout:
                            self.append(line.rstrip())
                    p.wait()
            self.append("Fertig.")
        except Exception as e:
            self.append(f"Fehler: {e}")
        finally:
            self.enable_start.emit(True)
            self.status_text.emit("Bereit")
            self.progress_range.emit(0, 100)
            self.progress_value.emit(0)
            self.progress_format.emit("")

    def _prompt_text(self, title: str, label: str, text: str = ""):
        val, ok = QInputDialog.getText(
            self, title, label, QLineEdit.EchoMode.Normal, text
        )
        return val.strip(), bool(ok)

    def _write_text_file(self, path: Path, content: str, overwrite: bool = False):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Datei existiert bereits: {path.name}")
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Launcher()
    w.show()
    app.exec()
