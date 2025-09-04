import os
import time
import json
import sys, subprocess
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

# Project paths / logs
BASE = Path(__file__).resolve().parent.parent
LOG_DIR = BASE / "logs"
CUR_FILE = LOG_DIR / "current.json"
LOG_FILE = LOG_DIR / "results.jsonl"


# --- helper to maybe launch and focus app window ---
def _maybe_launch_and_focus(cfg: dict):
    """Launch the target app (if provided) and try to focus its window.
    Works on macOS (preferred), Windows, and Linux (best effort).
    """
    path = str(cfg.get("app_path", "")).strip()
    if not path:
        return
    try:
        # Launch
        if sys.platform == "darwin":
            p = Path(path)
            if p.exists():
                subprocess.Popen(["/usr/bin/open", str(p)])
            else:
                subprocess.Popen(["/usr/bin/open", "-a", path])
            # Activate frontmost
            app_name = (
                (p.stem if p.suffix == ".app" else p.name)
                if p.name
                else Path(path).stem
            )
            try:
                subprocess.run(
                    [
                        "/usr/bin/osascript",
                        "-e",
                        f'tell application "{app_name}" to activate',
                    ],
                    check=False,
                )
            except Exception:
                pass
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen([path])
    except Exception:
        pass

    # Small delay to allow window to appear
    time.sleep(1.2)

    # Best-effort focus via click inside configured region
    try:
        x, y, w, h = tuple(int(v) for v in cfg.get("region", [0, 0, 400, 300]))
        cx, cy = x + max(5, w // 2), y + max(5, h // 2)
        pyautogui.click(cx, cy)
    except Exception:
        pass


# PyAutoGUI safety tweaks
pyautogui.FAILSAFE = False  # Verhindert Abbruch bei Maus in Ecke
pyautogui.PAUSE = 0  # Keine zusätzliche Wartezeit zwischen Events


class DesktopEnv(gym.Env):
    """Generic Desktop Gym Env using screen capture + key sends.
    Observations: 84x84x1 (grayscale)
    Actions: discrete mapping aus config["actions"] (keys)
    Reward: pixel-diff zu vorherigem Frame (normalisiert)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.region = tuple(int(x) for x in self.cfg.get("region", [0, 0, 400, 300]))
        self.fps = int(self.cfg.get("fps", 5))
        # Action mapping
        self.keys = list(self.cfg.get("actions", {}).keys())
        if not self.keys:
            self.keys = ["noop"]
        self.action_space = spaces.Discrete(len(self.keys))
        # Observation space (H, W, C)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        # Internals
        self._sct = mss.mss()
        self._last = None
        self._t_last = time.time()

    # --- helpers ---
    def _grab(self) -> np.ndarray:
        x, y, w, h = self.region
        shot = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
        img = np.array(shot)[:, :, :3]  # BGRA -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        return img[..., None]

    def _send_action(self, idx: int):
        key = self.keys[idx]
        if key == "noop":
            return
        seq = self.cfg.get("actions", {}).get(key, [])
        # Drücke alle Tasten der Sequenz, dann lasse sie los (um simultane Tasten zu erlauben)
        for k in seq:
            pyautogui.keyDown(k)
        for k in reversed(seq):
            pyautogui.keyUp(k)

    # --- gym API ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._grab()
        self._last = obs
        return obs, {}

    def step(self, action):
        self._send_action(int(action))
        # pace to fps
        wait = max(1.0 / max(1, self.fps) - (time.time() - self._t_last), 0)
        if wait > 0:
            time.sleep(wait)
        self._t_last = time.time()

        obs = self._grab()
        # Reward = durchschnittliche Pixeländerung
        if self._last is None:
            rew = 0.0
        else:
            dif = np.abs(obs.astype(np.int32) - self._last.astype(np.int32)).mean()
            rew = float(dif / 255.0)
        self._last = obs

        terminated = False
        truncated = False
        info = {}
        return obs, rew, terminated, truncated, info


# --- helpers ---


def _load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- training entry used by launcher ---


def train(timesteps: int, log, *, progress=None):
    """Train PPO auf dem DesktopEnv. Liest Pfad aus ENV DESKTOP_CFG.
    - timesteps: gesamte Lernschritte
    - log: callable(str) → Ausgabe in Launcher
    - progress: optional callable(int 0..100)
    """
    cfg_path = os.environ.get("DESKTOP_CFG")
    if not cfg_path:
        raise RuntimeError("DESKTOP_CFG nicht gesetzt.")

    cfg = _load_config(cfg_path)
    _maybe_launch_and_focus(cfg)
    env = DummyVecEnv([lambda: DesktopEnv(cfg)])
    model = PPO("CnnPolicy", env, verbose=0)

    # Schritthäppchen für Progress-Anzeige
    total = int(max(1, timesteps))
    if progress:
        progress(0)
    done = 0
    chunk = max(256, total // 100)  # genügend fein für Balken, nicht zu klein

    while done < total:
        step = min(chunk, total - done)
        model.learn(total_timesteps=step, progress_bar=False)
        done += step
        if progress:
            pct = int(min(100, (100 * done) // total))
            progress(pct)

    # Simple „Eval“: Random-Rollout als grobe Prozentmetrik (0..100)
    env2 = DesktopEnv(cfg)
    tot = 0.0
    n = 0
    obs, _ = env2.reset()
    for _ in range(100):
        a = env2.action_space.sample()
        obs, r, term, trunc, _ = env2.step(a)
        tot += r
        n += 1
        if term or trunc:
            break
    avg = (tot / max(1, n)) * 100.0

    # Loggen wie bestehendes System
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": int(time.time()),
        "game": cfg.get("display_name", "Desktop"),
        "episode": 0,
        "percent": float(avg),
        "steps": int(done),
    }
    with open(CUR_FILE, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"Train done: ~{avg:.1f}%")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=int, default=3)
    p.add_argument("--cfg", type=str, required=True)
    args = p.parse_args()

    cfg = _load_config(args.cfg)
    _maybe_launch_and_focus(cfg)
    env = DesktopEnv(cfg)

    import random

    for ep in range(args.eval):
        tot = 0.0
        obs, _ = env.reset()
        for t in range(120):
            a = random.randrange(env.action_space.n)
            obs, r, term, trunc, _ = env.step(a)
            tot += r
            if term or trunc:
                break
        avg = tot / max(1, t + 1) * 100.0
        rec = {
            "ts": int(time.time()),
            "game": cfg.get("display_name", "Desktop"),
            "episode": ep + 1,
            "percent": float(avg),
            "steps": int(t + 1),
        }
        with open(CUR_FILE, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"Eval Ep {ep + 1}: {avg:.1f}% ({t + 1} steps)")
