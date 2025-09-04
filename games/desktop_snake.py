import os, sys
from pathlib import Path
from importlib import import_module

GAME_NAME = 'Snake'

# We only proxy to the generic desktop engine. You never need to edit this file.

def train(timesteps: int, log, *, progress=None):
    os.environ["DESKTOP_CFG"] = str(Path(__file__).resolve().parent.parent / "configs" / ("snake.json"))
    eng = import_module("games.desktop_env")
    return eng.train(timesteps, log, progress=progress)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=int, default=3)
    args = p.parse_args()
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "snake.json"
    os.environ["DESKTOP_CFG"] = str(cfg_path)
    target = Path(__file__).resolve().parent / "desktop_env.py"
    os.execv(sys.executable, [sys.executable, str(target), "--eval", str(args.eval), "--cfg", str(cfg_path)])
