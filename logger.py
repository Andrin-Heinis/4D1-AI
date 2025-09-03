import os, json, time
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "logs"
LOG_FILE = LOG_DIR / "results.jsonl"
CUR_FILE = LOG_DIR / "current.json"


def _ensure():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def push_current(game, percent, extra=None):
    _ensure()
    rec = {
        "ts": int(time.time()),
        "game": str(game),
        "percent": float(percent),
        "extra": extra or {},
    }
    tmp = CUR_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f, ensure_ascii=False)
    os.replace(tmp, CUR_FILE)


def log_result(game, episode, percent, steps, duration_s, extra=None):
    _ensure()
    rec = {
        "ts": int(time.time()),
        "game": str(game),
        "episode": int(episode),
        "percent": float(percent),
        "steps": int(steps),
        "duration_s": float(duration_s),
        "extra": extra or {},
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
