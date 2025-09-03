import time, argparse, json, numpy as np, gymnasium as gym, warnings
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import push_current, log_result

GAME_NAME = "LunarLander"


def _make_env(render_mode=None):
    # Prefer v3. Fall back to older IDs only if needed, but mute deprecation spam.
    cands = [
        ("LunarLander-v3", False),
        ("LunarLander-v2", True),
        ("LunarLander-v1", True),
        ("LunarLander-v0", True),
    ]
    last_err = None
    for cid, mute_depr in cands:
        try:
            with warnings.catch_warnings():
                if mute_depr:
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                if render_mode is not None:
                    env = gym.make(cid, render_mode=render_mode)
                else:
                    env = gym.make(cid)
            return env, cid
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        'Kein kompatibles LunarLander-Env gefunden. Installiere: pip install "gymnasium[box2d]>=0.29" und ggf. `brew install swig`.'
    ) from last_err


def _paths():
    base = Path(__file__).resolve().parents[1]
    models = base / "models"
    models.mkdir(exist_ok=True)
    best = models / "lunar_best.zip"
    meta = models / "lunar_best_meta.json"
    cand = models / "lunar_candidate.zip"
    return models, best, meta, cand


def _load_meta(meta_path):
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except:
            return {}
    return {}


def _save_meta(meta_path, best_percent):
    meta = {"best_percent": float(best_percent), "updated_ts": int(time.time())}
    meta_path.write_text(json.dumps(meta))


def _eval_percent_lunar(model, n_episodes=5):
    env, _ = _make_env()
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(a))
            done = terminated or truncated
            ret += float(r)
        pct = float(np.clip((ret + 100.0) / 300.0, 0.0, 1.0) * 100.0)
        scores.append(pct)
    env.close()
    return float(np.mean(scores))


def train(timesteps, log, progress=None):
    models, best, meta, cand = _paths()
    env, env_id = _make_env()
    if best.exists():
        model = PPO.load(str(best), env=env)
        log("Weitertrainieren vom BEST-Modell...")
    else:
        model = PPO("MlpPolicy", env, verbose=1)
        log("Neues Modell...")
    log(f"Env: {env_id}")
    # Optionaler Fortschritts-Callback für den Launcher
    cb = None
    if progress is not None:
        total = int(max(1, timesteps))

        class _ProgressCB(BaseCallback):
            def __init__(self, total_steps, cb_fn):
                super().__init__()
                self.total = int(max(1, total_steps))
                self.cb_fn = cb_fn
                self._last_emit = -1
                self._last_t = 0.0

            def _on_training_start(self) -> None:
                try:
                    self.cb_fn(0)
                except Exception:
                    pass
                self._last_t = time.time()
                return True

            def _on_step(self) -> bool:
                try:
                    pct = int(min(100, (100 * self.num_timesteps) // self.total))
                    now = time.time()
                    if pct != self._last_emit and (now - self._last_t) >= 0.25:
                        self.cb_fn(pct)
                        self._last_emit = pct
                        self._last_t = now
                except Exception:
                    pass
                return True

            def _on_training_end(self) -> None:
                try:
                    self.cb_fn(100)
                except Exception:
                    pass
                return True

        cb = _ProgressCB(total, progress)
    model.learn(total_timesteps=int(timesteps), callback=cb)
    model.save(str(cand))
    env.close()
    new_score = _eval_percent_lunar(PPO.load(str(cand), env=_make_env()[0]))
    old_score = float(_load_meta(meta).get("best_percent", 0.0))
    if new_score > old_score:
        os.replace(str(cand), str(best))
        _save_meta(meta, new_score)
        ts = int(time.time())
        PPO.load(str(best), env=_make_env()[0]).save(str(models / f"lunar_best_{ts}"))
        log(f"Verbessert: {new_score:.2f}% > {old_score:.2f}% → BEST aktualisiert.")
    else:
        try:
            os.remove(str(cand))
        except:
            pass
        log(f"Nicht besser: {new_score:.2f}% <= {old_score:.2f}% → verwerfe.")


def eval_once(model, render_mode="human"):
    # render_mode: "human" for window, None for headless
    env, _ = _make_env(render_mode=render_mode)
    obs, _ = env.reset()
    done = False
    steps = 0
    ret = 0.0
    t0 = time.time()
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(a))
        done = terminated or truncated
        ret += float(r)
        steps += 1
        pct = float(np.clip((ret + 100.0) / 300.0, 0.0, 1.0) * 100.0)
        push_current(GAME_NAME, pct)
        if render_mode == "human":
            try:
                env.render()
            except Exception:
                pass
    pct = float(np.clip((ret + 100.0) / 300.0, 0.0, 1.0) * 100.0)
    log_result(GAME_NAME, 1, pct, steps, time.time() - t0, extra={"return": ret})
    env.close()
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=0)
    ap.add_argument("--silenteval", action="store_true")
    ap.add_argument("--minimizeeval", action="store_true")
    ap.add_argument("--promote_on_eval", action="store_true")
    args = ap.parse_args()

    if args.eval > 0:
        # Fenster-/Headless-Konfiguration
        render_mode = None if args.silenteval else "human"
        if not args.silenteval:
            os.environ["SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"
            os.environ["SDL_VIDEO_WINDOW_POS"] = (
                "4000,4000" if args.minimizeeval else "200,80"
            )

        models, best, meta, cand = _paths()
        env, _ = _make_env()
        meta_d = _load_meta(meta)

        # Entscheide, welches Modell ausgewertet wird
        use_label = ""
        model_path = None
        if args.promote_on_eval and cand.exists():
            model_path = cand
            use_label = "CANDIDATE"
        elif best.exists():
            model_path = best
            use_label = "BEST"
        else:
            print(
                "[Eval] Kein BEST-Modell vorhanden – trainiere Kurzstart (5k Schritte)."
            )
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=5000)
            model.save(str(best))
            _save_meta(meta, 0.0)
            model_path = best
            use_label = "BEST"

        print(
            f"[Eval] Verwende {use_label}-Modell: {Path(model_path).name} (beste Prozent: {float(meta_d.get('best_percent',0.0)):.2f}%)"
        )
        model = PPO.load(str(model_path), env=env)

        pcts = []
        for i in range(args.eval):
            pct = eval_once(model, render_mode=render_mode)
            pcts.append(pct)
            print(f"[Eval] Episode {i+1}/{args.eval}: {pct:.2f}%")
        mean_pct = float(np.mean(pcts)) if pcts else 0.0
        print(f"[Eval] Mittelwert über {len(pcts)} Episoden: {mean_pct:.2f}%")

        # Optionales Befördern auf Basis der EVAL-Durchschnitte
        if args.promote_on_eval and str(model_path).endswith("lunar_candidate.zip"):
            old = float(meta_d.get("best_percent", 0.0))
            if mean_pct > old:
                os.replace(str(cand), str(best))
                _save_meta(meta, mean_pct)
                ts = int(time.time())
                PPO.load(str(best), env=_make_env()[0]).save(
                    str(models / f"lunar_best_{ts}")
                )
                print(
                    f"[Eval] Verbessert durch Eval: {mean_pct:.2f}% > {old:.2f}% → BEST aktualisiert."
                )
            else:
                print(
                    f"[Eval] Kandidat nicht besser: {mean_pct:.2f}% <= {old:.2f}% – kein Update."
                )

        env.close()


if __name__ == "__main__":
    main()
