import time, argparse, json, numpy as np, gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import push_current, log_result

GAME_NAME = "CartPole"


def _paths():
    base = Path(__file__).resolve().parents[1]
    models = base / "models"
    models.mkdir(exist_ok=True)
    best = models / "cartpole_best.zip"
    meta = models / "cartpole_best_meta.json"
    cand = models / "cartpole_candidate.zip"
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


def _eval_percent_cartpole(model, n_episodes=5):
    env = gym.make("CartPole-v1")
    x_thr = float(env.unwrapped.x_threshold)
    th_thr = float(env.unwrapped.theta_threshold_radians)
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        qs = []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(a))
            done = terminated or truncated
            q = 1.0 - max(abs(obs[0]) / x_thr, abs(obs[2]) / th_thr)
            qs.append(float(np.clip(q, 0.0, 1.0)))
        scores.append(float(np.mean(qs) * 100.0) if qs else 0.0)
    env.close()
    return float(np.mean(scores))


def train(timesteps, log, progress=None):
    models, best, meta, cand = _paths()
    env = gym.make("CartPole-v1")
    if best.exists():
        model = PPO.load(str(best), env=env)
        log("Weitertrainieren vom BEST-Modell...")
    else:
        model = PPO("MlpPolicy", env, verbose=1)
        log("Neues Modell...")

    class _ProgressCB(BaseCallback):
        def __init__(self, total_steps, cb_fn):
            super().__init__()
            self.total = int(max(1, total_steps))
            self.cb_fn = cb_fn
            self._last = -1

        def _on_training_start(self) -> None:
            if self.cb_fn:
                try:
                    self.cb_fn(0)
                except Exception:
                    pass
            return True

        def _on_step(self) -> bool:
            if not self.cb_fn:
                return True
            pct = int(min(100, (100 * self.num_timesteps) // self.total))
            if pct != self._last:
                try:
                    self.cb_fn(pct)
                except Exception:
                    pass
                self._last = pct
            return True

    model.learn(
        total_timesteps=int(timesteps),
        callback=_ProgressCB(int(timesteps), progress),
    )
    if progress:
        try:
            progress(100)
        except Exception:
            pass
    model.save(str(cand))
    env.close()
    new_score = _eval_percent_cartpole(PPO.load(str(cand), env=gym.make("CartPole-v1")))
    old_score = float(_load_meta(meta).get("best_percent", 0.0))
    if new_score > old_score:
        os.replace(str(cand), str(best))
        _save_meta(meta, new_score)
        ts = int(time.time())
        PPO.load(str(best), env=gym.make("CartPole-v1")).save(
            str(models / f"cartpole_best_{ts}")
        )
        log(f"Verbessert: {new_score:.2f}% > {old_score:.2f}% → BEST aktualisiert.")
    else:
        try:
            os.remove(str(cand))
        except:
            pass
        log(f"Nicht besser: {new_score:.2f}% <= {old_score:.2f}% → verwerfe.")


def eval_once(model):
    env = gym.make("CartPole-v1", render_mode="human")
    x_thr = float(env.unwrapped.x_threshold)
    th_thr = float(env.unwrapped.theta_threshold_radians)
    obs, _ = env.reset()
    done = False
    steps = 0
    qs = []
    t0 = time.time()
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(a))
        done = terminated or truncated
        q = 1.0 - max(abs(obs[0]) / x_thr, abs(obs[2]) / th_thr)
        qs.append(float(np.clip(q, 0.0, 1.0)))
        steps += 1
        push_current(GAME_NAME, float(np.mean(qs) * 100.0))
    pct = float(np.mean(qs) * 100.0) if qs else 0.0
    log_result(GAME_NAME, 1, pct, steps, time.time() - t0)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=0)
    args = ap.parse_args()
    if args.eval > 0:
        _, best, meta, _ = _paths()
        env = gym.make("CartPole-v1")
        if best.exists():
            model = PPO.load(str(best), env=env)
        else:
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=5000)
            model.save(str(best))
        for _ in range(args.eval):
            eval_once(model)
        env.close()


if __name__ == "__main__":
    main()
