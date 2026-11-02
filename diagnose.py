"""Is the BC result an input-path bug? Compare what the policy sees in rollouts with what it saw in training."""
import json, numpy as np, torch, gymnasium as gym, gym_pusht  # noqa: F401
from ptlib import *
from eval import load_policy
S, A, ends = load_data(); out = {}
for name in ("bc_s0", "bcc_s0", "dp_s0"):
    pol, c = load_policy(name)
    tr = np.concatenate([S[l:h] for l, h in zip(*frame_ranges(ends, np.array(c["train_eps"])))])
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    obs, acts = [], []
    for i in range(10):
        o = reset_env(env, 100000 + i); h = [o, o]
        for _ in range(30):
            a = pol.act(np.array([h], np.float32))[0, 0]; obs.append(h[1]); acts.append(a)
            o, *_ = env.step(np.clip(a, 0, 512).astype(np.float32)); h = [h[1], o]
    obs = np.array(obs); z = pol.nobs.enc(obs)
    out[name] = dict(frac_obs_outside_train_range=float(((z < -1) | (z > 1)).any(1).mean()),
                     rollout_obs_mean=obs.mean(0).round(1).tolist(), train_obs_mean=tr.mean(0).round(1).tolist(),
                     rollout_action_mean=np.mean(acts, 0).round(1).tolist(), train_action_mean=A.mean(0).round(1).tolist(),
                     action_dtype_shape_ok=bool(np.array(acts).shape[1] == 2))
json.dump(out, open("results/diagnose.json", "w"), indent=1); print(json.dumps(out, indent=1))
