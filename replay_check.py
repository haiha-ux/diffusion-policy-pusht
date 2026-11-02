"""Sanity check: does the pip sim reproduce the demonstrations? Replay the recorded actions open-loop from each
episode's first state and compare the final block pose / coverage with the recording."""
import json, numpy as np, gymnasium as gym, gym_pusht  # noqa: F401
from ptlib import load_data, set_state
S, A, ends = load_data()
starts = np.concatenate([[0], ends[:-1]])
env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
rows = []
for e in range(0, len(ends), 5):
    lo, hi = starts[e], ends[e]
    env.reset(seed=0); set_state(env, S[lo])
    best, drift = 0.0, []
    for k in range(hi - lo):
        o, _, term, _, info = env.step(A[lo + k])
        best = max(best, info["coverage"])
        if lo + k + 1 < hi:
            drift.append(np.linalg.norm(o[2:4] - S[lo + k + 1, 2:4]))
        if term:
            break
    rows.append(dict(ep=int(e), max_coverage=float(best), success=bool(best > 0.95), mean_block_pos_err_px=float(np.mean(drift)),
                     final_block_pos_err_px=float(drift[-1]) if drift else 0.0))
out = dict(n_episodes=len(rows), mean_max_coverage=float(np.mean([r["max_coverage"] for r in rows])),
           success_rate=float(np.mean([r["success"] for r in rows])),
           median_mean_block_err_px=float(np.median([r["mean_block_pos_err_px"] for r in rows])), episodes=rows)
import os; os.makedirs("results", exist_ok=True)
json.dump(out, open("results/replay_check.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "episodes"})
