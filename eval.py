"""Evaluate trained models: closed-loop PushT rollouts + open-loop multimodality on held-out demos.
python eval.py [name ...]   (no names = every ckpt/*_s*.pt without a result yet); results/eval/<name>.json"""
import glob, json, os, sys, time, numpy as np, torch
import gymnasium as gym, gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
from ptlib import *

N_EPS, SEED0 = 50, 100000            # 50 initial states, the same for every model
K_SAMPLES, N_STATES = 16, 200        # multimodality: samples per state, held-out states


def load_policy(name, sampler="ddim", steps=10, dev="cpu"):
    c = torch.load(f"ckpt/{name}.pt", weights_only=False)
    m = make_model(c["algo"]); m.load_state_dict(c["model"]); m.to(dev)
    return Policy(c["algo"], m, Normalizer(*c["nobs"]), Normalizer(*c["nact"]), dev, sampler, steps, seed=c["seed"] + 1000), c


def rollout(pol, n=N_EPS, seed0=SEED0, max_steps=MAX_STEPS, record_idx=()):
    """n episodes in lockstep (one batched policy call per chunk). Returns per-episode max coverage, success, steps, frames."""
    envs = [gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array" if i in record_idx else None) for i in range(n)]
    hist = []
    for i, e in enumerate(envs):
        o = reset_env(e, seed0 + i)
        hist.append([o, o])
    cov, done, steps = np.zeros(n), np.zeros(n, bool), np.zeros(n, int)
    frames = {i: [envs[i].render()] for i in record_idx}
    t = 0
    while t < max_steps and not done.all():
        acts = pol.act(np.array(hist, np.float32))                     # (n, ex, 2)
        for k in range(acts.shape[1]):
            if t >= max_steps:
                break
            for i, e in enumerate(envs):
                if done[i]:
                    continue
                o, r, term, _, info = e.step(np.clip(acts[i, k], 0, 512).astype(np.float32))
                hist[i] = [hist[i][1], o]
                cov[i] = max(cov[i], info["coverage"]); steps[i] += 1
                if i in frames:
                    frames[i].append(e.render())
                if term:
                    done[i] = True
            t += 1
            if done.all():
                break
    for e in envs:
        e.close()
    return dict(coverage=cov, success=done, steps=steps), frames


def multimodality(pol, c, nobs_state_idx=None):
    """On held-out demo states: spread of K sampled chunks, best-of-K error to the demo chunk, bimodal fraction."""
    S, A, ends = load_data()
    va_ep = np.array(c["val_eps"])
    g, lo, hi = build_samples(ends, va_ep)
    sel = np.random.RandomState(0).choice(len(g), N_STATES, replace=False)
    g, lo, hi = g[sel], lo[sel], hi[sel]
    obs = S[obs_indices(g, lo)]
    cfg = pol.cfg
    demo = A[chunk_indices(g, lo, hi, cfg["offset"], cfg["H"])][:, cfg["ex0"]:cfg["ex0"] + cfg["ex"]]       # (N, ex, 2) raw
    pol_samples = pol.chunks(obs, K_SAMPLES)[:, :, cfg["ex0"]:cfg["ex0"] + cfg["ex"]]
    s = pol.nact.dec(pol_samples.cpu().numpy())                                                             # (N, K, ex, 2) raw px
    err = np.linalg.norm(s - demo[:, None], axis=-1).mean(-1)                                               # (N, K) mean px error per step
    end = s[:, :, -1]                                                                                       # (N, K, 2) last executed target
    spread = end.std(1).mean(-1)                                                                            # px
    bimodal = np.zeros(len(g), bool)
    for i in range(len(g)):                                                                                 # 2-means on chunk endpoints
        x = end[i]; c0 = x[np.argmax(np.linalg.norm(x - x.mean(0), axis=1))]; c1 = x[np.argmax(np.linalg.norm(x - c0, axis=1))]
        for _ in range(10):
            lab = np.linalg.norm(x - c0, axis=1) > np.linalg.norm(x - c1, axis=1)
            if lab.all() or (~lab).all():
                break
            c0, c1 = x[~lab].mean(0), x[lab].mean(0)
        frac = lab.mean()
        bimodal[i] = 0.15 <= frac <= 0.85 and np.linalg.norm(c0 - c1) > 40
    return dict(endpoint_spread_px=float(spread.mean()), single_sample_err_px=float(err.mean()),
                best_of_k_err_px=float(err.min(1).mean()), bimodal_frac=float(bimodal.mean()), k=K_SAMPLES, n_states=N_STATES)


def evaluate(name, sampler, steps):
    pol, c = load_policy(name, sampler, steps)
    t0 = time.time()
    r, _ = rollout(pol)
    out = dict(name=name, algo=c["algo"], seed=c["seed"], sampler=sampler, steps=steps if sampler == "ddim" else NOISE_STEPS,
               coverage=float(r["coverage"].mean()), success=float(r["success"].mean()),
               score=float(np.minimum(r["coverage"] / 0.95, 1).mean()),
               episode_coverage=[round(float(x), 4) for x in r["coverage"]], rollout_sec=round(time.time() - t0))
    if c["algo"] == "dp" or sampler == "ddim":
        out["multimodality"] = multimodality(pol, c)
    return out


if __name__ == "__main__":
    torch.set_num_threads(4)
    os.makedirs("results/eval", exist_ok=True)
    names = sys.argv[1:] or sorted(os.path.basename(p)[:-3] for p in glob.glob("ckpt/*_s[0-9]*.pt") if "pilot" not in p)
    for n in names:
        variants = [("ddim", 10), ("ddpm", 100)] if n.startswith("dp") else [("ddim", 10)]
        for sampler, steps in variants:
            f = f"results/eval/{n}_{sampler}{steps if sampler == 'ddim' else ''}.json"
            if os.path.exists(f):
                continue
            r = evaluate(n, sampler, steps)
            json.dump(r, open(f, "w"))
            print(n, sampler, {k: r[k] for k in ("coverage", "success", "rollout_sec")}, r.get("multimodality"), flush=True)
