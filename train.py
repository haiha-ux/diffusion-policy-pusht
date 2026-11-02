"""Train one model: python train.py --algo dp|bc|bcc --seed N [--steps N]."""
import argparse, copy, csv, json, os, time, numpy as np, torch, torch.nn.functional as F
from ptlib import *

p = argparse.ArgumentParser()
p.add_argument("--algo", required=True, choices=list(ALGOS))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--tag", default="")
a = p.parse_args()
torch.set_num_threads(4)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.3)
torch.manual_seed(a.seed); np.random.seed(a.seed)
lr = a.lr or (3e-4 if a.algo == "dp" else 1e-3)
cfg = ALGOS[a.algo]

S, A, ends = load_data()
tr_ep, va_ep = split_episodes(ends)
tr_lo, tr_hi = frame_ranges(ends, tr_ep)
nobs = Normalizer.fit(np.concatenate([S[l:h] for l, h in zip(tr_lo, tr_hi)]))
nact = Normalizer.fit(np.concatenate([A[l:h] for l, h in zip(tr_lo, tr_hi)]))
Sn, An = torch.tensor(nobs.enc(S), device=dev), torch.tensor(nact.enc(A), device=dev)


def batch_tensors(g, lo, hi):
    return Sn[obs_indices(g, lo)], An[chunk_indices(g, lo, hi, cfg["offset"], cfg["H"])]


tr = build_samples(ends, tr_ep)
va = build_samples(ends, va_ep)
sch = Schedule()
ab_dev = sch.abar.to(dev, torch.float32)
model = make_model(a.algo).to(dev)
ema = copy.deepcopy(model).requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
warm = 500
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
nparam = sum(p.numel() for p in model.parameters())


def loss_fn(m, o, act, g=None):
    if a.algo != "dp":
        return F.mse_loss(m(o), act)
    t = torch.randint(0, NOISE_STEPS, (act.shape[0],), device=dev, generator=g)
    eps = torch.randn(act.shape, device=dev, generator=g)
    x = sch.add_noise(act, eps, t)
    return F.mse_loss(m(x, t, o), eps)


@torch.no_grad()
def val_loss(m):
    g = torch.Generator(device=dev).manual_seed(123)      # same noise/timesteps every call
    idx = np.random.RandomState(0).choice(len(va[0]), min(4096, len(va[0])), replace=False)
    o, act = batch_tensors(va[0][idx], va[1][idx], va[2][idx])
    return float(loss_fn(m, o, act, g))


name = f"{a.algo}{a.tag}_s{a.seed}"
os.makedirs("ckpt", exist_ok=True); os.makedirs("logs", exist_ok=True)
rng = np.random.RandomState(a.seed)
rows, t0 = [], time.time()
for step in range(1, a.steps + 1):
    i = rng.randint(0, len(tr[0]), a.batch)
    o, act = batch_tensors(tr[0][i], tr[1][i], tr[2][i])
    loss = loss_fn(model, o, act)
    opt.zero_grad(set_to_none=True); loss.backward()
    opt.step(); sched.step()
    d = min(0.999, (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - d)
    if step % 1000 == 0 or step == a.steps:
        rows.append(dict(step=step, train_loss=float(loss), val_loss_ema=val_loss(ema), sec=round(time.time() - t0)))
        print(name, rows[-1], flush=True)
torch.save(dict(model=ema.state_dict(), algo=a.algo, nobs=(nobs.lo, nobs.hi), nact=(nact.lo, nact.hi), seed=a.seed,
                steps=a.steps, lr=lr, params=nparam, train_eps=tr_ep.tolist(), val_eps=va_ep.tolist()), f"ckpt/{name}.pt")
with open(f"logs/{name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(dict(name=name, sec=round(time.time() - t0), params=nparam, lr=lr, steps=a.steps, batch=a.batch), open(f"logs/{name}.time.json", "w"))
