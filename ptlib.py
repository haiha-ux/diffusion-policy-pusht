"""Diffusion Policy and BC baselines for state-based PushT, in plain PyTorch.

Data layout (the public Diffusion Policy PushT demonstrations, low-dimensional part):
  state  (N, 5)  agent x, agent y, block x, block y, block angle   (workspace 0..512)
  action (N, 2)  absolute target position of the agent (PD-controlled)
  ends   (E,)    exclusive end index of every episode in the flat arrays
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

T_OBS, H_DP, HA = 2, 16, 8          # observation frames, prediction horizon, executed actions per chunk
NOISE_STEPS = 100
MAX_STEPS = 300


# ---------------------------------------------------------------- data
def load_data(path="data/pusht_lowdim.npz"):
    d = np.load(path)
    return d["state"].astype(np.float32), d["action"].astype(np.float32), d["ends"].astype(np.int64)


def split_episodes(ends, val_frac=0.1, seed=0):
    """Fixed episode-level split: returns (train_ep_ids, val_ep_ids)."""
    ids = np.random.RandomState(seed).permutation(len(ends))
    nv = max(1, int(round(len(ends) * val_frac)))
    return np.sort(ids[nv:]), np.sort(ids[:nv])


def frame_ranges(ends, ep_ids):
    starts = np.concatenate([[0], ends[:-1]])
    return starts[ep_ids], ends[ep_ids]


def chunk_indices(g, lo, hi, offset, H):
    """Flat indices of the H actions of the chunk for current frame g in an episode [lo, hi).
    Chunk starts at g + offset; indices outside the episode are clamped to its first/last frame (edge padding)."""
    k = np.arange(H)[None, :]
    return np.clip(g[:, None] + offset + k, lo[:, None], hi[:, None] - 1)


def obs_indices(g, lo):
    """The T_OBS most recent frames ending at g (older ones clamped to the episode start), oldest first."""
    return np.clip(g[:, None] - (T_OBS - 1) + np.arange(T_OBS)[None, :], lo[:, None], None)


def build_samples(ends, ep_ids):
    """One sample per frame of the chosen episodes: (g, lo, hi) arrays."""
    lo_e, hi_e = frame_ranges(ends, ep_ids)
    g = np.concatenate([np.arange(a, b) for a, b in zip(lo_e, hi_e)])
    lo = np.concatenate([np.full(b - a, a) for a, b in zip(lo_e, hi_e)])
    hi = np.concatenate([np.full(b - a, b) for a, b in zip(lo_e, hi_e)])
    return g, lo, hi


class Normalizer:
    """Per-dimension min/max scaling to [-1, 1] (as in the reference implementation)."""
    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
        self.scale = np.maximum(self.hi - self.lo, 1e-6)

    @classmethod
    def fit(cls, x):
        return cls(x.min(0), x.max(0))

    def enc(self, x):
        return 2 * (x - self.lo) / self.scale - 1

    def dec(self, x):
        return (x + 1) / 2 * self.scale + self.lo


# ---------------------------------------------------------------- noise schedule + samplers
def cosine_betas(n=NOISE_STEPS, s=0.008):
    ab = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return torch.tensor([min(1 - ab((i + 1) / n) / ab(i / n), 0.999) for i in range(n)], dtype=torch.float64)


class Schedule:
    def __init__(self, n=NOISE_STEPS):
        self.n = n
        self.betas = cosine_betas(n)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)

    def add_noise(self, x0, eps, t):
        ab = self.abar.to(x0.device, x0.dtype)[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


@torch.no_grad()
def ddpm_sample(eps_fn, shape, sch, gen=None, device="cpu"):
    """Ancestral sampling over all sch.n steps (fixed-small variance), x0 clipped to [-1, 1] at every step."""
    x = torch.randn(shape, generator=gen, device=device)
    ab, al, be = [v.to(device, torch.float32) for v in (sch.abar, sch.alphas, sch.betas)]
    for t in range(sch.n - 1, -1, -1):
        e = eps_fn(x, torch.full((shape[0],), t, device=device, dtype=torch.long))
        x0 = ((x - (1 - ab[t]).sqrt() * e) / ab[t].sqrt()).clamp(-1, 1)
        abp = ab[t - 1] if t > 0 else torch.ones((), device=device)
        mean = (abp.sqrt() * be[t] / (1 - ab[t])) * x0 + (al[t].sqrt() * (1 - abp) / (1 - ab[t])) * x
        if t > 0:
            var = ((1 - abp) / (1 - ab[t]) * be[t]).clamp(min=1e-20)
            x = mean + var.sqrt() * torch.randn(shape, generator=gen, device=device)
        else:
            x = mean
    return x


def ddim_timesteps(sch, steps):
    return list(range(sch.n - sch.n // steps, -1, -(sch.n // steps)))   # 100, 10 steps: 90, 80, ..., 0


@torch.no_grad()
def ddim_sample(eps_fn, shape, sch, steps=10, gen=None, device="cpu", x_init=None):
    """Deterministic DDIM (eta = 0), x0 clipped to [-1, 1]; the last step lands exactly on the x0 estimate."""
    x = torch.randn(shape, generator=gen, device=device) if x_init is None else x_init
    ab = sch.abar.to(device, torch.float32)
    ts = ddim_timesteps(sch, steps)
    for i, t in enumerate(ts):
        e = eps_fn(x, torch.full((shape[0],), t, device=device, dtype=torch.long))
        x0 = ((x - (1 - ab[t]).sqrt() * e) / ab[t].sqrt()).clamp(-1, 1)
        if i + 1 < len(ts):
            abp = ab[ts[i + 1]]
            x = abp.sqrt() * x0 + (1 - abp).sqrt() * e
        else:
            x = x0
    return x


# ---------------------------------------------------------------- models
class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)


class Block1D(nn.Module):
    def __init__(self, i, o, k=5, g=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.GroupNorm(g, o), nn.Mish())

    def forward(self, x):
        return self.net(x)


class ResBlock1D(nn.Module):
    """Two conv blocks with FiLM (scale and bias) conditioning, plus a residual path."""
    def __init__(self, i, o, cond):
        super().__init__()
        self.b1, self.b2 = Block1D(i, o), Block1D(o, o)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond, 2 * o))
        self.res = nn.Conv1d(i, o, 1) if i != o else nn.Identity()
        self.o = o

    def forward(self, x, c):
        h = self.b1(x)
        s, b = self.film(c).view(-1, 2, self.o, 1).unbind(1)
        return self.b2(s * h + b) + self.res(x)


class CondUNet1D(nn.Module):
    """Noise model eps(x_t, t | obs): 1D conditional U-Net over the action-chunk (time) axis."""
    def __init__(self, ad=2, od=5, dims=(64, 128, 256), t_dim=64):
        super().__init__()
        self.temb = nn.Sequential(SinEmb(t_dim), nn.Linear(t_dim, t_dim * 4), nn.Mish(), nn.Linear(t_dim * 4, t_dim))
        c = t_dim + od * T_OBS
        io = list(zip((ad,) + dims[:-1], dims))
        self.down = nn.ModuleList()
        for n, (i, o) in enumerate(io):
            last = n == len(io) - 1
            self.down.append(nn.ModuleList([ResBlock1D(i, o, c), ResBlock1D(o, o, c),
                                            nn.Identity() if last else nn.Conv1d(o, o, 3, 2, 1)]))
        self.mid = nn.ModuleList([ResBlock1D(dims[-1], dims[-1], c), ResBlock1D(dims[-1], dims[-1], c)])
        self.up = nn.ModuleList()
        for i, o in reversed(io[1:]):
            self.up.append(nn.ModuleList([ResBlock1D(o * 2, i, c), ResBlock1D(i, i, c), nn.ConvTranspose1d(i, i, 4, 2, 1)]))
        self.out = nn.Sequential(Block1D(dims[0], dims[0]), nn.Conv1d(dims[0], ad, 1))

    def forward(self, x, t, obs):
        """x (B, H, ad), t (B,), obs (B, T_OBS, od) all normalised -> predicted noise (B, H, ad)."""
        c = torch.cat([self.temb(t), obs.flatten(1)], -1)
        x = x.transpose(1, 2)
        skips = []
        for r1, r2, down in self.down:
            x = r2(r1(x, c), c)
            skips.append(x)
            x = down(x)
        for m in self.mid:
            x = m(x, c)
        for r1, r2, up in self.up:
            x = r2(r1(torch.cat([x, skips.pop()], 1), c), c)
            x = up(x)
        return self.out(x).transpose(1, 2)


class BCMLP(nn.Module):
    """Deterministic regressor obs -> action chunk of length H (H = 1 is plain single-step BC)."""
    def __init__(self, H, ad=2, od=5, w=512, depth=3):
        super().__init__()
        layers, i = [], od * T_OBS
        for _ in range(depth):
            layers += [nn.Linear(i, w), nn.ReLU()]
            i = w
        self.net = nn.Sequential(*layers, nn.Linear(w, H * ad))
        self.H, self.ad = H, ad

    def forward(self, obs):
        return self.net(obs.flatten(1)).view(-1, self.H, self.ad)


# ---------------------------------------------------------------- algorithms
ALGOS = {   # chunk offset relative to the current frame, chunk length, executed slice start, executed length
    "dp":  dict(offset=-(T_OBS - 1), H=H_DP, ex0=T_OBS - 1, ex=HA),
    "bcc": dict(offset=0, H=H_DP, ex0=0, ex=HA),
    "bc":  dict(offset=0, H=1, ex0=0, ex=1),
}


def make_model(algo):
    return CondUNet1D() if algo == "dp" else BCMLP(ALGOS[algo]["H"])


class Policy:
    """Maps a raw observation history (B, T_OBS, 5) to the raw actions to execute (B, ex, 2)."""
    def __init__(self, algo, model, nobs, nact, device, sampler="ddim", steps=10, seed=0):
        self.algo, self.model, self.nobs, self.nact, self.dev = algo, model.eval(), nobs, nact, device
        self.sampler, self.steps, self.sch = sampler, steps, Schedule()
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.cfg = ALGOS[algo]

    @torch.no_grad()
    def chunks(self, obs, n_samples=1):
        """Full normalised chunk(s): (B, n_samples, H, 2). Deterministic algos repeat their single output."""
        o = torch.as_tensor(self.nobs.enc(obs), dtype=torch.float32, device=self.dev)
        B = o.shape[0]
        if self.algo != "dp":
            return self.model(o)[:, None].expand(-1, n_samples, -1, -1)
        o = o[:, None].expand(-1, n_samples, -1, -1).reshape(B * n_samples, T_OBS, -1)
        eps = lambda x, t: self.model(x, t, o)
        shape = (B * n_samples, H_DP, 2)
        if self.sampler == "ddpm":
            a = ddpm_sample(eps, shape, self.sch, self.gen, self.dev)
        else:
            a = ddim_sample(eps, shape, self.sch, self.steps, self.gen, self.dev)
        return a.view(B, n_samples, H_DP, 2)

    def act(self, obs):
        a = self.chunks(obs)[:, 0, self.cfg["ex0"]:self.cfg["ex0"] + self.cfg["ex"]]
        return self.nact.dec(a.cpu().numpy())


# ---------------------------------------------------------------- simulator helpers
def set_state(env, s):
    """Put a gym-pusht env into state s = (agent xy, block xy, block angle) so that get_obs() returns s.
    The angle must be set BEFORE the position: pymunk rotates about the T's centre of gravity, so the other order
    (what the env's own reset does) moves the body origin by up to ~45 px. The demonstrations match this order
    (one-step block error 0.25 px median on moving steps; see results/replay_check.json)."""
    e = env.unwrapped
    e.agent.position = list(map(float, s[:2]))
    e.block.angle = float(s[4])
    e.block.position = list(map(float, s[2:4]))
    e.space.step(e.dt)
    return e.get_obs()


def reset_env(env, seed):
    """Initial state drawn like the reference PushT env (agent 50..450, block 100..400, angle uniform), seed -> state."""
    rs = np.random.RandomState(seed)
    s = np.array([rs.randint(50, 450), rs.randint(50, 450), rs.randint(100, 400), rs.randint(100, 400), rs.uniform(-np.pi, np.pi)])
    env.reset(seed=seed)
    return set_state(env, s)
