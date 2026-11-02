import os, sys, numpy as np, torch, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptlib import *


def test_noise_schedule():
    s = Schedule()
    assert len(s.betas) == NOISE_STEPS and float(s.betas.max()) <= 0.999 and float(s.betas.min()) > 0
    assert float(s.abar[0]) > 0.999 and float(s.abar[-1]) < 1e-3          # starts clean, ends as pure noise
    assert bool((s.abar[1:] < s.abar[:-1]).all())


def test_add_noise_matches_formula():
    s = Schedule(); x0, e = torch.randn(4, 16, 2), torch.randn(4, 16, 2); t = torch.tensor([0, 10, 50, 99])
    xt = s.add_noise(x0, e, t)
    for i, ti in enumerate(t):
        ab = float(s.abar[ti]); assert torch.allclose(xt[i], ab ** 0.5 * x0[i] + (1 - ab) ** 0.5 * e[i], atol=1e-5)


def test_sampling_shapes_and_range():
    net = CondUNet1D(); net.eval(); obs = torch.randn(6, T_OBS, 5)
    eps = lambda x, t: net(x, t, obs)
    for out in (ddim_sample(eps, (6, H_DP, 2), Schedule(), 5), ddpm_sample(eps, (6, H_DP, 2), Schedule())):
        assert out.shape == (6, H_DP, 2) and float(out.abs().max()) <= 1.0 + 1e-6


def test_ddim_recovers_x0_with_oracle_noise_model():
    """If the noise model is exact for a fixed x0, deterministic DDIM must return x0 for any starting noise."""
    s = Schedule(); x0 = torch.rand(3, H_DP, 2) * 1.6 - 0.8
    ab = s.abar.float()
    oracle = lambda x, t: (x - ab[t].sqrt().view(-1, 1, 1) * x0) / (1 - ab[t]).sqrt().view(-1, 1, 1)
    for steps in (10, 100):
        assert torch.allclose(ddim_sample(oracle, x0.shape, s, steps), x0, atol=1e-4)
    assert ddim_timesteps(s, 10) == list(range(90, -1, -10))


def test_unet_and_bc_output_shapes():
    assert CondUNet1D()(torch.randn(3, H_DP, 2), torch.tensor([1, 2, 3]), torch.randn(3, T_OBS, 5)).shape == (3, H_DP, 2)
    assert BCMLP(H_DP)(torch.randn(3, T_OBS, 5)).shape == (3, H_DP, 2) and BCMLP(1)(torch.randn(3, T_OBS, 5)).shape == (3, 1, 2)


def test_normalizer_round_trip():
    x = np.random.RandomState(0).rand(100, 5).astype(np.float32) * [512, 512, 400, 400, 6.28]
    n = Normalizer.fit(x); z = n.enc(x)
    assert z.min() >= -1 - 1e-6 and z.max() <= 1 + 1e-6 and np.allclose(n.dec(z), x, atol=1e-3)


def test_chunk_alignment():
    """Action a_g is executed at frame g. Diffusion chunk starts T_OBS-1 frames earlier and its executed slice must be a_g.. ;
    BC chunk starts at g. Edge frames are padded with the first/last action of the episode."""
    ends = np.array([10, 25]); A = np.arange(25, dtype=np.float32)       # action value == its frame index
    g, lo, hi = build_samples(ends, np.array([0, 1]))
    assert len(g) == 25
    for algo in ("dp", "bcc", "bc"):
        c = ALGOS[algo]; idx = chunk_indices(g, lo, hi, c["offset"], c["H"]); ex = A[idx][:, c["ex0"]:c["ex0"] + c["ex"]]
        mid = (g >= 1) & (g <= 2) | (g >= 11) & (g <= 17)                # chunk fully inside the episode
        assert (ex[mid] == g[mid, None] + np.arange(c["ex"])).all(), algo
        assert (idx >= lo[:, None]).all() and (idx < hi[:, None]).all()   # never leaks into the neighbouring episode
    i = np.where(g == 9)[0][0]; assert A[chunk_indices(g, lo, hi, 0, 4)][i].tolist() == [9, 9, 9, 9]   # padded with last action


def test_obs_window_is_oldest_first_and_clamped():
    ends = np.array([10, 25]); g, lo, hi = build_samples(ends, np.array([0, 1])); S = np.arange(25)
    o = S[obs_indices(g, lo)]
    assert (o[:, -1] == g).all() and (o[g == 12, 0] == 11).all() and (o[g == 10, 0] == 10).all() and (o[g == 0, 0] == 0).all()


def test_episode_split_disjoint():
    tr, va = split_episodes(np.arange(1, 207) * 100)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == 206 and len(va) == 21


def test_sim_state_round_trip():
    gym = pytest.importorskip("gymnasium"); pytest.importorskip("gym_pusht")
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    s = np.array([222.0, 97.0, 222.99, 381.6, 3.008]); env.reset(seed=0); o = set_state(env, s)
    assert np.allclose(o, s, atol=1e-3)
    o2 = reset_env(env, 100000); o3 = reset_env(env, 100000)
    assert np.allclose(o2, o3) and 0 <= o2[4] < 2 * np.pi


def test_policy_act_shapes():
    nobs = Normalizer(np.zeros(5), np.ones(5) * 512); nact = Normalizer(np.zeros(2), np.ones(2) * 512)
    for algo, n in (("dp", HA), ("bcc", HA), ("bc", 1)):
        pol = Policy(algo, make_model(algo), nobs, nact, "cpu")
        assert pol.act(np.random.rand(3, T_OBS, 5).astype(np.float32) * 512).shape == (3, n, 2)
