"""Reference for tests/node_check.js: DDIM-10 with the PyTorch fp32 model from fixed noise and a fixed state."""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptlib import *
from eval import load_policy
pol, c = load_policy("dp_s0")
state = np.array([300.0, 200.0, 250.0, 280.0, 2.0], np.float32)
o = torch.tensor(pol.nobs.enc(np.stack([state, state])), dtype=torch.float32)[None].expand(8, -1, -1)
xi = torch.randn(8, H_DP, 2, generator=torch.Generator().manual_seed(1))
out = ddim_sample(lambda x, t: pol.model(x, t, o), xi.shape, pol.sch, 10, x_init=xi.clone())
json.dump(dict(state=state.tolist(), x_init=xi.flatten().tolist(), out=out.flatten().tolist()), open("tests/js_ref.json", "w"))
