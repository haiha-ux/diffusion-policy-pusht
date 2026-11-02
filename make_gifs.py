"""Pre-rendered closed-loop episodes (fixed init seed 100000, first of the evaluation set, not selected) -> docs/rollouts/*.gif."""
import json, numpy as np
from PIL import Image
from eval import load_policy, rollout
man = []
for name in ("dp_s0", "bcc_s0", "bc_s0"):
    pol, _ = load_policy(name)
    r, fr = rollout(pol, n=1, record_idx=(0,))
    f = [Image.fromarray(x).resize((300, 300)) for x in fr[0][::2]]
    f[0].save(f"docs/rollouts/{name}.gif", save_all=True, append_images=f[1:], duration=100, loop=0, optimize=True)
    man.append(dict(name=name, coverage=float(r["coverage"][0]), success=bool(r["success"][0]), steps=int(r["steps"][0])))
    print(man[-1])
json.dump(man, open("docs/rollouts/manifest.json", "w"))
