"""results/eval/*.json -> results/summary.md (+ results/summary.json, results/coverage.png)."""
import glob, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROWS = [("Diffusion Policy, DDIM-10", "dp", "ddim"), ("Diffusion Policy, DDPM-100", "dp", "ddpm"),
        ("BC with action chunks (16, execute 8)", "bcc", "ddim"), ("BC (single step MLP)", "bc", "ddim")]
R = [json.load(open(f)) for f in sorted(glob.glob("results/eval/*.json"))]
ms = lambda v, d=3: f"{np.mean(v):.{d}f} +- {np.std(v, ddof=1):.{d}f}"
per = lambda v, d=3: "(" + ", ".join(f"{x:.{d}f}" for x in v) + ")"
out, summ, fig = ["| Method | coverage (max over episode) | paper-style score | success (cov > 0.95) | success per seed (of 50) |", "|---|---|---|---|---|"], {}, []
mm = ["| Method | endpoint spread (px) | single-sample error (px) | best-of-16 error (px) | states with 2 modes |", "|---|---|---|---|---|"]
for lab, algo, smp in ROWS:
    rs = sorted([r for r in R if r["algo"] == algo and r["sampler"] == smp], key=lambda r: r["seed"])
    cov, sc, su = [r["coverage"] for r in rs], [r["score"] for r in rs], [r["success"] for r in rs]
    out.append(f"| {lab} | {ms(cov)} {per(cov)} | {ms(sc)} {per(sc)} | {ms(su, 2)} | {', '.join(str(round(s * 50)) for s in su)} |")
    m = [r["multimodality"] for r in rs]
    mm.append(f"| {lab} | {ms([x['endpoint_spread_px'] for x in m], 2)} | {ms([x['single_sample_err_px'] for x in m], 1)} | {ms([x['best_of_k_err_px'] for x in m], 1)} | {ms([x['bimodal_frac'] for x in m], 3)} |")
    summ[lab] = dict(seeds=[r["seed"] for r in rs], coverage=cov, score=sc, success=su, n_episodes=len(rs[0]["episode_coverage"]))
    fig.append((lab, cov))
open("results/summary.md", "w").write("\n".join(out) + "\n\nOpen-loop, 200 held-out demo states, 16 samples each (executed slice of the chunk):\n\n" + "\n".join(mm) + "\n")
json.dump(summ, open("results/summary.json", "w"), indent=1)
plt.figure(figsize=(7, 3.2))
for i, (lab, cov) in enumerate(fig):
    plt.scatter([i] * len(cov), cov, c="k", zorder=3); plt.hlines(np.mean(cov), i - .25, i + .25, colors="C3")
plt.xticks(range(len(fig)), [l.replace(", ", "\n").replace(" (", "\n(") for l, _ in fig], fontsize=7); plt.ylabel("mean max coverage (50 episodes)")
plt.axhline(0.9, ls=":", c="gray"); plt.text(3.3, 0.905, "demos replayed: 0.90", fontsize=7, ha="right"); plt.ylim(0.4, 1); plt.tight_layout(); plt.savefig("results/coverage.png", dpi=130)
print(open("results/summary.md").read())
