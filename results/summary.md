| Method | coverage (max over episode) | paper-style score | success (cov > 0.95) | success per seed (of 50) |
|---|---|---|---|---|
| Diffusion Policy, DDIM-10 | 0.866 +- 0.039 (0.900, 0.874, 0.823) | 0.907 +- 0.041 (0.942, 0.915, 0.862) | 0.55 +- 0.08 | 27, 32, 24 |
| Diffusion Policy, DDPM-100 | 0.854 +- 0.038 (0.897, 0.827, 0.839) | 0.895 +- 0.040 (0.941, 0.867, 0.877) | 0.50 +- 0.09 | 24, 21, 30 |
| BC with action chunks (16, execute 8) | 0.640 +- 0.063 (0.572, 0.652, 0.696) | 0.671 +- 0.065 (0.600, 0.683, 0.729) | 0.25 +- 0.05 | 10, 14, 14 |
| BC (single step MLP) | 0.544 +- 0.079 (0.474, 0.529, 0.629) | 0.570 +- 0.082 (0.497, 0.555, 0.659) | 0.27 +- 0.08 | 10, 13, 18 |

Open-loop, 200 held-out demo states, 16 samples each (executed slice of the chunk):

| Method | endpoint spread (px) | single-sample error (px) | best-of-16 error (px) | states with 2 modes |
|---|---|---|---|---|
| Diffusion Policy, DDIM-10 | 3.19 +- 0.30 | 23.7 +- 0.4 | 19.3 +- 0.8 | 0.012 +- 0.003 |
| Diffusion Policy, DDPM-100 | 2.93 +- 0.27 | 23.8 +- 0.3 | 19.7 +- 0.4 | 0.002 +- 0.003 |
| BC with action chunks (16, execute 8) | 0.00 +- 0.00 | 30.5 +- 0.8 | 30.5 +- 0.8 | 0.000 +- 0.000 |
| BC (single step MLP) | 0.00 +- 0.00 | 10.5 +- 0.1 | 10.5 +- 0.1 | 0.000 +- 0.000 |
