// DDIM sampling of action chunks with the exported noise model; same maths as ddim_sample in ptlib.py.
// ort is passed in so the same file runs in the browser (onnxruntime-web) and under Node (tests/node_check.js).
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(rng) { const u = 1 - rng(), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function encObs(M, key, s) {            // raw state (5) -> normalised, two identical frames (agent assumed at rest)
  const [lo, hi] = M[key].obs, f = new Float32Array(2 * 5);
  for (let k = 0; k < 5; k++) { const z = 2 * (s[k] - lo[k]) / Math.max(hi[k] - lo[k], 1e-6) - 1; f[k] = z; f[5 + k] = z; }
  return f;
}
function decAct(M, key, z) { const [lo, hi] = M[key].act; return [(z[0] + 1) / 2 * (hi[0] - lo[0]) + lo[0], (z[1] + 1) / 2 * (hi[1] - lo[1]) + lo[1]]; }
async function ddimSample(ort, sess, M, obs, n, rng, xInit) {   // -> Float32Array (n, H, 2), normalised
  const H = M.H, size = n * H * 2, ab = M.abar, ts = M.ddim_timesteps;
  let x = xInit ? Float32Array.from(xInit) : Float32Array.from({ length: size }, () => gauss(rng));
  const ob = new Float32Array(n * obs.length); for (let i = 0; i < n; i++) ob.set(obs, i * obs.length);
  for (let i = 0; i < ts.length; i++) {
    const t = ts[i], tt = new BigInt64Array(n).fill(BigInt(t));
    const r = await sess.run({ x: new ort.Tensor("float32", x, [n, H, 2]), t: new ort.Tensor("int64", tt, [n]), obs: new ort.Tensor("float32", ob, [n, M.T_OBS, 5]) });
    const e = r.out.data, nx = new Float32Array(size), at = ab[t], abp = i + 1 < ts.length ? ab[ts[i + 1]] : null;
    for (let j = 0; j < size; j++) {
      const x0 = Math.max(-1, Math.min(1, (x[j] - Math.sqrt(1 - at) * e[j]) / Math.sqrt(at)));
      nx[j] = abp === null ? x0 : Math.sqrt(abp) * x0 + Math.sqrt(1 - abp) * e[j];
    }
    x = nx;
  }
  return x;
}
async function bcChunk(ort, sess, M, obs) {
  const r = await sess.run({ obs: new ort.Tensor("float32", obs, [1, M.T_OBS, 5]) });
  return Float32Array.from(r.out.data);
}
if (typeof module !== "undefined") module.exports = { ddimSample, bcChunk, encObs, decAct, mulberry32 };
