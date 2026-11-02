// Runs docs/policy.js under Node with onnxruntime-web 1.22.0 and compares with the PyTorch reference (tests/js_ref.json).
// Usage: ORT_DIR=/path/with/node_modules node tests/node_check.js
const path = require("path"), fs = require("fs");
const ort = require(require.resolve("onnxruntime-web", { paths: [process.env.ORT_DIR || "."] }));
const P = require("../docs/policy.js"), M = JSON.parse(fs.readFileSync(path.join(__dirname, "../docs/model.json"))), R = JSON.parse(fs.readFileSync(path.join(__dirname, "js_ref.json")));
(async () => {
  ort.env.wasm.numThreads = 1;
  const sess = await ort.InferenceSession.create(path.join(__dirname, "../docs/dp_noise.onnx"));
  const out = await P.ddimSample(ort, sess, M, P.encObs(M, "dp_s0", R.state), 8, null, R.x_init);
  let worst = 0; out.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - R.out[i])); });
  const bc = await P.bcChunk(ort, await ort.InferenceSession.create(path.join(__dirname, "../docs/bc_chunk.onnx")), M, P.encObs(M, "bcc_s0", R.state));
  console.log("onnxruntime-web", ort.env.versions ? ort.env.versions.web : "?", "| samples", out.length / 32, "| max abs diff JS vs torch (normalised units, [-1,1] range):", worst.toExponential(2), "| bc chunk len", bc.length);
  if (!(worst < 0.05)) { console.error("FAIL"); process.exit(1); }
})();
