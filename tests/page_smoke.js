// Runs the inline script of docs/index.html under Node with stubbed DOM/canvas and real onnxruntime-web 1.22.0 models:
// checks that loading, sampling and drawing complete without exceptions. Does NOT check pixels. ORT_DIR as in node_check.js.
const fs = require("fs"), path = require("path"), vm = require("vm");
const ort = require(require.resolve("onnxruntime-web", { paths: [process.env.ORT_DIR || "."] })); ort.env.wasm.numThreads = 1;
const docs = path.join(__dirname, "../docs"), html = fs.readFileSync(path.join(docs, "index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[0].slice(8, -9);
const calls = { lines: 0 };
const el = () => new Proxy(function () {}, { get: (t, k) => k === "getContext" ? () => ctx : k === "getBoundingClientRect" ? () => ({ left: 0, top: 0, width: 512, height: 512 }) : t[k] ?? (t[k] = 0), set: (t, k, v) => (t[k] = v, true), apply: () => undefined });
const ctx = new Proxy({}, { get: (t, k) => k === "lineTo" ? () => calls.lines++ : () => {}, set: () => true });
const elements = {}; const document = { getElementById: id => elements[id] ??= el() };
const fetchLocal = async u => { const b = fs.readFileSync(path.join(docs, u)); return { json: async () => JSON.parse(b), arrayBuffer: async () => b }; };
const P = require("../docs/policy.js"); Object.assign(globalThis, P, { document, fetch: fetchLocal });
const realCreate = ort.InferenceSession.create.bind(ort.InferenceSession);
globalThis.ort = { ...ort, env: { wasm: {}, versions: ort.env.versions }, Tensor: ort.Tensor, InferenceSession: { create: f => realCreate(fs.readFileSync(path.join(docs, f))) } };
(0, eval)(script.replace(/^const /gm, "var ").replace(/^let /gm, "var "));
setTimeout(() => { console.log("page smoke: status:", String(elements["st"].innerHTML).slice(0, 110), "| polyline segments drawn:", calls.lines); process.exit(calls.lines > 100 ? 0 : 1); }, 8000);
