"""Export the seed-0 diffusion noise model and the seed-0 chunked BC model to ONNX (dynamic batch) and check parity."""
import json, numpy as np, torch, onnxruntime as ort
from ptlib import *
from eval import load_policy
sch = Schedule()
meta = dict(T_OBS=T_OBS, H=H_DP, HA=HA, ddim_steps=10, ddim_timesteps=ddim_timesteps(sch, 10), abar=sch.abar.tolist())
for name, f in (("dp_s0", "dp_noise"), ("bcc_s0", "bc_chunk")):
    pol, c = load_policy(name)
    net = pol.model.eval()
    if name.startswith("dp"):
        ex = (torch.randn(2, H_DP, 2), torch.tensor([5, 50]), torch.randn(2, T_OBS, 5)); names = ["x", "t", "obs"]
    else:
        ex = (torch.randn(2, T_OBS, 5),); names = ["obs"]
    torch.onnx.export(net, ex, f"docs/{f}.onnx", dynamo=False, opset_version=17, input_names=names, output_names=["out"],
                      dynamic_axes={n: {0: "b"} for n in names + ["out"]})
    sess = ort.InferenceSession(f"docs/{f}.onnx"); worst = 0
    for b in (1, 8):
        x = tuple(torch.randn(b, H_DP, 2) if n == "x" else torch.randint(0, 100, (b,)) if n == "t" else torch.randn(b, T_OBS, 5) for n in names)
        ref = net(*x).detach().numpy()
        worst = max(worst, float(np.abs(ref - sess.run(None, {n: v.numpy() for n, v in zip(names, x)})[0]).max()))
    print(name, "max abs diff torch vs onnxruntime:", worst)
    meta[name] = dict(obs=[c["nobs"][0].tolist(), c["nobs"][1].tolist()], act=[c["nact"][0].tolist(), c["nact"][1].tolist()], parity=worst)
json.dump(meta, open("docs/model.json", "w"))

# The fp32 U-Net is 16.5 MB (repo limit 10 MB per file): store weights as fp16 and cast back to fp32 inside the graph.
import onnx
from onnx import numpy_helper, TensorProto, helper
m = onnx.load("docs/dp_noise.onnx"); casts = []
for init in list(m.graph.initializer):
    if init.data_type == TensorProto.FLOAT and init.dims and int(np.prod(init.dims)) > 16:
        a = numpy_helper.to_array(init); h = numpy_helper.from_array(a.astype(np.float16), init.name + "_h16")
        m.graph.initializer.remove(init); m.graph.initializer.append(h)
        casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT))
m.graph.node.insert(0, *casts) if len(casts) == 1 else [m.graph.node.insert(i, c) for i, c in enumerate(casts)]
onnx.save(m, "docs/dp_noise.onnx")
net = load_policy("dp_s0")[0].model.eval(); sess = ort.InferenceSession("docs/dp_noise.onnx"); worst = 0
for b in (1, 8):
    x = (torch.randn(b, H_DP, 2), torch.randint(0, 100, (b,)), torch.randn(b, T_OBS, 5))
    worst = max(worst, float(np.abs(net(*x).detach().numpy() - sess.run(None, dict(zip(["x", "t", "obs"], [v.numpy() for v in x])))[0]).max()))
print("dp_s0 fp16-weight model, max abs diff vs fp32 torch:", worst)
meta["dp_s0"]["parity_fp16"] = worst; json.dump(meta, open("docs/model.json", "w"))
