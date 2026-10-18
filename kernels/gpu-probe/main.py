# bench-gas-gpu-probe: GPU training throughput of the candidate models on synthetic 128x128 crops (no inputs, no data). Run FIRST in the GPU window (about 4 GPU-minutes);
# its numbers set the step budgets of the control / compact / wide kernels in protocol/GPU-PLAN.md.
import json, subprocess, time, torch
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
log(subprocess.run("nvidia-smi; lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True, capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "no GPU in this session"
env = dict(torch=torch.__version__, cuda=torch.version.cuda, arch_list=torch.cuda.get_arch_list(), capability=list(torch.cuda.get_device_capability(0)), n_gpu=torch.cuda.device_count(), cpus=__import__("os").cpu_count())
for mod in ("rasterio", "pandas", "numpy", "onnx", "onnxruntime"):  # what the test/edge stages of G2/G3 need: a missing module would only show up after hours of training
    try: env[mod] = getattr(__import__(mod), "__version__", "?")
    except Exception as e: env[mod] = "MISSING " + repr(e)[:80]
log("env", json.dumps(env))
if CFG.get("max_gpus") is not None and env["n_gpu"] > CFG["max_gpus"]:  # billed per GPU (Amendment 5): do not burn quota measuring a shape the budget cannot afford
    json.dump(dict(env=env, aborted=f"{env['n_gpu']} GPUs visible, budget assumes at most {CFG['max_gpus']}", device=torch.cuda.get_device_name(0), secs=round(time.time() - T0, 1)), open("/kaggle/working/gpuprobe.json", "w"), indent=1)
    log("ABORT: more GPUs than budgeted"); raise SystemExit(0)
from starcop_edge.models import EdgeUNet, SMPLikeUnet, count_params
CASES = {"mnv2_paper": lambda: SMPLikeUnet(4, 1), "d2_8-128_s2": lambda: EdgeUNet(4, (8, 16, 32, 64, 128), stem_stride=2), "d1_4-64_s2": lambda: EdgeUNet(4, (4, 8, 16, 32, 64), stem_stride=2),
         "d3_16-256_s2": lambda: EdgeUNet(4, (16, 32, 64, 128, 256), stem_stride=2), "d5_4-64_s1": lambda: EdgeUNet(4, (4, 8, 16, 32, 64), stem_stride=1),
         "d2w_8-128_s1": lambda: EdgeUNet(4, (8, 16, 32, 64, 128), stem_stride=1), "d3w_16-256_s1": lambda: EdgeUNet(4, (16, 32, 64, 128, 256), stem_stride=1)}
res = {}
for amp in (True, False):
    for name, mk in CASES.items():
        for bs in (32,):
            net = mk().cuda().train(); opt = torch.optim.AdamW(net.parameters(), 1e-3); scaler = torch.amp.GradScaler("cuda", enabled=amp)
            x = torch.rand(bs, 4, 128, 128, device="cuda"); y = (torch.rand(bs, 1, 128, 128, device="cuda") > 0.9).float(); torch.cuda.reset_peak_memory_stats()
            try:
                for i in range(8 + 40):
                    if i == 8: torch.cuda.synchronize(); t = time.time()
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp): out = net(x)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y)
                    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                torch.cuda.synchronize(); sps = 40 / (time.time() - t)
                res[f"{name}|bs{bs}|amp{int(amp)}"] = dict(steps_per_s=round(sps, 2), params=count_params(net), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            except Exception as e:
                res[f"{name}|bs{bs}|amp{int(amp)}"] = dict(error=repr(e)[:200])
            log(name, "bs", bs, "amp", amp, json.dumps(res[f"{name}|bs{bs}|amp{int(amp)}"])); del net, opt
            torch.cuda.empty_cache()
res["env"] = env; res["device"] = torch.cuda.get_device_name(0); res["torch"] = torch.__version__; res["secs"] = round(time.time() - T0, 1)
json.dump(res, open("/kaggle/working/gpuprobe.json", "w"), indent=1); log("DONE", res["device"])
