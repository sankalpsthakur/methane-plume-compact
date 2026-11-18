# bench-gas-gpu-probe-wv3 (G0 of Amendment 7): GPU training throughput of the WV3 candidates and of the paper-recipe control on synthetic 3-channel 128x128 crops (no inputs, no data), hardware readback, imports the test stage needs,
# and the ImageNet-weights check (download, SHA-256 prefix, strict load into the 3-channel released architecture): a failure there costs minutes here instead of the control stage.
# Twin of kernels/gpu-probe/main.py with 3 input channels, AMP only (the planner uses the AMP rates), internet on.
import hashlib, json, subprocess, time, urllib.request, torch
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
log(subprocess.run("nvidia-smi; lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True, capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "no GPU in this session"
env = dict(torch=torch.__version__, cuda=torch.version.cuda, arch_list=torch.cuda.get_arch_list(), capability=list(torch.cuda.get_device_capability(0)), n_gpu=torch.cuda.device_count(), cpus=__import__("os").cpu_count())
for mod in ("rasterio", "pandas", "numpy", "onnx", "onnxruntime"):  # rasterio and pandas are needed by the WV3 test loader; a missing module would only show up after the control has trained
    try: env[mod] = getattr(__import__(mod), "__version__", "?")
    except Exception as e: env[mod] = "MISSING " + repr(e)[:80]
log("env", json.dumps(env))
if CFG.get("max_gpus") is not None and env["n_gpu"] > CFG["max_gpus"]:  # billed per GPU (Amendment 5): do not burn quota measuring a shape the budget cannot afford
    json.dump(dict(env=env, aborted=f"{env['n_gpu']} GPUs visible, budget assumes at most {CFG['max_gpus']}", device=torch.cuda.get_device_name(0), secs=round(time.time() - T0, 1)), open("/kaggle/working/gpuprobe.json", "w"), indent=1)
    log("ABORT: more GPUs than budgeted"); raise SystemExit(0)
from starcop_edge.models import EdgeUNet, SMPLikeUnet, count_params, load_imagenet_encoder
res = {}
if CFG.get("imagenet"):
    spec = CFG["imagenet"]; chk = dict(url=spec["url"], expected_sha256_prefix=spec["sha256_prefix"])
    try:
        with urllib.request.urlopen(spec["url"], timeout=120) as resp: raw = resp.read()
        dg = hashlib.sha256(raw).hexdigest(); open("/kaggle/working/imagenet_mobilenet_v2.pth", "wb").write(raw)
        chk.update(bytes=len(raw), sha256=dg, hash_ok=dg.startswith(spec["sha256_prefix"]))
        sd = torch.load("/kaggle/working/imagenet_mobilenet_v2.pth", map_location="cpu")
        chk["n_keys"] = len(sd); chk["load_report"] = load_imagenet_encoder(SMPLikeUnet(3, 1), sd); chk["load_ok"] = True
    except Exception as e:
        chk.update(load_ok=False, error=repr(e)[:300])
    res["imagenet_check"] = chk; log("imagenet check", json.dumps(chk))
CASES = {"mnv2_paper": lambda: SMPLikeUnet(3, 1), "d2_8-128_s2": lambda: EdgeUNet(3, (8, 16, 32, 64, 128), stem_stride=2), "d1_4-64_s2": lambda: EdgeUNet(3, (4, 8, 16, 32, 64), stem_stride=2),
         "d3_16-256_s2": lambda: EdgeUNet(3, (16, 32, 64, 128, 256), stem_stride=2), "d5_4-64_s1": lambda: EdgeUNet(3, (4, 8, 16, 32, 64), stem_stride=1),
         "d2w_8-128_s1": lambda: EdgeUNet(3, (8, 16, 32, 64, 128), stem_stride=1)}
amp = True
for name, mk in CASES.items():
    bs = 32
    net = mk().cuda().train(); opt = torch.optim.AdamW(net.parameters(), 1e-3); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.rand(bs, 3, 128, 128, device="cuda"); y = (torch.rand(bs, 1, 128, 128, device="cuda") > 0.9).float(); torch.cuda.reset_peak_memory_stats()
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
