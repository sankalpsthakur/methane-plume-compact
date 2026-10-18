"""Kernel-side probes: CPU training throughput layouts and an end-to-end edge-export self-test (random weights)."""
import importlib, json, os, subprocess, sys, time

VARIANTS = {"s2c8": dict(widths=(8, 16, 32, 64, 128), stem_stride=2), "s2c16": dict(widths=(16, 32, 64, 128, 256), stem_stride=2),
            "s1c4": dict(widths=(4, 8, 16, 32, 64), stem_stride=1)}


def worker(variant, threads, steps=12):
    """Run in a child interpreter: synthetic batch-32 128x128 training steps; prints one JSON line."""
    import torch
    from .models import EdgeUNet
    torch.set_num_threads(int(threads))
    v = VARIANTS[variant]
    net = EdgeUNet(4, v["widths"], stem_stride=v["stem_stride"]); opt = torch.optim.Adam(net.parameters(), 1e-3); net.train()
    x = torch.rand(32, 4, 128, 128); y = (torch.rand(32, 1, 128, 128) > 0.9).float()
    for i in range(3 + int(steps)):
        if i == 3: t = time.time()
        opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(net(x), y).backward(); opt.step()
    print(json.dumps({"variant": variant, "threads": int(threads), "steps_per_s": int(steps) / (time.time() - t)}), flush=True)


def run_cpu_probe(log, out="/kaggle/working/cpuprobe.json"):
    res = {}
    code = "import sys; sys.path.insert(0, '/tmp/pkg'); from starcop_edge import probe; probe.worker(sys.argv[1], sys.argv[2], sys.argv[3])"
    for variant in ("s2c8", "s2c16", "s1c4"):
        for nproc, th in ((1, 4), (2, 2), (4, 1)):
            ps = [subprocess.Popen([sys.executable, "-c", code, variant, str(th), "12"], stdout=subprocess.PIPE, text=True) for _ in range(nproc)]
            r = [json.loads(p.communicate()[0].strip().splitlines()[-1])["steps_per_s"] for p in ps]
            res[f"{variant}|procs{nproc}|threads{th}"] = {"per_proc": [round(v, 2) for v in r], "total": round(sum(r), 2)}
            log(f"cpu probe {variant} procs={nproc} threads/proc={th}: per-proc steps/s {[round(v, 2) for v in r]} total {sum(r):.2f}")
    json.dump(res, open(out, "w"), indent=1)
    return res


def edge_selftest(log, outdir="/kaggle/working/edge_selftest"):
    """Random-weight EdgeUNet -> BN fold -> ONNX -> static int8 -> onnxruntime; checks numerics, sizes and latency. Needs internet for pip."""
    import numpy as np, torch
    os.makedirs(outdir, exist_ok=True)
    log(subprocess.run("pip install -q onnxruntime onnx onnxscript 2>&1 | tail -1", shell=True, capture_output=True, text=True).stdout.strip())
    importlib.invalidate_caches()
    from . import edge
    from .models import EdgeUNet, count_params
    torch.manual_seed(0)
    net = EdgeUNet(4, (8, 16, 32, 64, 128), stem_stride=2)
    net.train()
    with torch.no_grad():
        for _ in range(3): net(torch.rand(4, 4, 128, 128) * 2)  # populate BN running stats
    net.eval()
    p32, p8 = f"{outdir}/m_fp32.onnx", f"{outdir}/m_int8.onnx"
    edge.export_onnx(net, p32)
    x = torch.rand(1, 4, 512, 512) * 2
    with torch.no_grad(): ref = net(x).numpy()
    s32 = edge.ort_session(p32, 1); o32 = edge.ort_logits(s32, x.numpy())
    log("selftest fp32 onnx vs torch max abs diff:", float(np.abs(o32 - ref).max()), "| params", count_params(net))
    calib = [(torch.rand(1, 4, 512, 512) * 2).numpy() for _ in range(8)]
    edge.quantize_int8(p32, p8, calib)
    s8 = edge.ort_session(p8, 1); o8 = edge.ort_logits(s8, x.numpy())
    log("selftest int8 onnx vs fp32 max abs diff:", float(np.abs(o8 - o32).max()), "| logit std", float(o32.std()))
    out = {"params": count_params(net), "fp32_bytes": os.path.getsize(p32), "int8_bytes": os.path.getsize(p8),
           "lat_fp32_ms_1thread(median,min)": edge.latency_ms(s32), "lat_int8_ms_1thread(median,min)": edge.latency_ms(s8),
           "fp32_vs_torch_maxabs": float(np.abs(o32 - ref).max()), "int8_vs_fp32_maxabs": float(np.abs(o8 - o32).max())}
    json.dump(out, open(f"{outdir}/selftest.json", "w"), indent=1)
    log("edge selftest:", json.dumps(out))
    return out
