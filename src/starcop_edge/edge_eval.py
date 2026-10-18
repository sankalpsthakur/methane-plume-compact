"""Final-kernel tail: export every seed to ONNX (fp32, BN folded) and static int8 (QDQ, S8/S8, per-channel, MinMax calibration on
48 TRAIN tiles), score both on the official test tiles with onnxruntime, record sizes and the Kaggle-CPU latency."""
import importlib, json, os, subprocess, time
import numpy as np
import torch

from . import data as D
from . import edge
from . import metrics as M
from .models import EdgeUNet, SMPLikeUnet, count_params


def load_run(run_dir):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    net = EdgeUNet(4, tuple(ck["widths"]), ck["dec_widths"], stem_stride=ck.get("stem_stride", 1)) if ck.get("arch", "edge") == "edge" else SMPLikeUnet(4, 1)
    net.load_state_dict(ck["state_dict"]); return net.eval(), ck


def calib_tiles(cache_dir, n_each=24, seed=0):
    X, Y, Wt, meta = D.load_split(cache_dir, "train", mmap=True)
    rng = np.random.default_rng(seed)
    pl = rng.choice(np.where(meta.has_plume.values)[0], n_each, replace=False); ng = rng.choice(np.where(~meta.has_plume.values)[0], n_each, replace=False)
    return [np.asarray(X[t], dtype=np.float32)[None] for t in np.concatenate([pl, ng])]


def score_onnx(sess, X, Y, qplume):
    sums = []
    for i in range(len(X)):
        lg = edge.ort_logits(sess, np.asarray(X[i], dtype=np.float32)[None])[0, 0]
        sums.append(M.tile_summary(lg, Y[i]))
    S = M.stack_summaries(sums)
    return M.scalar_metrics(S, qplume), S


def _export_and_score(name, net, params, rd_bytes, outdir, calib, Xt, Yt, q, mt, log):
    p32, p8 = f"{outdir}/{name}_fp32.onnx", f"{outdir}/{name}_int8.onnx"
    net = net.eval(); fused = edge.fuse_conv_bn(net)
    xt = torch.from_numpy(np.asarray(Xt[0], dtype=np.float32))[None]
    with torch.no_grad(): d_fuse = float((net(xt) - fused(xt)).abs().max())
    edge.export_onnx(net, p32); edge.quantize_int8(p32, p8, calib, method="minmax", per_channel=True)
    rec = {"params": params, "pt_state_bytes": rd_bytes, "onnx_fp32_bytes": os.path.getsize(p32), "onnx_int8_bytes": os.path.getsize(p8), "bn_fold_maxabs": d_fuse}
    s32 = edge.ort_session(p32, threads=1)
    with torch.no_grad(): ref = net(xt).numpy()
    rec["onnx_fp32_vs_torch_maxabs"] = float(np.abs(edge.ort_logits(s32, xt.numpy()) - ref).max())
    for tag, path in (("onnx_fp32", p32), ("onnx_int8", p8)):
        sess = edge.ort_session(path, threads=4); t = time.time()
        m, S = score_onnx(sess, Xt, Yt, q)
        np.savez_compressed(f"{outdir}/{name}_{tag}_test_summaries.npz", **S, qplume=q, ids=mt.id.values.astype(str))
        rec[tag] = {k: round(m[k], 3) for k in ["F1_strong_q", "F1_weak_q", "F1_strong_px", "F1_weak_px", "FPR_tile_text10", "FPR_tile_official640", "AUPRC_ap", "AUPRC_trapz16"]}
        rec[tag + "_secs_per_tile_4thr"] = round((time.time() - t) / len(Xt), 3)
        rec[tag + "_latency_ms_1thread_kaggle(median,min)"] = edge.latency_ms(edge.ort_session(path, threads=1), n=8, warm=2)
        log(name, tag, json.dumps(rec[tag]))
    return rec


def edge_pipeline(run_dirs, cache_dir, mirror, assets, outdir, log, released_ckpt=None):
    os.makedirs(outdir, exist_ok=True)
    log(subprocess.run("pip install -q onnxruntime onnxscript 2>&1 | tail -1", shell=True, capture_output=True, text=True).stdout.strip())
    importlib.invalidate_caches()
    Xt, Yt, _, mt = D.load_test_from_mirror(mirror, assets)
    q = mt.qplume.values.astype(np.float64)
    calib = calib_tiles(cache_dir)
    report = {}
    for rd in run_dirs:
        name = os.path.basename(rd.rstrip("/"))
        try:
            net, ck = load_run(rd)
            report[name] = _export_and_score(name, net, count_params(net), os.path.getsize(os.path.join(rd, "model.pt")), outdir, calib, Xt, Yt, q, mt, log)
        except Exception as e:
            import traceback; traceback.print_exc(); report[name] = {"error": repr(e)}
        json.dump(report, open(f"{outdir}/edge_report.json", "w"), indent=1)
    if released_ckpt:  # the leader (6.6M params) through the same edge pipeline, for an apples-to-apples size/latency/score row
        try:
            from .models import load_released_checkpoint
            sd, ck = load_released_checkpoint(released_ckpt)
            net = SMPLikeUnet(4, 1); net.load_state_dict(sd, strict=True)
            report["released_hyperstarcop_mag1c_rgb"] = _export_and_score("released_mag1c_rgb", net, count_params(net), 4 * sum(v.numel() for v in sd.values()), outdir, calib, Xt, Yt, q, mt, log)
        except Exception as e:
            import traceback; traceback.print_exc(); report["released_hyperstarcop_mag1c_rgb"] = {"error": repr(e)}
        json.dump(report, open(f"{outdir}/edge_report.json", "w"), indent=1)
    return report
