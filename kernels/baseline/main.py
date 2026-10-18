# bench-gas-baseline: CPU kernel (private, internet on). Reproduces the published HyperSTARCOP mag1c+RGB evaluation on the
# 342 official test tiles with the RELEASED checkpoint, (A) with our scorer and (B) with the official starcop.validation.run_validation.
import collections, io, json, os, subprocess, sys, time, zlib, types
import numpy as np
import pandas as pd

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
def sh(c, t=1800):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return (r.stdout + r.stderr).strip()
import torch
from starcop_edge.tif import read_tif
torch.set_num_threads(4)
from starcop_edge.models import SMPLikeUnet, load_released_checkpoint, count_params, INPUT_ORDER, INPUT_FACTORS
from starcop_edge import metrics as M

IN = "/kaggle/input"
from starcop_edge.kio import find_dir
root = find_dir("STARCOP_test"); ASSETS = find_dir("zenodo_STARCOP_test.tsv")
assert root and ASSETS, (root, ASSETS)
TEST_DIR = os.path.join(root, "STARCOP_test", "STARCOP_test")
OUT = "/kaggle/working"
te = pd.read_csv(os.path.join(ASSETS, "test.csv"))
zen = {}
for l in open(os.path.join(ASSETS, "zenodo_STARCOP_test.tsv")):
    n, s, crc, ct = l.rstrip("\n").split("\t"); p = n.split("/")
    if len(p) == 3: zen[(p[1], p[2])] = (int(s), crc)

sd, ck = load_released_checkpoint(os.path.join(ASSETS, "hyperstarcop_mag1c_rgb_final_checkpoint_model.ckpt"))
net = SMPLikeUnet(4, 1); net.load_state_dict(sd, strict=True); net.eval()
nrm = {k[len("normalizer."):]: v.flatten().tolist() for k, v in ck["state_dict"].items() if k.startswith("normalizer.")}
assert nrm["factors_input"] == list(INPUT_FACTORS) and nrm["clip_max_input"] == [2] * 4 and nrm["offsets_input"] == [0] * 4, nrm
log("released model params", count_params(net), "| normalizer", nrm)

# ------------------------------------------------------------------ (A) our pipeline on raw float TIFFs
def load_tile(tid):
    chans = []
    for b in INPUT_ORDER:
        f = b + ".tif"; raw = open(os.path.join(TEST_DIR, tid, f), "rb").read()
        assert f"{zlib.crc32(raw):08x}" == zen[(tid, f)][1], (tid, f)
        chans.append(read_tif(raw).astype(np.float32))  # official: torch.cat(...).float()
    x = np.stack(chans)
    fac = np.array(INPUT_FACTORS, dtype=np.float32)[:, None, None]
    return np.clip(x / fac, 0.0, 2.0).astype(np.float32)
def load_label(tid):
    raw = open(os.path.join(TEST_DIR, tid, "labelbinary.tif"), "rb").read()
    assert f"{zlib.crc32(raw):08x}" == zen[(tid, "labelbinary.tif")][1]
    return read_tif(raw).astype(np.uint8)

TILES = np.stack([load_tile(t) for t in te.id]); LABELS = np.stack([load_label(t) for t in te.id]); log("test tiles loaded", TILES.shape, LABELS.shape)
sums, logits16, t_inf = [], np.zeros((len(te), 512, 512), np.float16), []
for i, tid in enumerate(te.id):
    x = torch.from_numpy(TILES[i])[None]; y = LABELS[i]
    t = time.time()
    with torch.inference_mode(): lg = net(x)[0, 0].numpy()
    t_inf.append(time.time() - t)
    logits16[i] = lg.astype(np.float16); sums.append(M.tile_summary(lg, y))
    if (i + 1) % 100 == 0: log("A:", i + 1, "tiles, mean inference s/tile", round(float(np.mean(t_inf)), 3))
S = M.stack_summaries(sums); q = te.qplume.values.astype(np.float64)
mine = M.scalar_metrics(S, q)
np.savez_compressed(f"{OUT}/baseline_test_summaries.npz", **S, qplume=q, ids=te.id.values.astype(str))
np.save(f"{OUT}/baseline_test_logits_fp16.npy", logits16)
boot = M.bootstrap(S, q, B=2000, seed=0)
ci = {k: [round(v, 2) for v in M.summarize_bootstrap(boot, [k])[k]] for k in mine}
log("(A) our scorer, released checkpoint, official test split:")
for k in ["F1_strong_px", "F1_weak_px", "F1_strong_q", "F1_weak_q", "FPR_tile_official640", "FPR_tile_text10", "AUPRC_ap", "AUPRC_trapz16", "FPR_px_no_plume", "F1_all_plume", "TPR_tile_official640_all_plume"]:
    log(f"   {k:34s} {mine[k]:8.2f}   95% tile-bootstrap CI {ci[k]}")
grp = pd.crosstab(pd.Series(np.where(S["n_pos"] == 0, "none", np.where(S["n_pos"] > 1000, "strong_px", "weak_px")), name="pixel-group"),
                  pd.Series(np.where(q == 0, "none", np.where(q >= 1000, "strong_q", "weak_q")), name="qplume-group"))
log("group crosstab (pixel-count vs emission-rate):\n" + grp.to_string())
paper = {"F1_strong": (81.96, 3.71), "F1_weak": (43.42, 5.72), "FPR_tile": (43.66, 7.36), "AUPRC": (51.99, 2.76)}

# ---- the second released model: mag1c only (best published weak-plume F1 in Table 2: 47.57 +- 4.17)
try:
    log(sh("curl -sL -o /tmp/mag1c_only.ckpt https://huggingface.co/isp-uv-es/starcop/resolve/main/models/hyperstarcop_mag1c_only/final_checkpoint_model.ckpt && sha256sum /tmp/mag1c_only.ckpt"))
    sd1, ck1 = load_released_checkpoint("/tmp/mag1c_only.ckpt")
    net1 = SMPLikeUnet(1, 1); net1.load_state_dict(sd1, strict=True); net1.eval()
    sums1 = []
    for i in range(len(te)):
        with torch.inference_mode(): lg = net1(torch.from_numpy(TILES[i, :1])[None])[0, 0].numpy()
        sums1.append(M.tile_summary(lg, LABELS[i]))
    S1 = M.stack_summaries(sums1); mine1 = M.scalar_metrics(S1, q)
    np.savez_compressed(f"{OUT}/baseline_mag1c_only_test_summaries.npz", **S1, qplume=q, ids=te.id.values.astype(str))
    boot1 = M.bootstrap(S1, q, B=2000, seed=0)
    ci1 = {k: [round(v, 2) for v in M.summarize_bootstrap(boot1, [k])[k]] for k in mine1}
    log("(A') released mag1c-only checkpoint, official test split:", {k: round(mine1[k], 2) for k in ["F1_strong_px", "F1_weak_px", "F1_strong_q", "F1_weak_q", "FPR_tile_official640", "FPR_tile_text10", "AUPRC_ap", "AUPRC_trapz16"]})
    mag1c_only = dict(params=count_params(net1), mine=mine1, ci=ci1, paper_table2={"F1_strong": (74.15, 6.10), "F1_weak": (47.57, 4.17), "FPR_tile": (52.11, 10.98), "AUPRC": (49.41, 5.49)})
except Exception as e:
    import traceback; traceback.print_exc(); mag1c_only = dict(error=repr(e)); log("mag1c-only failed:", repr(e))

res = dict(mine=mine, ci=ci, paper_table2=paper, crosstab=json.loads(grp.to_json()), n_tiles=len(te), infer_s_per_tile_4threads=round(float(np.mean(t_inf)), 3), mag1c_only=mag1c_only)

# ------------------------------------------------------------------ cross-check against the real smp model (numerical)
try:
    log(sh("pip install -q segmentation-models-pytorch 2>&1 | tail -1"))
    import segmentation_models_pytorch as smp
    ref = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=4, classes=1, activation=None); ref.load_state_dict(sd, strict=True); ref.eval()
    d = 0.0
    for tid in te.id[:8]:
        x = torch.from_numpy(load_tile(tid))[None]
        with torch.inference_mode(): d = max(d, (ref(x) - net(x)).abs().max().item())
    res["smp_vs_ours_max_abs_logit_diff_8tiles"] = d; res["smp_version"] = smp.__version__; log("smp vs ours max abs logit diff:", d)
except Exception as e:
    res["smp_error"] = repr(e); log("smp cross-check failed:", repr(e))

# ------------------------------------------------------------------ (B) official code path
try:
    log(sh("pip install -q rasterio pytorch-lightning torchmetrics omegaconf kornia wandb fsspec requests scikit-learn 2>&1 | tail -2"))
    log(sh("rm -rf /tmp/starcop_repo && mkdir /tmp/starcop_repo && cd /tmp/starcop_repo && git init -q && git remote add origin https://github.com/spaceml-org/STARCOP.git && "
           "git fetch -q --depth 1 origin c4789268a3fa0395f92357429052f6f5fc748acb && git checkout -q FETCH_HEAD && git log -1 --format=%H"))
    sys.path.insert(0, "/tmp/starcop_repo")
    os.environ["WANDB_MODE"] = "disabled"
    import torchmetrics, rasterio.windows
    from omegaconf import OmegaConf
    _Bin = torchmetrics.classification.BinaryConfusionMatrix
    torchmetrics.ConfusionMatrix = lambda num_classes=2, task=None, **kw: _Bin(**kw)  # shim: legacy call signature used by the 2022 code
    from starcop.models.model_module import ModelModule
    from starcop.data.dataset import STARCOPDataset
    from starcop.validation import run_validation
    from torch.utils.data import DataLoader
    cfg = OmegaConf.load("/tmp/starcop_repo/starcop/config.yaml")
    cfg.dataset.input_products = list(INPUT_ORDER); cfg.model.pos_weight = 1
    model = ModelModule(cfg); missing = model.load_state_dict(ck["state_dict"], strict=True); model.eval(); log("official ModelModule loaded:", missing)
    df = te.copy(); df["folder"] = [os.path.join(TEST_DIR, i) for i in df.id]
    df["window"] = [rasterio.windows.Window(col_off=0, row_off=0, width=512, height=512)] * len(df)
    df = df.sort_values(["has_plume", "qplume"], ascending=False).set_index("id")  # same ordering as official datamodule
    ds = STARCOPDataset(df, input_products=list(INPUT_ORDER), output_products=["labelbinary"], weight_loss="weight_mag1c")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    t = time.time()
    out_data, mo = run_validation(model, dl, products_plot=[], verbose=False, show_plots=False, path_save_results=f"{OUT}/official_eval", skip_saving_plots=True)
    log("official run_validation secs", round(time.time() - t, 1))
    cmc = mo["classification_confusion_matrix"].numpy()
    th = mo["thresholded"]; prec = np.array([float(x["precision"]) for x in th]); rec = np.array([float(x["recall"]) for x in th])
    o = np.argsort(rec); auc16 = float(np.trapz(prec[o], rec[o])) * 100
    off = dict(F1_strong_px=mo["f1score_easy"] * 100, F1_weak_px=mo["f1score_hard"] * 100, FPR_tile_official640=100.0 * cmc[0, 1] / (cmc[0, 0] + cmc[0, 1]),
               FPR_px_no_plume=mo["FPR_no_plume"] * 100, AUPRC_trapz16_from_official_thresholded=auc16, F1_all_tiles=mo["f1score"] * 100)
    log("(B) official run_validation:", json.dumps({k: round(float(v), 3) for k, v in off.items()}))
    # per-tile exact comparison with (A)
    od = out_data.reindex(te.id.values)
    mism = int(((od[["TP", "FP", "TN", "FN"]].values != np.stack([S["tp"], S["fp"], S["tn"], S["fn"]], 1)).any(1)).sum())
    res["official"] = {k: float(v) for k, v in off.items()}; res["official_vs_ours_tiles_with_different_TPFPTNFN"] = mism
    log("tiles whose TP/FP/TN/FN differ between official code and our scorer:", mism, "of", len(te))
    res["official_commit"] = "c4789268a3fa0395f92357429052f6f5fc748acb"
except Exception as e:
    import traceback; traceback.print_exc(); res["official_error"] = repr(e); log("official path failed:", repr(e))

res["secs"] = round(time.time() - T0, 1)
json.dump(res, open(f"{OUT}/baseline_report.json", "w"), indent=1, default=str)
log("DONE")
