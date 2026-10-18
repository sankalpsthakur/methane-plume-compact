"""Run the OFFICIAL spaceml-org/STARCOP `run_validation` (pinned commit c4789268) on the released mag1c+RGB checkpoint and the 342 test tiles
inside a kernel with internet, and compare with the numbers of our scorer (same checkpoint, same files)."""
import json, os, subprocess, sys, time
import numpy as np

COMMIT = "c4789268a3fa0395f92357429052f6f5fc748acb"


def run(assets, mirror, outdir, ref, log):
    """ref: dict of our-scorer numbers for the same model: F1_strong_px, F1_weak_px, FPR_tile_official640, AUPRC_trapz16."""
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    log(sh("pip install -q segmentation-models-pytorch pytorch-lightning torchmetrics omegaconf kornia wandb fsspec requests scikit-learn 2>&1 | tail -1"))
    log(sh(f"rm -rf /tmp/starcop_repo && mkdir /tmp/starcop_repo && cd /tmp/starcop_repo && git init -q && git remote add origin https://github.com/spaceml-org/STARCOP.git && "
           f"git fetch -q --depth 1 origin {COMMIT} && git checkout -q FETCH_HEAD && git log -1 --format=%H"))
    import importlib; importlib.invalidate_caches()
    sys.path.insert(0, "/tmp/starcop_repo"); os.environ["WANDB_MODE"] = "disabled"
    import pandas as pd, torch, torchmetrics, rasterio.windows
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    _Bin = torchmetrics.classification.BinaryConfusionMatrix
    torchmetrics.ConfusionMatrix = lambda num_classes=2, task=None, **kw: _Bin(**kw)  # legacy signature used by the 2022 code
    from starcop.models.model_module import ModelModule
    from starcop.data.dataset import STARCOPDataset
    from starcop.validation import run_validation
    from .models import INPUT_ORDER, load_released_checkpoint
    sd, ck = load_released_checkpoint(os.path.join(assets, "hyperstarcop_mag1c_rgb_final_checkpoint_model.ckpt"))
    cfg = OmegaConf.load("/tmp/starcop_repo/starcop/config.yaml"); cfg.dataset.input_products = list(INPUT_ORDER); cfg.model.pos_weight = 1
    model = ModelModule(cfg); log("official ModelModule:", model.load_state_dict(ck["state_dict"], strict=True)); model.eval()
    te = pd.read_csv(os.path.join(assets, "test.csv")); base = os.path.join(mirror, "STARCOP_test", "STARCOP_test")
    df = te.copy(); df["folder"] = [os.path.join(base, i) for i in df.id]
    df["window"] = [rasterio.windows.Window(col_off=0, row_off=0, width=512, height=512)] * len(df)
    df = df.sort_values(["has_plume", "qplume"], ascending=False).set_index("id")
    ds = STARCOPDataset(df, input_products=list(INPUT_ORDER), output_products=["labelbinary"], weight_loss="weight_mag1c")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    os.makedirs(f"{outdir}/official_eval", exist_ok=True)  # the official function does not create it
    t = time.time()
    out_data, mo = run_validation(model, dl, products_plot=[], verbose=False, show_plots=False, path_save_results=f"{outdir}/official_eval", skip_saving_plots=True)
    cmc = mo["classification_confusion_matrix"].numpy()
    th = mo["thresholded"]; prec = np.array([float(x["precision"]) for x in th]); rec = np.array([float(x["recall"]) for x in th]); o = np.argsort(rec)
    off = dict(F1_strong_px=mo["f1score_easy"] * 100, F1_weak_px=mo["f1score_hard"] * 100, FPR_tile_official640=100.0 * cmc[0, 1] / (cmc[0, 0] + cmc[0, 1]),
               AUPRC_trapz16=float(np.trapz(prec[o], rec[o])) * 100 if hasattr(np, "trapz") else float(np.trapezoid(prec[o], rec[o])) * 100,
               F1_all_tiles=mo["f1score"] * 100, FPR_px_no_plume=mo["FPR_no_plume"] * 100)
    res = {"official": {k: float(v) for k, v in off.items()}, "ours": ref, "secs": round(time.time() - t, 1), "commit": COMMIT,
           "abs_diff": {k: abs(float(off[k]) - float(ref[k])) for k in ref if k in off}}
    json.dump(res, open(f"{outdir}/official_check.json", "w"), indent=1)
    log("OFFICIAL run_validation vs our scorer:", json.dumps(res))
    return res
