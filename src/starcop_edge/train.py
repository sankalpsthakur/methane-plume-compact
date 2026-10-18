"""Train a compact U-Net on the STARCOP cache (PyTorch, CPU or GPU). One process = one (config, seed)."""
import argparse, contextlib, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import metrics as M
from .models import EdgeUNet, SMPLikeUnet, count_params


def parse_widths(s):
    return tuple(int(v) for v in s.split(","))


@torch.no_grad()
def infer_tiles(model, X, idx, device="cpu", bs=4):
    """Full-tile inference. X: (N,4,512,512) fp16 normalised. Yields (tile_index, logits (H,W) float32)."""
    model.eval()
    for s in range(0, len(idx), bs):
        sel = idx[s:s + bs]
        xb = torch.from_numpy(np.stack([np.asarray(X[t]) for t in sel]).astype(np.float32)).to(device)
        out = model(xb)[:, 0].float().cpu().numpy()
        for t, o in zip(sel, out):
            yield t, o


def evaluate(model, X, Y, meta, idx, device="cpu"):
    sums = []
    for t, logit in infer_tiles(model, X, idx, device):
        sums.append(M.tile_summary(logit, np.asarray(Y[t])))
    S = M.stack_summaries(sums)
    q = meta["qplume"].values[idx].astype(np.float64)
    return M.scalar_metrics(S, q), S


def lr_at(step, total, peak, warm=200, floor=0.02):
    if step < warm:
        return peak * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--test_mirror", default=""); ap.add_argument("--assets", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="8,16,32,64,128"); ap.add_argument("--dec_widths", default=""); ap.add_argument("--stem_stride", type=int, default=1)
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--pos_weight", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_dates", default="", help="comma list of dates held out as validation; empty = train on everything")
    ap.add_argument("--eval_every", type=int, default=3000); ap.add_argument("--val_max", type=int, default=0); ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--jitter", type=int, default=32); ap.add_argument("--hardneg", type=float, default=0.0)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--tag", default="run")
    ap.add_argument("--arch", default="edge", choices=["edge", "mnv2"], help="mnv2 = released-architecture control (6.6M params)")
    ap.add_argument("--sched", default="cosine", choices=["cosine", "plateau"], help="plateau = paper recipe: x0.5 after 4 epochs w/o val-loss gain (val = held-out dates, never test)")
    ap.add_argument("--plateau_every", type=int, default=5245); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--test_eval", action="store_true", help="after training, score the 342 official test tiles ONCE and save per-tile summaries")
    ap.add_argument("--allow_test_with_val", action="store_true", help="ONLY for the fixed-recipe paper control (GPU plan): validation dates drive the plateau LR schedule, nothing is selected on them")
    a = ap.parse_args(argv)
    if a.test_eval and a.val_dates and not a.allow_test_with_val:
        raise SystemExit("integrity guard: --test_eval must not be combined with --val_dates (dev runs never touch the test split)")
    if a.allow_test_with_val and a.sched != "plateau":
        raise SystemExit("--allow_test_with_val is reserved for the fixed paper-recipe control (--sched plateau)")
    if a.test_eval and not (a.test_mirror and a.assets):
        raise SystemExit("--test_eval needs --test_mirror (Kaggle mirror root) and --assets")
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    X, Y, Wt, meta = D.load_split(a.data, "train", mmap=True)  # page-cache shared between concurrent processes
    val_dates = [d for d in a.val_dates.split(",") if d]
    is_val = meta["date"].isin(val_dates).values
    tr_idx = np.where(~is_val)[0]; va_idx = np.where(is_val)[0]
    if a.val_max and len(va_idx) > a.val_max:
        va_idx = np.sort(np.random.default_rng(123).choice(va_idx, a.val_max, replace=False))
    dec = parse_widths(a.dec_widths) if a.dec_widths else None
    model = (SMPLikeUnet(4, 1) if a.arch == "mnv2" else EdgeUNet(4, parse_widths(a.widths), dec, stem_stride=a.stem_stride)).to(a.device)
    nparams = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    samp = D.CropSampler(X, Y, Wt, tr_idx, seed=a.seed, jitter=a.jitter, batch=a.batch, hardneg=a.hardneg)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    def emit(rec):
        rec["t"] = round(time.time() - t0, 1); log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
    t0 = time.time()
    emit(dict(event="start", tag=a.tag, args=vars(a), params=nparams, n_train_tiles=len(tr_idx), n_val_tiles=len(va_idx),
              pos_windows=samp.n_pos_windows, neg_windows=samp.n_neg_windows, hard_neg_windows=samp.n_hard_windows))
    pw = torch.tensor(a.pos_weight, device=a.device)
    use_amp = a.amp and str(a.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    plateau = None
    if a.sched == "plateau":
        assert len(va_idx), "plateau schedule needs --val_dates"
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
        vs = D.CropSampler(X, Y, Wt, va_idx, seed=999, jitter=0, batch=a.batch)
        val_crops = [vs.next() for _ in range(64)]; vs.close()  # 2048 fixed validation crops (val dates only)
    run_loss = 0.0
    for step in range(a.steps):
        if plateau is None:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.steps, a.lr)
        x, y, w = (torch.from_numpy(v).to(a.device) for v in samp.next())
        model.train()
        with (torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()):
            logit = model(x)
        loss = (F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pw, reduction="none") * w).mean()
        opt.zero_grad(set_to_none=True)
        if scaler is not None: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
        run_loss += loss.item()
        if plateau is not None and (step + 1) % a.plateau_every == 0:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([(F.binary_cross_entropy_with_logits(model(torch.from_numpy(cx).to(a.device)).float(), torch.from_numpy(cy).to(a.device), pos_weight=pw, reduction="none") * torch.from_numpy(cw).to(a.device)).mean().item() for cx, cy, cw in val_crops]))
            plateau.step(vl); emit(dict(event="plateau", step=step + 1, val_loss=round(vl, 6), lr=opt.param_groups[0]["lr"]))
        if (step + 1) % a.log_every == 0:
            emit(dict(event="train", step=step + 1, loss=round(run_loss / a.log_every, 5), lr=round(opt.param_groups[0]["lr"], 6))); run_loss = 0.0
        if len(va_idx) and (step + 1) % a.eval_every == 0 and step + 1 != a.steps:
            m, _ = evaluate(model, X, Y, meta, va_idx, a.device)
            emit(dict(event="val", step=step + 1, **{k: round(m[k], 3) for k in ["F1_strong_q", "F1_weak_q", "F1_strong_px", "F1_weak_px", "FPR_tile_text10", "FPR_tile_official640", "AUPRC_ap", "AUPRC_trapz16"]}))
    samp.close()
    torch.save({"state_dict": model.state_dict(), "arch": a.arch, "widths": parse_widths(a.widths), "dec_widths": dec, "stem_stride": a.stem_stride, "args": vars(a), "params": nparams},
               os.path.join(a.out, "model.pt"))
    if len(va_idx):
        m, S = evaluate(model, X, Y, meta, va_idx, a.device)
        np.savez_compressed(os.path.join(a.out, "val_summaries.npz"), **S, qplume=meta["qplume"].values[va_idx], ids=meta["id"].values[va_idx].astype(str))
        json.dump(m, open(os.path.join(a.out, "val_metrics.json"), "w"), indent=1)
        emit(dict(event="val", step=a.steps, **{k: round(m[k], 3) for k in ["F1_strong_q", "F1_weak_q", "F1_strong_px", "F1_weak_px", "FPR_tile_text10", "FPR_tile_official640", "AUPRC_ap", "AUPRC_trapz16"]}))
    if a.test_eval:
        Xt, Yt, _, mt = D.load_test_from_mirror(a.test_mirror, a.assets)
        m, S = evaluate(model, Xt, Yt, mt, np.arange(len(mt)), a.device)
        np.savez_compressed(os.path.join(a.out, "test_summaries.npz"), **S, qplume=mt["qplume"].values, ids=mt["id"].values.astype(str))
        json.dump(m, open(os.path.join(a.out, "test_metrics.json"), "w"), indent=1)
        emit(dict(event="test", **{k: round(m[k], 3) for k in ["F1_strong_q", "F1_weak_q", "F1_strong_px", "F1_weak_px", "FPR_tile_text10", "FPR_tile_official640", "AUPRC_ap", "AUPRC_trapz16"]}))
    emit(dict(event="done", secs=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
