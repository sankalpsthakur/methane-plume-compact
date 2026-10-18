# bench-gas-cpuprobe: how to get the most training throughput out of a 4-vCPU Kaggle CPU kernel (synthetic data, no inputs).
import json, multiprocessing as mp, os, subprocess, sys, time
import numpy as np
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

def worker(widths, threads, steps, channels_last, q, idx):
    import torch
    from starcop_edge.models import EdgeUNet
    torch.set_num_threads(threads)
    net = EdgeUNet(4, widths); opt = torch.optim.Adam(net.parameters(), 1e-3); net.train()
    x = torch.rand(32, 4, 128, 128); y = (torch.rand(32, 1, 128, 128) > 0.9).float()
    if channels_last:
        net = net.to(memory_format=torch.channels_last); x = x.contiguous(memory_format=torch.channels_last)
    for i in range(3 + steps):
        if i == 3: t = time.time()
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y); loss.backward(); opt.step()
    q.put((idx, steps / (time.time() - t)))

def run(widths, nproc, threads, steps=15, cl=False):
    q = mp.Queue(); ps = [mp.Process(target=worker, args=(widths, threads, steps, cl, q, i)) for i in range(nproc)]
    [p.start() for p in ps]; r = [q.get() for _ in ps]; [p.join() for p in ps]
    return sorted(round(v, 2) for _, v in r)

res = {}
log(subprocess.run("lscpu | egrep 'Model name|Thread|Core|Socket|L2|L3'", shell=True, capture_output=True, text=True).stdout)
for widths in [(4, 8, 16, 32, 64), (8, 16, 32, 64, 128)]:
    for nproc, threads in [(1, 4), (1, 2), (2, 2), (2, 1), (4, 1), (3, 1)]:
        for cl in (False, True):
            r = run(widths, nproc, threads, cl=cl)
            res[f"{widths[0]}|p{nproc}t{threads}|cl{int(cl)}"] = r
            log(f"c{widths[0]} procs={nproc} threads/proc={threads} channels_last={cl}: steps/s per proc {r}  total {sum(r):.2f}")
json.dump(res, open("/kaggle/working/cpuprobe.json", "w"), indent=1)
log("DONE")
