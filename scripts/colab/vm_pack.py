# Wave 3 (Amendment 6), inside the Colab VM: pack the small outputs of one job (JOB = kernel name) for `colab download`.
# Only json / jsonl / logs / model.pt / summaries (a few MB); no STARCOP data.
import hashlib, os, tarfile
job = os.environ["JOB"]
out = f"/content/w3/out_{job}.tgz"
keep = (".json", ".jsonl", ".log", ".out", ".pt", ".npz", ".csv", ".txt")
with tarfile.open(out, "w:gz") as tf:
    for root in ("/kaggle/working", f"/content/w3/{job}"):
        for dp, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(dp, f)
                if f.endswith(keep) and os.path.getsize(p) < 64 * 2**20:
                    tf.add(p, arcname=os.path.relpath(p, "/"))
    if os.path.exists("/content/w3/hw.json"):
        tf.add("/content/w3/hw.json", arcname="content/w3/hw.json")
print("PACKED", out, os.path.getsize(out), "B sha256", hashlib.sha256(open(out, "rb").read()).hexdigest())
