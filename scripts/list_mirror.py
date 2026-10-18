"""List every file of the Kaggle STARCOP mirror (chrisphamda/starcop) via the Kaggle API.
Metadata only: nothing is downloaded. Writes manifests/mirror_files.tsv (name<TAB>size)."""
import sys, time
from kaggle import api  # authenticates from ~/.kaggle

DS = "chrisphamda/starcop"
out = sys.argv[1] if len(sys.argv) > 1 else "manifests/mirror_files.tsv"
tok, n, t0 = None, 0, time.time()
with open(out, "w") as fh:
    while True:
        for attempt in range(6):
            try:
                res = api.dataset_list_files(DS, page_token=tok, page_size=200)
                break
            except Exception as e:  # transient
                print("retry", attempt, repr(e)[:200], flush=True); time.sleep(2 + 3 * attempt)
        else:
            raise SystemExit("giving up")
        for f in res.files:
            fh.write(f"{f.name}\t{f.total_bytes}\n"); n += 1
        fh.flush()
        tok = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
        if n % 2000 < 200:
            print(f"{n} files, {time.time()-t0:.0f}s", flush=True)
        if not tok:
            break
print("DONE", n, "files", f"{time.time()-t0:.0f}s", flush=True)
