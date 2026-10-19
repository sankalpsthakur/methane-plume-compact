# Wave 3 (Amendment 6), runs INSIDE the Colab VM through `colab exec -f`: hardware readback gate, Kaggle CLI, route A cache download,
# sha256 check against results/prep_cache/prep_report.json, and the /kaggle layout the generated kernels expect.
# The Kaggle credential is a short-lived OAuth access token uploaded to /root/.kaggle/access_token just before this runs; it is deleted
# in the finally block whatever happens (the refresh token never leaves the Mac). Nothing from the test split is fetched here.
import hashlib, json, os, shutil, subprocess, sys, time

TOKEN = "/root/.kaggle/access_token"
CACHE_PARENT = "/kaggle/input/notebooks/sankalpsthakur/bench-gas-prep-cache"
W3 = "/content/w3"
EXPECT = {  # results/prep_cache/prep_report.json (bench-gas-prep-cache v1)
    "train_X.npy": "5e6f1a9aab35992e5ff540eca6f804312e64cc2ef386c345ec0a2101f83da0e8",
    "train_Y.npy": "d55da93f701c2372c67a1efe16d56ed410692d2176fc6d98cc7f4d6c83f4bc69",
    "train_W.npy": "0c553912d906f27e47004b1f2efaf43f58f72b8a5a7a5dae82c37bd72bdffa5c",
    "train_meta.csv": "14ae7f44a842c816d20ddeb087dd29618e844d59e46e522f24f137a7fde5ca17",
}


def sh(cmd, timeout=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def find(name):
    for root, _, files in os.walk(CACHE_PARENT):
        if name in files:
            return os.path.join(root, name)
    return None


def output_urls():
    """file basename -> signed download URL of bench-gas-prep-cache's output (short-lived; never printed)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest(); req.user_name = "sankalpsthakur"; req.kernel_slug = "bench-gas-prep-cache"; req.page_size = 50
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
    return {it.file_name.split("/")[-1]: it.url for it in (resp.files or [])}


def remote_size(url):
    import requests
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60, stream=True)
    total = int(r.headers["Content-Range"].split("/")[-1]); r.close()
    return total


def main():
    os.makedirs(W3, exist_ok=True)
    gpus = [l.strip() for l in sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").stdout.splitlines() if l.strip()]
    hw = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), gpus=gpus, nproc=os.cpu_count(),
              ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
              disk_free_gb=round(shutil.disk_usage("/content").free / 2**30, 1), python=sys.version.split()[0],
              torch=sh("python3 -c 'import torch; print(torch.__version__)'").stdout.strip())
    json.dump(hw, open(f"{W3}/hw.json", "w"), indent=1)
    print("HARDWARE", json.dumps(hw), flush=True)
    if len(gpus) != 1 or "T4" not in gpus[0]:
        raise SystemExit(f"HARDWARE GATE FAILED: {gpus} (Amendment 6 needs exactly one T4)")
    r = sh("pip install -q kaggle==2.2.4", timeout=600)
    if r.returncode:
        raise SystemExit("pip install kaggle failed: " + r.stderr[-400:])
    # On Colab /kaggle/input is an empty read-only mount (ext4, ro); a writable directory is bind-mounted over it so the driver's
    # find_dir("train_X.npy") (base /kaggle/input) finds the cache without any change to the generated scripts (Amendment 6 note 2).
    try:
        os.makedirs(CACHE_PARENT, exist_ok=True)
    except OSError:
        os.makedirs("/content/kaggle_input", exist_ok=True)
        r = sh("mount --bind /content/kaggle_input /kaggle/input")
        if r.returncode:
            raise SystemExit("bind mount over /kaggle/input failed: " + (r.stdout + r.stderr)[-300:])
        os.makedirs(CACHE_PARENT, exist_ok=True)
        print("bind-mounted /content/kaggle_input over the read-only /kaggle/input", flush=True)
    if all(find(f) and sha256(find(f)) == h for f, h in EXPECT.items()):
        print("cache already present and verified", flush=True)
    else:
        if not os.path.isfile(TOKEN):
            raise SystemExit("no Kaggle access token on the VM")
        # The Kaggle CLI holds a whole file in memory and cannot resume (the 7.2 GB train_X.npy broke at 4.8-6.7 GB or was OOM-killed):
        # list the signed URLs through the same API and stream each file to disk with curl, resuming from the last byte (Amendment 6 note 4).
        for f, h in EXPECT.items():
            dest = os.path.join(CACHE_PARENT, "cache", f)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            for attempt in range(1, 9):
                url = output_urls()[f]
                total = remote_size(url)
                have = os.path.getsize(dest) if os.path.exists(dest) else 0
                if have == total and sha256(dest) == h:
                    break
                if have >= total:  # complete size but wrong bytes, or larger: start over
                    os.remove(dest); have = 0
                t = time.time()
                try:
                    r = subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-delay", "5", "--connect-timeout", "30", "-C", "-", "-o", dest, url],
                                       capture_output=True, text=True, timeout=2400)
                    rc, err = r.returncode, r.stderr[-160:].replace("\n", " ")
                except subprocess.TimeoutExpired:
                    rc, err = "timeout", "curl ran past 2400 s"
                now = os.path.getsize(dest) if os.path.exists(dest) else 0
                print(f"download {f} attempt {attempt}: curl rc={rc} in {time.time() - t:.0f} s, {have} -> {now} of {total} B", err, flush=True)
    bad = {}
    for f, h in EXPECT.items():
        p = find(f)
        got = sha256(p) if p else None
        print(f"sha256 {f}: {'OK' if got == h else 'MISMATCH'} ({p}, {os.path.getsize(p) if p else 0} B)", flush=True)
        if got != h:
            bad[f] = got
    if bad:
        raise SystemExit(f"CACHE HASH CHECK FAILED: {bad}")
    os.makedirs("/kaggle/working", exist_ok=True)
    print("SETUP OK", flush=True)


try:
    main()
finally:
    for t in (TOKEN, "/content/root/.kaggle/access_token"):  # the second only if an upload path was resolved under /content
        if os.path.exists(t):
            os.remove(t)
    print("token file on VM removed:", not any(os.path.exists(t) for t in (TOKEN, "/content/root/.kaggle/access_token")), flush=True)
