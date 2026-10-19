# Wave 3 (Amendment 6), inside the Colab VM: state of one job (JOB = kernel name). Prints PROGRESS lines and a final JOBSTATE=running|done|dead.
import json, os
job = os.environ["JOB"]
d = f"/content/w3/{job}"
pid = int(open(f"{d}/driver.pid").read())
state = "gone"
try:
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("State:"):
            state = line.split()[1]
except FileNotFoundError:
    pass
alive = state not in ("gone", "Z", "X")
rs = None
if os.path.exists("/kaggle/working/RUN_STATUS.json"):
    rs = json.load(open("/kaggle/working/RUN_STATUS.json"))
runs_dir = [n for n in os.listdir("/kaggle/working") if os.path.isfile(f"/kaggle/working/{n}/log.jsonl")]
for n in runs_dir:
    last = None
    for line in open(f"/kaggle/working/{n}/log.jsonl"):
        if '"event": "train"' in line or '"event":"train"' in line:
            last = line
    if last:
        e = json.loads(last)
        print(f"PROGRESS {n} step {e.get('step')} loss {e.get('loss')} lr {e.get('lr')}")
tail = open(f"{d}/driver.out", errors="replace").read()[-600:].replace("\n", " | ") if os.path.exists(f"{d}/driver.out") else ""
print("DRIVER_TAIL", tail[-400:])
if alive:
    print("JOBSTATE=running")
elif rs is not None and not rs.get("partial"):
    for n, st in (rs.get("runs") or {}).items():
        for fn in ("val_metrics.json", "test_metrics.json"):
            if os.path.exists(f"/kaggle/working/{n}/{fn}"):
                print("METRICS", n, fn, "rc", st.get("rc"), json.dumps(json.load(open(f"/kaggle/working/{n}/{fn}"))))
    print("JOBSTATE=done", json.dumps(rs.get("runs")))
else:
    print("JOBSTATE=dead", json.dumps(rs) if rs else "(no RUN_STATUS.json)")
