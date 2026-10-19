# Wave 3 (Amendment 6), inside the Colab VM: start one generated kernel script (JOB = kernel name) as a detached process, so it survives
# the notebook kernel; outputs go to /kaggle/working exactly as on Kaggle. A stale RUN_STATUS.json of an earlier job is removed first.
import os, subprocess
job = os.environ["JOB"]
d = f"/content/w3/{job}"
assert os.path.isfile(f"{d}/script.py"), f"missing {d}/script.py"
os.makedirs("/kaggle/working", exist_ok=True)
if os.path.exists("/kaggle/working/RUN_STATUS.json"):
    os.remove("/kaggle/working/RUN_STATUS.json")
p = subprocess.Popen(["bash", "-c", f"cd {d} && exec python3 script.py > {d}/driver.out 2>&1"], start_new_session=True)
open(f"{d}/driver.pid", "w").write(str(p.pid))
print("STARTED", job, "pid", p.pid)
