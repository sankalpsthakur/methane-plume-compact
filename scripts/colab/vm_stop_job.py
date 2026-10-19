# Wave 3 (Amendment 6, note 1 b), inside the Colab VM: stop one job (JOB = kernel name) on a throughput-only projection.
# vm_start.py started the driver as the leader of its own process group, so SIGTERM reaches the training processes too.
import os, signal
job = os.environ["JOB"]
pid = int(open(f"/content/w3/{job}/driver.pid").read())
try:
    os.killpg(pid, signal.SIGTERM)
    print("SIGTERM sent to process group", pid)
except ProcessLookupError:
    print("process group", pid, "already gone")
