# Wave 3 (Amendment 6), inside the Colab VM: confirm the kernel scripts landed where the runner put them (JOBS = comma-separated kernel
# names) BEFORE the runner uploads the Kaggle token to an absolute path.
import os
jobs = [j for j in os.environ.get("JOBS", "").split(",") if j]
print("UPLOADS", bool(jobs) and all(os.path.isfile(f"/content/w3/{j}/script.py") for j in jobs), jobs)
