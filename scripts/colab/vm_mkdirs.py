# Wave 3 (Amendment 6), inside the Colab VM: directories the uploads go to. JOBS = comma-separated kernel names.
import os
os.makedirs("/root/.kaggle", mode=0o700, exist_ok=True)
for job in os.environ.get("JOBS", "").split(","):
    if job:
        os.makedirs(f"/content/w3/{job}", exist_ok=True)
print("dirs ready:", sorted(os.listdir("/content/w3")))
