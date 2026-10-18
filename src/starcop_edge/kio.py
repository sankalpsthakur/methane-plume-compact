"""Kaggle input discovery (the 2026 image mounts datasets under /kaggle/input/datasets/<owner>/<slug>/...)."""
import os


def find_dir(marker, base="/kaggle/input", max_depth=5, skip_prefixes=("STARCOP_",)):
    """First directory under `base` that directly contains a file/dir called `marker` (does not descend into
    the big STARCOP mirror archive folders)."""
    base = os.path.abspath(base)
    for root, dirs, files in os.walk(base):
        if marker in dirs or marker in files:
            return root
        depth = root[len(base):].count(os.sep)
        dirs[:] = sorted(d for d in dirs if not d.startswith(skip_prefixes)) if depth < max_depth else []
    return None
