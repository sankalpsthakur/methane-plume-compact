"""TIFF decoding via rasterio (GDAL), the same reader the official STARCOP dataset class uses."""
import numpy as np
import rasterio


def read_tif(raw: bytes) -> np.ndarray:
    """raw TIFF bytes -> (H,W) array for single-band files, (C,H,W) otherwise (dtype as stored)."""
    with rasterio.MemoryFile(raw) as mf:
        with mf.open() as ds:
            a = ds.read()
    return a[0] if a.shape[0] == 1 else a
