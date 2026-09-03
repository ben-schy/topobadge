"""DEM fetch from the USGS 3DEP ImageServer, pre-aligned to a GridSpec.

The exportImage request is built so the server reprojects and resamples for
us: bbox/bboxSR/imageSR are all in the working UTM CRS and `size` matches the
GridSpec's vertex grid exactly, so the returned array can be used directly as
a heightfield with no local resampling.
"""
from __future__ import annotations

import io
import time

import numpy as np
import requests
import tifffile

from .grid import GridSpec

EXPORT_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"

# 3DEP bare-earth coverage is roughly -100m (below sea level basins) to ~6200m
# (highest US peaks); anything outside this is treated as missing data.
_PLAUSIBLE_MIN_M = -100.0
_PLAUSIBLE_MAX_M = 6500.0


def fetch_dem(grid: GridSpec, timeout: float = 60.0, retries: int = 3) -> np.ndarray:
    """Fetch elevation (meters) at every grid vertex.

    Returns an array of shape (grid.n_vertex_rows, grid.n_vertex_cols).
    """
    params = {
        "bbox": f"{grid.min_x},{grid.min_y},{grid.max_x},{grid.max_y}",
        "bboxSR": grid.utm_epsg,
        "imageSR": grid.utm_epsg,
        "size": f"{grid.n_vertex_cols},{grid.n_vertex_rows}",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    expected_shape = (grid.n_vertex_rows, grid.n_vertex_cols)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(EXPORT_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("application/json"):
                raise RuntimeError(f"3DEP service returned an error payload: {resp.text[:300]}")
            arr = np.asarray(tifffile.imread(io.BytesIO(resp.content)), dtype=np.float64)
            if arr.shape != expected_shape:
                raise ValueError(f"Unexpected DEM shape {arr.shape}, expected {expected_shape}")
            return _fill_nodata(arr)
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            last_err = e
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch DEM from 3DEP after {retries} attempts") from last_err


def _fill_nodata(arr: np.ndarray) -> np.ndarray:
    """Replace implausible/no-data pixels with the nearest valid value."""
    invalid = ~np.isfinite(arr) | (arr < _PLAUSIBLE_MIN_M) | (arr > _PLAUSIBLE_MAX_M)
    if not invalid.any():
        return arr
    if invalid.all():
        raise RuntimeError("DEM fetch returned no valid elevation data for this area")
    from scipy import ndimage

    idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    filled = arr[tuple(idx)]
    return filled
