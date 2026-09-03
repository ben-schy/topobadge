"""USGS National Land Cover Database (NLCD): a 30m-resolution land-cover
classification raster, used here for vegetation (forest, shrub, grassland,
pasture, wetland - see FOREST_CLASSES) and permanent ice/snow. Fetched via
MRLC's WCS for the grid's working area, then resampled with
nearest-neighbor (never bilinear - this is categorical data, not a
continuous field like elevation) onto the exact mesh grid.

US-only, same as the elevation source (USGS 3DEP) - not a new limitation.
"""
from __future__ import annotations

import io
import time

import numpy as np
import pyproj
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .grid import GridSpec

_WCS_URL = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
_COVERAGE_ID = "mrlc_download__NLCD_2021_Land_Cover_L48"

# NLCD legend codes: https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description
# Each is its own independently-assignable source class (see pipeline.py's
# layer_assignment) rather than a single hardcoded "forest" bucket - so
# grassland, say, can be sent to a different color (or hidden into the base)
# without dragging trees along with it.
SOURCE_CLASSES: dict[str, tuple[int, ...]] = {
    "trees": (41, 42, 43),  # deciduous, evergreen, mixed forest
    "shrub": (52,),  # shrub/scrub
    "grassland": (71,),  # grassland/herbaceous
    "pasture_crops": (81, 82),  # pasture/hay, cultivated crops
    "wetland": (90, 95),  # woody wetlands, emergent herbaceous wetlands
    "ice": (12,),  # perennial ice/snow
    "water_nlcd": (11,),  # open water, per NLCD - blended alongside the vector water sources (hydrography.py)
}


def fetch_nlcd_classes(grid: GridSpec, timeout: float = 30.0, retries: int = 3) -> np.ndarray:
    """NLCD class code at every grid vertex, shape (n_vertex_rows, n_vertex_cols).
    Code 0 (or any code outside the legend) means outside NLCD's Lower-48 coverage.
    """
    # Query in the grid's own lat/lon bounding box; the response keeps its
    # native georeferencing, which we then resample onto the exact UTM grid
    # ourselves (WCS doesn't offer the same "give me exactly this pixel
    # grid" export that 3DEP's ImageServer does).
    to_lonlat = pyproj.Transformer.from_crs(f"EPSG:{grid.utm_epsg}", "EPSG:4326", always_xy=True)
    lons, lats = zip(
        *[
            to_lonlat.transform(x, y)
            for x, y in [(grid.min_x, grid.min_y), (grid.max_x, grid.max_y), (grid.min_x, grid.max_y), (grid.max_x, grid.min_y)]
        ]
    )
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": _COVERAGE_ID,
        "subset": [f"Lat({min_lat},{max_lat})", f"Long({min_lon},{max_lon})"],
        "subsettingCrs": "http://www.opengis.net/def/crs/EPSG/0/4326",
        "format": "image/geotiff",
    }

    last_err: Exception | None = None
    content: bytes | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(_WCS_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            content = resp.content
            break
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            last_err = e
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    if content is None:
        raise RuntimeError(f"Failed to fetch NLCD coverage after {retries} attempts") from last_err

    shape = (grid.n_vertex_rows, grid.n_vertex_cols)
    dest = np.zeros(shape, dtype=np.uint8)
    with rasterio.open(io.BytesIO(content)) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.vertex_transform(),
            dst_crs=f"EPSG:{grid.utm_epsg}",
            resampling=Resampling.nearest,
        )
    return dest


def source_class_masks(classes: np.ndarray) -> dict[str, np.ndarray]:
    """One boolean vertex mask per NLCD-derived source class (see
    SOURCE_CLASSES), for the caller to group into output layers however it
    likes (pipeline.compose_masks)."""
    return {name: np.isin(classes, codes) for name, codes in SOURCE_CLASSES.items()}
