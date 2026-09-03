"""Roads and trails other than the user's own uploaded GPX route, from
USGS's National Map Transportation service - same reliable ArcGIS REST
infrastructure as hydrography.py and landcover.py.

US-only, same as the elevation source (USGS 3DEP) - not a new limitation.
"""
from __future__ import annotations

import pyproj
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from .arcgis import query_layer
from .gpx import Track

_BASE = "https://carto.nationalmap.gov/arcgis/rest/services/transportation/MapServer"

# Deliberately skipping interstates/controlled-access highways (layer 29) and
# secondary highways (30) as irrelevant to a hiking map; these are the road
# classes actually likely to be near a trailhead.
_ROAD_LAYERS = (31, 32, 35)  # Local Connecting Roads, Local Roads, 4WD Roads
_TRAIL_LAYER = 37  # "Trails" (distinct from the user's own GPX route)


def _fetch_lines(layers: tuple[int, ...], bbox_latlon, to_utm, timeout: float, retries: int) -> BaseGeometry | None:
    geoms = []
    for layer in layers:
        for f in query_layer(_BASE, layer, bbox_latlon, timeout, retries):
            geoms.append(shapely_transform(to_utm, shape(f["geometry"])))
    return unary_union(geoms) if geoms else None


def fetch_roads(track: Track, timeout: float = 30.0, retries: int = 3) -> BaseGeometry | None:
    bbox = track.working_polygon_latlon.bounds
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", track.utm_crs, always_xy=True).transform
    return _fetch_lines(_ROAD_LAYERS, bbox, to_utm, timeout, retries)


def fetch_other_trails(track: Track, timeout: float = 30.0, retries: int = 3) -> BaseGeometry | None:
    bbox = track.working_polygon_latlon.bounds
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", track.utm_crs, always_xy=True).transform
    return _fetch_lines((_TRAIL_LAYER,), bbox, to_utm, timeout, retries)
