"""Named landmarks (summits, passes, notable landforms) from the USGS
Geographic Names Information System (GNIS), via the same reliable ArcGIS
REST infrastructure as hydrography.py/transportation.py.

US-only, same as the elevation source (USGS 3DEP) - not a new limitation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyproj
from shapely.geometry import Point, shape
from shapely.ops import transform as shapely_transform

from .arcgis import feature_name, query_layer
from .gpx import Track

_BASE = "https://carto.nationalmap.gov/arcgis/rest/services/geonames/MapServer"
_LANDFORMS_LAYER = 5  # summits, passes, valleys, pillars, ridges, etc.


@dataclass
class NamedPoint:
    name: str
    point: Point  # in the track's UTM CRS


def fetch_landmarks(track: Track, timeout: float = 30.0, retries: int = 3) -> list[NamedPoint]:
    bbox = track.working_polygon_latlon.bounds
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", track.utm_crs, always_xy=True).transform

    points: list[NamedPoint] = []
    for f in query_layer(_BASE, _LANDFORMS_LAYER, bbox, timeout, retries):
        name = feature_name(f.get("properties") or {})
        geom = shapely_transform(to_utm, shape(f["geometry"]))
        points.append(NamedPoint(name=name or "Unnamed", point=geom))
    return points
