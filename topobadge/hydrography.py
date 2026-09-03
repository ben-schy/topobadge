"""Water features (standing water and streams/rivers), queried directly from
government/public ArcGIS REST services via arcgis.query_layer. No API key,
and no geometry library beyond shapely/pyproj (already dependencies).

Two independent sources, tried in order, so an outage on one (this has
happened) doesn't take water off the map entirely:

1. USGS National Hydrography Dataset (NHD) - the primary source: most
   complete, includes minor tributaries.
2. Esri Living Atlas "USA Detailed Water Bodies" / "USA Rivers and Streams" -
   hosted on entirely separate infrastructure from USGS, so it stays up
   through a USGS-side outage. Less exhaustive (fewer small streams) but a
   real, independent fallback rather than nothing.

US-only, same as the elevation source (USGS 3DEP) - not a new limitation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pyproj
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from .arcgis import feature_name, query_layer
from .gpx import Track

_NHD_BASE = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
_NHD_WATERBODY_LAYER = 12  # "Waterbody - Large Scale"
_NHD_FLOWLINE_LAYER = 6  # "Flowline - Large Scale"

_ESRI_WATERBODY_URL = "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Detailed_Water_Bodies/FeatureServer"
_ESRI_STREAMS_URL = "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Rivers_and_Streams/FeatureServer"


@dataclass
class WaterFeatures:
    water: BaseGeometry | None  # lakes/ponds/wetlands (polygons)
    rivers: BaseGeometry | None  # flowlines named "River" - wider
    streams: BaseGeometry | None  # all other flowlines (creeks, unnamed, etc.) - narrower


def _features_to_water(waterbody_features: list[dict], flowline_features: list[dict], to_utm) -> WaterFeatures:
    water = None
    if waterbody_features:
        geoms = [shapely_transform(to_utm, shape(f["geometry"])) for f in waterbody_features]
        water = unary_union(geoms).buffer(0)

    river_geoms: list = []
    stream_geoms: list = []
    for f in flowline_features:
        name = feature_name(f.get("properties") or {})
        geom = shapely_transform(to_utm, shape(f["geometry"]))
        (river_geoms if "river" in name.lower() else stream_geoms).append(geom)

    rivers = unary_union(river_geoms) if river_geoms else None
    streams = unary_union(stream_geoms) if stream_geoms else None
    return WaterFeatures(water=water, rivers=rivers, streams=streams)


def _fetch_nhd(bbox, to_utm, timeout: float, retries: int) -> WaterFeatures:
    waterbody = query_layer(_NHD_BASE, _NHD_WATERBODY_LAYER, bbox, timeout, retries)
    flowline = query_layer(_NHD_BASE, _NHD_FLOWLINE_LAYER, bbox, timeout, retries)
    return _features_to_water(waterbody, flowline, to_utm)


def _fetch_esri(bbox, to_utm, timeout: float, retries: int) -> WaterFeatures:
    waterbody = query_layer(_ESRI_WATERBODY_URL, 0, bbox, timeout, retries)
    flowline = query_layer(_ESRI_STREAMS_URL, 0, bbox, timeout, retries)
    return _features_to_water(waterbody, flowline, to_utm)


def fetch_water_features(
    track: Track,
    timeout: float = 30.0,
    retries: int = 3,
    on_log: Callable[[str], None] = lambda s: None,
) -> WaterFeatures:
    bbox = track.working_polygon_latlon.bounds
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", track.utm_crs, always_xy=True).transform

    try:
        result = _fetch_nhd(bbox, to_utm, timeout, retries)
        on_log("  water source: USGS NHD")
        return result
    except Exception as nhd_err:  # noqa: BLE001 - fall back to an independent source
        on_log(f"  USGS NHD unavailable ({nhd_err}); trying Esri Living Atlas fallback...")
        try:
            result = _fetch_esri(bbox, to_utm, timeout, retries)
            on_log("  water source: Esri Living Atlas (fallback)")
            return result
        except Exception as esri_err:  # noqa: BLE001
            raise RuntimeError(
                f"Both water sources failed - NHD: {nhd_err}; Esri Living Atlas: {esri_err}"
            ) from esri_err
