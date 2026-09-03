"""Shared helper for querying ArcGIS REST FeatureServer/MapServer `query`
endpoints (used by hydrography.py, transportation.py, landmarks.py) - no API
key, geojson in, retried with backoff, and treating an ArcGIS-style "HTTP 200
with an error embedded in the JSON body" response as the failure it actually
is rather than silently reading it as zero features.
"""
from __future__ import annotations

import time

import requests

_NAME_FIELDS = ("gnis_name", "gaz_name", "Name", "NAME", "name")


def feature_name(properties: dict) -> str:
    for key in _NAME_FIELDS:
        val = properties.get(key)
        if val:
            return str(val)
    return ""


def query_layer(
    base_url: str,
    layer: int,
    bbox_latlon: tuple[float, float, float, float],
    timeout: float = 30.0,
    retries: int = 3,
) -> list[dict]:
    """GeoJSON features from one ArcGIS REST layer intersecting a lon/lat bbox."""
    minx, miny, maxx, maxy = bbox_latlon
    params = {
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "geojson",
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(f"{base_url}/{layer}/query", params=params, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            # ArcGIS REST services often respond HTTP 200 with an error
            # embedded in the JSON body rather than a 4xx/5xx status - treat
            # that as a real failure (and retry/fall back), not "nothing
            # here", or an outage silently renders as an empty map.
            if "error" in body:
                raise RuntimeError(f"{base_url} layer {layer} returned an error: {body['error']}")
            return body.get("features", [])
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            last_err = e
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to query {base_url} layer {layer} after {retries} attempts") from last_err
