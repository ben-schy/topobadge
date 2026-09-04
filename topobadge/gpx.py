"""GPX parsing, working-area bounding box, and CRS selection.

All downstream modules (elevation, landcover, grid, mesh) work in one shared
projected CRS chosen here (UTM, auto-selected from the track's centroid) so
that DEM pixels, land-cover polygons, and the trail line all land in the same
coordinate space without repeated reprojection.
"""
from __future__ import annotations

from dataclasses import dataclass

import gpxpy
import pyproj
from shapely.geometry import MultiLineString, Polygon, box
from shapely.ops import transform as shapely_transform


@dataclass(frozen=True)
class UtmBBox:
    """A bounding box in a projected (UTM) CRS, units = meters."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    def squared(self) -> "UtmBBox":
        """Expand the shorter dimension to match the longer one, keeping the
        same center - used for non-rectangular output shapes (e.g. a
        hexagon) so the shape isn't clipped by a narrower fetch extent."""
        cx, cy = self.center
        half = max(self.width, self.height) / 2
        return UtmBBox(cx - half, cy - half, cx + half, cy + half)

    def buffered(self, buffer_m: float) -> "UtmBBox":
        return UtmBBox(
            min_x=self.min_x - buffer_m,
            min_y=self.min_y - buffer_m,
            max_x=self.max_x + buffer_m,
            max_y=self.max_y + buffer_m,
        )

    def polygon(self) -> Polygon:
        return box(self.min_x, self.min_y, self.max_x, self.max_y)


@dataclass
class Track:
    source_label: str  # human-readable description, for logging only
    utm_epsg: int
    utm_crs: pyproj.CRS
    lines_latlon: MultiLineString  # raw track, EPSG:4326, (lon, lat)
    lines_utm: MultiLineString  # raw track, working UTM CRS
    track_bbox_utm: UtmBBox  # tight bbox around the track only, in UTM
    working_bbox_utm: UtmBBox  # track bbox buffered by buffer_km, in UTM
    working_polygon_latlon: Polygon  # working_bbox_utm reprojected to EPSG:4326, for OSM queries


def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _densified_box_coords(bbox: UtmBBox, points_per_edge: int = 16) -> list[tuple[float, float]]:
    """Corners of the bbox plus points along each edge, so reprojecting a
    large box to lon/lat (where UTM->WGS84 is not affine) stays a good
    approximation of the true buffered area rather than just its 4 corners.
    """
    xs = [bbox.min_x + (bbox.max_x - bbox.min_x) * i / points_per_edge for i in range(points_per_edge + 1)]
    ys = [bbox.min_y + (bbox.max_y - bbox.min_y) * i / points_per_edge for i in range(points_per_edge + 1)]
    coords = []
    coords += [(x, bbox.min_y) for x in xs]
    coords += [(bbox.max_x, y) for y in ys]
    coords += [(x, bbox.max_y) for x in reversed(xs)]
    coords += [(bbox.min_x, y) for y in reversed(ys)]
    return coords


def load_track(gpx_path: str, buffer_km: float) -> Track:
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    lines: list[list[tuple[float, float]]] = []
    all_points: list[tuple[float, float]] = []

    for trk in gpx.tracks:
        for seg in trk.segments:
            pts = [(p.longitude, p.latitude) for p in seg.points]
            if len(pts) >= 2:
                lines.append(pts)
            all_points.extend(pts)

    if not lines:
        for route in gpx.routes:
            pts = [(p.longitude, p.latitude) for p in route.points]
            if len(pts) >= 2:
                lines.append(pts)
            all_points.extend(pts)

    if not lines:
        raise ValueError(f"No track or route segments with >= 2 points found in {gpx_path}")

    lons = [p[0] for p in all_points]
    lats = [p[1] for p in all_points]
    centroid_lon = (min(lons) + max(lons)) / 2
    centroid_lat = (min(lats) + max(lats)) / 2
    utm_epsg = _utm_epsg_for_lonlat(centroid_lon, centroid_lat)
    utm_crs = pyproj.CRS.from_epsg(utm_epsg)

    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_lonlat = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    lines_latlon = MultiLineString(lines)
    lines_utm = shapely_transform(to_utm.transform, lines_latlon)

    txs, tys = zip(*[to_utm.transform(lon, lat) for lon, lat in all_points])
    track_bbox_utm = UtmBBox(min(txs), min(tys), max(txs), max(tys))
    working_bbox_utm = track_bbox_utm.buffered(buffer_km * 1000.0)

    ring_utm = _densified_box_coords(working_bbox_utm)
    ring_latlon = [to_lonlat.transform(x, y) for x, y in ring_utm]
    working_polygon_latlon = Polygon(ring_latlon)

    return Track(
        source_label=gpx_path,
        utm_epsg=utm_epsg,
        utm_crs=utm_crs,
        lines_latlon=lines_latlon,
        lines_utm=lines_utm,
        track_bbox_utm=track_bbox_utm,
        working_bbox_utm=working_bbox_utm,
        working_polygon_latlon=working_polygon_latlon,
    )


def load_point_area(lat: float, lon: float, buffer_km: float) -> Track:
    """Build a synthetic Track centered on a bare lat/lon point instead of a
    GPX line - for the map-picker starting mode. Everything downstream
    (fetch_stage, compose_masks, mesh_stage) only ever reads the working
    bbox/CRS, except the "trail" mask which reads lines_utm - left empty
    here, so a point-area build simply has no trail layer.
    """
    utm_epsg = _utm_epsg_for_lonlat(lon, lat)
    utm_crs = pyproj.CRS.from_epsg(utm_epsg)

    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_lonlat = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    cx, cy = to_utm.transform(lon, lat)
    track_bbox_utm = UtmBBox(cx, cy, cx, cy)
    working_bbox_utm = track_bbox_utm.buffered(buffer_km * 1000.0)

    ring_utm = _densified_box_coords(working_bbox_utm)
    ring_latlon = [to_lonlat.transform(x, y) for x, y in ring_utm]
    working_polygon_latlon = Polygon(ring_latlon)

    empty_lines = MultiLineString([])

    return Track(
        source_label=f"{lat:.5f}, {lon:.5f}",
        utm_epsg=utm_epsg,
        utm_crs=utm_crs,
        lines_latlon=empty_lines,
        lines_utm=empty_lines,
        track_bbox_utm=track_bbox_utm,
        working_bbox_utm=working_bbox_utm,
        working_polygon_latlon=working_polygon_latlon,
    )
