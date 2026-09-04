"""Shared GridSpec used by every module (DEM heightfield, land-cover masks,
trail ribbon) so all vertex positions line up exactly without independent
resampling logic in each module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import Point, Polygon

from .gpx import UtmBBox


@dataclass(frozen=True)
class GridSpec:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    ncols: int  # number of cells across (x)
    nrows: int  # number of cells across (y)
    utm_epsg: int
    mm_per_meter: float  # print-space horizontal scale factor

    @property
    def cell_size_x(self) -> float:
        return (self.max_x - self.min_x) / self.ncols

    @property
    def cell_size_y(self) -> float:
        return (self.max_y - self.min_y) / self.nrows

    @property
    def n_vertex_rows(self) -> int:
        return self.nrows + 1

    @property
    def n_vertex_cols(self) -> int:
        return self.ncols + 1

    def vertex_xy_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """X, Y vertex coordinates in print-space millimeters, shape
        (n_vertex_rows, n_vertex_cols). Row 0 = geographic north.
        """
        height_m = self.max_y - self.min_y
        xs = (np.arange(self.n_vertex_cols) * self.cell_size_x) * self.mm_per_meter
        ys = (height_m - np.arange(self.n_vertex_rows) * self.cell_size_y) * self.mm_per_meter
        return np.meshgrid(xs, ys)

    def vertex_transform(self) -> Affine:
        """Affine mapping array (col,row) -> UTM coords, with pixel CENTERS
        landing exactly on grid vertices (min_x + i*cs_x, max_y - j*cs_y).
        """
        cs_x, cs_y = self.cell_size_x, self.cell_size_y
        return Affine(cs_x, 0, self.min_x - cs_x / 2, 0, -cs_y, self.max_y + cs_y / 2)


def build_grid_spec(working_bbox_utm: UtmBBox, utm_epsg: int, size_mm: float, mm_per_cell: float) -> GridSpec:
    width_m = working_bbox_utm.width
    height_m = working_bbox_utm.height
    longest_m = max(width_m, height_m)
    mm_per_meter = size_mm / longest_m
    cells_across_longest = size_mm / mm_per_cell
    ncols = max(4, round(width_m / longest_m * cells_across_longest))
    nrows = max(4, round(height_m / longest_m * cells_across_longest))
    return GridSpec(
        min_x=working_bbox_utm.min_x,
        min_y=working_bbox_utm.min_y,
        max_x=working_bbox_utm.max_x,
        max_y=working_bbox_utm.max_y,
        ncols=ncols,
        nrows=nrows,
        utm_epsg=utm_epsg,
        mm_per_meter=mm_per_meter,
    )


def hexagon_polygon(center_x: float, center_y: float, circumradius: float) -> Polygon:
    """A regular, "pointy-top" hexagon (a vertex points along +x), so its
    horizontal point-to-point width is exactly 2*circumradius - matching how
    size_mm is used as a diameter for the rectangular footprint."""
    angles = np.deg2rad(np.arange(6) * 60.0)
    coords = [(center_x + circumradius * np.cos(a), center_y + circumradius * np.sin(a)) for a in angles]
    return Polygon(coords)


def circle_polygon(center_x: float, center_y: float, radius: float) -> Polygon:
    """A circular footprint whose diameter is 2*radius - sized the same way
    hexagon_polygon's circumradius is, so size_mm stays the model's longest
    horizontal dimension whichever shape is chosen."""
    return Point(center_x, center_y).buffer(radius, quad_segs=48)


def rasterize_mask(grid: GridSpec, geometry) -> np.ndarray:
    """Rasterize a shapely geometry (already in the grid's UTM CRS) onto grid
    vertices as a boolean mask of shape (n_vertex_rows, n_vertex_cols)."""
    shape = (grid.n_vertex_rows, grid.n_vertex_cols)
    if geometry is None or geometry.is_empty:
        return np.zeros(shape, dtype=bool)
    mask = rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=grid.vertex_transform(),
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return mask.astype(bool)


def clean_mask(mask: np.ndarray, min_component_vertices: int = 6) -> np.ndarray:
    """Morphological cleanup for print quality: close small gaps, remove
    single-vertex pinches, drop dust-sized connected components."""
    if not mask.any():
        return mask
    structure = np.ones((3, 3), dtype=bool)
    closed = ndimage.binary_closing(mask, structure=structure)
    opened = ndimage.binary_opening(closed, structure=structure)
    labeled, n = ndimage.label(opened, structure=structure)
    if n == 0:
        return opened
    sizes = ndimage.sum(opened, labeled, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_component_vertices
    return keep[labeled]


def smooth_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Round off blocky, grid-cell-sized corners on a region boundary: blur
    the mask as a float field and rethreshold at 0.5. `sigma` is in grid
    cells; 0 (or no True cells) is a no-op. Applied per-layer, after merging
    source classes but before clean_mask's dust/pinch cleanup, so smoothing
    only ever changes the boundary's shape, not its topology-fixing role.
    """
    if sigma <= 0 or not mask.any():
        return mask
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return blurred > 0.5


def resolve_overlaps(masks: dict[str, np.ndarray], priority: list[str]) -> dict[str, np.ndarray]:
    """Apply an explicit priority order so overlapping categories (e.g. trail
    crossing forest) don't produce interpenetrating overlay shells."""
    any_mask = next((m for m in masks.values() if m is not None), None)
    if any_mask is None:
        return dict(masks)
    claimed = np.zeros_like(any_mask, dtype=bool)
    resolved: dict[str, np.ndarray] = {}
    for name in priority:
        m = masks.get(name)
        if m is None:
            continue
        m2 = m & ~claimed
        resolved[name] = m2
        claimed = claimed | m2
    for name, m in masks.items():
        if name not in resolved:
            resolved[name] = m
    return resolved
