"""Base terrain solid: a heightmap surface - optionally clipped to an
arbitrary footprint mask (e.g. a hexagon) instead of the full rectangular
grid - closed with boundary skirt walls and a flat bottom cap anchored below
the lowest terrain point.

Wherever a colored overlay (water/forest/ice/trail/...) sits, the base's top
is "notched": recessed by `pocket_depth_mm` so the overlay sits inlaid
rather than stacked on top of a full-height base - only the overlay's own
rise above that (e.g. the trail) reads as physically raised. Symmetrically,
the flat bottom cap can be notched too (raised locally by
`bottom_pocket_depth_mm`) for an underside text plaque (see plaque.py) to
sit flush in.
"""
from __future__ import annotations

import numpy as np
import trimesh

from ..grid import GridSpec
from .common import build_region_solid


def build_terrain_mesh(
    grid: GridSpec,
    z_mm: np.ndarray,
    base_thickness_mm: float,
    footprint_mask: np.ndarray | None = None,
    pocket_vertex_mask: np.ndarray | None = None,
    pocket_depth_mm: float = 0.0,
    bottom_pocket_vertex_mask: np.ndarray | None = None,
    bottom_pocket_depth_mm: float = 0.0,
) -> trimesh.Trimesh:
    """z_mm: heightfield of shape (n_vertex_rows, n_vertex_cols), already in
    print-space millimeters (see mesh.common.heights_to_mm), all >= 0.

    footprint_mask: optional vertex mask restricting the base to a shape
    other than the full rectangular grid (e.g. a hexagon); defaults to the
    full grid, giving the original rectangular footprint.

    pocket_vertex_mask: vertices where the top surface is lowered by
    pocket_depth_mm, so an overlay built at the same vertex positions (see
    mesh.overlay) sits flush with the surrounding surface instead of on top
    of it. Should be the union of every overlay category's actual (post
    diagonal-pinch-fix) footprint - see mesh.common.resolve_face_inclusion /
    vertex_mask_from_face_inclusion - so no pocket is left uncovered.

    bottom_pocket_vertex_mask / bottom_pocket_depth_mm: the same idea
    applied to the flat bottom cap instead of the terrain surface - raised
    (made less negative) locally so an underside text plaque sits flush in
    the floor instead of poking out below it.
    """
    if footprint_mask is None:
        footprint_mask = np.ones_like(z_mm, dtype=bool)

    top_z = z_mm
    if pocket_vertex_mask is not None and pocket_depth_mm > 0:
        top_z = z_mm.copy()
        top_z[pocket_vertex_mask] -= pocket_depth_mm

    floor_z = -base_thickness_mm
    bottom_z: np.ndarray | float = floor_z
    if bottom_pocket_vertex_mask is not None and bottom_pocket_depth_mm > 0:
        bottom_z = np.full_like(z_mm, floor_z)
        bottom_z[bottom_pocket_vertex_mask] += bottom_pocket_depth_mm

    mesh = build_region_solid(grid, top_z, bottom_z, footprint_mask, name="base")
    if mesh is None:
        raise ValueError(
            "Base terrain footprint selects no area - check size_mm / mm_per_cell / footprint shape"
        )
    return mesh
