"""Colored overlay shells (water/forest/ice/trail): thin shells built at the
same vertex grid as the base terrain (mesh.terrain), inlaid into it rather
than stacked on top - the top sits `rise_mm` above the surrounding terrain
height (0 for water/forest/ice, so they read flush; positive for the trail,
so it alone reads as physically raised) and the bottom sits `inlay_depth_mm`
below it, matching the pocket notched into the base's own top surface for
that same footprint (see mesh.terrain) so the two parts sit flush with no
CSG boolean operation needed.
"""
from __future__ import annotations

import numpy as np
import trimesh

from ..grid import GridSpec
from .common import build_region_solid


def build_overlay_mesh(
    grid: GridSpec,
    z_mm: np.ndarray,
    vertex_mask: np.ndarray,
    name: str,
    rise_mm: float = 0.0,
    inlay_depth_mm: float = 0.8,
) -> trimesh.Trimesh | None:
    """z_mm, vertex_mask: shape (n_vertex_rows, n_vertex_cols).

    Returns None if the mask selects no complete face (nothing to build).
    """
    top_z_mm = z_mm + rise_mm if rise_mm else z_mm
    bottom_z_mm = z_mm - inlay_depth_mm
    return build_region_solid(grid, top_z_mm, bottom_z_mm, vertex_mask, name=name)
