"""Shared helpers for mesh construction: consistent height scaling, and a
general "restricted region solid" builder used by both the base terrain and
the colored overlay shells, so every part ends up in the same coordinate
space and shares the same manifold-construction logic.
"""
from __future__ import annotations

import numpy as np
import trimesh

from ..grid import GridSpec


def heights_to_mm(
    elev_m: np.ndarray,
    mm_per_meter: float,
    vertical_exaggeration: float,
    min_elev_m: float,
) -> np.ndarray:
    """Elevation (meters) -> print-space Z (millimeters).

    The same horizontal mm_per_meter scale is applied here so x/y/z are never
    scaled independently of each other.
    """
    return (elev_m - min_elev_m) * mm_per_meter * vertical_exaggeration


def _break_diagonal_pinches(face_included: np.ndarray) -> np.ndarray:
    """A face only ever has 4 orthogonal neighbors, but two faces diagonally
    across a 2x2 block CAN still both be included while their two orthogonal
    neighbors are not - the classic checkerboard case. The two included faces
    then touch only at their single shared corner vertex, which breaks
    manifoldness (a non-manifold vertex/pinch) rather than being merely a
    print-quality wrinkle. Resolve each such block by dropping one of the two
    diagonal faces, iterating until no pinch remains (clearing faces only, so
    this always terminates).
    """
    result = face_included.copy()
    for _ in range(10):
        tl, tr = result[:-1, :-1], result[:-1, 1:]
        bl, br = result[1:, :-1], result[1:, 1:]
        pinch_tl_br = tl & br & ~tr & ~bl
        pinch_tr_bl = tr & bl & ~tl & ~br
        if not (pinch_tl_br.any() or pinch_tr_bl.any()):
            break
        clear = np.zeros_like(result)
        clear[1:, 1:] |= pinch_tl_br  # drop the bottom-right face of a "\" pinch
        clear[1:, :-1] |= pinch_tr_bl  # drop the bottom-left face of a "/" pinch
        result = result & ~clear
    return result


def resolve_face_inclusion(vertex_mask: np.ndarray) -> np.ndarray:
    """A face is included only if all 4 corner vertices are masked, with
    diagonal/"checkerboard" pinches broken (see _break_diagonal_pinches).
    This is the actual, final footprint a region solid will be built from -
    exposed so callers can precompute it (e.g. to notch another part's
    surface to match) without duplicating this logic.
    """
    face_included = (
        vertex_mask[:-1, :-1] & vertex_mask[:-1, 1:] & vertex_mask[1:, :-1] & vertex_mask[1:, 1:]
    )
    return _break_diagonal_pinches(face_included)


def vertex_mask_from_face_inclusion(face_included: np.ndarray) -> np.ndarray:
    """Vertices that are a corner of at least one included face - i.e. the
    face-level mask "dilated" back out to vertex granularity."""
    nrows, ncols = face_included.shape
    vmask = np.zeros((nrows + 1, ncols + 1), dtype=bool)
    vmask[:-1, :-1] |= face_included
    vmask[:-1, 1:] |= face_included
    vmask[1:, :-1] |= face_included
    vmask[1:, 1:] |= face_included
    return vmask


def build_region_solid(
    grid: GridSpec,
    top_z_mm: np.ndarray,
    bottom_z_mm: np.ndarray | float,
    vertex_mask: np.ndarray,
    name: str,
) -> trimesh.Trimesh | None:
    """A watertight solid over the region of `grid` selected by
    `vertex_mask`: a top surface at `top_z_mm`, a bottom surface at
    `bottom_z_mm` (a flat float for a flat floor, e.g. the base terrain; or
    an array shaped like top_z_mm to hug another surface, e.g. an overlay
    sitting just under the terrain top), and side walls closing the boundary
    of the included region.

    Inclusion is decided at FACE granularity (all 4 corner vertices masked),
    not per-vertex/pixel: a face only has 4 orthogonal neighbors, which
    structurally rules out the diagonal/"checkerboard" adjacency ambiguity
    that pixel-corner approaches run into. Returns None if the mask selects
    no complete face.
    """
    nrv, ncv = grid.n_vertex_rows, grid.n_vertex_cols
    nrows, ncols = nrv - 1, ncv - 1

    face_included = resolve_face_inclusion(vertex_mask)
    if not face_included.any():
        return None

    x_mm, y_mm = grid.vertex_xy_mm()
    bottom_z_mm = np.broadcast_to(np.asarray(bottom_z_mm, dtype=float), top_z_mm.shape)

    def idx_top(r: int, c: int) -> int:
        return r * ncv + c

    n_top = nrv * ncv

    def idx_bottom(r: int, c: int) -> int:
        return n_top + r * ncv + c

    top_verts = np.column_stack([x_mm.ravel(), y_mm.ravel(), top_z_mm.ravel()])
    bottom_verts = np.column_stack([x_mm.ravel(), y_mm.ravel(), bottom_z_mm.ravel()])
    vertices = np.vstack([top_verts, bottom_verts])

    faces: list[tuple[int, int, int]] = []
    rows, cols = np.nonzero(face_included)
    for r, c in zip(rows.tolist(), cols.tolist()):
        v00, v01 = idx_top(r, c), idx_top(r, c + 1)
        v10, v11 = idx_top(r + 1, c), idx_top(r + 1, c + 1)
        faces.append((v00, v10, v11))
        faces.append((v00, v11, v01))

        b00, b01 = idx_bottom(r, c), idx_bottom(r, c + 1)
        b10, b11 = idx_bottom(r + 1, c), idx_bottom(r + 1, c + 1)
        faces.append((b00, b11, b10))
        faces.append((b00, b01, b11))

        # Boundary walls: emitted only on edges shared with a non-included
        # neighbor face (or the grid edge) - i.e. only around the outline.
        if r == 0 or not face_included[r - 1, c]:
            faces.append((idx_top(r, c), idx_top(r, c + 1), idx_bottom(r, c + 1)))
            faces.append((idx_top(r, c), idx_bottom(r, c + 1), idx_bottom(r, c)))
        if r == nrows - 1 or not face_included[r + 1, c]:
            faces.append((idx_top(r + 1, c), idx_bottom(r + 1, c + 1), idx_top(r + 1, c + 1)))
            faces.append((idx_top(r + 1, c), idx_bottom(r + 1, c), idx_bottom(r + 1, c + 1)))
        if c == 0 or not face_included[r, c - 1]:
            faces.append((idx_top(r, c), idx_bottom(r + 1, c), idx_top(r + 1, c)))
            faces.append((idx_top(r, c), idx_bottom(r, c), idx_bottom(r + 1, c)))
        if c == ncols - 1 or not face_included[r, c + 1]:
            faces.append((idx_top(r, c + 1), idx_top(r + 1, c + 1), idx_bottom(r + 1, c + 1)))
            faces.append((idx_top(r, c + 1), idx_bottom(r + 1, c + 1), idx_bottom(r, c + 1)))

    return finalize_mesh(vertices, np.asarray(faces, dtype=np.int64), name=name)


def finalize_mesh(vertices: np.ndarray, faces: np.ndarray, name: str) -> trimesh.Trimesh:
    """Build a trimesh from raw vertex/face arrays and repair it into a clean
    watertight solid, ready for STL/3MF export."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh, multibody=True)
    mesh.metadata["name"] = name
    return mesh
