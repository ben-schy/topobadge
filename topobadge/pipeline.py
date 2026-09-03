"""The end-to-end GPX -> multicolor topo map pipeline, shared by the CLI
(cli.py) and the local web UI (web.py) so there is exactly one place that
knows how to turn a GPX file + options into STL/3MF/preview output.

Split into three stages so a caller (the web UI) can fetch data once and
recompose/re-preview it many times cheaply, without re-hitting any network
service:

  fetch_stage   - slow, network-bound: DEM (3DEP), water (NHD/Esri), land
                  cover classification (NLCD), roads/other-trails (USGS
                  Transportation), landmarks (GNIS). Depends only on
                  FetchOptions (the working area/grid).
  compose_masks - fast, local: groups every independently-fetched "source
                  class" (trees, shrub, grassland, water, ice, roads,
                  landmarks, ...) into whichever output "layer" the caller
                  assigned it to (AdjustOptions.layer_assignment), smooths
                  and cleans each layer's mask, and resolves overlaps. No
                  mesh building.
  mesh_stage    - fast, local: builds the actual meshes and writes
                  STL/3MF/preview from the resolved masks, using each
                  layer's own color and rise_mm.

build_topo() chains all three for the CLI, which only ever wants one shot.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import trimesh
from shapely.ops import unary_union

from .elevation import fetch_dem
from .export3mf import ColoredPart, write_3mf
from .gpx import Track, load_track
from .grid import GridSpec, build_grid_spec, clean_mask, hexagon_polygon, rasterize_mask, resolve_overlaps, smooth_mask
from .hydrography import WaterFeatures, fetch_water_features
from .landcover import SOURCE_CLASSES, fetch_nlcd_classes, source_class_masks
from .landmarks import NamedPoint, fetch_landmarks
from .mesh.common import heights_to_mm, resolve_face_inclusion, vertex_mask_from_face_inclusion
from .mesh.overlay import build_overlay_mesh
from .mesh.terrain import build_terrain_mesh
from .plaque import plaque_vertex_mask, recommended_max_mm_per_cell
from .preview import render_preview
from .transportation import fetch_other_trails, fetch_roads

# --------------------------------------------------------------------------
# Layers: the output colors/heights a source class can be assigned to.
# --------------------------------------------------------------------------


@dataclass
class LayerSpec:
    color_hex: str
    rise_mm: float = 0.0  # 0 = flush/inlaid; positive = physically raised above the surrounding surface


def default_layers() -> dict[str, LayerSpec]:
    return {
        "base": LayerSpec("808080", 0.0),  # implicit - never has its own overlay mesh, just exposed rock
        "vegetation": LayerSpec("3C8C3C", 0.0),
        "water": LayerSpec("2E75B6", 0.0),
        "ice": LayerSpec("FFFFFF", 0.0),
        "trail": LayerSpec("8B5A2B", 2.0),
        "roads": LayerSpec("4A4A4A", 0.0),
        "other_trails": LayerSpec("C9A66B", 0.0),
        "landmarks": LayerSpec("D4AF37", 0.6),
    }


# Every independently-fetched source class, and which layer it targets by
# default. Reassigning e.g. "ice" -> "water" (or -> "base" to hide it) or
# "grassland" -> "base" (while leaving "trees" on "vegetation") is exactly
# the kind of per-type control this map is built around. Roads/other_trails/
# landmarks default onto "base" (hidden) - opt-in extras, not a change to
# the default look of a plain build.
def default_layer_assignment() -> dict[str, str]:
    return {
        "trees": "vegetation",
        "shrub": "vegetation",
        "grassland": "vegetation",
        "pasture_crops": "vegetation",
        "wetland": "vegetation",
        "water": "water",
        "ice": "ice",
        "trail": "trail",
        "roads": "base",
        "other_trails": "base",
        "landmarks": "base",
    }


# Display labels for the web UI's source-class -> layer assignment table.
SOURCE_CLASS_LABELS: dict[str, str] = {
    "trees": "Trees",
    "shrub": "Shrub / scrub",
    "grassland": "Grassland",
    "pasture_crops": "Pasture / crops",
    "wetland": "Wetland",
    "water": "Water (lakes, streams, rivers)",
    "ice": "Ice / permanent snow",
    "trail": "Your hike route",
    "roads": "Roads",
    "other_trails": "Other trails",
    "landmarks": "Named landmarks",
}

LAYER_LABELS: dict[str, str] = {
    "base": "Base (hidden)",
    "vegetation": "Vegetation",
    "water": "Water",
    "ice": "Ice",
    "trail": "Trail (raised)",
    "roads": "Roads",
    "other_trails": "Other trails",
    "landmarks": "Landmarks",
}


# trail wins ties over land-cover, then water, then ice, then forest -
# without this, e.g. a trail through forest would produce two
# independently-watertight overlay shells that interpenetrate. Superseded at
# runtime by compose_masks, which sorts any layer with rise_mm > 0 first.
OVERLAY_PRIORITY = ["trail", "water", "ice", "vegetation", "roads", "other_trails", "landmarks"]


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


@dataclass
class FetchOptions:
    """Anything that changes the working area/grid - changing one of these
    requires re-fetching from every data source."""

    buffer_km: float = 1.2
    size_mm: float = 80.0
    mm_per_cell: float = 0.1
    base_shape: str = "hexagon"  # "rectangle" | "hexagon" - hex keepsake tile is the primary use case


@dataclass
class AdjustOptions:
    """Everything that only affects local compositing/meshing - safe to
    change and recompute without touching the network."""

    vertical_exaggeration: float = 2.0
    base_thickness_mm: float = 1.0
    inlay_depth_mm: float = 1.0
    trail_width_mm: float = 1.0
    water_width_mm: float = 0.6
    river_width_mm: float = 0.6
    road_width_mm: float = 1.0
    other_trail_width_mm: float = 1.0
    landmark_radius_mm: float = 1.5
    smoothing: float = 2.0  # gaussian-blur-then-rethreshold strength, in grid cells; 0 = off
    use_vector_water: bool = True  # NHD, falling back to Esri Living Atlas
    use_nlcd_water: bool = True  # blend in NLCD's own "open water" class too
    layers: dict[str, LayerSpec] = field(default_factory=default_layers)
    layer_assignment: dict[str, str] = field(default_factory=default_layer_assignment)

    # Optional engraved text on the underside (hike name, date, who was
    # there, ...) - blank disables it entirely. Use "\n" for multiple lines.
    plaque_text: str = ""
    plaque_width_mm: float = 50.0
    plaque_depth_mm: float = 0.8
    plaque_color_hex: str = "8B5A2B"


@dataclass
class FetchedData:
    gpx_path: str
    fetch_options: FetchOptions
    track: Track
    grid: GridSpec
    footprint_mask: np.ndarray | None
    dem: np.ndarray
    water_features: WaterFeatures
    nlcd_classes: np.ndarray | None
    roads: object | None  # shapely geometry or None
    other_trails: object | None
    landmarks: list[NamedPoint]
    warnings: list[str] = field(default_factory=list)


@dataclass
class BuildResult:
    out_dir: str
    parts: list[tuple[str, trimesh.Trimesh]]
    mf_path: str
    preview_path: str | None
    elapsed_s: float
    landcover_warning: str | None = field(default=None)


# --------------------------------------------------------------------------
# Stage 1: fetch (slow, network-bound)
# --------------------------------------------------------------------------


def fetch_stage(gpx_path: str, options: FetchOptions, on_log: Callable[[str], None] = print) -> FetchedData:
    on_log(f"Loading GPX track: {gpx_path}")
    track = load_track(gpx_path, buffer_km=options.buffer_km)

    # A hexagon footprint needs a square fetch/working extent underneath it -
    # otherwise the hexagon would get clipped along the shorter axis of a
    # non-square working bbox.
    working_bbox = track.working_bbox_utm.squared() if options.base_shape == "hexagon" else track.working_bbox_utm

    on_log(f"Building working grid (UTM EPSG:{track.utm_epsg}, {options.base_shape} footprint)...")
    grid = build_grid_spec(working_bbox, track.utm_epsg, size_mm=options.size_mm, mm_per_cell=options.mm_per_cell)
    on_log(f"  {grid.ncols} x {grid.nrows} cells, {grid.cell_size_x:.1f} x {grid.cell_size_y:.1f} m/cell")

    footprint_mask = None
    if options.base_shape == "hexagon":
        cx = (grid.min_x + grid.max_x) / 2
        cy = (grid.min_y + grid.max_y) / 2
        circumradius_m = (options.size_mm / 2) / grid.mm_per_meter
        footprint_mask = rasterize_mask(grid, hexagon_polygon(cx, cy, circumradius_m))

    on_log("Fetching elevation data (USGS 3DEP)...")
    dem = fetch_dem(grid)
    on_log(f"  elevation range {dem.min():.0f}m - {dem.max():.0f}m")

    warnings: list[str] = []

    on_log("Fetching water features (USGS NHD, falling back to Esri Living Atlas)...")
    try:
        water_features = fetch_water_features(track, on_log=on_log)
    except Exception as e:  # noqa: BLE001 - keep going without water rather than failing the whole fetch
        msg = f"Water feature fetch failed ({e}); continuing without water/streams."
        warnings.append(msg)
        on_log(f"  WARNING: {msg}")
        water_features = WaterFeatures(water=None, rivers=None, streams=None)

    on_log("Fetching land cover classification (USGS NLCD)...")
    try:
        nlcd_classes = fetch_nlcd_classes(grid)
    except Exception as e:  # noqa: BLE001 - keep going without vegetation/ice rather than failing the whole fetch
        msg = f"Land cover fetch failed ({e}); continuing without vegetation/ice."
        warnings.append(msg)
        on_log(f"  WARNING: {msg}")
        nlcd_classes = None

    on_log("Fetching roads and other trails (USGS Transportation)...")
    try:
        roads = fetch_roads(track)
        other_trails = fetch_other_trails(track)
    except Exception as e:  # noqa: BLE001
        msg = f"Roads/trails fetch failed ({e}); continuing without them."
        warnings.append(msg)
        on_log(f"  WARNING: {msg}")
        roads, other_trails = None, None

    on_log("Fetching named landmarks (USGS GNIS)...")
    try:
        landmarks = fetch_landmarks(track)
        on_log(f"  {len(landmarks)} landmark(s) found" + (f": {', '.join(p.name for p in landmarks[:6])}{'...' if len(landmarks) > 6 else ''}" if landmarks else ""))
    except Exception as e:  # noqa: BLE001
        msg = f"Landmark fetch failed ({e}); continuing without them."
        warnings.append(msg)
        on_log(f"  WARNING: {msg}")
        landmarks = []

    return FetchedData(
        gpx_path=gpx_path,
        fetch_options=options,
        track=track,
        grid=grid,
        footprint_mask=footprint_mask,
        dem=dem,
        water_features=water_features,
        nlcd_classes=nlcd_classes,
        roads=roads,
        other_trails=other_trails,
        landmarks=landmarks,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Stage 2: compose (fast, local)
# --------------------------------------------------------------------------


def _source_class_masks(fetched: FetchedData, adjust: AdjustOptions) -> dict[str, np.ndarray]:
    """One boolean vertex mask per independently-assignable source class."""
    grid = fetched.grid
    shape = (grid.n_vertex_rows, grid.n_vertex_cols)
    empty = np.zeros(shape, dtype=bool)

    masks: dict[str, np.ndarray] = {name: empty for name in SOURCE_CLASSES}
    nlcd_water = empty
    if fetched.nlcd_classes is not None:
        nlcd_masks = source_class_masks(fetched.nlcd_classes)
        nlcd_water = nlcd_masks.pop("water_nlcd")
        masks.update(nlcd_masks)

    # Water: vector sources (NHD/Esri), buffered into a visible ribbon
    # (rivers wider than small streams; standing water gets the stream-width
    # buffer as a minimum size so a small pond isn't lost to clean_mask at
    # coarser resolutions), optionally blended with NLCD's own water class.
    water_geoms = []
    if adjust.use_vector_water:
        wf = fetched.water_features
        water_real_width_m = adjust.water_width_mm / grid.mm_per_meter
        river_real_width_m = adjust.river_width_mm / grid.mm_per_meter
        if wf.water is not None:
            water_geoms.append(wf.water.buffer(water_real_width_m / 2))
        if wf.streams is not None:
            water_geoms.append(wf.streams.buffer(water_real_width_m / 2))
        if wf.rivers is not None:
            water_geoms.append(wf.rivers.buffer(river_real_width_m / 2))
    water_geometry = unary_union(water_geoms) if water_geoms else None
    water_mask = rasterize_mask(grid, water_geometry)
    if adjust.use_nlcd_water:
        water_mask = water_mask | nlcd_water
    masks["water"] = water_mask

    # The user's own hike route.
    trail_real_width_m = adjust.trail_width_mm / grid.mm_per_meter
    trail_polygon = fetched.track.lines_utm.buffer(trail_real_width_m / 2)
    masks["trail"] = rasterize_mask(grid, trail_polygon)

    # Roads / other trails: centerlines, buffered into a ribbon.
    if fetched.roads is not None:
        road_poly = fetched.roads.buffer((adjust.road_width_mm / grid.mm_per_meter) / 2)
        masks["roads"] = rasterize_mask(grid, road_poly)
    else:
        masks["roads"] = empty

    if fetched.other_trails is not None:
        ot_poly = fetched.other_trails.buffer((adjust.other_trail_width_mm / grid.mm_per_meter) / 2)
        masks["other_trails"] = rasterize_mask(grid, ot_poly)
    else:
        masks["other_trails"] = empty

    # Landmarks: each point buffered into a small disc.
    if fetched.landmarks:
        radius_m = adjust.landmark_radius_mm / grid.mm_per_meter
        discs = unary_union([lm.point.buffer(radius_m) for lm in fetched.landmarks])
        masks["landmarks"] = rasterize_mask(grid, discs)
    else:
        masks["landmarks"] = empty

    return masks


def compose_masks(fetched: FetchedData, adjust: AdjustOptions) -> dict[str, np.ndarray]:
    """Resolved (non-overlapping) per-layer vertex masks - no network, no
    mesh building, cheap enough to call on every UI adjustment."""
    grid = fetched.grid
    shape = (grid.n_vertex_rows, grid.n_vertex_cols)
    empty = np.zeros(shape, dtype=bool)

    source_masks = _source_class_masks(fetched, adjust)

    # Group sources into whichever layer each one is assigned to.
    layer_names = [name for name in adjust.layers if name != "base"]
    grouped: dict[str, np.ndarray] = {name: empty.copy() for name in layer_names}
    for source_name, mask in source_masks.items():
        target = adjust.layer_assignment.get(source_name, "base")
        if target in grouped:
            grouped[target] = grouped[target] | mask

    if fetched.footprint_mask is not None:
        grouped = {k: v & fetched.footprint_mask for k, v in grouped.items()}

    smoothed = {k: smooth_mask(v, adjust.smoothing) for k, v in grouped.items()}
    cleaned = {k: clean_mask(v) for k, v in smoothed.items()}

    # Layers that physically rise above the surface always win ties over
    # flush ones (e.g. the trail must print on top of a road it crosses);
    # otherwise fall back to OVERLAY_PRIORITY's order, then anything else.
    def sort_key(name: str) -> tuple[bool, int]:
        rise = adjust.layers.get(name, LayerSpec("000000")).rise_mm
        order = OVERLAY_PRIORITY.index(name) if name in OVERLAY_PRIORITY else len(OVERLAY_PRIORITY)
        return (rise <= 0, order)

    priority = sorted(cleaned.keys(), key=sort_key)
    return resolve_overlaps(cleaned, priority=priority)


def mask_area_percentages(resolved_masks: dict[str, np.ndarray], footprint_mask: np.ndarray | None, grid: GridSpec) -> dict[str, float]:
    """Share of the model's footprint occupied by each layer - handy for a
    live "here's what changed" readout without building any mesh."""
    total = float(footprint_mask.sum()) if footprint_mask is not None else float(grid.n_vertex_rows * grid.n_vertex_cols)
    if total <= 0:
        return {name: 0.0 for name in resolved_masks}
    return {name: 100.0 * float(mask.sum()) / total for name, mask in resolved_masks.items()}


def build_plaque_mask(fetched: FetchedData, adjust: AdjustOptions) -> tuple[np.ndarray | None, str | None]:
    """The underside text plaque's vertex mask (mirrored, ready to build into
    the mesh), or (None, warning) if the text is blank or didn't resolve at
    this grid's resolution. Shared by mesh_stage and the web UI's live
    preview, so "will this text fit" is answered identically in both.
    """
    if not adjust.plaque_text.strip():
        return None, None

    grid = fetched.grid
    plaque_width_mm = min(adjust.plaque_width_mm, fetched.fetch_options.size_mm * 0.85)
    mask = plaque_vertex_mask(grid, adjust.plaque_text, plaque_width_mm)
    if fetched.footprint_mask is not None:
        mask = mask & fetched.footprint_mask

    if not resolve_face_inclusion(mask).any():
        rec_mm_per_cell = recommended_max_mm_per_cell(adjust.plaque_text, plaque_width_mm)
        warning = (
            f"Plaque text didn't render at this grid resolution ({fetched.fetch_options.mm_per_cell:.2f}mm/cell) - "
            f"try mm-per-cell {rec_mm_per_cell:.2f} or finer (needs a re-fetch), a larger size, a wider plaque, "
            "or shorter/fewer lines of text."
        )
        return None, warning

    return mask, None


def render_preview_for(
    fetched: FetchedData,
    resolved_masks: dict[str, np.ndarray],
    layers: dict[str, LayerSpec],
    out_path: str,
    plaque_mask: np.ndarray | None = None,
    plaque_color_hex: str | None = None,
) -> None:
    colors = {name: spec.color_hex for name, spec in layers.items()}
    render_preview(
        fetched.dem,
        resolved_masks,
        fetched.footprint_mask,
        out_path,
        colors=colors,
        plaque_mask=plaque_mask,
        plaque_color_hex=plaque_color_hex,
    )


# --------------------------------------------------------------------------
# Stage 3: mesh (fast, local)
# --------------------------------------------------------------------------


def mesh_stage(
    fetched: FetchedData,
    adjust: AdjustOptions,
    resolved_masks: dict[str, np.ndarray],
    out_dir: str,
    write_preview: bool = True,
    on_log: Callable[[str], None] = print,
) -> BuildResult:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    grid = fetched.grid

    min_elev = float(fetched.dem.min())
    z_mm = heights_to_mm(fetched.dem, grid.mm_per_meter, adjust.vertical_exaggeration, min_elev)

    layer_names = list(resolved_masks.keys())

    # Notch the base wherever any layer sits, so flush layers are inlaid
    # into it rather than stacked on top of a full-height base - only each
    # layer's own `rise_mm` then reads as physically raised. The pocket
    # footprint is the union of every layer's actual (post diagonal-pinch-
    # fix) face inclusion, not the raw pre-cleanup mask, so every notch in
    # the base is fully covered by the layer above it.
    pocket_depth_mm = min(adjust.inlay_depth_mm, adjust.base_thickness_mm * 0.5)
    any_face_included = np.zeros((grid.nrows, grid.ncols), dtype=bool)
    for name in layer_names:
        any_face_included |= resolve_face_inclusion(resolved_masks[name])
    pocket_vertex_mask = vertex_mask_from_face_inclusion(any_face_included)

    # Underside text plaque (optional): notch the base's flat bottom cap the
    # same way its top gets notched for overlays, then build the plaque
    # part to fill that notch flush. Text lives on the same shared grid as
    # everything else, so it needs the grid fine enough to resolve individual
    # letter strokes - warn with an actionable fix rather than silently
    # building nothing if it isn't.
    plaque_mask, plaque_warning = build_plaque_mask(fetched, adjust)
    if plaque_warning is not None:
        on_log(f"  WARNING: {plaque_warning}")
    bottom_pocket_vertex_mask = None
    bottom_pocket_depth_mm = 0.0
    if plaque_mask is not None:
        bottom_pocket_depth_mm = min(adjust.plaque_depth_mm, adjust.base_thickness_mm * 0.5)
        bottom_pocket_vertex_mask = vertex_mask_from_face_inclusion(resolve_face_inclusion(plaque_mask))

    on_log("Building base terrain mesh...")
    base_layer = adjust.layers.get("base", LayerSpec("808080"))
    base_mesh = build_terrain_mesh(
        grid,
        z_mm,
        base_thickness_mm=adjust.base_thickness_mm,
        footprint_mask=fetched.footprint_mask,
        pocket_vertex_mask=pocket_vertex_mask,
        pocket_depth_mm=pocket_depth_mm,
        bottom_pocket_vertex_mask=bottom_pocket_vertex_mask,
        bottom_pocket_depth_mm=bottom_pocket_depth_mm,
    )

    parts: list[tuple[str, trimesh.Trimesh]] = [("base", base_mesh)]
    colors: dict[str, str] = {"base": base_layer.color_hex}
    for name in layer_names:
        spec = adjust.layers.get(name, LayerSpec("999999"))
        on_log(f"Building {name} layer mesh...")
        mesh = build_overlay_mesh(
            grid, z_mm, resolved_masks[name], name, rise_mm=spec.rise_mm, inlay_depth_mm=pocket_depth_mm
        )
        if mesh is not None:
            parts.append((name, mesh))
            colors[name] = spec.color_hex
        else:
            on_log(f"  (no {name} area in this map, skipping)")

    if plaque_mask is not None:
        on_log("Building plaque text mesh...")
        floor_z_full = np.full_like(z_mm, -adjust.base_thickness_mm)
        plaque_mesh = build_overlay_mesh(
            grid, floor_z_full, plaque_mask, "plaque", rise_mm=bottom_pocket_depth_mm, inlay_depth_mm=0.0
        )
        if plaque_mesh is not None:
            parts.append(("plaque", plaque_mesh))
            colors["plaque"] = adjust.plaque_color_hex
        else:
            on_log("  (plaque text produced no area - check plaque_width_mm)")

    on_log("\nPart QA:")
    on_log(f"{'part':<14}{'faces':>10}{'watertight':>12}{'volume(mm3)':>14}")
    for name, mesh in parts:
        on_log(f"{name:<14}{len(mesh.faces):>10}{str(mesh.is_watertight):>12}{mesh.volume:>14.1f}")

    on_log(f"\nWriting output to {out_dir}")
    for name, mesh in parts:
        stl_path = os.path.join(out_dir, f"{name}.stl")
        mesh.export(stl_path)
        on_log(f"  {stl_path}")

    colored_parts = [ColoredPart(name, mesh, colors[name]) for name, mesh in parts]
    mf_path = os.path.join(out_dir, "combined.3mf")
    write_3mf(mf_path, colored_parts)
    on_log(f"  {mf_path}")

    preview_path = None
    if write_preview:
        preview_path = os.path.join(out_dir, "preview.png")
        render_preview_for(
            fetched,
            resolved_masks,
            adjust.layers,
            preview_path,
            plaque_mask=plaque_mask,
            plaque_color_hex=adjust.plaque_color_hex,
        )
        on_log(f"  {preview_path}")

    elapsed_s = time.time() - t0
    on_log(f"\nDone in {elapsed_s:.1f}s")

    return BuildResult(
        out_dir=out_dir,
        parts=parts,
        mf_path=mf_path,
        preview_path=preview_path,
        elapsed_s=elapsed_s,
        landcover_warning=" ".join(fetched.warnings) if fetched.warnings else None,
    )


# --------------------------------------------------------------------------
# Convenience: fetch + compose + mesh in one call (what the CLI uses)
# --------------------------------------------------------------------------


def build_topo(
    gpx_path: str,
    fetch_options: FetchOptions,
    adjust_options: AdjustOptions,
    out_dir: str | None = None,
    write_preview: bool = True,
    on_log: Callable[[str], None] = print,
) -> BuildResult:
    fetched = fetch_stage(gpx_path, fetch_options, on_log=on_log)
    resolved_masks = compose_masks(fetched, adjust_options)
    if out_dir is None:
        base_name = os.path.splitext(os.path.basename(gpx_path))[0]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(gpx_path)), f"{base_name}_topo")
    return mesh_stage(fetched, adjust_options, resolved_masks, out_dir, write_preview=write_preview, on_log=on_log)
