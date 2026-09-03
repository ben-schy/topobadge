"""topobadge CLI: turn a GPX hike track into a multicolor 3D-printable topo
map (grey rock base + assignable color layers for vegetation/water/ice/
roads/trails/landmarks + a raised trail line), exported as both per-color
STLs and a combined 3MF.
"""
from __future__ import annotations

import json
import webbrowser

import click

from .pipeline import AdjustOptions, FetchOptions, LayerSpec, build_topo, default_layer_assignment, default_layers


@click.group()
def cli() -> None:
    """topobadge: GPX hike track -> multicolor 3D-printable topo map."""


@cli.command()
@click.argument("gpx_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--buffer-km", default=1.2, show_default=True, help="Surrounding context beyond the track's bounding box, in km.")
@click.option("--size-mm", default=80.0, show_default=True, help="Longest horizontal dimension of the printed model, in mm.")
@click.option("--vertical-exaggeration", default=2.0, show_default=True, help="Multiplier applied to relief so terrain reads clearly at this horizontal scale.")
@click.option("--base-thickness-mm", default=1.0, show_default=True, help="Minimum floor thickness below the lowest terrain point.")
@click.option(
    "--inlay-depth-mm",
    default=1.0,
    show_default=True,
    help="How deep flush layers (vegetation/water/ice/roads/...) are inlaid below the surface - they sit "
    "flush with the surrounding terrain, not raised. Also every layer's bonding depth into the base "
    "(~4 printed layers at a typical 0.2mm layer height).",
)
@click.option(
    "--trail-height-mm",
    default=2.0,
    show_default=True,
    help="How far the trail alone rises above the surrounding (flush) surface "
    "(~10 printed layers at a typical 0.2mm layer height).",
)
@click.option("--trail-width-mm", default=1.0, show_default=True, help="Printed width of the raised trail ribbon.")
@click.option(
    "--water-width-mm",
    default=0.6,
    show_default=True,
    help="Minimum printed width for streams/canals (NHD flowlines) and small ponds/lakes, so they "
    "survive cleanup and stay visible at this grid resolution.",
)
@click.option(
    "--river-width-mm",
    default=0.6,
    show_default=True,
    help="Printed width for named rivers (NHD flowlines), wider than --water-width-mm.",
)
@click.option("--road-width-mm", default=1.0, show_default=True, help="Printed width for roads, if enabled (see --layers-json).")
@click.option("--other-trail-width-mm", default=1.0, show_default=True, help="Printed width for other (non-hike) trails, if enabled.")
@click.option("--landmark-radius-mm", default=1.5, show_default=True, help="Printed radius of each named-landmark marker, if enabled.")
@click.option(
    "--smoothing",
    default=2.0,
    show_default=True,
    help="Rounds off blocky grid-cell corners on region boundaries (gaussian-blur-then-rethreshold "
    "strength, in grid cells). 0 disables smoothing.",
)
@click.option("--mm-per-cell", default=0.1, show_default=True, help="Target grid resolution in print-space mm per cell.")
@click.option(
    "--base-shape",
    type=click.Choice(["hexagon", "rectangle"]),
    default="hexagon",
    show_default=True,
    help="Outer footprint of the printed model. Hexagon is sized so its point-to-point width equals --size-mm.",
)
@click.option(
    "--use-vector-water/--no-use-vector-water",
    default=True,
    show_default=True,
    help="Use vector water/stream sources (NHD, falling back to Esri Living Atlas).",
)
@click.option(
    "--use-nlcd-water/--no-use-nlcd-water",
    default=True,
    show_default=True,
    help="Blend in NLCD's own \"open water\" class alongside the vector water sources.",
)
@click.option(
    "--vegetation-preset",
    type=click.Choice(["all", "trees-only"]),
    default="all",
    show_default=True,
    help="'all' treats shrub/grassland/pasture/wetland as vegetation too; 'trees-only' hides them "
    "(shown as bare rock). For finer per-type control (including roads/other trails/landmarks, and "
    "custom colors), use --layers-json.",
)
@click.option(
    "--layers-json",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to a JSON file overriding layer colors/heights and/or source-class assignment, e.g. '
    '{"layers": {"vegetation": {"color_hex": "3C8C3C", "rise_mm": 0}}, '
    '"layer_assignment": {"ice": "water", "roads": "roads"}}. '
    "Source classes: trees, shrub, grassland, pasture_crops, wetland, water, ice, trail, roads, "
    "other_trails, landmarks. Applied after --vegetation-preset.",
)
@click.option(
    "--plaque-text",
    default=None,
    help='Optional text engraved flush on the underside, in a separate color (e.g. a hike name/date/who '
    'was there). Use literal "\\n" for multiple lines, e.g. --plaque-text "James Peak\\n9/15/26\\nBen, Lillie, Brynn". '
    "For longer text, use --plaque-text-file instead. Mirrored automatically so it reads correctly when "
    "the model is flipped over like a page to view its underside. Needs a fine enough --mm-per-cell to "
    "resolve legibly - topobadge will tell you the required value if it doesn't fit.",
)
@click.option(
    "--plaque-text-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Read --plaque-text from a file instead (one line per printed line) - easier for multi-line text "
    "than escaping newlines on the command line.",
)
@click.option("--plaque-width-mm", default=50.0, show_default=True, help="Printed width of the text plaque.")
@click.option("--plaque-depth-mm", default=0.8, show_default=True, help="How deep the plaque is inlaid into the floor.")
@click.option("--plaque-color", default="8B5A2B", show_default=True, help="Hex color (no '#') for the plaque text.")
@click.option("--out-dir", default=None, type=click.Path(file_okay=False), help="Output directory (default: <gpx name>_topo next to the GPX file).")
@click.option("--preview/--no-preview", default=True, show_default=True, help="Write a top-down colored preview.png alongside the STL/3MF output.")
def build(
    gpx_path: str,
    buffer_km: float,
    size_mm: float,
    vertical_exaggeration: float,
    base_thickness_mm: float,
    inlay_depth_mm: float,
    trail_height_mm: float,
    trail_width_mm: float,
    water_width_mm: float,
    river_width_mm: float,
    road_width_mm: float,
    other_trail_width_mm: float,
    landmark_radius_mm: float,
    smoothing: float,
    mm_per_cell: float,
    base_shape: str,
    use_vector_water: bool,
    use_nlcd_water: bool,
    vegetation_preset: str,
    layers_json: str | None,
    plaque_text: str | None,
    plaque_text_file: str | None,
    plaque_width_mm: float,
    plaque_depth_mm: float,
    plaque_color: str,
    out_dir: str | None,
    preview: bool,
) -> None:
    """Build STL + 3MF outputs for GPX_PATH."""
    fetch_options = FetchOptions(
        buffer_km=buffer_km,
        size_mm=size_mm,
        mm_per_cell=mm_per_cell,
        base_shape=base_shape,
    )

    layers = default_layers()
    layers["trail"].rise_mm = trail_height_mm
    layer_assignment = default_layer_assignment()
    if vegetation_preset == "trees-only":
        for source in ("shrub", "grassland", "pasture_crops", "wetland"):
            layer_assignment[source] = "base"

    if layers_json:
        with open(layers_json, encoding="utf-8") as f:
            overrides = json.load(f)
        for name, spec in overrides.get("layers", {}).items():
            layers[name] = LayerSpec(color_hex=spec["color_hex"], rise_mm=float(spec.get("rise_mm", 0.0)))
        layer_assignment.update(overrides.get("layer_assignment", {}))

    if plaque_text_file:
        with open(plaque_text_file, encoding="utf-8") as f:
            plaque_text = f.read()
    plaque_text = (plaque_text or "").replace("\\n", "\n")

    adjust_options = AdjustOptions(
        vertical_exaggeration=vertical_exaggeration,
        base_thickness_mm=base_thickness_mm,
        inlay_depth_mm=inlay_depth_mm,
        trail_width_mm=trail_width_mm,
        water_width_mm=water_width_mm,
        river_width_mm=river_width_mm,
        road_width_mm=road_width_mm,
        other_trail_width_mm=other_trail_width_mm,
        landmark_radius_mm=landmark_radius_mm,
        smoothing=smoothing,
        use_vector_water=use_vector_water,
        use_nlcd_water=use_nlcd_water,
        layers=layers,
        layer_assignment=layer_assignment,
        plaque_text=plaque_text,
        plaque_width_mm=plaque_width_mm,
        plaque_depth_mm=plaque_depth_mm,
        plaque_color_hex=plaque_color,
    )
    build_topo(gpx_path, fetch_options, adjust_options, out_dir=out_dir, write_preview=preview, on_log=click.echo)


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind the local web server to.")
@click.option("--port", default=5151, show_default=True, help="Port for the local web server.")
@click.option("--open-browser/--no-open-browser", default=True, show_default=True, help="Open the page in your default browser on start.")
def serve(host: str, port: int, open_browser: bool) -> None:
    """Start the local web UI: upload a GPX, fill in a form, download the result."""
    from .web import create_app

    app = create_app()
    url = f"http://{host}:{port}/"
    click.echo(f"topobadge web UI running at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    cli()
