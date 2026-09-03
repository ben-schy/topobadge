# topobadge

Turn a GPX hike track into a multicolor 3D-printable topo map.

Feed it a `.gpx` file and topobadge pulls real elevation and land-cover data,
builds a small hexagonal (or rectangular) relief map of the hike and its
surroundings, and hands you a ready-to-print 3MF — grey rock, blue water,
green forest, and white ice inlaid flush into the terrain, with the trail
itself the only thing raised above the surface.

Everything runs locally. Your GPX file and the generated models never leave
your machine except to fetch public elevation/map data (US government APIs,
no API key required).

## Quick start

**Use `serve` — it's the intended way to run topobadge.** The web UI is
where you actually customize a build: reassign source classes to layers,
tweak colors/rise/widths, and see the preview update in under a second per
change, all without touching the network again after the initial fetch. The
`build` CLI command skips all of that and just applies whatever flags you
pass, so it's mainly useful for scripting or debugging, not day-to-day use.

**Windows, no command line needed:** double-click [run.bat](run.bat). The
first run sets up a virtual environment and installs dependencies
automatically; every run after that just starts the app and opens your
browser to the local web UI — drag in a `.gpx` file, pick a size, click
Fetch, then adjust and generate.

**Command line:**

```sh
pip install -e .
topobadge serve
```

Requires Python 3.11+.

## How it works

Five stages, all running on your machine:

1. **Track & frame** — parse the GPX, pick a local UTM projection, and frame
   a hexagonal (or rectangular) working area around the track plus a buffer
   of surrounding context.
2. **Elevation** — fetch a real elevation grid for that area from USGS 3DEP.
3. **Land cover & overlays** — fetch lakes/streams/rivers (USGS NHD, falling
   back to Esri Living Atlas), a vegetation/ice classification (NLCD), roads
   and other trails (USGS Transportation), and named landmarks (USGS GNIS).
4. **Compose** — group every source class into whichever output layer it's
   assigned to, smooth the resulting boundaries, and build a solid grey
   terrain base notched wherever a layer sits, so flush layers inlay into
   the surface instead of stacking on top of it.
5. **Export** — write each colored part as its own STL, bundle them into one
   colored 3MF, and render a top-down preview image.

Each colored region is built directly from the same elevation grid as the
base terrain (no boolean/CSG operations), at whole-grid-cell granularity so
diagonal-only touching regions can't produce a non-manifold, unprintable
seam.

## Layers & sources

Every piece of geometry topobadge can draw — trees, shrub, grassland, water,
ice, roads, other trails, named landmarks, and your own hike route — is a
**source class**. Every source class is assigned to an **output layer**,
and it's the layer that has a color and a height. Multiple source classes
can share one layer, or you can split them apart, merge one into another, or
hide any of them by assigning it to `base`.

| Source class | Default layer |
|---|---|
| Trees, shrub, grassland, pasture/crops, wetland | Vegetation (green) |
| Water (lakes, streams, rivers) | Water (blue) |
| Ice / permanent snow | Ice (white) |
| Your hike route | Trail (brown, raised) |
| Roads, other trails, named landmarks | Base (hidden) — opt in to show them |

Each layer also has its own color and **rise** — how far it sits above the
surrounding surface. 0 means flush/inlaid (water and vegetation by default);
positive means physically raised, like the trail.

From the CLI, use `--vegetation-preset` for the common case, or
`--layers-json` for full control, e.g.:

```json
{
  "layers": {
    "roads": {"color_hex": "4A4A4A", "rise_mm": 0}
  },
  "layer_assignment": {
    "ice": "water",
    "grassland": "base",
    "roads": "roads",
    "landmarks": "trail"
  }
}
```

## Underside plaque

Optional text — a hike name, date, who was there — engraved flush into the
underside in its own color. Set it in the web UI's "Underside text plaque"
panel, or on the CLI:

```sh
topobadge build my_hike.gpx --mm-per-cell 0.35 \
  --plaque-text "James Peak\n9/15/26\nBen, Lillie, Brynn" \
  --plaque-width-mm 70
```

The text is mirrored automatically, since the model is meant to be flipped
over like a page to read it. If a build logs a warning that the plaque
didn't render, it tells you exactly which `--mm-per-cell` value would fit
it — as a rule of thumb, aim for roughly 8 grid cells per character across
the longest line.

## CLI reference

```
topobadge build TRACK.gpx [OPTIONS]
topobadge serve [--host 127.0.0.1] [--port 5151] [--no-open-browser]
```

`build` is a debug/scripting escape hatch, not the recommended way to use
topobadge — it takes a fixed set of flags and produces output in one shot,
with none of the web UI's live preview or per-layer adjustment. Reach for
`serve` unless you specifically need a non-interactive, scriptable build.
Every option has a sensible default — the shortest invocation is just
`topobadge build my_hike.gpx`. Key options:

| Flag | Default | What it controls |
|---|---|---|
| `--base-shape` | hexagon | Outer footprint: `hexagon` or `rectangle`. |
| `--size-mm` | 80 | Longest horizontal dimension of the printed model. |
| `--vertical-exaggeration` | 2.0 | Relief multiplier so terrain reads clearly at this scale. |
| `--buffer-km` | 1.2 | Surrounding context fetched beyond the track's own bounding box. |
| `--mm-per-cell` | 0.1 | Mesh resolution in print-space mm per grid cell. |
| `--base-thickness-mm` | 1.0 | Minimum floor thickness below the lowest terrain point. |
| `--inlay-depth-mm` | 1.0 | How deep water/forest/ice are inlaid below the surface. |
| `--trail-height-mm` | 2.0 | How far the trail rises above the surrounding surface. |
| `--trail-width-mm` | 1.0 | Printed width of the raised trail ribbon. |
| `--water-width-mm` / `--river-width-mm` | 0.6 | Minimum printed width for streams/small ponds vs. named rivers. |
| `--smoothing` | 2.0 | Rounds off blocky grid-cell corners on region boundaries. 0 disables it. |
| `--vegetation-preset` | all | `all` includes shrub/grassland/pasture/wetland; `trees-only` hides them. |
| `--layers-json PATH` | — | Full override for layer colors/heights and source-class assignment. |
| `--out-dir` | `<gpx name>_topo` | Where output files are written. |

Run `topobadge build --help` for the full list, including plaque options.

## Output files

Written to `<gpx name>_topo/` (or `--out-dir`) — only the parts actually
present in the hike's area are written:

| File | Contents |
|---|---|
| `base.stl` | The grey terrain solid — always present. |
| `water.stl` | Lakes, ponds, streams, and rivers, where present. |
| `vegetation.stl` | Trees/shrub/grassland/etc. assigned to the vegetation layer. |
| `ice.stl` | Glaciers/permanent snow, where present. |
| `trail.stl` | The hike route itself, raised above everything else. |
| `roads.stl` / `other_trails.stl` / `landmarks.stl` | Only written if assigned to a visible layer. |
| `combined.3mf` | All present parts in one file, with color assigned per part. |
| `preview.png` | A top-down colored render for a quick sanity check. |

## Printing it

For a single-material printer, `base.stl` alone already prints a clean
relief tile. For multi-material printing (Bambu AMS, Prusa MMU, or
similar):

- **Recommended: import the STLs together.** Drag all the `.stl` files into
  your slicer at once — they share one coordinate origin, so they land in
  the right place automatically. Assign a filament color to each object.
- **Convenience: open `combined.3mf`.** One file, colors already assigned.
  Per-object color import in Bambu Studio specifically has had
  version-dependent bugs — if colors don't come through right, fall back to
  the STL method above.

## Data sources

| Data | Source | Coverage |
|---|---|---|
| Elevation | USGS 3DEP (The National Map) | United States only |
| Lakes, ponds, streams, rivers | USGS NHD, falling back to Esri Living Atlas | United States only |
| Vegetation, permanent snow/ice | USGS National Land Cover Database (NLCD) | United States only (lower 48) |
| Roads, other trails | USGS Transportation (The National Map) | United States only |
| Named landmarks | USGS GNIS gazetteer (The National Map) | United States only |

All sources are US government/public APIs, no key required, so topobadge
currently only produces full-color terrain for US hikes. Water is
deliberately layered across three independent sources so a single outage
doesn't take water off the map.

## Troubleshooting

**Fetch finished, but there's no water/forest color.** These are network
services topobadge doesn't control. It continues with whatever it has
rather than failing the fetch, and prints a warning. Re-fetching later
fills things back in.

**Colors look wrong in my slicer.** Import the individual STL files
together and assign filament colors by hand rather than relying on the
combined 3MF's embedded color.

**No terrain shows up at all / elevation looks flat.** Confirm the hike is
within the United States — USGS 3DEP doesn't cover other countries yet.
