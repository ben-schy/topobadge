"""Hand-rolled 3MF writer: bundles multiple colored trimesh parts into a
single .3mf file (core spec + Materials/Properties extension for per-object
color), all sharing the same coordinate space as the individual STL exports
so it can be dropped straight into a slicer as one file.

Uses the Materials/Properties extension's <m:colorgroup> resource (color
referenced per-object via pid/pindex), not <basematerials>: Bambu Studio's
"Standard 3MF" color parser specifically reads colorgroup, and does not
reliably surface basematerials displaycolor as per-object color. Even so,
3MF color import behavior has changed across Bambu Studio versions and has
had real bugs (e.g. bambulab/BambuStudio#9666) - importing the individual
per-part STL files together and assigning filament colors by hand in the
slicer's object list is the reliable fallback if a given Bambu Studio
version still doesn't render this correctly.

Not using trimesh's own 3MF writer: its multi-object per-color support is
inconsistent across versions, and the format is small/stable enough to
generate directly.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass

import trimesh

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    "</Types>"
)

_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
    "</Relationships>"
)


@dataclass
class ColoredPart:
    name: str
    mesh: trimesh.Trimesh
    color_hex: str  # "RRGGBB", no leading '#'


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _mesh_xml(mesh: trimesh.Trimesh) -> str:
    vertices = "".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in mesh.vertices)
    triangles = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces)
    return f"<mesh><vertices>{vertices}</vertices><triangles>{triangles}</triangles></mesh>"


def write_3mf(path: str, parts: list[ColoredPart]) -> None:
    if not parts:
        raise ValueError("No parts to write to 3MF")

    color_entries = "".join(f'<m:color color="#{p.color_hex.upper()}FF"/>' for p in parts)

    object_entries = []
    build_entries = []
    object_id = 2  # id 1 is reserved for the colorgroup resource
    for pindex, part in enumerate(parts):
        object_entries.append(
            f'<object id="{object_id}" type="model" name="{_escape(part.name)}" pid="1" pindex="{pindex}">'
            f"{_mesh_xml(part.mesh)}</object>"
        )
        build_entries.append(f'<item objectid="{object_id}"/>')
        object_id += 1

    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">'
        f'<resources><m:colorgroup id="1">{color_entries}</m:colorgroup>'
        f"{''.join(object_entries)}</resources>"
        f"<build>{''.join(build_entries)}</build>"
        "</model>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("3D/3dmodel.model", model_xml)
