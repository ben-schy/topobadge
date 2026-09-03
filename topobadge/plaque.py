"""Optional engraved text on the underside of the model (a hike name, date,
who-was-there, etc.) - rendered to a bitmap, thresholded, and placed as a
vertex mask on the same grid as everything else, so it builds through the
exact same restricted-region-solid machinery as the top-surface overlays
(mesh.overlay / mesh.common.build_region_solid), just applied to the base's
flat bottom cap instead of the terrain surface.

The text is mirrored left-right before placement: the physical model is
meant to be flipped over like a page (rotated about the vertical axis) to
read its underside, so the source art has to be authored mirrored ahead of
time - the same convention used for a coin's reverse or a PCB's bottom
silkscreen.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .grid import GridSpec

_RENDER_FONT_PX = 120  # oversample generously; downsampling onto the grid gives clean anti-aliased edges
_LINE_SPACING = int(_RENDER_FONT_PX * 0.3)
_PAD_PX = 6
_TARGET_CELLS_PER_CHAR = 8.0  # rough rule of thumb for a legible engraved stroke width


def _normalize_newlines(text: str) -> str:
    """Collapse CRLF/CR line endings to plain \\n. A browser <textarea> sends
    "\\r\\n" per line; left as-is, the stray "\\r" has no glyph in the render
    font and shows up as a tofu box at the end of every line but the last."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def recommended_max_mm_per_cell(text: str, width_mm: float) -> float:
    """The coarsest mm_per_cell that should still render `text` legibly at
    `width_mm` wide, based on the longest line's character count - a rough
    rule of thumb (not a hard limit) used to give an actionable warning
    when a plaque comes out too sparse to build."""
    text = _normalize_newlines(text)
    lines = [ln for ln in text.strip("\n").split("\n") if ln.strip()]
    longest = max((len(ln) for ln in lines), default=1)
    return width_mm / max(1, longest) / _TARGET_CELLS_PER_CHAR


def plaque_vertex_mask(grid: GridSpec, text: str, width_mm: float) -> np.ndarray:
    """Boolean vertex mask (grid.n_vertex_rows x grid.n_vertex_cols): `text`
    centered in the grid, `width_mm` wide in print space, mirrored left-right.
    Returns an all-False mask if `text` is blank.
    """
    text = _normalize_newlines(text).strip("\n")
    if not text.strip():
        return np.zeros((grid.n_vertex_rows, grid.n_vertex_cols), dtype=bool)

    font = ImageFont.load_default(size=_RENDER_FONT_PX)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = probe.multiline_textbbox((0, 0), text, font=font, align="center", spacing=_LINE_SPACING)
    w = int(bbox[2] - bbox[0]) + _PAD_PX * 2
    h = int(bbox[3] - bbox[1]) + _PAD_PX * 2

    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (_PAD_PX - bbox[0], _PAD_PX - bbox[1]), text, fill=255, font=font, align="center", spacing=_LINE_SPACING
    )
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    width_cells = max(1, round(width_mm / (grid.cell_size_x * grid.mm_per_meter)))
    height_cells = max(1, round(width_cells * h / w))
    img = img.resize((width_cells, height_cells), Image.LANCZOS)
    ink = np.asarray(img) > 127

    mask = np.zeros((grid.n_vertex_rows, grid.n_vertex_cols), dtype=bool)
    r0 = (grid.n_vertex_rows - height_cells) // 2
    c0 = (grid.n_vertex_cols - width_cells) // 2
    r1, c1 = r0 + height_cells, c0 + width_cells
    # Clip against the grid bounds rather than erroring if the plaque (or a
    # very small model) doesn't leave room for the full requested width.
    sr0, sr1 = max(0, r0), min(grid.n_vertex_rows, r1)
    sc0, sc1 = max(0, c0), min(grid.n_vertex_cols, c1)
    if sr0 < sr1 and sc0 < sc1:
        mask[sr0:sr1, sc0:sc1] = ink[sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0]
    return mask
