"""Top-down colored preview image: a quick way to sanity-check a build
(terrain shape, land-cover placement, trail routing, footprint clipping)
without opening a slicer or STL viewer. When an underside text plaque is
set, a second panel is appended showing it in "readable" orientation (i.e.
un-mirrored, exactly as you'd see it after actually flipping the model
over) alongside the top-down view.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_OUTSIDE_FOOTPRINT_RGB = (30, 30, 30)
_UPSCALE = 2
_PANEL_GAP = 14
_LABEL_FONT_PX = 13


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _labeled(img: Image.Image, label: str) -> Image.Image:
    """Add a small caption strip above a panel so two side-by-side panels
    are unambiguous about which is which."""
    font = ImageFont.load_default(size=_LABEL_FONT_PX)
    strip_h = _LABEL_FONT_PX + 8
    out = Image.new("RGB", (img.width, img.height + strip_h), (20, 20, 20))
    draw = ImageDraw.Draw(out)
    draw.text((4, 3), label, fill=(190, 190, 190), font=font)
    out.paste(img, (0, strip_h))
    return out


def render_preview(
    dem: np.ndarray,
    masks: dict[str, np.ndarray],
    footprint_mask: np.ndarray | None,
    out_path: str,
    colors: dict[str, str],
    plaque_mask: np.ndarray | None = None,
    plaque_color_hex: str | None = None,
) -> None:
    """masks: resolved (mutually-exclusive) per-layer vertex masks, e.g. from
    pipeline.compose_masks. colors: layer name -> "RRGGBB" hex string.

    plaque_mask (optional): the underside text mask exactly as built into
    the mesh (mirrored for printing) - this function un-mirrors it for
    display, since the preview should show what a person actually reads,
    not the mirrored build-time geometry.
    """
    znorm = (dem - dem.min()) / max(1e-6, float(dem.max() - dem.min()))
    gray = (80 + znorm * 140).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1)

    # Masks are already mutually exclusive (resolve_overlaps), so draw order
    # doesn't affect correctness - just iterate whatever layers are present.
    for name, mask in masks.items():
        if name == "base" or mask is None or not mask.any():
            continue
        color_hex = colors.get(name)
        if color_hex is None:
            continue
        rgb[mask] = _hex_to_rgb(color_hex)

    if footprint_mask is not None:
        rgb[~footprint_mask] = _OUTSIDE_FOOTPRINT_RGB

    top_img = Image.fromarray(rgb, "RGB")
    top_img = top_img.resize((top_img.width * _UPSCALE, top_img.height * _UPSCALE), Image.NEAREST)

    if plaque_mask is None or not plaque_mask.any():
        top_img.save(out_path)
        return

    base_rgb = _hex_to_rgb(colors.get("base", "808080"))
    bottom_rgb = np.full((*plaque_mask.shape, 3), base_rgb, dtype=np.uint8)
    readable_mask = plaque_mask[:, ::-1]  # undo the build-time mirror for display
    bottom_rgb[readable_mask] = _hex_to_rgb(plaque_color_hex or "8B5A2B")
    if footprint_mask is not None:
        bottom_rgb[~footprint_mask[:, ::-1]] = _OUTSIDE_FOOTPRINT_RGB

    bottom_img = Image.fromarray(bottom_rgb, "RGB")
    bottom_img = bottom_img.resize((bottom_img.width * _UPSCALE, bottom_img.height * _UPSCALE), Image.NEAREST)

    top_labeled = _labeled(top_img, "top")
    bottom_labeled = _labeled(bottom_img, "underside (flipped over, readable)")

    h = max(top_labeled.height, bottom_labeled.height)
    combined = Image.new("RGB", (top_labeled.width + _PANEL_GAP + bottom_labeled.width, h), (20, 20, 20))
    combined.paste(top_labeled, (0, 0))
    combined.paste(bottom_labeled, (top_labeled.width + _PANEL_GAP, 0))
    combined.save(out_path)
