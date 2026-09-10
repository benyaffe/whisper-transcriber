#!/usr/bin/env python3
"""
Generate the PodcastNotesWT app icon.

A placeholder, and deliberately a plain one: a page with a waveform through
it, which is what the app does. Replace resources/icon.png with anything
better and re-run this to rebuild the .icns.

    venv/bin/python scripts/create_icon.py
"""

import os
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "resources")

SIZE = 1024
INK = (24, 31, 46)
PAPER = (255, 255, 255)
ACCENT = (61, 122, 255)


def draw_icon() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin, radius = 76, 210
    draw.rounded_rectangle(
        [margin, margin, SIZE - margin, SIZE - margin], radius=radius, fill=INK
    )

    # A page, slightly inset and off-centre so the waveform has room.
    page = [286, 236, SIZE - 286, SIZE - 236]
    draw.rounded_rectangle(page, radius=34, fill=PAPER)

    # Ruled lines, the top two full width and the rest interrupted by the wave.
    left, right = page[0] + 58, page[2] - 58
    y = page[1] + 92
    for i in range(4):
        width = right if i < 2 else left + (right - left) * 0.55
        draw.rounded_rectangle([left, y, width, y + 16], radius=8,
                               fill=(212, 218, 228))
        y += 74

    # The waveform: uneven on purpose, because a symmetrical one reads as a
    # logo for an equaliser rather than for speech.
    heights = [0.30, 0.62, 0.44, 0.92, 0.55, 0.74, 0.36]
    bar_w, gap = 30, 26
    total = len(heights) * bar_w + (len(heights) - 1) * gap
    x = (SIZE - total) / 2
    centre = page[3] - 150
    for h in heights:
        half = (h * 190) / 2
        draw.rounded_rectangle(
            [x, centre - half, x + bar_w, centre + half], radius=15, fill=ACCENT
        )
        x += bar_w + gap

    return img


def main():
    os.makedirs(RESOURCES, exist_ok=True)
    icon = draw_icon()
    png_path = os.path.join(RESOURCES, "icon.png")
    icon.save(png_path)
    print(f"wrote {png_path}")

    # macOS wants an .icns, which iconutil builds from a sized iconset.
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "icon.iconset")
        os.makedirs(iconset)
        for size in (16, 32, 128, 256, 512):
            icon.resize((size, size), Image.LANCZOS).save(
                os.path.join(iconset, f"icon_{size}x{size}.png"))
            icon.resize((size * 2, size * 2), Image.LANCZOS).save(
                os.path.join(iconset, f"icon_{size}x{size}@2x.png"))

        icns_path = os.path.join(RESOURCES, "icon.icns")
        result = subprocess.run(
            ["iconutil", "-c", "icns", iconset, "-o", icns_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("iconutil failed:", result.stderr.strip())
            return 1
        print(f"wrote {icns_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
