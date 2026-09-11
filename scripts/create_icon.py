#!/usr/bin/env python3
"""
Generate the PodcastNotesWT app icon and the installer background.

A placeholder, and deliberately a plain one: a page with a waveform through
it, which is what the app does. Replace resources/icon.png with anything
better and re-run this to rebuild the .icns.

The installer background lives here too, and used to live in a second script
that was left behind when the app was renamed. It kept generating "Drag to
install Whisper Transcriber", which is what every person who downloaded the
DMG saw first. One script, so the name can only be wrong in one place.

    venv/bin/python scripts/create_icon.py
"""

import os
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "resources")

SIZE = 1024
INK = (24, 31, 46)
PAPER = (255, 255, 255)
ACCENT = (61, 122, 255)

# The name on the installer window. Read from nowhere else on purpose: the
# app is called this, and a second copy of the string is how the last one got
# out of date.
APP_NAME = "PodcastNotesWT"

DMG_WIDTH, DMG_HEIGHT = 660, 400


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


def draw_dmg_background() -> Image.Image:
    """The window somebody sees when they open the DMG: drag this, into there."""
    img = Image.new("RGBA", (DMG_WIDTH, DMG_HEIGHT), (245, 245, 247, 255))
    draw = ImageDraw.Draw(img)

    try:
        big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        big = small = ImageFont.load_default()

    for text, font, y, colour in (
        (f"Drag to install {APP_NAME}", big, 30, (80, 80, 80)),
        (f"Then open Applications and run {APP_NAME}", small, 55, (120, 120, 120)),
    ):
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((DMG_WIDTH - (box[2] - box[0])) // 2, y), text, fill=colour, font=font)

    middle = DMG_HEIGHT // 2 + 20
    left = DMG_WIDTH // 2 - 60
    right = DMG_WIDTH // 2 + 60
    draw.rectangle([left, middle - 10, right - 30, middle + 10], fill=(100, 100, 100))
    draw.polygon(
        [(right - 40, middle - 35), (right, middle), (right - 40, middle + 35)],
        fill=(100, 100, 100),
    )
    return img


def main():
    os.makedirs(RESOURCES, exist_ok=True)

    dmg_path = os.path.join(RESOURCES, "dmg_background.png")
    draw_dmg_background().save(dmg_path)
    print(f"wrote {dmg_path}")

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
