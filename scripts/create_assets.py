#!/usr/bin/env python3
"""
Generate app icon and DMG background for Whisper Transcriber.
"""

from PIL import Image, ImageDraw, ImageFont
import os
import subprocess

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), '..', 'resources')
os.makedirs(RESOURCES_DIR, exist_ok=True)


def create_app_icon():
    """Create a professional app icon with waveform/microphone design."""
    size = 1024
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background - rounded square with gradient-like effect (blue/purple)
    margin = 80
    corner_radius = 180

    # Main background color - deep blue
    bg_color = (41, 98, 255)  # Vibrant blue

    # Draw rounded rectangle background
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=corner_radius,
        fill=bg_color
    )

    # Add a subtle lighter overlay at top for depth
    overlay = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin // 2],
        radius=corner_radius,
        fill=(255, 255, 255, 30)
    )
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Draw waveform bars (representing audio/transcription)
    center_x = size // 2
    center_y = size // 2
    bar_width = 45
    bar_spacing = 65
    bar_color = (255, 255, 255)  # White

    # Heights for waveform effect (symmetric)
    heights = [120, 200, 300, 380, 300, 200, 120]

    start_x = center_x - (len(heights) // 2) * bar_spacing

    for i, height in enumerate(heights):
        x = start_x + i * bar_spacing
        y1 = center_y - height // 2
        y2 = center_y + height // 2

        # Draw rounded bar
        draw.rounded_rectangle(
            [x - bar_width // 2, y1, x + bar_width // 2, y2],
            radius=bar_width // 2,
            fill=bar_color
        )

    # Add small "W" text at bottom for Whisper branding
    try:
        # Try to use SF Pro or Helvetica
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
    except:
        font = ImageFont.load_default()

    text = "W"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = center_x - text_width // 2
    text_y = size - margin - 180
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 200), font=font)

    # Save as PNG first
    png_path = os.path.join(RESOURCES_DIR, 'icon.png')
    img.save(png_path, 'PNG')
    print(f"Created {png_path}")

    # Create iconset for macOS
    iconset_path = os.path.join(RESOURCES_DIR, 'icon.iconset')
    os.makedirs(iconset_path, exist_ok=True)

    # Generate all required sizes
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for s in sizes:
        resized = img.resize((s, s), Image.Resampling.LANCZOS)
        resized.save(os.path.join(iconset_path, f'icon_{s}x{s}.png'))
        if s <= 512:
            # @2x versions
            resized_2x = img.resize((s * 2, s * 2), Image.Resampling.LANCZOS)
            resized_2x.save(os.path.join(iconset_path, f'icon_{s}x{s}@2x.png'))

    # Convert to icns
    icns_path = os.path.join(RESOURCES_DIR, 'icon.icns')
    subprocess.run(['iconutil', '-c', 'icns', iconset_path, '-o', icns_path], check=True)
    print(f"Created {icns_path}")

    # Cleanup iconset
    import shutil
    shutil.rmtree(iconset_path)

    return icns_path


def create_dmg_background():
    """Create DMG background with arrow pointing to Applications."""
    width = 660
    height = 400

    img = Image.new('RGBA', (width, height), (245, 245, 247, 255))  # Light gray
    draw = ImageDraw.Draw(img)

    # Add instruction text at top
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = font_large

    # Title text
    title = "Drag to install Whisper Transcriber"
    bbox = draw.textbbox((0, 0), title, font=font_large)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, 30), title, fill=(80, 80, 80), font=font_large)

    subtitle = "Then open Applications and run Whisper Transcriber"
    bbox = draw.textbbox((0, 0), subtitle, font=font_small)
    text_width = bbox[2] - bbox[0]
    draw.text(((width - text_width) // 2, 55), subtitle, fill=(120, 120, 120), font=font_small)

    # Draw arrow in the middle
    arrow_y = height // 2 + 20
    arrow_start_x = width // 2 - 60
    arrow_end_x = width // 2 + 60
    arrow_color = (100, 100, 100)

    # Arrow shaft
    shaft_height = 20
    draw.rectangle(
        [arrow_start_x, arrow_y - shaft_height // 2,
         arrow_end_x - 30, arrow_y + shaft_height // 2],
        fill=arrow_color
    )

    # Arrow head
    draw.polygon([
        (arrow_end_x - 40, arrow_y - 35),  # Top
        (arrow_end_x, arrow_y),             # Point
        (arrow_end_x - 40, arrow_y + 35),  # Bottom
    ], fill=arrow_color)

    # Save
    bg_path = os.path.join(RESOURCES_DIR, 'dmg_background.png')
    img.save(bg_path, 'PNG')
    print(f"Created {bg_path}")

    return bg_path


if __name__ == '__main__':
    create_app_icon()
    create_dmg_background()
    print("\nAssets created successfully!")
