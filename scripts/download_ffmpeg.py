#!/usr/bin/env python3
"""
Download static ffmpeg build for bundling.
Uses pre-built static binaries from evermeet.cx (macOS).
"""

import os
import sys
import urllib.request
import zipfile
import stat

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), '..', 'resources')
FFMPEG_DIR = os.path.join(RESOURCES_DIR, 'ffmpeg')

# Static ffmpeg builds for macOS (universal binaries)
FFMPEG_URL = "https://evermeet.cx/ffmpeg/getrelease/zip"
FFPROBE_URL = "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"


def download_and_extract(url: str, name: str) -> str:
    """Download zip and extract binary."""
    os.makedirs(FFMPEG_DIR, exist_ok=True)

    zip_path = os.path.join(FFMPEG_DIR, f'{name}.zip')
    binary_path = os.path.join(FFMPEG_DIR, name)

    if os.path.exists(binary_path):
        print(f"  {name} already exists, skipping download")
        return binary_path

    print(f"  Downloading {name}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"  Extracting {name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(FFMPEG_DIR)

    # Make executable
    os.chmod(binary_path, os.stat(binary_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Clean up zip
    os.remove(zip_path)

    return binary_path


def main():
    print("Downloading static ffmpeg binaries...")

    ffmpeg_path = download_and_extract(FFMPEG_URL, 'ffmpeg')
    ffprobe_path = download_and_extract(FFPROBE_URL, 'ffprobe')

    # Verify they work
    print("Verifying binaries...")
    import subprocess

    result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, text=True)
    if result.returncode == 0:
        version = result.stdout.split('\n')[0]
        print(f"  ffmpeg: {version}")
    else:
        print(f"  ERROR: ffmpeg failed to run")
        sys.exit(1)

    result = subprocess.run([ffprobe_path, '-version'], capture_output=True, text=True)
    if result.returncode == 0:
        version = result.stdout.split('\n')[0]
        print(f"  ffprobe: {version}")
    else:
        print(f"  ERROR: ffprobe failed to run")
        sys.exit(1)

    print(f"\nStatic binaries ready in: {FFMPEG_DIR}")
    print("These will be bundled into the app.")


if __name__ == '__main__':
    main()
