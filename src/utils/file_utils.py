"""
File handling utilities.
"""

import os
import re
import subprocess
import json
import sys


def get_bundled_binary(name: str) -> str:
    """
    Get path to a bundled binary (ffmpeg/ffprobe).
    Falls back to system PATH if not bundled.
    """
    # When running as .app bundle, binaries are in Resources/bin/
    if getattr(sys, 'frozen', False):
        # Running as bundled app
        bundle_dir = os.path.dirname(sys.executable)
        resources_dir = os.path.abspath(os.path.join(bundle_dir, '..', 'Resources'))
        bundled_path = os.path.join(resources_dir, 'bin', name)
        if os.path.exists(bundled_path):
            return bundled_path

    # Fall back to system binary
    return name


def get_supported_extensions() -> list:
    """Return list of supported audio/video file extensions."""
    return [
        # Video
        '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v',
        # Audio
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus'
    ]


def is_url(text: str) -> bool:
    """Check if text is a valid URL for video download."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP address
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return bool(url_pattern.match(text))


def get_file_info(filepath: str) -> dict:
    """
    Get media file information using ffprobe.
    Returns dict with format, duration, codec info.
    """
    info = {
        'format': 'Unknown',
        'duration': 0,
        'duration_str': 'Unknown',
        'video_codec': None,
        'audio_codec': None,
        'has_video': False,
        'has_audio': False
    }

    try:
        # Use ffprobe to get file info
        cmd = [
            get_bundled_binary('ffprobe'),
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            filepath
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return info

        data = json.loads(result.stdout)

        # Get format info
        if 'format' in data:
            fmt = data['format']
            info['format'] = fmt.get('format_long_name', fmt.get('format_name', 'Unknown'))

            duration = float(fmt.get('duration', 0))
            info['duration'] = duration

            # Format duration as human-readable
            mins, secs = divmod(int(duration), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                info['duration_str'] = f"{hours}h {mins}m {secs}s"
            elif mins:
                info['duration_str'] = f"{mins}m {secs}s"
            else:
                info['duration_str'] = f"{secs}s"

        # Get stream info
        for stream in data.get('streams', []):
            codec_type = stream.get('codec_type')
            if codec_type == 'video':
                info['has_video'] = True
                info['video_codec'] = stream.get('codec_name')
            elif codec_type == 'audio':
                info['has_audio'] = True
                info['audio_codec'] = stream.get('codec_name')

    except subprocess.TimeoutExpired:
        pass
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    except FileNotFoundError:
        # ffprobe not installed
        info['format'] = os.path.splitext(filepath)[1].upper().replace('.', '')

    return info


def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract audio from video file for faster transcription.
    Returns path to extracted audio file.
    """
    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = base + '_audio.wav'

    cmd = [
        get_bundled_binary('ffmpeg'),
        '-y',  # Overwrite
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '16000',  # 16kHz (Whisper's native rate)
        '-ac', '1',  # Mono
        output_path
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e.stderr.decode() if e.stderr else str(e)}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out (>5 minutes)")


def generate_output_paths(source_path: str) -> tuple:
    """
    Generate output paths for VTT and TXT files.
    Returns (vtt_path, txt_path)
    """
    base, _ = os.path.splitext(source_path)
    return base + '.vtt', base + '.txt'
