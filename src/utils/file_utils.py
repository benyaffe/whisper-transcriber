"""
File handling utilities.
"""

import os
import re
import subprocess
import json
import sys


def check_network_connectivity() -> tuple[bool, str]:
    """
    Check if network is available for HuggingFace downloads.
    Returns (is_connected, message).
    """
    import socket
    try:
        # Try to connect to HuggingFace
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True, "Network available"
    except (socket.timeout, OSError):
        pass

    # Fallback: try Google DNS
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True, "Network available (HuggingFace may be slow)"
    except (socket.timeout, OSError):
        return False, "No internet connection. Speaker ID requires network access for model downloads."


def check_ffmpeg_health() -> tuple[bool, str]:
    """
    Verify FFmpeg is available and functional.
    Returns (is_healthy, message).
    """
    ffmpeg_path = get_bundled_binary('ffmpeg')

    try:
        result = subprocess.run(
            [ffmpeg_path, '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Extract version from first line
            version_line = result.stdout.split('\n')[0] if result.stdout else "unknown"
            return True, f"FFmpeg OK: {version_line[:50]}"
        else:
            return False, f"FFmpeg error: {result.stderr[:50]}"
    except FileNotFoundError:
        return False, "FFmpeg not found. Audio/video processing will fail."
    except subprocess.TimeoutExpired:
        return False, "FFmpeg timed out during health check."
    except Exception as e:
        return False, f"FFmpeg check failed: {str(e)[:50]}"


def validate_input_file(filepath: str) -> tuple[bool, str]:
    """
    Validate an input file before queueing for transcription.
    Returns (is_valid, message).
    """
    # Check exists
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"

    # Check is file (not directory)
    if not os.path.isfile(filepath):
        return False, f"Not a file: {filepath}"

    # Check readable
    if not os.access(filepath, os.R_OK):
        return False, f"Cannot read file (permission denied): {os.path.basename(filepath)}"

    # Check not empty
    size = os.path.getsize(filepath)
    if size == 0:
        return False, f"File is empty (0 bytes): {os.path.basename(filepath)}"

    # Check extension
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in get_supported_extensions():
        return False, f"Unsupported format '{ext}'. Supported: mp4, mp3, wav, etc."

    # All checks passed
    return True, f"File OK: {os.path.basename(filepath)} ({size / 1024 / 1024:.1f} MB)"


def get_bundled_binary(name: str) -> str:
    """
    Get path to a bundled binary (ffmpeg/ffprobe).
    Falls back to system PATH if not bundled.
    """
    if getattr(sys, 'frozen', False):
        # Running as bundled app - check multiple possible locations
        bundle_dir = os.path.dirname(sys.executable)

        # PyInstaller puts data files in various locations
        possible_paths = [
            # PyInstaller: Contents/Frameworks/bin/ (common for COLLECT mode)
            os.path.abspath(os.path.join(bundle_dir, '..', 'Frameworks', 'bin', name)),
            # PyInstaller: Contents/MacOS/bin/
            os.path.join(bundle_dir, 'bin', name),
            # py2app: Contents/Resources/bin/
            os.path.abspath(os.path.join(bundle_dir, '..', 'Resources', 'bin', name)),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

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
