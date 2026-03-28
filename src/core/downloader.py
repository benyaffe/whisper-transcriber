"""
Video downloader using yt-dlp for YouTube, Instagram, TikTok, etc.
"""

import os
import re
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.file_utils import get_bundled_binary


class VideoDownloader(QThread):
    """Downloads video/audio from URLs using yt-dlp."""

    progress = pyqtSignal(float, str)  # percent, status message
    completed = pyqtSignal(str)  # output filepath
    error = pyqtSignal(str)  # error message

    def __init__(self, url: str, output_dir: str):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self._cancelled = False

    def run(self):
        try:
            import yt_dlp

            # Output template - use title and extension
            output_template = os.path.join(
                self.output_dir,
                '%(title)s.%(ext)s'
            )

            # Get bundled ffmpeg location for merging video+audio
            ffmpeg_path = get_bundled_binary('ffmpeg')
            ffmpeg_dir = os.path.dirname(ffmpeg_path) if os.path.isabs(ffmpeg_path) else None

            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': output_template,
                'progress_hooks': [self._progress_hook],
                'quiet': True,
                'no_warnings': True,
            }

            # Tell yt-dlp where to find ffmpeg
            if ffmpeg_dir:
                ydl_opts['ffmpeg_location'] = ffmpeg_dir

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if self._cancelled:
                    return

                # Extract info first to get filename
                info = ydl.extract_info(self.url, download=False)
                if self._cancelled:
                    return

                # Download
                ydl.download([self.url])

                if self._cancelled:
                    return

                # Determine the output file path
                filename = ydl.prepare_filename(info)

                # Handle potential format changes
                if not os.path.exists(filename):
                    # Try common alternatives
                    base, _ = os.path.splitext(filename)
                    for ext in ['.mp4', '.webm', '.mkv', '.m4a', '.mp3']:
                        alt = base + ext
                        if os.path.exists(alt):
                            filename = alt
                            break

                self.completed.emit(filename)

        except Exception as e:
            error_msg = str(e)

            # Provide helpful context for common errors
            if "Video unavailable" in error_msg:
                error_msg = (
                    "Video unavailable. This could mean:\n"
                    "• The video is private or deleted\n"
                    "• Geographic restrictions apply\n"
                    "• The URL is incorrect"
                )
            elif "Sign in" in error_msg:
                error_msg = (
                    "This content requires authentication.\n\n"
                    "Suggestion: Try a public video, or use browser cookies with yt-dlp."
                )
            elif "Unsupported URL" in error_msg:
                error_msg = (
                    f"URL not supported: {self.url}\n\n"
                    "Supported sites include YouTube, Instagram, TikTok, Twitter/X, "
                    "Vimeo, and many more. Check that the URL is complete."
                )

            self.error.emit(error_msg)

    def _progress_hook(self, d):
        if self._cancelled:
            raise Exception("Download cancelled")

        if d['status'] == 'downloading':
            # Parse progress
            percent_str = d.get('_percent_str', '0%')
            # Remove ANSI codes and parse
            percent_str = re.sub(r'\x1b\[[0-9;]*m', '', percent_str)
            try:
                percent = float(percent_str.replace('%', '').strip())
            except ValueError:
                percent = 0

            # Get ETA if available
            eta = d.get('_eta_str', '')
            eta = re.sub(r'\x1b\[[0-9;]*m', '', eta).strip()

            # Format status consistently
            if eta and eta != 'Unknown':
                status = f"ETA: {eta}"
            else:
                status = "Downloading..."

            self.progress.emit(percent, status)

        elif d['status'] == 'finished':
            self.progress.emit(100, "Processing...")

    def cancel(self):
        self._cancelled = True
