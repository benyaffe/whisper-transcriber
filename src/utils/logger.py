"""
Logging utility for Whisper Transcriber.
Provides file-based logging for debugging and error reporting.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path


# Log directory
LOG_DIR = Path.home() / "Library" / "Logs" / "WhisperTranscriber"
LOG_FILE = LOG_DIR / "whisper_transcriber.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_LOG_FILES = 3


def setup_logging(enable_file_logging: bool = True) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        enable_file_logging: If True, logs to file in ~/Library/Logs/

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("WhisperTranscriber")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (DEBUG level) - optional
    if enable_file_logging:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            # Rotate log if too large
            if LOG_FILE.exists() and LOG_FILE.stat().st_size > MAX_LOG_SIZE:
                _rotate_logs()

            file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)

            # Log startup
            logger.debug("=" * 60)
            logger.debug(f"Whisper Transcriber started at {datetime.now().isoformat()}")
            logger.debug(f"Python {sys.version}")
            logger.debug(f"Log file: {LOG_FILE}")
            logger.debug("=" * 60)

        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")

    return logger


def _rotate_logs():
    """Rotate log files when they get too large."""
    try:
        for i in range(MAX_LOG_FILES - 1, 0, -1):
            old_log = LOG_DIR / f"whisper_transcriber.{i}.log"
            new_log = LOG_DIR / f"whisper_transcriber.{i + 1}.log"
            if old_log.exists():
                if new_log.exists():
                    new_log.unlink()
                old_log.rename(new_log)

        if LOG_FILE.exists():
            backup = LOG_DIR / "whisper_transcriber.1.log"
            LOG_FILE.rename(backup)
    except Exception:
        pass  # Best effort rotation


def get_logger() -> logging.Logger:
    """Get the application logger."""
    logger = logging.getLogger("WhisperTranscriber")
    if not logger.handlers:
        setup_logging()
    return logger


def get_debug_info() -> str:
    """
    Collect debug information for error reports.
    Returns a formatted string suitable for copying.
    """
    import platform

    info_lines = [
        "=== Whisper Transcriber Debug Info ===",
        f"Date: {datetime.now().isoformat()}",
        f"OS: {platform.system()} {platform.release()}",
        f"Python: {sys.version}",
        f"Executable: {sys.executable}",
        f"Frozen: {getattr(sys, 'frozen', False)}",
    ]

    # Check for GPU
    try:
        import torch
        info_lines.append(f"PyTorch: {torch.__version__}")
        info_lines.append(f"CUDA available: {torch.cuda.is_available()}")
        info_lines.append(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        info_lines.append("PyTorch: Not installed")

    # Check FFmpeg
    try:
        from src.utils.file_utils import check_ffmpeg_health
        ffmpeg_ok, ffmpeg_msg = check_ffmpeg_health()
        info_lines.append(f"FFmpeg: {'OK' if ffmpeg_ok else 'ERROR'} - {ffmpeg_msg[:50]}")
    except Exception as e:
        info_lines.append(f"FFmpeg: Check failed - {str(e)[:30]}")

    # Recent log entries
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent = lines[-20:] if len(lines) > 20 else lines
                info_lines.append("")
                info_lines.append("=== Recent Log Entries ===")
                info_lines.extend([line.rstrip() for line in recent])
        except Exception:
            pass

    return "\n".join(info_lines)


def log_exception(exc: Exception, context: str = ""):
    """Log an exception with full traceback."""
    import traceback
    logger = get_logger()
    tb = traceback.format_exc()
    logger.error(f"Exception in {context}: {exc}")
    logger.debug(f"Traceback:\n{tb}")
