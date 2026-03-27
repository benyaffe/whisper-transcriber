"""
Smart error handling with suggestions for remediation.
"""


def get_error_suggestion(error_message: str) -> str:
    """
    Analyze error message and return helpful suggestion.
    """
    error_lower = error_message.lower()

    # Model/Memory issues
    if 'out of memory' in error_lower or 'oom' in error_lower or 'memory' in error_lower:
        return (
            "💡 Suggestion: Your system ran out of memory.\n"
            "• Close other applications to free up RAM\n"
            "• Try a smaller model (tiny or base) in Settings\n"
            "• For very long files, consider splitting them first\n\n"
            "🔧 Request feature: Add automatic file splitting for large media"
        )

    # FFmpeg issues
    if 'ffmpeg' in error_lower or 'ffprobe' in error_lower:
        if 'not found' in error_lower or 'no such file' in error_lower:
            return (
                "💡 Suggestion: FFmpeg is not installed.\n"
                "• Install via Homebrew: brew install ffmpeg\n"
                "• Or download from https://ffmpeg.org/download.html\n\n"
                "🔧 Request feature: Bundle FFmpeg with the app"
            )
        else:
            return (
                "💡 Suggestion: FFmpeg encountered an error.\n"
                "• The media file may be corrupted\n"
                "• Try re-downloading or re-encoding the file\n"
                "• Check if the file plays correctly in VLC"
            )

    # Codec/Format issues
    if 'codec' in error_lower or 'unsupported' in error_lower or 'invalid' in error_lower:
        return (
            "💡 Suggestion: The file format may not be supported.\n"
            "• Try converting to MP4 or WAV first\n"
            "• Use: ffmpeg -i input.file -c:a aac output.mp4\n\n"
            "🔧 Request feature: Add automatic format conversion"
        )

    # Network issues
    if 'network' in error_lower or 'connection' in error_lower or 'timeout' in error_lower:
        return (
            "💡 Suggestion: Network error occurred.\n"
            "• Check your internet connection\n"
            "• The server may be temporarily unavailable\n"
            "• Try again in a few minutes"
        )

    # Model download issues
    if 'download' in error_lower and 'model' in error_lower:
        return (
            "💡 Suggestion: Failed to download Whisper model.\n"
            "• Check your internet connection\n"
            "• The model will be cached after first download\n"
            "• Try using a smaller model (tiny, base, small)"
        )

    # Whisper/transcription issues
    if 'whisper' in error_lower or 'transcri' in error_lower:
        return (
            "💡 Suggestion: Transcription failed.\n"
            "• The audio quality may be too low\n"
            "• Try a larger model for better accuracy\n"
            "• Ensure the file contains audible speech"
        )

    # File access issues
    if 'permission' in error_lower or 'access denied' in error_lower:
        return (
            "💡 Suggestion: File access denied.\n"
            "• Check file permissions\n"
            "• Make sure the file isn't open in another app\n"
            "• Try moving the file to your Desktop or Downloads folder"
        )

    if 'no such file' in error_lower or 'not found' in error_lower or 'does not exist' in error_lower:
        return (
            "💡 Suggestion: File not found.\n"
            "• The file may have been moved or deleted\n"
            "• Check that the file path is correct\n"
            "• Try drag-dropping the file again"
        )

    # yt-dlp issues
    if 'yt-dlp' in error_lower or 'youtube' in error_lower:
        return (
            "💡 Suggestion: Video download issue.\n"
            "• Make sure the URL is correct and complete\n"
            "• The video may be private, deleted, or geo-restricted\n"
            "• Try updating yt-dlp: pip install -U yt-dlp"
        )

    # GPU/CUDA issues
    if 'cuda' in error_lower or 'gpu' in error_lower:
        return (
            "💡 Suggestion: GPU acceleration issue.\n"
            "• The app will fall back to CPU (slower but works)\n"
            "• For GPU support, ensure CUDA is properly installed\n"
            "• On Mac, GPU acceleration uses Metal (Apple Silicon)"
        )

    # Default suggestion
    return (
        "💡 If this error persists:\n"
        "• Try a different file to isolate the issue\n"
        "• Restart the application\n"
        "• Report the issue at: github.com/your-repo/whisper-transcriber/issues"
    )
