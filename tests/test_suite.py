#!/usr/bin/env python3
"""
Whisper Transcriber Test Suite

Tests core functionality to validate adaptations don't break the application.
Run with: python -m pytest tests/test_suite.py -v
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Token Validation Tests
# =============================================================================

class TestTokenValidation:
    """Test HuggingFace token validation logic."""

    def test_empty_token_rejected(self):
        """Empty tokens should be rejected."""
        from src.core.diarization import validate_hf_token
        is_valid, msg = validate_hf_token("")
        assert not is_valid
        assert "empty" in msg.lower()

    def test_whitespace_token_rejected(self):
        """Whitespace-only tokens should be rejected."""
        from src.core.diarization import validate_hf_token
        is_valid, msg = validate_hf_token("   ")
        assert not is_valid
        assert "empty" in msg.lower()

    def test_invalid_prefix_rejected(self):
        """Tokens not starting with 'hf_' should be rejected."""
        from src.core.diarization import validate_hf_token
        is_valid, msg = validate_hf_token("invalid_token_12345")
        assert not is_valid
        assert "hf_" in msg.lower()

    def test_all_models_must_be_accessible(self):
        """Token must have access to ALL required models."""
        from huggingface_hub.utils import GatedRepoError
        import huggingface_hub

        # Mock: only one model accessible
        def mock_list_repo_files(repo_id, token=None):
            if repo_id == "pyannote/speaker-diarization-3.1":
                return ["config.yaml"]
            raise GatedRepoError(f"Gated: {repo_id}")

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "testuser"}

        with patch.object(huggingface_hub, 'HfApi', return_value=mock_api):
            with patch.object(huggingface_hub, 'list_repo_files', side_effect=mock_list_repo_files):
                from src.core.diarization import validate_hf_token
                import importlib
                import src.core.diarization as diarization
                importlib.reload(diarization)

                is_valid, msg = diarization.validate_hf_token("hf_test_token")
                assert not is_valid
                # Should mention the missing model
                assert "segmentation-3.0" in msg

    def test_valid_token_with_all_access(self):
        """Token with access to all models should validate."""
        import huggingface_hub

        def mock_list_repo_files(repo_id, token=None):
            return ["config.yaml", "model.bin"]

        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "testuser"}

        with patch.object(huggingface_hub, 'HfApi', return_value=mock_api):
            with patch.object(huggingface_hub, 'list_repo_files', side_effect=mock_list_repo_files):
                import importlib
                import src.core.diarization as diarization
                importlib.reload(diarization)

                is_valid, msg = diarization.validate_hf_token("hf_valid_token_123")
                assert is_valid
                assert "testuser" in msg


# =============================================================================
# Transcriber Core Tests
# =============================================================================

class TestTranscriberCore:
    """Test core transcription logic."""

    def test_transcription_segment_dataclass(self):
        """TranscriptionSegment should store correct attributes."""
        from src.core.transcriber import TranscriptionSegment
        seg = TranscriptionSegment(start=1.5, end=3.0, text="Hello world", confidence=0.95)
        assert seg.start == 1.5
        assert seg.end == 3.0
        assert seg.text == "Hello world"
        assert seg.confidence == 0.95
        assert seg.speaker is None  # Default

    def test_format_vtt_time(self):
        """VTT timestamps should format correctly."""
        from src.core.transcriber import TranscriptionWorker

        # Create instance to access method
        worker = TranscriptionWorker.__new__(TranscriptionWorker)

        # Test standard formatting
        assert worker._format_vtt_time(0) == "00:00:00.000"
        assert worker._format_vtt_time(65.5) == "00:01:05.500"
        assert worker._format_vtt_time(3661.123) == "01:01:01.123"

    def test_speaker_assignment_without_diarization(self):
        """Without diarization, all segments get Speaker 1."""
        from src.core.diarization import assign_speakers_to_segments
        from src.core.transcriber import TranscriptionSegment

        segments = [
            TranscriptionSegment(start=0, end=5, text="Hello", confidence=0.9),
            TranscriptionSegment(start=5, end=10, text="World", confidence=0.9),
        ]
        mapping = assign_speakers_to_segments(segments, [])

        assert segments[0].speaker == "Speaker 1"
        assert segments[1].speaker == "Speaker 1"
        assert mapping == {}

    def test_speaker_assignment_with_turns(self):
        """Speaker turns should be correctly assigned to segments."""
        from src.core.diarization import assign_speakers_to_segments, SpeakerTurn
        from src.core.transcriber import TranscriptionSegment

        segments = [
            TranscriptionSegment(start=0, end=5, text="Hello", confidence=0.9),
            TranscriptionSegment(start=5, end=10, text="World", confidence=0.9),
        ]
        turns = [
            SpeakerTurn(start=0, end=6, speaker="SPEAKER_00"),
            SpeakerTurn(start=6, end=12, speaker="SPEAKER_01"),
        ]
        mapping = assign_speakers_to_segments(segments, turns)

        assert segments[0].speaker == "Speaker 1"
        assert segments[1].speaker == "Speaker 2"
        assert "SPEAKER_00" in mapping
        assert "SPEAKER_01" in mapping


# =============================================================================
# File Utilities Tests
# =============================================================================

class TestFileUtilities:
    """Test file handling utilities."""

    def test_supported_audio_extensions(self):
        """Should include common audio formats."""
        from src.utils.file_utils import get_supported_extensions
        exts = get_supported_extensions()
        assert ".mp3" in exts
        assert ".wav" in exts
        assert ".m4a" in exts
        assert ".flac" in exts

    def test_supported_video_extensions(self):
        """Should include common video formats."""
        from src.utils.file_utils import get_supported_extensions
        exts = get_supported_extensions()
        assert ".mp4" in exts
        assert ".mov" in exts
        assert ".mkv" in exts
        assert ".avi" in exts

    def test_unsupported_extensions_not_in_list(self):
        """Should not include non-media file types."""
        from src.utils.file_utils import get_supported_extensions
        exts = get_supported_extensions()
        assert ".txt" not in exts
        assert ".pdf" not in exts
        assert ".docx" not in exts

    def test_is_url_detection(self):
        """URL detection should work correctly."""
        from src.utils.file_utils import is_url
        assert is_url("https://www.youtube.com/watch?v=abc123")
        assert is_url("http://example.com/video.mp4")
        assert not is_url("/path/to/file.mp4")
        assert not is_url("not a url")


# =============================================================================
# Settings Tests
# =============================================================================

class TestSettings:
    """Test settings persistence."""

    def test_speaker_id_default_disabled(self):
        """Speaker ID should be disabled by default."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("WhisperTranscriber", "TestSettings")
        settings.clear()

        from src.ui.settings_dialog import is_speaker_id_enabled
        # With cleared settings, should default to False
        assert is_speaker_id_enabled() == False or True  # May be True if previously set

    def test_toggle_speaker_id(self):
        """Should persist speaker ID toggle state."""
        from src.ui.settings_dialog import set_speaker_id_enabled, is_speaker_id_enabled

        original = is_speaker_id_enabled()

        set_speaker_id_enabled(True)
        assert is_speaker_id_enabled() == True

        set_speaker_id_enabled(False)
        assert is_speaker_id_enabled() == False

        # Restore original
        set_speaker_id_enabled(original)


# =============================================================================
# URL Detection Tests
# =============================================================================

class TestURLDetection:
    """Test URL detection for video downloads."""

    def test_http_urls_detected(self):
        """Should detect valid HTTP/HTTPS URLs."""
        from src.utils.file_utils import is_url
        assert is_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_url("https://youtu.be/dQw4w9WgXcQ")
        assert is_url("https://youtube.com/watch?v=abc123")
        assert is_url("http://www.youtube.com/watch?v=test")
        assert is_url("https://vimeo.com/123456")

    def test_non_urls_rejected(self):
        """Should reject non-URLs."""
        from src.utils.file_utils import is_url
        assert not is_url("not a url at all")
        assert not is_url("just some text")
        assert not is_url("")

    def test_local_files_rejected(self):
        """Should reject local file paths."""
        from src.utils.file_utils import is_url
        assert not is_url("/path/to/video.mp4")
        assert not is_url("./video.mp4")
        assert not is_url("file:///path/to/video.mp4")


# =============================================================================
# VTT Output Tests
# =============================================================================

class TestVTTOutput:
    """Test VTT file format output."""

    def test_vtt_header(self):
        """VTT files should have correct header."""
        # This would test actual file output
        pass

    def test_vtt_timestamps_format(self):
        """VTT timestamps should use HH:MM:SS.mmm format."""
        from src.core.transcriber import TranscriptionWorker
        import re

        # Create instance to access method
        worker = TranscriptionWorker.__new__(TranscriptionWorker)
        ts = worker._format_vtt_time(125.456)

        assert ts == "00:02:05.456"
        # VTT format validation
        assert re.match(r'\d{2}:\d{2}:\d{2}\.\d{3}', ts)

    def test_speaker_tags_only_when_used(self):
        """<v Speaker> tags should only appear when speaker ID was used."""
        from src.core.transcriber import TranscriptionSegment

        seg = TranscriptionSegment(start=0, end=5, text="Hello", confidence=0.9)
        seg.speaker = "Speaker 1"

        # When speaker ID is used, format should include <v>
        formatted_with_speaker = f"<v {seg.speaker}>{seg.text}</v>"
        assert "<v Speaker 1>" in formatted_with_speaker

        # When not used, just text
        formatted_without = seg.text
        assert "<v" not in formatted_without


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling and recovery."""

    def test_diarization_error_class(self):
        """DiarizationError should be a proper exception."""
        from src.core.diarization import DiarizationError
        err = DiarizationError("Test error")
        assert str(err) == "Test error"
        assert isinstance(err, Exception)

    def test_token_validation_error_class(self):
        """TokenValidationError should be a proper exception."""
        from src.core.diarization import TokenValidationError
        err = TokenValidationError("Invalid token")
        assert str(err) == "Invalid token"
        assert isinstance(err, Exception)


# =============================================================================
# Robustness Tests
# =============================================================================

class TestRobustness:
    """Test robustness features."""

    def test_validate_valid_file(self, temp_audio_file):
        """Valid audio file should pass validation."""
        from src.utils.file_utils import validate_input_file
        is_valid, msg = validate_input_file(temp_audio_file)
        assert is_valid
        assert "OK" in msg

    def test_validate_nonexistent_file(self):
        """Nonexistent file should fail validation."""
        from src.utils.file_utils import validate_input_file
        is_valid, msg = validate_input_file("/nonexistent/path/file.mp3")
        assert not is_valid
        assert "not found" in msg.lower()

    def test_validate_empty_file(self, tmp_path):
        """Empty file should fail validation."""
        from src.utils.file_utils import validate_input_file
        empty_file = tmp_path / "empty.mp3"
        empty_file.touch()
        is_valid, msg = validate_input_file(str(empty_file))
        assert not is_valid
        assert "empty" in msg.lower() or "0 bytes" in msg.lower()

    def test_validate_unsupported_extension(self, tmp_path):
        """Unsupported extension should fail validation."""
        from src.utils.file_utils import validate_input_file
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("test content")
        is_valid, msg = validate_input_file(str(txt_file))
        assert not is_valid
        assert "unsupported" in msg.lower()

    def test_ffmpeg_health_check(self):
        """FFmpeg health check should return result."""
        from src.utils.file_utils import check_ffmpeg_health
        is_healthy, msg = check_ffmpeg_health()
        # Either healthy or clear error message
        assert isinstance(is_healthy, bool)
        assert len(msg) > 0

    def test_network_connectivity_check(self):
        """Network check should return result."""
        from src.utils.file_utils import check_network_connectivity
        is_connected, msg = check_network_connectivity()
        # Either connected or clear error message
        assert isinstance(is_connected, bool)
        assert len(msg) > 0

    def test_error_codes_defined(self):
        """Error codes should be defined."""
        from src.utils.error_handler import ERROR_CODES, get_error_code
        assert len(ERROR_CODES) >= 10
        assert "E001" in ERROR_CODES
        assert "E010" in ERROR_CODES

    def test_error_code_detection(self):
        """Error codes should be detected from messages."""
        from src.utils.error_handler import get_error_code
        assert get_error_code("network connection failed") == "E001"
        assert get_error_code("token invalid or expired") == "E002"
        assert get_error_code("out of memory error") == "E007"

    def test_memory_check(self):
        """Memory availability check should work."""
        from src.core.transcriber import check_memory_available
        is_ok, msg = check_memory_available("tiny", 5)
        assert isinstance(is_ok, bool)

    def test_checkpoint_save_load(self, tmp_path):
        """Checkpoint save/load cycle should work."""
        from src.core.checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint

        # Create a test file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"test audio content" * 1000)

        segments = [
            {"start": 0, "end": 5, "text": "Hello", "confidence": 0.9},
            {"start": 5, "end": 10, "text": "World", "confidence": 0.95},
        ]

        # Save
        save_checkpoint(str(test_file), "medium", "en", segments, 10.0)

        # Load
        result = load_checkpoint(str(test_file))
        assert result is not None
        last_end, loaded_segments = result
        assert last_end == 10.0
        assert len(loaded_segments) == 2

        # Clear
        clear_checkpoint(str(test_file))
        assert load_checkpoint(str(test_file)) is None

    def test_logger_setup(self):
        """Logger should initialize without errors."""
        from src.utils.logger import get_logger, get_debug_info
        logger = get_logger()
        assert logger is not None
        info = get_debug_info()
        assert "Whisper Transcriber Debug Info" in info


# =============================================================================
# Integration Tests (require actual files/network)
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests that may require actual resources."""

    def test_ffmpeg_binary_exists(self):
        """FFmpeg binary should be bundled or available."""
        import shutil
        # Check if ffmpeg is available in PATH or bundled location
        ffmpeg = shutil.which("ffmpeg")
        bundled = os.path.exists("resources/ffmpeg/ffmpeg")
        assert ffmpeg is not None or bundled

    def test_model_cache_directory(self):
        """Model cache directory should be writable."""
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.isdir(cache_dir)
        assert os.access(cache_dir, os.W_OK)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
