"""
Pytest configuration for Whisper Transcriber tests.
"""

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require network/files)"
    )


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    # Create a minimal valid WAV file header
    import struct

    audio_file = tmp_path / "test.wav"

    # Simple WAV header (44 bytes) + 1 second of silence at 44100Hz, 16-bit mono
    sample_rate = 44100
    bits_per_sample = 16
    channels = 1
    duration_seconds = 1

    data_size = sample_rate * duration_seconds * channels * (bits_per_sample // 8)

    with open(audio_file, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')

        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 1))   # PCM format
        f.write(struct.pack('<H', channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * channels * bits_per_sample // 8))
        f.write(struct.pack('<H', channels * bits_per_sample // 8))
        f.write(struct.pack('<H', bits_per_sample))

        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(b'\x00' * data_size)

    return str(audio_file)


@pytest.fixture
def mock_hf_token():
    """Provide a mock HuggingFace token for testing."""
    return "hf_mock_test_token_12345"
