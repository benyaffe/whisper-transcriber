"""
Pytest configuration for Whisper Transcriber tests.
"""

import sys
import os
from types import SimpleNamespace

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


@pytest.fixture(scope="session")
def qt_app():
    """One offscreen QApplication for widget tests.

    Session-scoped because Qt permits exactly one QApplication per process.
    Forced offscreen so a test run never steals focus or flashes a window;
    export QT_QPA_PLATFORM=cocoa to watch a widget test render for real.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


# --- Diarization test doubles -------------------------------------------------
#
# run_diarization() otherwise needs a HuggingFace token, three gated model
# downloads, torch and several minutes of CPU. These let it be exercised in
# milliseconds. WT-1 and WT-3 are expected to reuse them.


class FakeAnnotation:
    """Stands in for pyannote.core.Annotation, which is all run_diarization uses."""

    def __init__(self, turns=()):
        # turns: iterable of (start, end, speaker_label)
        self._turns = list(turns)

    def itertracks(self, yield_label=False):
        for start, end, speaker in self._turns:
            segment = SimpleNamespace(start=start, end=end)
            yield (segment, None, speaker) if yield_label else (segment, None)


class FakeAnnotationWithLabels(FakeAnnotation):
    """FakeAnnotation that also implements labels(), like pyannote's Annotation.

    Ordering matters: speaker_embeddings rows are aligned with labels(), and
    pyannote sorts labels by str, so the double sorts too.
    """

    def labels(self):
        return sorted({speaker for _, _, speaker in self._turns}, key=str)


class FakeDiarizeOutput:
    """Shape of pyannote 4.x's DiarizeOutput dataclass."""

    def __init__(self, turns=(), embeddings=None, exclusive_turns=None):
        self.speaker_diarization = FakeAnnotationWithLabels(turns)
        self.exclusive_speaker_diarization = FakeAnnotationWithLabels(
            turns if exclusive_turns is None else exclusive_turns
        )
        self.speaker_embeddings = embeddings


def make_embeddings(*rows, dimension=256):
    """Build a (n, dimension) float32 array from per-row fill values.

    A fill of 0 produces the all-zero row pyannote emits for a padded phantom
    speaker; float("nan") produces the degenerate row from its max_clusters<2
    shortcut. Both must be rejected as voice profiles.
    """
    import numpy as np

    return np.array(
        [np.full(dimension, fill, dtype=np.float32) for fill in rows],
        dtype=np.float32,
    )


class FakePipeline:
    """Replays a scripted hook sequence instead of doing any real work.

    Records what it was called with so tests can assert the hook was actually
    threaded through, which is the whole point of WT-2.
    """

    #  (step_name, kwargs) pairs, transcribed from pyannote 4.0.7's
    #  speaker_diarization.py. Mirrors PYANNOTE_SEQUENCE in
    #  tests/test_diarization_progress.py.
    DEFAULT_SEQUENCE = [
        ("segmentation", {"completed": 0, "total": 2}),
        ("segmentation", {"completed": 2, "total": 2}),
        ("segmentation", {}),
        ("speaker_counting", {}),
        ("embeddings", {"completed": 0, "total": 2}),
        ("embeddings", {"completed": 2, "total": 2}),
        ("embeddings", {}),
        ("discrete_diarization", {}),
    ]

    def __init__(self, output=None, sequence=None):
        self.output = output if output is not None else FakeDiarizeOutput(
            turns=[(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
        )
        self.sequence = self.DEFAULT_SEQUENCE if sequence is None else sequence
        self.called_with_file = None
        self.received_hook = None
        self.moved_to = []

    def to(self, device):
        self.moved_to.append(device)
        return self

    def __call__(self, file, hook=None, **kwargs):
        self.called_with_file = file
        self.received_hook = hook
        if hook is not None:
            for step_name, step_kwargs in self.sequence:
                hook(step_name, None, file=file, **step_kwargs)
        return self.output


@pytest.fixture
def fake_pipeline():
    """A FakePipeline instance, so a test can inspect it after the call."""
    return FakePipeline()


@pytest.fixture
def patched_diarization(fake_pipeline, monkeypatch):
    """run_diarization with every slow/networked seam replaced.

    Patches the three module-level helpers on src.core.diarization directly,
    and pyannote.audio.Pipeline on the library module -- the latter follows the
    existing idiom in tests/test_suite.py, and works because diarization.py
    imports pyannote inside the function rather than at module scope.

    Yields the FakePipeline so tests can assert on what it received.
    """
    import pyannote.audio
    from src.core import diarization

    monkeypatch.setattr(diarization, "validate_hf_token", lambda token: (True, "ok"))
    monkeypatch.setattr(diarization, "_ensure_models_downloaded", lambda token, log: None)
    monkeypatch.setattr(
        diarization,
        "_decode_audio_to_tensor",
        lambda path, sample_rate=16000: SimpleNamespace(shape=(1, sample_rate)),
    )
    # from_pretrained hands back the *same* instance the test holds, so the
    # test can inspect what the pipeline was called with afterwards.
    monkeypatch.setattr(
        pyannote.audio,
        "Pipeline",
        SimpleNamespace(from_pretrained=lambda *a, **k: fake_pipeline),
    )

    yield fake_pipeline
