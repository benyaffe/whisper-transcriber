"""
Tests for DiarizationProgressHook, the pyannote progress hook behind WT-2.

These are pure unit tests: no pyannote import, no torch, no network. The hook
only has to satisfy pyannote's callable contract, so it can be driven with a
scripted sequence of the calls pyannote actually makes.

Run with: python -m pytest tests/test_diarization_progress.py -v
"""

import pytest

from src.core.diarization import (
    DIARIZATION_STAGE_LABELS,
    DIARIZATION_STAGE_WEIGHTS,
    DiarizationProgressHook,
)


class FakeClock:
    """Manually advanced clock, so throttle behavior is testable without sleeping."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


@pytest.fixture
def recorder():
    """Collects (fraction, label) pairs emitted by the hook."""
    return []


@pytest.fixture
def clock():
    return FakeClock()


def make_hook(recorder, clock, min_interval_s=0.5):
    return DiarizationProgressHook(
        lambda fraction, label: recorder.append((fraction, label)),
        min_interval_s=min_interval_s,
        clock=clock,
    )


# The call sequence pyannote 4.x actually makes, from
# pyannote/audio/pipelines/speaker_diarization.py. Recorded here so the test
# fails loudly if a pyannote upgrade changes the contract.
PYANNOTE_SEQUENCE = [
    # segmentation, incremental via Inference
    ("segmentation", {"completed": 0, "total": 4}),
    ("segmentation", {"completed": 2, "total": 4}),
    ("segmentation", {"completed": 4, "total": 4}),
    # segmentation completion tick
    ("segmentation", {}),
    ("speaker_counting", {}),
    # embeddings, incremental
    ("embeddings", {"completed": 0, "total": 3}),
    ("embeddings", {"completed": 1, "total": 3}),
    ("embeddings", {"completed": 2, "total": 3}),
    ("embeddings", {"completed": 3, "total": 3}),
    # embeddings completion tick
    ("embeddings", {}),
    # clustering runs here, unhooked
    ("discrete_diarization", {}),
]


def test_scripted_sequence_is_monotonic_and_ends_at_one(recorder, clock):
    """A full pyannote run drives the bar from 0 to exactly 1.0, never backwards."""
    hook = make_hook(recorder, clock, min_interval_s=0.0)

    for step, kwargs in PYANNOTE_SEQUENCE:
        hook(step, None, file={}, **kwargs)

    fractions = [f for f, _ in recorder]
    assert fractions, "hook emitted nothing"
    assert fractions == sorted(fractions), f"progress went backwards: {fractions}"
    assert fractions[-1] == pytest.approx(1.0)
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_all_four_stage_labels_appear(recorder, clock):
    """Every stage the pipeline reports is surfaced to the UI."""
    hook = make_hook(recorder, clock, min_interval_s=0.0)

    for step, kwargs in PYANNOTE_SEQUENCE:
        hook(step, None, file={}, **kwargs)

    emitted = " | ".join(label for _, label in recorder)
    for label in DIARIZATION_STAGE_LABELS.values():
        assert label in emitted, f"stage label {label!r} never reached the callback"


def test_incremental_embeddings_maps_to_expected_fraction(recorder, clock):
    """Halfway through embeddings means everything before it, plus half its weight."""
    hook = make_hook(recorder, clock, min_interval_s=0.0)

    hook("segmentation", None, completed=4, total=4)
    hook("speaker_counting", None)
    hook("embeddings", None, completed=5, total=10)

    expected = (
        DIARIZATION_STAGE_WEIGHTS["segmentation"]
        + DIARIZATION_STAGE_WEIGHTS["speaker_counting"]
        + DIARIZATION_STAGE_WEIGHTS["embeddings"] * 0.5
    )
    fraction, label = recorder[-1]
    assert fraction == pytest.approx(expected)
    assert label == "Analyzing voices 5/10"


def test_overshooting_batch_count_reaches_neither_the_bar_nor_the_label(recorder, clock):
    """Inference reports completed = c + batch_size, which overshoots on the last batch."""
    hook = make_hook(recorder, clock, min_interval_s=0.0)

    # 10 chunks, batch size 4: the final batch reports completed=12.
    hook("segmentation", None, completed=12, total=10)

    fraction, label = recorder[-1]
    assert fraction == pytest.approx(DIARIZATION_STAGE_WEIGHTS["segmentation"])
    # The label must not read "12/10" to the user either. This half is the
    # load-bearing clamp; the fraction is protected by the completed-step path.
    assert label == "Finding speech 10/10"


def test_progress_never_retreats_when_a_step_is_revisited(recorder, clock):
    """A step reported out of order must not drag a further-along bar backwards.

    pyannote runs speaker_counting before embeddings, so this ordering is not
    what 4.0.7 does today. It is the invariant the monotonic guard exists for:
    a reordering in a future release should slow the bar, never rewind it.
    """
    hook = make_hook(recorder, clock, min_interval_s=0.0)

    hook("segmentation", None, completed=10, total=10)  # 0.45
    hook("embeddings", None, completed=5, total=10)     # 0.45 + 0.225 = 0.675
    peak = recorder[-1][0]
    assert peak == pytest.approx(0.675)

    # Now a stage worth far less reports in. Naively this recomputes to 0.50.
    hook("speaker_counting", None)

    assert recorder[-1][0] == pytest.approx(peak)


def test_fraction_stays_within_bounds_if_stage_weights_drift(recorder, clock, monkeypatch):
    """The bar is clamped at runtime, not just by the sum-to-one unit test."""
    monkeypatch.setitem(DIARIZATION_STAGE_WEIGHTS, "segmentation", 0.9)  # now sums to 1.45

    hook = make_hook(recorder, clock, min_interval_s=0.0)
    for step, kwargs in PYANNOTE_SEQUENCE:
        hook(step, None, **kwargs)

    assert max(f for f, _ in recorder) <= 1.0


def test_completion_tick_with_completed_none_finishes_step(recorder, clock):
    """The one-shot ticks (completed=None) mean the step is done, not 0% done."""
    hook = make_hook(recorder, clock, min_interval_s=0.0)

    hook("segmentation", "some-artifact", file={})

    fraction, label = recorder[-1]
    assert fraction == pytest.approx(DIARIZATION_STAGE_WEIGHTS["segmentation"])
    assert label == "Finding speech"


def test_throttle_suppresses_rapid_calls(recorder, clock):
    """The embeddings step fires per batch; the UI must not see every one."""
    hook = make_hook(recorder, clock, min_interval_s=0.5)

    # 100 mid-stage updates arriving faster than the throttle interval.
    for i in range(1, 101):
        clock.advance(0.01)
        hook("embeddings", None, completed=i, total=1000)

    # ~1 second of wall clock at a 0.5s interval: a couple of emissions, not 100.
    assert len(recorder) <= 4, f"throttle let {len(recorder)} calls through"
    assert recorder, "throttle suppressed everything"


def test_throttle_never_suppresses_step_changes_or_completions(recorder, clock):
    """A transition or a finished stage must always reach the UI, however fast."""
    hook = make_hook(recorder, clock, min_interval_s=1000.0)  # effectively never due

    hook("segmentation", None, completed=1, total=10)   # step change -> emits
    hook("segmentation", None, completed=2, total=10)   # throttled -> silent
    hook("segmentation", None, completed=10, total=10)  # completion -> emits
    hook("speaker_counting", None)                      # step change -> emits

    labels = [label for _, label in recorder]
    assert labels == [
        "Finding speech 1/10",
        "Finding speech 10/10",
        "Counting speakers",
    ]


def test_unknown_step_name_does_not_crash_or_exceed_one(recorder, clock):
    """A stage added by a future pyannote must not break the bar or overshoot it."""
    hook = make_hook(recorder, clock, min_interval_s=0.0)

    for step, kwargs in PYANNOTE_SEQUENCE:
        hook(step, None, **kwargs)
    hook("some_future_stage", None, completed=3, total=7)

    fractions = [f for f, _ in recorder]
    assert max(fractions) <= 1.0
    assert fractions == sorted(fractions)
    # Unknown stages still get a readable label rather than being hidden.
    assert recorder[-1][1] == "Some future stage 3/7"


def test_hook_is_a_context_manager(recorder, clock):
    """pyannote's own hooks are used as `with Hook() as h:`; ours must match."""
    with make_hook(recorder, clock, min_interval_s=0.0) as hook:
        hook("segmentation", None)
    assert recorder


def test_stage_weights_sum_to_one():
    """Otherwise the bar cannot reach 100% (or overshoots it)."""
    assert sum(DIARIZATION_STAGE_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(DIARIZATION_STAGE_WEIGHTS) == set(DIARIZATION_STAGE_LABELS)


# --- run_diarization wiring ---------------------------------------------------
#
# These use the patched_diarization fixture from conftest.py, which replaces
# token validation, model download, audio decode and pyannote.Pipeline. They
# import pyannote (slow, no network), so they are the only non-trivial-cost
# tests in this file.


def test_run_diarization_passes_hook_to_pipeline(patched_diarization):
    """The whole point of WT-2: the pipeline must actually receive a hook."""
    from src.core.diarization import run_diarization

    updates = []
    run_diarization(
        "/tmp/does-not-matter.wav",
        "hf_test_token",
        progress_callback=lambda fraction, label: updates.append((fraction, label)),
    )

    assert patched_diarization.received_hook is not None, "pipeline was called without hook="

    fractions = [f for f, _ in updates]
    assert fractions, "progress_callback was never called"
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert "Finding speech" in updates[0][1]


def test_run_diarization_without_callback_passes_no_hook(patched_diarization):
    """Existing callers get exactly the previous behavior, hook=None."""
    from src.core.diarization import run_diarization

    run_diarization("/tmp/does-not-matter.wav", "hf_test_token")

    assert patched_diarization.received_hook is None


def test_run_diarization_still_returns_turns(patched_diarization):
    """Adding the hook must not disturb the return value."""
    from src.core.diarization import run_diarization

    turns = run_diarization(
        "/tmp/does-not-matter.wav",
        "hf_test_token",
        progress_callback=lambda fraction, label: None,
    )

    assert [(t.start, t.end, t.speaker) for t in turns] == [
        (0.0, 1.0, "SPEAKER_00"),
        (1.0, 2.0, "SPEAKER_01"),
    ]
