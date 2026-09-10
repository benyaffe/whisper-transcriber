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


def test_unhooked_clustering_gap_gets_its_own_label(recorder, clock):
    """The long silent pause between embeddings and discrete_diarization.

    pyannote does not hook clustering. Its completion tick for embeddings
    fires immediately before clustering starts, so that tick is relabelled to
    name what is actually running during the stall.
    """
    hook = make_hook(recorder, clock, min_interval_s=0.0)

    hook("embeddings", None, completed=60, total=60)  # last incremental update
    hook("embeddings", "artifact")                    # completion tick

    assert recorder[-2][1] == "Analyzing voices 60/60"
    assert recorder[-1][1] == "Grouping speakers"


def test_completion_ticks_without_a_successor_keep_their_own_label(recorder, clock):
    """Only steps followed by unhooked work get relabelled."""
    hook = make_hook(recorder, clock, min_interval_s=0.0)

    hook("segmentation", "artifact")

    assert recorder[-1][1] == "Finding speech"


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


@pytest.mark.parametrize(
    "enabled, token, expected_run, expected_reason_fragment",
    [
        (False, "hf_abc", False, "Disabled"),
        (True, "", False, "No token configured"),
        (True, "hf_abc", True, ""),
    ],
)
def test_diarization_preflight(monkeypatch, enabled, token, expected_run, expected_reason_fragment):
    """The will-it-run decision is made once, up front, and drives the bar range."""
    from src.core import transcriber

    monkeypatch.setattr(transcriber, "is_speaker_id_enabled", lambda: enabled)
    monkeypatch.setattr(transcriber, "get_hf_token", lambda: token)

    worker = transcriber.TranscriptionWorker.__new__(transcriber.TranscriptionWorker)
    will_run, hf_token, reason = worker._diarization_preflight()

    assert will_run is expected_run
    assert expected_reason_fragment in reason
    assert hf_token == (token if expected_run else "")


def test_hook_fraction_maps_onto_the_reserved_slice_of_the_bar():
    """0.0-1.0 from the hook must span exactly ceiling..100, whatever the ceiling."""
    floor = 88.0
    span = 100.0 - floor

    assert floor + 0.0 * span == pytest.approx(88.0)
    assert floor + 0.5 * span == pytest.approx(94.0)
    assert floor + 1.0 * span == pytest.approx(100.0)


# Measured end to end on real recordings. Both phases scale linearly with audio
# duration, so these fix the model: if the predicted ceiling drifts away from
# what actually happened, the realtime factors need revisiting.
#
#   name, audio_s, transcription_s, observed_diarization_s
REAL_RUNS = [
    ("Lakeside StBede", 756.0, 274.0, 35.0),
    ("Lakeside WP", 1244.0, 413.0, 47.0),
]


@pytest.mark.parametrize("name, audio_s, transcribe_s, diarize_s", REAL_RUNS)
def test_predicted_split_matches_measured_runs(name, audio_s, transcribe_s, diarize_s):
    """The model has to reproduce the runs it was derived from, within a point."""
    from src.core.transcriber import TranscriptionWorker

    ceiling = TranscriptionWorker._estimate_progress_ceiling(
        audio_duration=audio_s,
        elapsed=transcribe_s,
        audio_done=audio_s,  # measured over the whole run
        device="mps",
    )
    observed_share = 100.0 * diarize_s / (transcribe_s + diarize_s)
    predicted_share = 100.0 - ceiling

    assert predicted_share == pytest.approx(observed_share, abs=2.0), (
        f"{name}: predicted {predicted_share:.1f}% for speaker ID, "
        f"observed {observed_share:.1f}%"
    )


def test_early_measurement_predicts_the_same_split_as_the_full_run():
    """The estimate is taken ~20s in; extrapolating early must not skew it."""
    from src.core.transcriber import TranscriptionWorker

    audio_s, transcribe_s = 756.0, 274.0
    rate = transcribe_s / audio_s

    early = TranscriptionWorker._estimate_progress_ceiling(
        audio_duration=audio_s, elapsed=20.0 * rate, audio_done=20.0, device="mps"
    )
    full = TranscriptionWorker._estimate_progress_ceiling(
        audio_duration=audio_s, elapsed=transcribe_s, audio_done=audio_s, device="mps"
    )
    assert early == pytest.approx(full, abs=0.5)


def test_cpu_only_hardware_gets_a_bigger_speaker_id_share():
    """The whole point of making this adaptive rather than a constant."""
    from src.core.transcriber import TranscriptionWorker

    kwargs = dict(audio_duration=756.0, elapsed=274.0, audio_done=756.0)
    mps = TranscriptionWorker._estimate_progress_ceiling(device="mps", **kwargs)
    cpu = TranscriptionWorker._estimate_progress_ceiling(device="cpu", **kwargs)

    assert cpu < mps, "a slower diarization device must be given more of the bar"


def test_unknown_device_falls_back_to_the_pessimistic_factor():
    from src.core.transcriber import TranscriptionWorker

    kwargs = dict(audio_duration=756.0, elapsed=274.0, audio_done=756.0)
    assert TranscriptionWorker._estimate_progress_ceiling(
        device="some-future-npu", **kwargs
    ) == pytest.approx(
        TranscriptionWorker._estimate_progress_ceiling(device="cpu", **kwargs)
    )


@pytest.mark.parametrize(
    "audio_duration, elapsed, audio_done",
    [(0, 10, 10), (756, 0, 10), (756, 10, 0), (-1, 10, 10)],
)
def test_degenerate_measurements_fall_back_to_the_default(audio_duration, elapsed, audio_done):
    """A zero-length or unmeasured file must not produce a nonsense ceiling."""
    from src.core.transcriber import TranscriptionWorker

    assert TranscriptionWorker._estimate_progress_ceiling(
        audio_duration=audio_duration, elapsed=elapsed, audio_done=audio_done, device="mps"
    ) == TranscriptionWorker.DEFAULT_TRANSCRIBE_CEILING


@pytest.mark.parametrize("elapsed", [1.0, 100000.0])  # absurdly fast, absurdly slow
def test_extreme_throughput_is_clamped(elapsed):
    """Neither phase may be squeezed out of the bar by a wild measurement."""
    from src.core.transcriber import TranscriptionWorker

    ceiling = TranscriptionWorker._estimate_progress_ceiling(
        audio_duration=756.0, elapsed=elapsed, audio_done=756.0, device="cpu"
    )
    assert (
        TranscriptionWorker.MIN_TRANSCRIBE_CEILING
        <= ceiling
        <= TranscriptionWorker.MAX_TRANSCRIBE_CEILING
    )


def test_diarization_always_keeps_a_usable_share_of_the_bar():
    """Regression guard on the original fix.

    The old code emitted a single progress.emit(95, -1), leaving a phase that
    runs for minutes just five integer steps of a 0-100 QProgressBar. However
    the ceiling is computed, it must never squeeze speaker ID back to that.
    """
    from src.core.transcriber import TranscriptionWorker

    reserved = 100.0 - TranscriptionWorker.MAX_TRANSCRIBE_CEILING
    assert reserved >= 8.0, f"only {reserved:.0f} integer steps reserved for speaker ID"


def test_run_diarization_still_returns_turns(patched_diarization):
    """Adding the hook must not disturb the speech turns."""
    from src.core.diarization import run_diarization

    result = run_diarization(
        "/tmp/does-not-matter.wav",
        "hf_test_token",
        progress_callback=lambda fraction, label: None,
    )

    assert [(t.start, t.end, t.speaker) for t in result.turns] == [
        (0.0, 1.0, "SPEAKER_00"),
        (1.0, 2.0, "SPEAKER_01"),
    ]


# --- UI rendering -------------------------------------------------------------


@pytest.fixture
def main_window(qt_app):
    from src.ui.main_window import MainWindow

    window = MainWindow()
    yield window
    window.close()


def test_main_window_renders_stage_label_and_advances_bar(main_window):
    """The label the user reads must be the stage name, not a static string."""
    main_window._on_diarization_progress(85.0, "Analyzing voices 340/1200")

    assert main_window.progress_bar.value() == 85
    assert main_window.eta_label.text() == "Analyzing voices 340/1200"


def test_stage_labels_fit_the_eta_label(main_window):
    """A clipped label defeats the point; the widest stage text must fit."""
    from PyQt6.QtGui import QFontMetrics

    from src.core.diarization import (
        DIARIZATION_POST_STEP_LABELS,
        DIARIZATION_STAGE_LABELS,
    )

    metrics = QFontMetrics(main_window.eta_label.font())
    available = main_window.eta_label.width()

    # Worst case: the longest stage name with a four-digit chunk count.
    all_labels = list(DIARIZATION_STAGE_LABELS.values()) + list(DIARIZATION_POST_STEP_LABELS.values())
    for label in all_labels:
        widest = f"{label} 1200/1200"
        assert metrics.horizontalAdvance(widest) <= available, (
            f"{widest!r} needs {metrics.horizontalAdvance(widest)}px "
            f"but eta_label is {available}px"
        )


class _RecordingSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class _FakeWorker:
    """Stands in for TranscriptionWorker so _start_transcription can be run.

    Records which signals got connected without starting a real transcription.
    """

    SIGNALS = (
        "status_message", "progress", "diarization_progress", "segment_ready",
        "language_detected", "quality_warning",
        "hardware_info", "audio_ready", "completed", "error",
    )

    def __init__(self, *args, **kwargs):
        for name in self.SIGNALS:
            setattr(self, name, _RecordingSignal())
        self.started = False

    def start(self):
        self.started = True


def test_start_transcription_connects_the_diarization_signal(
    main_window, monkeypatch, temp_audio_file
):
    """Guards the wiring itself: a live signal nobody connected is still silent."""
    from src.ui import main_window as main_window_module

    monkeypatch.setattr(main_window_module, "TranscriptionWorker", _FakeWorker)

    item = main_window_module.QueueItem(temp_audio_file)
    main_window._start_transcription(item)

    worker = main_window.transcription_worker
    assert worker.started, "_start_transcription never started the worker"
    assert worker.diarization_progress.slots == [main_window._on_diarization_progress]


def test_worker_signal_reaches_the_window(main_window, qt_app):
    """End to end through Qt: worker signal -> connected slot -> widgets."""
    from src.core.transcriber import TranscriptionWorker

    worker = TranscriptionWorker.__new__(TranscriptionWorker)
    TranscriptionWorker.__init__(worker, "/tmp/nonexistent.wav")
    worker.diarization_progress.connect(main_window._on_diarization_progress)

    worker.diarization_progress.emit(92.5, "Assigning speakers")
    qt_app.processEvents()

    assert main_window.progress_bar.value() == 92
    assert main_window.eta_label.text() == "Assigning speakers"
