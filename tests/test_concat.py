"""
Tests for joining a trip's recordings.

Uses real ffmpeg against small generated tones rather than mocks, because the
things most likely to break here are ffmpeg's own behaviours: stream copy
refusing mismatched inputs, and the concat list file's quoting rules.

Run with: python -m pytest tests/test_concat.py -v
"""

import json
import os
import subprocess

import pytest

from src.podcastnotes.concat import (
    BOUNDARIES_FILENAME,
    DRIFT_TOLERANCE_PER_FILE,
    AudioParams,
    Boundary,
    ConcatError,
    ConcatResult,
    _escape_for_list_file,
    compute_boundaries,
    concat,
    probe_compatibility,
    write_boundaries,
)
from src.utils.file_utils import get_bundled_binary, get_file_info


def make_tone(path, seconds=1.0, rate=48000, channels=1, freq=440, codec="aac"):
    """A real encoded file, so ffprobe and the stream copy have something true."""
    subprocess.run(
        [
            get_bundled_binary("ffmpeg"), "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}:sample_rate={rate}",
            "-ac", str(channels), "-c:a", codec, str(path),
        ],
        check=True, capture_output=True, timeout=120,
    )
    return str(path)


@pytest.fixture
def three_matching(tmp_path):
    return [
        make_tone(tmp_path / "one.m4a", seconds=1.0, freq=440),
        make_tone(tmp_path / "two.m4a", seconds=2.0, freq=550),
        make_tone(tmp_path / "three.m4a", seconds=1.5, freq=660),
    ]


# --- compatibility ------------------------------------------------------------


def test_matching_recordings_are_compatible(three_matching):
    compatible, reason = probe_compatibility(three_matching)

    assert compatible is True
    assert reason == ""


def test_a_single_recording_is_trivially_compatible(tmp_path):
    assert probe_compatibility([make_tone(tmp_path / "solo.m4a")]) == (True, "")


@pytest.mark.parametrize(
    "differing",
    [
        {"rate": 44100},
        {"channels": 2},
    ],
)
def test_mismatched_recordings_are_not_compatible(tmp_path, differing):
    """An iPhone and an Android recording of the same meeting look like this."""
    a = make_tone(tmp_path / "a.m4a", rate=48000, channels=1)
    b = make_tone(tmp_path / "b.m4a", **differing)

    compatible, reason = probe_compatibility([a, b])

    assert compatible is False
    assert "b.m4a" in reason and "a.m4a" in reason


def test_the_reason_names_the_offending_file(tmp_path):
    """"Your recordings don't match" is not actionable."""
    a = make_tone(tmp_path / "first.m4a", rate=48000)
    b = make_tone(tmp_path / "odd-one-out.m4a", rate=44100)

    _, reason = probe_compatibility([a, b])

    assert "odd-one-out.m4a" in reason
    assert "44100" in reason and "48000" in reason


def test_a_file_with_no_audio_stream_is_an_error(tmp_path):
    empty = tmp_path / "notaudio.m4a"
    empty.write_bytes(b"definitely not audio")

    with pytest.raises(ConcatError, match="No audio stream"):
        AudioParams.probe(str(empty))


# --- list file quoting --------------------------------------------------------


def test_a_plain_path_is_quoted():
    assert _escape_for_list_file("/tmp/audio.m4a") == "file '/tmp/audio.m4a'"


def test_spaces_need_no_special_handling():
    assert _escape_for_list_file("/tmp/Lakeside WP.m4a") == "file '/tmp/Lakeside WP.m4a'"


def test_an_apostrophe_is_escaped():
    """Voice memos are routinely named things like "Ben's notes"."""
    assert _escape_for_list_file("/tmp/Ben's notes.m4a") == r"file '/tmp/Ben'\''s notes.m4a'"


def test_a_newline_in_a_filename_is_refused():
    """It would inject an extra directive into the list file."""
    with pytest.raises(ConcatError, match="Newline"):
        _escape_for_list_file("/tmp/evil\nfile '/etc/passwd'\n.m4a")


def test_a_file_named_with_an_apostrophe_actually_joins(tmp_path):
    """The quoting rule, proven against real ffmpeg rather than asserted."""
    a = make_tone(tmp_path / "Ben's first.m4a", seconds=1.0)
    b = make_tone(tmp_path / "Ben's second.m4a", seconds=1.0)
    out = tmp_path / "out" / "combined.m4a"

    result = concat([a, b], str(out))

    assert os.path.exists(result.output_path)
    assert float(get_file_info(result.output_path)["duration"]) == pytest.approx(2.0, abs=0.2)


# --- boundaries ---------------------------------------------------------------


def test_boundaries_are_cumulative_and_gapless(three_matching):
    boundaries = compute_boundaries(three_matching)

    assert [b.source_name for b in boundaries] == ["one.m4a", "two.m4a", "three.m4a"]
    assert boundaries[0].start == 0.0
    for earlier, later in zip(boundaries, boundaries[1:]):
        assert later.start == earlier.end, "a gap or overlap between recordings"


def test_boundary_durations_match_the_sources(three_matching):
    boundaries = compute_boundaries(three_matching)

    assert boundaries[0].duration == pytest.approx(1.0, abs=0.15)
    assert boundaries[1].duration == pytest.approx(2.0, abs=0.15)
    assert boundaries[2].duration == pytest.approx(1.5, abs=0.15)


def test_boundaries_are_written_beside_the_audio(tmp_path, three_matching):
    out = tmp_path / "work" / "combined.m4a"
    result = concat(three_matching, str(out))

    path = write_boundaries(result, str(out.parent))
    payload = json.loads(open(path, encoding="utf-8").read())

    assert os.path.basename(path) == BOUNDARIES_FILENAME
    assert payload["combined_audio"] == "combined.m4a"
    assert [s["source_name"] for s in payload["sources"]] == [
        "one.m4a", "two.m4a", "three.m4a",
    ]
    assert payload["stream_copied"] is True
    json.dumps(payload)  # no numpy or other unserialisable types


def test_drift_is_measured_not_assumed(tmp_path, three_matching):
    """Joining AAC loses a fraction of a second per seam to frame alignment."""
    out = tmp_path / "combined.m4a"

    result = concat(three_matching, str(out))

    assert abs(result.drift) < DRIFT_TOLERANCE_PER_FILE * 3
    assert result.drift_is_acceptable


def test_excessive_drift_is_flagged():
    """Pure check on the threshold, without contriving a pathological file."""
    boundaries = [Boundary("a.m4a", "/a.m4a", 0.0, 10.0)]

    assert ConcatResult("/out.m4a", boundaries, drift=0.5).drift_is_acceptable
    assert not ConcatResult("/out.m4a", boundaries, drift=9.0).drift_is_acceptable


# --- joining ------------------------------------------------------------------


def test_matching_recordings_are_stream_copied(tmp_path, three_matching):
    out = tmp_path / "combined.m4a"

    result = concat(three_matching, str(out))

    assert result.stream_copied is True
    assert result.reencode_reason == ""
    assert float(get_file_info(str(out))["duration"]) == pytest.approx(4.5, abs=0.3)


def test_mismatched_recordings_are_re_encoded(tmp_path):
    a = make_tone(tmp_path / "a.m4a", seconds=1.0, rate=48000, channels=1)
    b = make_tone(tmp_path / "b.m4a", seconds=1.0, rate=44100, channels=2)
    out = tmp_path / "combined.m4a"

    result = concat([a, b], str(out))

    assert result.stream_copied is False
    assert "b.m4a" in result.reencode_reason
    # The point of the fallback: a playable file rather than a corrupt one.
    info = get_file_info(str(out))
    assert float(info["duration"]) == pytest.approx(2.0, abs=0.3)
    assert info["has_audio"] is True


def test_recordings_are_joined_in_the_order_given(tmp_path):
    """Play order is the operator's decision, not alphabetical."""
    first = make_tone(tmp_path / "zzz.m4a", seconds=1.0)
    second = make_tone(tmp_path / "aaa.m4a", seconds=2.0)
    out = tmp_path / "combined.m4a"

    result = concat([first, second], str(out))

    assert [b.source_name for b in result.boundaries] == ["zzz.m4a", "aaa.m4a"]
    assert result.boundaries[0].duration == pytest.approx(1.0, abs=0.15)


def test_the_scratch_list_file_is_cleaned_up(tmp_path, three_matching):
    out = tmp_path / "work" / "combined.m4a"

    concat(three_matching, str(out))

    assert os.listdir(out.parent) == ["combined.m4a"]


def test_a_missing_recording_is_reported_by_name(tmp_path, three_matching):
    os.remove(three_matching[1])

    with pytest.raises(ConcatError, match="two.m4a"):
        concat(three_matching, str(tmp_path / "combined.m4a"))


def test_nothing_to_join_is_an_error(tmp_path):
    with pytest.raises(ConcatError, match="nothing to join"):
        concat([], str(tmp_path / "combined.m4a"))


def test_a_failed_join_leaves_no_partial_file(tmp_path):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio at all")
    out = tmp_path / "combined.m4a"

    with pytest.raises(ConcatError):
        concat([str(bad)], str(out))

    assert not os.path.exists(out)


def test_a_join_that_dies_midway_leaves_no_playable_stub(tmp_path, monkeypatch, three_matching):
    """ffmpeg killed part-way leaves a truncated file on disk.

    The natural failure above never reaches this, because ffmpeg rejects a
    non-audio input before creating any output. Simulated here, because a
    half-written combined.m4a that looks openable is worse than no file: the
    transcription stage would happily process it and silently lose the rest of
    the trip.
    """
    import subprocess as real_subprocess
    from types import SimpleNamespace

    out = tmp_path / "combined.m4a"
    real_run = real_subprocess.run

    def dies_midway(cmd, **kwargs):
        # Only intercept the join itself. The compatibility probe and the
        # duration lookup are ffprobe and must stay real, or this stops
        # exercising the code path it claims to.
        if "-f" in cmd and "concat" in cmd:
            with open(cmd[-1], "wb") as f:
                f.write(b"\x00" * 4096)  # partial output, then interrupted
            return SimpleNamespace(returncode=255, stderr=b"Received signal 15")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr("src.podcastnotes.concat.subprocess.run", dies_midway)

    with pytest.raises(ConcatError, match="signal 15"):
        concat(three_matching, str(out))

    assert not os.path.exists(out), "a truncated recording was left where it could be used"


def test_status_messages_explain_which_path_was_taken(tmp_path, three_matching):
    said = []

    concat(three_matching, str(tmp_path / "combined.m4a"), on_status=said.append)

    assert any("without re-encoding" in m for m in said)
    assert any("Joined" in m for m in said)


def test_the_source_recordings_are_never_modified(tmp_path, three_matching):
    import hashlib

    before = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in three_matching}

    concat(three_matching, str(tmp_path / "out" / "combined.m4a"))

    for path, original in before.items():
        assert os.path.exists(path)
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == original


def test_concat_module_imports_no_qt():
    import ast
    import inspect

    from src.podcastnotes import concat as mod

    tree = ast.parse(inspect.getsource(mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    qt = sorted(m for m in imported if m.split(".")[0] in {"PyQt6", "PyQt5", "PySide6"})
    assert qt == [], f"concat imports Qt: {qt}"
