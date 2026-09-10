"""
Tests for the trip project: where a trip's work lives, and what it remembers.

The load-bearing guarantee here is that the source recordings are never
written to. They are the only thing in this pipeline that cannot be
regenerated, so there is a test that hashes them before and after.

Run with: python -m pytest tests/test_trip_project.py -v
"""

import hashlib
import json
import os

import pytest

from src.podcastnotes.project import (
    WORK_DIR_NAME,
    SourceRecording,
    TripProject,
    resolve_work_dir,
    slugify,
)


@pytest.fixture
def recordings(tmp_path):
    """Three files standing in for a trip's recordings."""
    folder = tmp_path / "Ashford Notes" / "Audio Files"
    folder.mkdir(parents=True)
    paths = []
    for name in ("Ridgeline Northgate.m4a", "Lakeside StBede.m4a", "Lakeside WP.m4a"):
        p = folder / name
        p.write_bytes(b"pretend audio " + name.encode())
        paths.append(str(p))
    return paths


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --- slugs --------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Ashford", "ashford"),
        ("Ashford Hospital Tour", "ashford-hospital-tour"),
        ("  Ashford / Q3  ", "ashford-q3"),
        ("Ben's trip", "bens-trip"),
        ("Trip #4: Merrow!", "trip-4-merrow"),
        ("under_scores and-dashes", "under-scores-and-dashes"),
        ("", "trip"),
        ("///", "trip"),
    ],
)
def test_slugify(name, expected):
    assert slugify(name) == expected


def test_slug_never_produces_a_path_separator():
    """A slug becomes a folder name, so a slash would escape the work dir."""
    assert "/" not in slugify("a/b/c")
    assert os.sep not in slugify("a/b/c")


# --- where the work goes ------------------------------------------------------


def test_work_dir_sits_beside_the_recordings(recordings):
    work = resolve_work_dir(recordings, "ashford")

    assert os.path.dirname(os.path.dirname(work)) == os.path.dirname(recordings[0])
    assert os.path.basename(os.path.dirname(work)) == WORK_DIR_NAME
    assert os.path.basename(work) == "ashford"


def test_recordings_in_sibling_folders_resolve_to_their_shared_parent(tmp_path):
    a = tmp_path / "trip" / "day1"
    b = tmp_path / "trip" / "day2"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "one.m4a").write_bytes(b"x")
    (b / "two.m4a").write_bytes(b"x")

    work = resolve_work_dir([str(a / "one.m4a"), str(b / "two.m4a")], "trip")

    assert work == str(tmp_path / "trip" / WORK_DIR_NAME / "trip")


def test_a_second_trip_of_the_same_name_does_not_overwrite_the_first(recordings):
    first = resolve_work_dir(recordings, "ashford")
    os.makedirs(first)

    second = resolve_work_dir(recordings, "ashford")

    assert second != first
    assert os.path.basename(second) == "ashford-2"


def test_a_single_recording_still_resolves(tmp_path):
    p = tmp_path / "solo.m4a"
    p.write_bytes(b"x")

    work = resolve_work_dir([str(p)], "solo")

    assert work == str(tmp_path / WORK_DIR_NAME / "solo")


def test_no_recordings_is_an_error():
    with pytest.raises(ValueError, match="at least one recording"):
        resolve_work_dir([], "empty")


# --- the project --------------------------------------------------------------


def test_create_records_each_recording(recordings):
    trip = TripProject.create("Ashford", recordings, description="Hospital tour")

    assert trip.name == "Ashford"
    assert trip.description == "Hospital tour"
    assert [s.name for s in trip.sources] == [
        "Ridgeline Northgate.m4a", "Lakeside StBede.m4a", "Lakeside WP.m4a",
    ]
    assert all(os.path.isabs(s.path) for s in trip.sources)
    assert trip.created_at.endswith("+00:00")


def test_source_order_is_the_order_given(recordings):
    """Play order is a user decision, so it must not be re-sorted."""
    reversed_order = list(reversed(recordings))

    trip = TripProject.create("Ashford", reversed_order)

    assert [s.name for s in trip.sources] == [
        "Lakeside WP.m4a", "Lakeside StBede.m4a", "Ridgeline Northgate.m4a",
    ]


def test_create_with_no_recordings_is_an_error():
    with pytest.raises(ValueError, match="at least one recording"):
        TripProject.create("Empty", [])


def test_derived_paths_stay_inside_the_work_dir(recordings):
    trip = TripProject.create("Ashford", recordings)

    assert trip.combined_audio_path.startswith(trip.work_dir + os.sep)
    assert trip.state_path.startswith(trip.work_dir + os.sep)
    assert trip.path_for("boundaries.json").startswith(trip.work_dir + os.sep)


def test_state_round_trips(recordings):
    trip = TripProject.create("Ashford Hospital Tour", recordings, description="Q3 visit")
    trip.save()

    loaded = TripProject.load(trip.work_dir)

    assert loaded.name == trip.name
    assert loaded.description == trip.description
    assert loaded.created_at == trip.created_at
    assert [s.path for s in loaded.sources] == [s.path for s in trip.sources]
    assert loaded.work_dir == trip.work_dir


def test_a_moved_trip_folder_still_opens(recordings, tmp_path):
    """work_dir is not stored, so it is wherever the state file was found."""
    trip = TripProject.create("Ashford", recordings)
    trip.save()

    moved = tmp_path / "somewhere-else"
    os.rename(trip.work_dir, moved)

    loaded = TripProject.load(str(moved))

    assert loaded.work_dir == str(moved)
    assert loaded.name == "Ashford"


def test_state_is_readable_json(recordings):
    trip = TripProject.create("Ashford", recordings)
    trip.save()

    payload = json.loads(open(trip.state_path, encoding="utf-8").read())

    assert payload["version"] == 1
    assert payload["name"] == "Ashford"
    assert len(payload["sources"]) == 3


def test_the_work_dir_is_deliberately_not_stored(recordings):
    """This is what makes a moved trip folder still open.

    Recording the path inside the file that lives at that path would go stale
    the moment anyone drags the folder somewhere else.
    """
    trip = TripProject.create("Ashford", recordings)
    trip.save()

    payload = json.loads(open(trip.state_path, encoding="utf-8").read())

    assert "work_dir" not in payload


# --- missing recordings -------------------------------------------------------


def test_a_missing_recording_is_reported_by_name(recordings):
    """Report what went, not a dead path."""
    trip = TripProject.create("Ashford", recordings)
    os.remove(recordings[1])

    missing = trip.missing_sources()

    assert [s.name for s in missing] == ["Lakeside StBede.m4a"]


def test_nothing_missing_when_everything_is_present(recordings):
    trip = TripProject.create("Ashford", recordings)

    assert trip.missing_sources() == []


# --- the guarantee ------------------------------------------------------------


def test_the_original_recordings_are_never_written_to(recordings):
    """The one artefact that cannot be regenerated.

    Creating a trip, saving it and reloading it must leave every source byte
    for byte as it was, and must not rename or move them either.
    """
    before = {p: digest(p) for p in recordings}

    trip = TripProject.create("Ashford", recordings, description="tour")
    trip.save()
    TripProject.load(trip.work_dir)

    for path, original in before.items():
        assert os.path.exists(path), f"{path} was moved or deleted"
        assert digest(path) == original, f"{path} was modified"


def test_everything_written_lands_under_the_work_dir(recordings, tmp_path):
    """Nothing should appear next to the recordings except the work dir itself."""
    audio_folder = os.path.dirname(recordings[0])
    before = set(os.listdir(audio_folder))

    trip = TripProject.create("Ashford", recordings)
    trip.save()

    added = set(os.listdir(audio_folder)) - before
    assert added == {WORK_DIR_NAME}


def test_project_module_imports_no_qt():
    import ast
    import inspect

    from src.podcastnotes import project

    tree = ast.parse(inspect.getsource(project))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    qt = sorted(m for m in imported if m.split(".")[0] in {"PyQt6", "PyQt5", "PySide6"})
    assert qt == [], f"project imports Qt: {qt}"
