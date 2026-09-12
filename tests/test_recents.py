"""
Tests for the list of trips this machine has worked on.

The reason this exists at all is that `WriteUp.resume` worked and nothing could
reach it. A trip closed part way through was finished or abandoned, with no
third option, and fifteen minutes of Glean research sat on disk unreachable.

What is worth defending here is that the list is a cache and behaves like one.
It is written from paths as they are used, so it will go stale, and the stale
entries have to disappear quietly rather than either erroring or being
"repaired" into something that points at nothing.

Run with: python -m pytest tests/test_recents.py -v
"""

import json
import os
import shutil

from src.podcastnotes import recents
from src.podcastnotes.project import TripProject


def _trip(tmp_path, name="Ashford", folder=None):
    """A trip folder with a real state file in it."""
    work_dir = tmp_path / (folder or name.lower())
    work_dir.mkdir(parents=True, exist_ok=True)
    recording = tmp_path / f"{name}.m4a"
    recording.write_bytes(b"pretend audio")
    trip = TripProject.create(name, [str(recording)])
    trip.work_dir = str(work_dir)
    trip.save()
    return str(work_dir)


def _where(tmp_path):
    return str(tmp_path / "recent.json")


# --- remembering -----------------------------------------------------------------


def test_a_trip_is_listed_once_it_has_been_seen(tmp_path):
    where = _where(tmp_path)
    work_dir = _trip(tmp_path)

    recents.remember(work_dir, where)

    assert [t["work_dir"] for t in recents.load(where)] == [work_dir]
    assert recents.load(where)[0]["name"] == "Ashford"


def test_the_newest_is_first(tmp_path):
    """It is what somebody is most likely to be coming back to."""
    where = _where(tmp_path)
    first = _trip(tmp_path, "Ashford")
    second = _trip(tmp_path, "Cleveland")

    recents.remember(first, where)
    recents.remember(second, where)

    assert [t["name"] for t in recents.load(where)] == ["Cleveland", "Ashford"]


def test_reopening_a_trip_moves_it_up_rather_than_listing_it_twice(tmp_path):
    where = _where(tmp_path)
    first = _trip(tmp_path, "Ashford")
    second = _trip(tmp_path, "Cleveland")
    recents.remember(first, where)
    recents.remember(second, where)

    recents.remember(first, where)

    assert [t["name"] for t in recents.load(where)] == ["Ashford", "Cleveland"]


def test_the_list_stops_growing(tmp_path):
    """Past a dozen it stops being a list somebody reads and Open... is the
    right tool instead."""
    where = _where(tmp_path)
    for n in range(recents.LIMIT + 5):
        recents.remember(_trip(tmp_path, f"Trip {n}", folder=f"t{n}"), where)

    assert len(recents.load(where)) == recents.LIMIT


def test_the_same_folder_written_two_ways_is_one_trip(tmp_path):
    """A path from a file dialog and a path built by the app differ by a
    trailing slash or a `.`, and neither is a different trip."""
    where = _where(tmp_path)
    work_dir = _trip(tmp_path)

    recents.remember(work_dir, where)
    recents.remember(work_dir + "/", where)
    recents.remember(os.path.join(work_dir, "."), where)

    assert len(recents.load(where)) == 1


def test_remembering_nothing_changes_nothing(tmp_path):
    where = _where(tmp_path)
    recents.remember(_trip(tmp_path), where)

    recents.remember("", where)

    assert len(recents.load(where)) == 1


# --- going stale -----------------------------------------------------------------


def test_a_trip_whose_folder_has_gone_drops_out(tmp_path):
    """The usual reason a trip folder disappears is that somebody deleted the
    recordings on purpose, so it goes quietly rather than erroring."""
    where = _where(tmp_path)
    work_dir = _trip(tmp_path)
    recents.remember(work_dir, where)

    shutil.rmtree(work_dir)

    assert recents.load(where) == []


def test_a_folder_with_no_state_file_is_not_a_trip(tmp_path):
    """Somebody's Downloads folder, picked in an Open dialog by mistake."""
    where = _where(tmp_path)
    empty = tmp_path / "downloads"
    empty.mkdir()
    recents.remember(str(empty), where)

    assert recents.load(where) == []


def test_a_half_written_state_file_is_still_listed(tmp_path):
    """From a crash mid-save. The folder is still openable, and the failure
    belongs on the screen that opens it rather than in the middle of drawing a
    menu, where it would take every other trip down with it."""
    where = _where(tmp_path)
    good = _trip(tmp_path, "Ashford")
    broken = _trip(tmp_path, "Cleveland")
    (tmp_path / "cleveland" / "project.json").write_text("{ this is not json")
    recents.remember(good, where)
    recents.remember(broken, where)

    listed = recents.load(where)

    assert [t["name"] for t in listed] == ["cleveland", "Ashford"]
    assert listed[0]["readable"] is False


def test_a_list_file_that_is_not_a_list_is_ignored(tmp_path):
    where = _where(tmp_path)
    with open(where, "w") as handle:
        json.dump({"trips": []}, handle)

    assert recents.load(where) == []


def test_no_list_file_yet_is_an_empty_list(tmp_path):
    assert recents.load(_where(tmp_path)) == []


def test_a_corrupt_list_file_does_not_stop_the_app_opening(tmp_path):
    where = _where(tmp_path)
    with open(where, "w") as handle:
        handle.write("{{{")

    assert recents.load(where) == []


# --- forgetting ------------------------------------------------------------------


def test_a_trip_can_be_dropped_from_the_list(tmp_path):
    where = _where(tmp_path)
    first = _trip(tmp_path, "Ashford")
    second = _trip(tmp_path, "Cleveland")
    recents.remember(first, where)
    recents.remember(second, where)

    recents.forget(first, where)

    assert [t["name"] for t in recents.load(where)] == ["Cleveland"]


def test_forgetting_a_trip_leaves_the_trip_alone(tmp_path):
    """Deleting somebody's recordings because they tidied a menu would be a
    spectacular overreach."""
    where = _where(tmp_path)
    work_dir = _trip(tmp_path)
    recents.remember(work_dir, where)

    recents.forget(work_dir, where)

    assert os.path.isdir(work_dir)
    assert os.path.isfile(os.path.join(work_dir, "project.json"))
