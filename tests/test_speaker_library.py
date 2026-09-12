"""
Tests for the speaker library.

Two things carry the weight. The threshold is calibrated against real measured
embeddings rather than chosen, so the tests use the numbers actually observed
on the Ashford recording. And matching is best-of across a person's samples
rather than an average, which is the difference between a library that gets
better with every trip and one that gets steadily blurrier.

Every test uses its own database file. Nothing here touches the real library
in Application Support.

Run with: python -m pytest tests/test_speaker_library.py -v
"""

import math

import pytest

from src.podcastnotes import speaker_library as lib


@pytest.fixture
def library(tmp_path):
    with lib.Library(str(tmp_path / "speakers.db")) as store:
        yield store


def _vec(*values, width=8):
    """A padded embedding, so dimensions match unless a test says otherwise."""
    out = list(values) + [0.0] * (width - len(values))
    return out[:width]


REFERENCE = _vec(1.0, 0.0)


def _at_cosine(target, width=8):
    """A vector whose cosine with REFERENCE is exactly `target`.

    Built from the definition rather than by rotating and hoping: on an
    orthonormal pair, cos(t)*e0 + sin(t)*e1 has similarity t with e0. An
    earlier version interpolated a fraction of the way towards an orthogonal
    vector, which is not linear in the angle and landed a "middling" match
    above the pre-fill threshold.
    """
    out = [0.0] * width
    out[0] = target
    out[1] = math.sqrt(max(1.0 - target * target, 0.0))
    return out


# --- the measurement this is calibrated against -------------------------------


def test_the_threshold_sits_between_the_measured_populations():
    """Measured on Ashford, where the diarizer split one person across two
    labels and so handed over a known same-person pair. The plan's original
    0.75 would have matched nobody, since the same-person score was 0.624."""
    same_person = 0.624
    different_people = [0.029, 0.029, 0.038, 0.078, 0.129, 0.244]

    assert max(different_people) < lib.CONSIDER < lib.MATCH < same_person


def test_cosine_agrees_with_the_measured_figure():
    """A guard on the maths itself, so a refactor cannot quietly change what
    the thresholds are calibrated against."""
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]

    assert lib.cosine(a, a) == pytest.approx(1.0)
    assert lib.cosine(a, b) == pytest.approx(0.0)
    assert lib.cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))


def test_a_vector_with_no_magnitude_scores_zero_rather_than_dividing_by_zero():
    """pyannote returns one of these for a speaker it heard for a fraction of
    a second."""
    assert lib.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_vectors_of_different_widths_do_not_compare():
    assert lib.cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


# --- remembering --------------------------------------------------------------


def test_a_remembered_voice_is_found_again(library):
    voice = _vec(1.0, 0.2, 0.1)
    library.remember("Anya Petrov-Hale", voice, trip="Ashford")

    found = library.identify(voice)

    assert found[0].name == "Anya Petrov-Hale"
    assert found[0].confident is True


def test_a_voice_survives_closing_and_reopening(tmp_path):
    """The whole point is the next trip, which is a different process."""
    path = str(tmp_path / "speakers.db")
    voice = _vec(1.0, 0.2)
    with lib.Library(path) as first:
        first.remember("Devan Shaw", voice)

    with lib.Library(path) as second:
        assert second.identify(voice)[0].name == "Devan Shaw"


def test_an_empty_name_is_refused(library):
    assert library.remember("   ", _vec(1.0)) is False
    assert library.names() == []


def test_an_embedding_with_no_magnitude_is_refused(library):
    """Stored, it would sit there matching nothing and could never be
    noticed."""
    assert library.remember("Nobody", [0.0, 0.0, 0.0]) is False
    assert library.names() == []


def test_an_absent_embedding_is_refused(library):
    assert library.remember("Nobody", None) is False


# --- several samples per person, matched best-of ------------------------------


def test_a_person_can_have_several_samples(library):
    library.remember("Anya", _vec(1.0, 0.0), trip="Ashford")
    library.remember("Anya", _vec(0.0, 1.0), trip="Houston")

    assert library.names() == [("Anya", 2)]


def test_the_best_sample_wins_rather_than_the_average(library):
    """An average of a quiet room and a busy ED describes neither, and gets
    worse with every trip added."""
    quiet = _vec(1.0, 0.0)
    noisy = _vec(0.0, 1.0)
    library.remember("Anya", quiet)
    library.remember("Anya", noisy)

    found = library.identify(quiet)

    assert found[0].score == pytest.approx(1.0), "the samples were averaged"
    assert found[0].samples == 2


def test_a_new_sample_cannot_make_an_existing_match_worse(library):
    voice = _vec(1.0, 0.1)
    library.remember("Anya", voice)
    before = library.identify(voice)[0].score

    library.remember("Anya", _vec(0.0, 0.0, 1.0))

    assert library.identify(voice)[0].score == pytest.approx(before)


# --- identifying --------------------------------------------------------------


def test_the_strongest_match_comes_first(library):
    library.remember("Anya", _at_cosine(1.0))
    library.remember("Devan", _at_cosine(0.60))

    found = library.identify(REFERENCE)

    assert [m.name for m in found] == ["Anya", "Devan"]


def test_somebody_below_the_floor_is_not_offered_at_all(library):
    """The measured different-person scores run up to 0.244, so anything under
    the floor is noise and putting it on the screen only invites a mistake."""
    library.remember("Devan", _at_cosine(0.24))

    assert library.identify(REFERENCE) == []


def test_a_middling_match_is_offered_but_not_pre_filled(library):
    """The case this two-threshold design exists for: the same person recorded
    in a different room months later."""
    library.remember("Anya", _at_cosine(0.38))

    found = library.identify(REFERENCE)

    assert found[0].score == pytest.approx(0.38, abs=1e-3)
    assert lib.CONSIDER <= found[0].score < lib.MATCH
    assert found[0].confident is False


def test_only_samples_of_the_same_width_are_considered(library):
    """A different diarization model produces vectors of a different width,
    whose similarity to these is not wrong but meaningless."""
    library.remember("Old Model", [1.0, 0.0, 0.0])

    assert library.identify([1.0, 0.0, 0.0, 0.0]) == []


def test_identifying_an_empty_voice_returns_nothing(library):
    library.remember("Anya", _vec(1.0))

    assert library.identify([]) == []
    assert library.identify([0.0, 0.0]) == []


def test_only_so_many_candidates_are_offered(library):
    voice = _vec(1.0, 0.0)
    for name in ("A", "B", "C", "D"):
        library.remember(name, voice)

    assert len(library.identify(voice, limit=2)) == 2


# --- forgetting ---------------------------------------------------------------


def test_forgetting_removes_every_sample_of_one_person(library):
    voice = _vec(1.0, 0.0)
    library.remember("Anya", voice)
    library.remember("Anya", voice)
    library.remember("Devan", _vec(0.0, 1.0))

    removed = library.forget("Anya")

    assert removed == 2
    assert [n for n, _ in library.names()] == ["Devan"]


# --- the trip-level helpers ---------------------------------------------------


def _payload():
    return {
        "speakers": [
            {"id": "Speaker 1", "pyannote_label": "P3", "embedding": _vec(1.0, 0.0)},
            {"id": "Speaker 2", "pyannote_label": "P1", "embedding": _vec(0.0, 1.0)},
            {"id": "Speaker 3", "pyannote_label": "P2"},
        ]
    }


def test_only_confirmed_speakers_are_remembered(tmp_path):
    """An unconfirmed label is exactly the one whose identity is uncertain.
    Remembering it teaches the library a name it then suggests forever."""
    path = str(tmp_path / "s.db")

    stored = lib.remember_trip(_payload(), {"Speaker 1": "Marcus"}, trip="Ashford", path=path)

    assert stored == 1
    with lib.Library(path) as store:
        assert [n for n, _ in store.names()] == ["Marcus"]


def test_a_speaker_with_no_embedding_is_skipped(tmp_path):
    path = str(tmp_path / "s.db")

    stored = lib.remember_trip(_payload(), {"Speaker 3": "Devan"}, path=path)

    assert stored == 0


def test_the_second_trip_pre_fills_from_the_first(tmp_path):
    """The demonstrable for this stage."""
    path = str(tmp_path / "s.db")
    lib.remember_trip(_payload(), {"Speaker 1": "Marcus", "Speaker 2": "Anya"},
                      trip="Ashford", path=path)

    # A second trip, same people, labels numbered differently by the diarizer.
    second = {"speakers": [
        {"id": "Speaker 1", "embedding": _vec(0.0, 1.0)},
        {"id": "Speaker 2", "embedding": _vec(1.0, 0.0)},
    ]}

    suggestions = lib.suggest_names(second, path=path)

    assert suggestions["Speaker 1"][0].name == "Anya"
    assert suggestions["Speaker 2"][0].name == "Marcus"


def test_an_empty_library_suggests_nothing_and_does_not_fail(tmp_path):
    suggestions = lib.suggest_names(_payload(), path=str(tmp_path / "s.db"))

    assert suggestions == {"Speaker 1": [], "Speaker 2": [], "Speaker 3": []}


# --- seeing what is remembered ----------------------------------------------------


def test_a_remembered_voice_can_be_listed_with_its_context(tmp_path):
    """The count alone does not answer the only question somebody asks on this
    screen, which is "is this the wrong one?". The trips it was heard on is
    what tells them."""
    with lib.Library(str(tmp_path / "v.db")) as library:
        library.remember("Anya Petrov-Hale", [1.0, 0.0], trip="Ashford")
        library.remember("Anya Petrov-Hale", [0.9, 0.1], trip="Cleveland")

        entries = library.entries()

    assert entries == [{
        "name": "Anya Petrov-Hale", "samples": 2,
        "last_heard": entries[0]["last_heard"],
        "trips": ["Cleveland", "Ashford"],
    }]
    assert entries[0]["last_heard"], "no idea when it was last heard"


def test_one_trip_heard_twice_is_listed_once(tmp_path):
    """Three recordings from one trip is three samples and one trip, and a row
    reading "Ashford, Ashford, Ashford" helps nobody."""
    with lib.Library(str(tmp_path / "v.db")) as library:
        library.remember("Marcus", [1.0, 0.0], trip="Ashford")
        library.remember("Marcus", [0.9, 0.1], trip="Ashford")

        assert library.entries()[0]["trips"] == ["Ashford"]


def test_a_voice_remembered_with_no_trip_still_lists(tmp_path):
    with lib.Library(str(tmp_path / "v.db")) as library:
        library.remember("Marcus", [1.0, 0.0])

        assert library.entries()[0]["trips"] == []


def test_nothing_remembered_is_an_empty_list(tmp_path):
    with lib.Library(str(tmp_path / "v.db")) as library:
        assert library.entries() == []
