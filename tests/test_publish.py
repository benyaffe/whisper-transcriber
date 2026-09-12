"""
Tests for getting the write-up out of the app.

The requirement is not "publishing works". It is that somebody who has spent
an hour on a trip is never left with nothing to show for it. So the order of
operations is the thing under test: the file lands first, and everything after
it is a convenience that is allowed to fail.

Run with: python -m pytest tests/test_publish.py -v
"""

import os

import pytest

from src.podcastnotes import publish

DOCUMENT = "# Ashford\n\nWe met the team.\n"


@pytest.fixture(autouse=True)
def no_real_clipboard_or_browser(monkeypatch):
    """Nothing here should touch the developer's clipboard or open a tab."""
    monkeypatch.setattr(publish, "copy_to_clipboard", lambda text: True)
    monkeypatch.setattr(publish, "open_new_google_doc", lambda: True)


# --- the file lands first -----------------------------------------------------


def test_the_document_is_written_to_disk(tmp_path):
    path = str(tmp_path / "ashford.md")

    result = publish.publish(DOCUMENT, path)

    assert result.saved
    assert open(path).read() == DOCUMENT


def test_a_clipboard_failure_does_not_lose_the_document(tmp_path, monkeypatch):
    """The whole reason the file is written first."""
    monkeypatch.setattr(publish, "copy_to_clipboard", lambda text: False)
    path = str(tmp_path / "ashford.md")

    result = publish.publish(DOCUMENT, path)

    assert result.saved, "an hour of work lost to a clipboard problem"
    assert result.copied is False
    assert any("clipboard" in p for p in result.problems)


def test_a_browser_failure_does_not_lose_the_document(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "open_new_google_doc", lambda: False)
    path = str(tmp_path / "ashford.md")

    result = publish.publish(DOCUMENT, path)

    assert result.saved
    assert result.copied is True
    assert any(publish.NEW_DOC_URL in p for p in result.problems)


def test_everything_downstream_can_fail_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "copy_to_clipboard", lambda text: False)
    monkeypatch.setattr(publish, "open_new_google_doc", lambda: False)
    path = str(tmp_path / "ashford.md")

    result = publish.publish(DOCUMENT, path)

    assert result.saved
    assert len(result.problems) == 2
    assert "Saved as ashford.md" in result.summary()


def test_a_missing_folder_is_created(tmp_path):
    path = str(tmp_path / "nested" / "deeper" / "ashford.md")

    assert publish.publish(DOCUMENT, path).saved


def test_an_unwritable_path_reports_rather_than_raising(tmp_path):
    """Failing here is the one real failure, and it still must not throw into
    the middle of a finished trip."""
    path = str(tmp_path / "ashford.md" / "impossible.md")
    (tmp_path / "ashford.md").write_text("in the way")

    result = publish.publish(DOCUMENT, path)

    assert result.saved is False
    assert result.problems


def test_nothing_is_attempted_after_the_file_fails(tmp_path, monkeypatch):
    """There is nothing to paste, so opening a blank document would just be
    confusing."""
    tried = []
    monkeypatch.setattr(publish, "open_new_google_doc", lambda: tried.append(True))
    path = str(tmp_path / "ashford.md" / "impossible.md")
    (tmp_path / "ashford.md").write_text("in the way")

    publish.publish(DOCUMENT, path)

    assert tried == []


# --- what a person is told ----------------------------------------------------


def test_the_summary_says_what_actually_happened(tmp_path):
    result = publish.publish(DOCUMENT, str(tmp_path / "ashford.md"))

    summary = result.summary()
    assert "ashford.md" in summary
    assert "copied" in summary.lower()


def test_the_instructions_name_the_menu_item(tmp_path):
    """"Paste from Markdown" is not where anybody would look, and Google hides
    it behind a preference that is off by default."""
    assert "Paste from Markdown" in publish.PASTE_INSTRUCTIONS
    assert "Enable Markdown" in publish.PASTE_INSTRUCTIONS


def test_publishing_can_skip_the_browser(tmp_path, monkeypatch):
    """Opening a tab from a test, or from a batch run, is rude."""
    opened = []
    monkeypatch.setattr(publish, "open_new_google_doc", lambda: opened.append(True))

    result = publish.publish(DOCUMENT, str(tmp_path / "a.md"), open_browser=False)

    assert opened == []
    assert result.saved
    assert result.browser_opened is False


# --- no Google credentials anywhere near this ---------------------------------


def test_publishing_imports_nothing_from_google():
    """The point of the clipboard route. Publishing through Drive would need a
    scope only an administrator can grant, and it would fail at the very end
    of a long job for a reason the person running it cannot act on.

    Checked against the imports rather than the prose, since the prose says
    "Drive" a good deal while explaining why it is not used.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(publish))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "googleapiclient" not in imported
    assert "google" not in imported
    assert not any(name.startswith("src.podcastnotes.auth") for name in imported)
