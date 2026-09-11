"""
Dependency coverage test.

Scans all source files for third-party imports and verifies every one of them
is importable in the current environment. This catches the class of bug where
code uses a library that was never added to requirements.txt.

Run with:
    python -m pytest tests/test_dependencies.py -v
"""

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

# ── Packages that are part of the Python standard library ──────────────────────
# We only want to flag third-party imports, so stdlib modules are excluded.
STDLIB_MODULES = set(sys.stdlib_module_names)

# ── Known import-name → install-name mappings ──────────────────────────────────
# Some packages are imported under a different name than they are installed as.
IMPORT_TO_PACKAGE = {
    "faster_whisper": "faster-whisper",
    "yt_dlp": "yt-dlp",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "google.cloud": "google-cloud",
    "pkg_resources": "setuptools",
    # "google" is a namespace shared by several packages. Everything this
    # project imports under it (google.oauth2, google.auth.transport) comes
    # from google-auth.
    "google": "google-auth",
    "googleapiclient": "google-api-python-client",
    "glean": "glean-api-client",
}

# ── Imports that are intentionally optional / platform-specific ────────────────
# These are allowed to be missing without failing the test.
OPTIONAL_IMPORTS = {
    "pyannote",          # requires HuggingFace token; heavy optional dep
    "pyannote.audio",
    "torchcodec",        # optional video decoding backend
    "asteroid_filterbanks",
}

# ── Local packages (part of this project, not third-party) ────────────────────
LOCAL_PACKAGES = {"src"}


# ── Source directories to scan ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIRS = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "main.py",
]


def collect_imports(path: Path) -> set[str]:
    """Return all top-level third-party import names found in a .py file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # skip relative imports
                imports.add(node.module.split(".")[0])
    return imports


def gather_all_source_imports() -> dict[str, list[str]]:
    """Scan all source files and return {module_name: [files that import it]}."""
    result: dict[str, list[str]] = {}
    paths = []
    for entry in SOURCE_DIRS:
        entry = Path(entry)
        if entry.is_file():
            paths.append(entry)
        else:
            paths.extend(entry.rglob("*.py"))

    for path in paths:
        for name in collect_imports(path):
            if name in STDLIB_MODULES:
                continue
            result.setdefault(name, []).append(str(path.relative_to(PROJECT_ROOT)))
    return result


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_all_imports_are_importable():
    """Every third-party module used in source can be imported."""
    source_imports = gather_all_source_imports()
    failures = []

    for module_name, files in sorted(source_imports.items()):
        if module_name in OPTIONAL_IMPORTS or module_name in LOCAL_PACKAGES:
            continue
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            install_name = IMPORT_TO_PACKAGE.get(module_name, module_name)
            failures.append(
                f"  '{module_name}' (install: '{install_name}') — "
                f"used in {', '.join(files)}\n    Error: {e}"
            )

    assert not failures, (
        f"\n\n{len(failures)} missing import(s) found:\n"
        + "\n".join(failures)
        + "\n\nAdd the missing package(s) to requirements.txt."
    )


def test_no_unlisted_third_party_imports():
    """
    Every third-party import in source is accounted for in requirements.txt.
    Parses requirements.txt and checks that each imported package maps to
    a listed requirement (by install name).
    """
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        return  # nothing to check

    # Parse install names from requirements.txt (strip version specifiers/comments)
    listed = set()
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # strip version specifier: psutil>=5.9.0 → psutil
        name = line.split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("[")[0]
        listed.add(name.lower().replace("-", "_"))

    source_imports = gather_all_source_imports()
    unlisted = []

    for module_name, files in sorted(source_imports.items()):
        if module_name in OPTIONAL_IMPORTS or module_name in LOCAL_PACKAGES:
            continue

        install_name = IMPORT_TO_PACKAGE.get(module_name, module_name)
        normalized = install_name.lower().replace("-", "_")

        # Check if it's in requirements (match on install name or import name)
        if normalized not in listed and module_name.lower() not in listed:
            unlisted.append(
                f"  '{module_name}' — used in {', '.join(files)}, "
                f"not found in requirements.txt"
            )

    assert not unlisted, (
        f"\n\n{len(unlisted)} import(s) missing from requirements.txt:\n"
        + "\n".join(unlisted)
        + "\n\nAdd the missing package(s) to requirements.txt."
    )


# --- the app has one name -------------------------------------------------------


def test_the_installer_window_uses_the_current_app_name():
    """The DMG background said "Drag to install Whisper Transcriber" for every
    build after the rename, which is the first thing anybody downloading it
    saw. It was generated by a second script that the rename missed."""
    from scripts.create_icon import APP_NAME

    assert APP_NAME == "PodcastNotesWT"


def test_the_installer_background_is_built_by_the_script_that_is_still_used():
    """It used to live in create_assets.py, which nothing else called and
    which also still produced the old blue microphone icon. Two scripts is how
    one of them gets left behind."""
    import pathlib

    from scripts import create_icon

    assert hasattr(create_icon, "draw_dmg_background")
    assert not (pathlib.Path(__file__).parent.parent / "scripts" / "create_assets.py").exists()


def test_no_user_facing_string_still_says_the_old_name():
    """Comments explaining the rename are allowed to mention it. Strings a
    person reads are not."""
    import pathlib
    import re

    root = pathlib.Path(__file__).parent.parent
    offenders = []
    for path in list((root / "src").rglob("*.py")) + [root / "main.py"]:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "Whisper Transcriber" not in line:
                continue
            if re.search(r'["\']([^"\']*Whisper Transcriber[^"\']*)["\']', line):
                offenders.append(f"{path.relative_to(root)}:{number}")

    assert offenders == [], f"old name in a user-facing string: {offenders}"


# --- the theme reaches every widget ---------------------------------------------


def test_no_widget_sets_its_own_stylesheet():
    """An inline stylesheet beats the application one on exactly the widget it
    is set on, so a single stray call silently exempts that widget from the
    theme. This is the property the whole tokens-in-a-dict approach rests on."""
    import pathlib

    root = pathlib.Path(__file__).parent.parent
    offenders = [
        f"{p.relative_to(root)}:{n}"
        for p in (root / "src").rglob("*.py")
        if p.name != "theme.py"
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if "setStyleSheet" in line and not line.strip().startswith("#")
    ]

    assert offenders == [], f"inline styling escapes the theme: {offenders}"


def test_the_transcript_pane_is_styled_from_the_same_tokens():
    """A QTextBrowser lays out with Qt's rich-text engine, so the application
    stylesheet does not reach inside it. Its colours were written out by hand,
    which made it the one widget that ignored any change to the theme."""
    from src.ui.theme import TOKENS, document_stylesheet

    css = document_stylesheet()

    assert TOKENS["accent"] in css
    assert TOKENS["text_muted"] in css
    assert TOKENS["warning"] in css
    assert "#2962ff" not in css, "an old hardcoded colour survived"


def test_the_transcript_pane_actually_uses_it(qt_app):
    from src.ui.theme import TOKENS
    from src.ui.widgets import ClickablePreview

    view = ClickablePreview()
    try:
        assert TOKENS["accent"] in view.document().defaultStyleSheet()
    finally:
        view.close()


def test_changing_a_token_moves_the_transcript_pane_too(monkeypatch):
    """The point of the exercise: one value, every screen."""
    from src.ui import theme

    monkeypatch.setitem(theme.TOKENS, "accent", "#ff0000")

    assert "#ff0000" in theme.document_stylesheet()
