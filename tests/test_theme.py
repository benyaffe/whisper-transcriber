"""
Tests for the one place the app decides what it looks like.

Most of this file measures contrast. That is worth doing as a test rather than
once by hand because a colour is the easiest thing in the codebase to nudge:
somebody lightens a grey to make a hint look calmer, the ratio drops under the
threshold, and nothing anywhere says so. The numbers here were measured against
WCAG 2.1, and every foreground is checked against every surface it can land on
rather than against the one it was designed on.

The rest checks that the tokens are actually used. A scale nothing draws from
is a comment, and the reason this file exists is that eleven distinct spacing
values had accumulated with no scale at all.

Run with: python -m pytest tests/test_theme.py -v
"""

import re

import pytest

from src.ui import theme

# WCAG 2.1 AA. 4.5:1 for body text, 3:1 for large text and for the boundary of
# a control somebody has to find with a keyboard.
AA_TEXT = 4.5
AA_LARGE = 3.0

# Every background a foreground can land on. Checking against only the canvas
# is how `accent` passed on paper and failed on the sunken panels it is
# actually drawn on.
SURFACES = ("canvas", "surface", "sunken")


def _luminance(colour: str) -> float:
    """Relative luminance, per WCAG 2.1."""
    value = colour.lstrip("#")

    def channel(pair):
        v = int(pair, 16) / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(value[0:2])
        + 0.7152 * channel(value[2:4])
        + 0.0722 * channel(value[4:6])
    )


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _token(name: str) -> str:
    return theme.TOKENS[name]


# --- contrast --------------------------------------------------------------------


def test_the_measurement_agrees_with_the_specification():
    """A contrast function that is subtly wrong would pass every test below
    while the screen stayed unreadable, so it is pinned to values anybody can
    check: black on white is 21, and a colour against itself is 1."""
    assert round(contrast("#000000", "#ffffff"), 2) == 21.0
    assert round(contrast("#777777", "#777777"), 2) == 1.0
    assert round(contrast("#767676", "#ffffff"), 1) == 4.5


@pytest.mark.parametrize("name", ["text", "text_muted", "success", "danger", "warning"])
@pytest.mark.parametrize("surface", SURFACES)
def test_every_text_colour_is_readable_on_every_surface(name, surface):
    got = contrast(_token(name), _token(surface))

    assert got >= AA_TEXT, f"{name} on {surface} is {got:.2f}:1"


@pytest.mark.parametrize("surface", SURFACES)
def test_the_accent_is_readable_as_text(surface):
    """It is not decoration. The quiet buttons and the transcript timestamps
    are accent-coloured text at the body size, and at 4.23:1 they were the
    least readable thing on the screen."""
    got = contrast(_token("accent"), _token(surface))

    assert got >= AA_TEXT, f"accent on {surface} is {got:.2f}:1"


def test_the_accent_is_readable_on_its_own_tint():
    """A checked toggle draws accent text on `accent_soft`, which is the
    tightest pairing in the palette and the one a canvas-only check misses."""
    got = contrast(_token("accent"), _token("accent_soft"))

    assert got >= AA_TEXT, f"accent on accent_soft is {got:.2f}:1"


def test_a_primary_button_is_readable():
    got = contrast("#ffffff", _token("accent"))

    assert got >= AA_TEXT, f"white on accent is {got:.2f}:1"


def test_there_is_no_third_grey_pretending_to_pass():
    """There is not room for one. Above 4.5:1 on a near-white canvas every
    usable grey sits within about a point of `text_muted`, close enough that
    the two read as the same colour, so the third level of the hierarchy is
    carried by size instead. A `text_faint` token would be an invitation to
    reintroduce the 2.88:1 grey that hints, evidence lines and speaker samples
    were set in."""
    assert "text_faint" not in theme.TOKENS
    assert "text_faint" not in theme.STYLESHEET


def test_the_faint_role_is_smaller_rather_than_paler():
    rule = _rule('QLabel[role="faint"]')

    assert "{size_small}" in rule
    assert "{text_muted}" in rule


def test_a_disabled_control_is_still_legible():
    """Greyed is not the same as gone. Somebody has to be able to read the
    label to work out what would happen if they could press it, and a disabled
    primary button sits on `border` rather than on any of the surfaces."""
    for name in ("QPushButton:disabled", "QLineEdit:disabled, QPlainTextEdit:disabled",
                 'QPushButton[role="primary"]:disabled'):
        rule = _rule(name)
        # Not `border-color`, which the naive pattern matched first and which
        # made this assert that a background was readable against a surface.
        colour = re.search(r"(?<![-\w])color:\s*\{(\w+)\}", rule)
        assert colour, f"{name} sets no text colour"
        for behind in ("sunken", "border"):
            got = contrast(_token(colour.group(1)), _token(behind))
            assert got >= AA_TEXT, f"{name} on {behind} is {got:.2f}:1"


def test_a_selected_row_is_still_readable():
    """`accent_soft` is the background of a selected list row, a checked
    toggle, and a text selection, so everything drawn on it has to clear the
    bar there too."""
    for name in ("text", "text_muted", "accent"):
        got = contrast(_token(name), _token("accent_soft"))
        assert got >= AA_TEXT, f"{name} on accent_soft is {got:.2f}:1"


# --- the scales ------------------------------------------------------------------


def _rule(selector: str) -> str:
    """The body of one stylesheet rule, unformatted, tokens and all."""
    at = theme.STYLESHEET.index(selector + " {{")
    return theme.STYLESHEET[at:theme.STYLESHEET.index("}}", at)]


def _lengths(sheet: str) -> list:
    """Every hardcoded pixel length left in the sheet."""
    return [int(n) for n in re.findall(r"(?<![\w-])(\d+)px", sheet)]


def test_spacing_comes_from_a_scale():
    """Eleven distinct values had accumulated, which is not a system, it is
    eleven separate decisions nobody can remember the reasons for."""
    scale = [v for k, v in theme.TOKENS.items() if k.startswith("space_")]

    assert scale, "there is no spacing scale"
    assert all(int(v.rstrip("px")) % 4 == 0 for v in scale), scale


def test_nothing_in_the_sheet_measures_itself():
    """Every length is a token, so changing the rhythm is one edit here rather
    than a search through the file. Hairlines and radii excepted: a 1px border
    is not spacing, and the radii have their own tokens."""
    stray = [n for n in _lengths(theme.STYLESHEET) if n > 2]

    assert stray == [], f"hardcoded lengths left in the sheet: {sorted(set(stray))}"


def test_the_radii_come_from_tokens():
    assert "border-radius: 6px" not in theme.STYLESHEET
    assert "border-radius: 5px" not in theme.STYLESHEET


def test_the_type_scale_has_a_line_height_for_every_size():
    """13px prose set solid is the densest thing on the screen, and the summary
    is several hundred words of it."""
    sizes = {k for k in theme.TOKENS if k.startswith("size_")}
    leading = {k.replace("leading_", "size_") for k in theme.TOKENS
               if k.startswith("leading_")}

    assert sizes <= leading, f"no line-height for {sorted(sizes - leading)}"


# --- being able to see where you are ---------------------------------------------


def test_a_keyboard_focus_is_visible_on_every_control_you_can_reach():
    """`QListWidget` sets `outline: none`, which is right for the mouse and
    removes the only cue a keyboard has. Every one of these is in the tab
    order, and a focus ring nobody can see is the same as no focus at all."""
    for control in ("QPushButton", "QCheckBox", "QListWidget::item", "QComboBox"):
        assert f"{control}:focus" in theme.STYLESHEET, f"{control} has no focus ring"


def test_the_focus_ring_is_visible_against_what_it_surrounds():
    got = min(contrast(_token("focus"), _token(s)) for s in SURFACES)

    assert got >= AA_LARGE, f"the focus ring is {got:.2f}:1"


def test_no_focus_rule_moves_what_it_surrounds():
    """A focus style that thickens a border or adds padding reflows the
    control, so tabbing along a row of buttons makes each one jump as it is
    reached. Every rule here changes colour and nothing else."""
    moves = ("border-width", "padding", "margin", "width:", "height:")
    for line in theme.STYLESHEET.splitlines():
        if ":focus" not in line:
            continue
        body = line.split("{{", 1)[-1]
        assert not any(m in body for m in moves), line
        assert "border:" not in body, f"{line}\nuse border-color, which cannot resize"


def test_a_ring_has_somewhere_to_be_drawn():
    """Changing `border-color` only avoids a reflow if there is already a
    border of that width to recolour. On a control drawn without one, the rule
    creates the border and shifts the contents by a pixel as it is reached,
    which is the same jump the colour-only rules exist to avoid."""
    for line in theme.STYLESHEET.splitlines():
        if ":focus" not in line or "border-color" not in line:
            continue
        base = line.split(":focus")[0].strip()
        rule = _rule(base)
        declared = re.search(r"(?<![-\w])border:\s*([^;]+)", rule)
        assert declared, f"{base} is given a focus ring but declares no border"
        # `border: none` counts as no border. Recolouring it creates one, which
        # is the reflow this is here to prevent.
        assert "none" not in declared.group(1), (
            f"{base} has no border to recolour, so its ring would resize it"
        )


# --- the combo box arrow ----------------------------------------------------------


def test_a_combo_box_says_it_is_one(qt_app):
    """Styling a QComboBox at all makes Qt stop painting the native control, so
    the speaker name box had no affordance whatsoever: a list of seven people
    behind something that looked exactly like a text field. Rendered and
    caught."""
    assert "QComboBox::down-arrow" in theme.STYLESHEET

    built = theme.stylesheet()
    assert 'image: url("")' not in built, "no arrow was drawn"


def test_the_arrow_is_drawn_in_the_palette_colour(qt_app, tmp_path):
    """Otherwise it is the one thing on the screen that keeps its own colour
    when the theme changes, which is what this whole file exists to prevent."""
    from PyQt6.QtGui import QColor, QImage

    where = str(tmp_path / "arrow.png")
    theme.chevron("#c02a20", where=where)
    image = QImage(where)

    used = {
        QColor(image.pixel(x, y)).name()
        for x in range(image.width()) for y in range(image.height())
        if QColor.fromRgba(image.pixel(x, y)).alpha() > 200
    }
    assert used, "nothing was drawn"
    assert used <= {"#c02a20"}, sorted(used)


def test_the_sheet_still_builds_without_a_running_application(monkeypatch):
    """The packaging checks and half the tests read the sheet with no
    QApplication, and a QPixmap cannot be made without one."""
    from PyQt6.QtGui import QGuiApplication

    monkeypatch.setattr(QGuiApplication, "instance", staticmethod(lambda: None))

    assert theme.chevron() == ""
    theme.stylesheet()


# --- the sheet stays buildable ----------------------------------------------------


def test_every_token_the_sheet_asks_for_exists():
    """A missing token is a KeyError at startup, on the machine of whoever
    opens the app next rather than here."""
    theme.stylesheet()
    theme.document_stylesheet()


def test_the_document_sheet_draws_on_the_same_palette():
    """A QTextBrowser lays out with the rich-text engine and ignores the widget
    stylesheet entirely, so it had its own hardcoded blue and its own
    hardcoded grey, quietly ignoring every change made here."""
    built = theme.document_stylesheet()

    assert _token("accent") in built
    assert _token("text_muted") in built
