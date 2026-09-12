"""
One place the app decides what it looks like.

QSS has no variables, so the tokens live in a Python dict and the stylesheet
is built from them. That is the whole trick: change a value here and every
screen follows, instead of thirteen scattered setStyleSheet calls drifting
apart.

Two rules that matter more than they sound:

**Inline setStyleSheet beats the application stylesheet**, on exactly the
widget it is set on. A widget with its own inline style silently ignores the
theme, so the old inline colours were removed rather than left to fight this
file. Use `role(widget, "muted")` instead; it sets a property the stylesheet
can select on, which keeps the cascade intact.

**Semantic names, not colour names.** `MUTED` rather than `GREY`, so a dark
variant is a change of values here and nothing else.

**Every value in the sheet is a token.** Not for tidiness: eleven distinct
spacing values had accumulated here, which is not a system but eleven separate
decisions nobody can remember the reasons for. Lengths come from a four-point
scale, radii and type from their own scales, and `tests/test_theme.py` fails on
a hardcoded pixel.

**The palette is measured, not chosen.** Every foreground is checked against
every surface it can land on, at WCAG AA. That check found three failures that
had shipped: `text_faint` at 2.88:1, which is what hints, evidence lines and
the speaker samples were set in; `accent` at 4.23:1, which is the timestamps
and every quiet button; and accent on its own tint at 3.94:1, which is a
checked toggle.

Two things Qt will not do, which shape what is here. QSS ignores `line-height`,
so the leading tokens reach only the rich-text document sheet, and the rhythm
of the widget screens comes from spacing instead. And a `:focus` rule that
changes a border *width* reflows the control, so every focus style here keeps
the width and changes only the colour.
"""

import os
import tempfile

from PyQt6.QtWidgets import QWidget

TOKENS = {
    # Surfaces, lightest to darkest.
    "canvas": "#f7f8fa",
    "surface": "#ffffff",
    "sunken": "#f0f2f5",
    "border": "#dfe3e9",
    "border_strong": "#c6ccd6",

    # Text, in descending prominence. Three levels is enough; a fourth just
    # produces arguments about which to use.
    # Two, not three. Above 4.5:1 on a near-white canvas every usable grey
    # sits within about a point of `text_muted`, close enough that the pair
    # read as one colour, so the third level of the hierarchy is carried by
    # size. The `faint` role is `text_muted` at the small size.
    "text": "#171b22",
    "text_muted": "#5b6472",

    # 4.87:1 on the canvas, 4.61:1 on the sunken panels and 4.53:1 on its own
    # tint. The old #2f6bff was 4.23:1, and it is text: the timestamps in the
    # transcript and every quiet button are set in it.
    "accent": "#1c5dff",
    "accent_hover": "#255ae0",
    "accent_pressed": "#1c49bb",
    "accent_soft": "#eaf0ff",

    "success": "#137a3a",
    "danger": "#c02a20",
    "warning": "#8a5a00",

    # A ring, not a border. Distinct from `accent` so that focusing a control
    # already drawn in the accent colour still shows.
    "focus": "#0b3ea8",

    # Four-point spacing. Six steps: anything a screen needs is one of these,
    # and a seventh would be the start of the drift this replaced.
    "space_1": "4px",
    "space_2": "8px",
    "space_3": "12px",
    "space_4": "16px",
    "space_5": "24px",
    "space_6": "32px",

    "radius_sm": "6px",
    "radius": "8px",
    "radius_lg": "12px",
    "hairline": "1px",

    "font": '-apple-system, "SF Pro Text", "Helvetica Neue", sans-serif',
    "size_small": "12px",
    "size_body": "13px",
    "size_h2": "16px",
    "size_h1": "24px",

    # Roughly 1.5 for prose and tighter as the type grows, which is the usual
    # shape. These reach the rich-text document sheet only: QSS ignores
    # `line-height`, so a widget label cannot be given one.
    "leading_small": "18px",
    "leading_body": "20px",
    "leading_h2": "22px",
    "leading_h1": "30px",
}


def role(widget: QWidget, name: str) -> QWidget:
    """Tag a widget so the stylesheet can style it.

    Returns the widget so it can be used inline while building a layout.
    """
    widget.setProperty("role", name)
    return widget


def restyle(widget: QWidget):
    """Make a widget pick up a role changed after it was first shown.

    Qt does not re-evaluate property selectors on its own, so a row that
    switches from failed to ok keeps its old colour without this.
    """
    widget.style().unpolish(widget)
    widget.style().polish(widget)


STYLESHEET = """
QWidget {{
    background: {canvas};
    color: {text};
    font-family: {font};
    font-size: {size_body};
}}

QLabel {{ background: transparent; }}
QLabel[role="h1"] {{ font-size: {size_h1}; font-weight: 600; }}
QLabel[role="h2"] {{ font-size: {size_h2}; font-weight: 600; }}
QLabel[role="muted"] {{ color: {text_muted}; }}
QLabel[role="faint"] {{ color: {text_muted}; font-size: {size_small}; }}
QLabel[role="success"] {{ color: {success}; }}
QLabel[role="danger"] {{ color: {danger}; }}

/* Fields ------------------------------------------------------------- */

QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {{
    background: {surface};
    border: 1px solid {border};
    border-radius: {radius};
    padding: {space_2} {space_3};
    selection-background-color: {accent_soft};
    selection-color: {text};
}}
/* Colour only, never width. A focus rule that thickens a border reflows the
   control, so tabbing along a row makes each one jump as it is reached. */
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{
    border: {hairline} solid {focus};
}}
QLineEdit:disabled, QPlainTextEdit:disabled {{
    background: {sunken};
    color: {text_muted};
}}
QLineEdit[role="title"] {{
    font-size: {size_h2};
    padding: {space_3};
}}

/* Styling a QComboBox at all makes Qt stop painting the native control, so
   without an arrow the box has no affordance whatsoever and reads as a plain
   text field: the list is there and nothing on screen says so. Found by
   rendering it. The image is drawn from the palette rather than shipped, so
   the arrow cannot be the one thing that ignores a colour change here, and so
   there is no asset for the packaging step to leave behind. */
QComboBox::drop-down {{ border: none; width: {space_6}; }}
QComboBox::down-arrow {{ image: url("{chevron}"); width: {space_3}; height: {space_3}; }}
QComboBox QAbstractItemView {{
    background: {surface};
    border: 1px solid {border};
    selection-background-color: {accent_soft};
    selection-color: {text};
    outline: none;
}}

/* Buttons ------------------------------------------------------------ */

QPushButton {{
    background: {surface};
    border: {hairline} solid {border_strong};
    border-radius: {radius};
    padding: {space_2} {space_4};
    color: {text};
}}
QPushButton:hover {{ background: {sunken}; }}
QPushButton:pressed {{ background: {border}; }}
QPushButton:disabled {{ color: {text_muted}; border-color: {border}; }}
QPushButton:focus {{ border-color: {focus}; }}
QCheckBox:focus {{ color: {focus}; }}

/* A checkable button used as a tab. Without this a checked button is drawn
   exactly like an unchecked one, so a two-way switch shows nothing at all
   about which side is showing. Found by rendering the screen, not by a test. */
/* Colour and border only, deliberately not a heavier font. The button sizes
   itself to its label once, and turning the text bold on check makes it wider
   than the space already reserved, so the last character clips. */
QPushButton:checked {{
    background: {accent_soft};
    border-color: {accent};
    color: {accent};
}}
QPushButton:checked:hover {{ background: {accent_soft}; }}

QPushButton[role="primary"] {{
    background: {accent};
    border: {hairline} solid {accent};
    color: #ffffff;
    font-weight: 600;
    padding: {space_2} {space_5};
}}
QPushButton[role="primary"]:focus {{ border-color: {focus}; }}
QPushButton[role="primary"]:hover {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton[role="primary"]:pressed {{ background: {accent_pressed}; }}
QPushButton[role="primary"]:disabled {{
    background: {border};
    border-color: {border};
    color: {text_muted};
}}

QPushButton[role="quiet"] {{
    background: transparent;
    border: none;
    color: {accent};
    padding: {space_1} {space_2};
}}
QPushButton[role="quiet"]:focus {{ background: {accent_soft}; color: {focus}; }}
/* Without this a disabled quiet button keeps the full accent and reads as
   clickable. Seen on "Stopping...", where looking pressable is the worst
   possible thing for it to do. */
QPushButton[role="quiet"]:disabled {{ color: {text_muted}; }}
QPushButton[role="quiet"]:hover {{ background: {accent_soft}; }}

/* Containers --------------------------------------------------------- */

/* The one elevation step. A card is lifted by a lighter surface against the
   canvas and a hairline, not by a shadow: QSS has no box-shadow, and a second
   step would need one, so there is exactly one. */
QFrame[role="card"] {{
    background: {surface};
    border: {hairline} solid {border};
    border-radius: {radius_lg};
    padding: {space_4};
}}
QFrame[role="rule"] {{
    background: {border};
    max-height: {hairline};
    border: none;
}}

QListWidget {{
    background: {surface};
    border: {hairline} solid {border};
    border-radius: {radius};
    padding: {space_1};
    outline: none;
}}
QListWidget:focus {{ border-color: {focus}; }}
/* The transparent border is load-bearing. `outline: none` above is right for
   the mouse and takes away the only cue a keyboard has, so the focused row is
   given one back; adding a border on focus alone would shift the row's
   contents by a pixel as it is reached. */
QListWidget::item {{
    padding: {space_2};
    border: {hairline} solid transparent;
    border-radius: {radius_sm};
    color: {text};
}}
QListWidget::item:focus {{ border-color: {focus}; }}
QListWidget::item:selected {{ background: {accent_soft}; color: {text}; }}
QListWidget::item:hover {{ background: {sunken}; }}

QGroupBox {{
    background: {surface};
    border: {hairline} solid {border};
    border-radius: {radius_lg};
    margin-top: {space_4};
    padding: {space_4};
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {space_3};
    padding: 0 {space_1};
    background: {canvas};
}}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: {space_3}; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {border_strong}; border-radius: {radius_sm}; min-height: {space_6};
}}
QScrollBar::handle:vertical:hover {{ background: {text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QFrame[role="dropzone"] {{
    border: {hairline} dashed {border_strong};
    border-radius: {radius_lg};
    background: {sunken};
}}
QFrame[role="dropzone"][hover="true"] {{
    border: {hairline} solid {accent};
    background: {accent_soft};
}}

/* The indicator is deliberately left to macOS. Styling the box without
   supplying a tick image produces a filled square that reads as
   indeterminate, and the native control is better than anything drawn
   from QSS here. */
QCheckBox {{ spacing: {space_2}; background: transparent; }}

QProgressBar {{
    background: {sunken};
    border: none;
    border-radius: {radius_sm};
    height: {space_2};
    text-align: center;
}}
QProgressBar::chunk {{ background: {accent}; border-radius: {radius_sm}; }}

QMenuBar, QMenu {{ background: {surface}; }}
QMenu::item:selected {{ background: {accent_soft}; }}
"""


# QTextBrowser lays its content out with Qt's rich-text engine, not with the
# widget stylesheet, so nothing in STYLESHEET above reaches inside one. It
# needs its own sheet, set on the document, and that sheet understands only a
# small subset of CSS: no variables, no custom properties, and selectors
# limited to tags and classes.
#
# It is generated from the same tokens anyway, because the alternative is what
# was here before: a transcript pane with its own hardcoded blue and its own
# hardcoded grey, quietly ignoring every change made to this file.
DOCUMENT_STYLESHEET = """
body {{
    font-family: {font};
    font-size: {size_body};
    line-height: {leading_body};
    color: {text};
}}
h1 {{ font-size: {size_h1}; line-height: {leading_h1}; }}
h2, h3 {{ font-size: {size_h2}; line-height: {leading_h2}; }}
small {{ font-size: {size_small}; line-height: {leading_small}; }}
a.ts {{
    font-family: Menlo, Monaco, Courier;
    color: {accent};
    text-decoration: none;
}}
.status {{
    color: {text_muted};
    font-style: italic;
}}
.warning {{
    color: {warning};
    font-style: italic;
}}
"""


def chevron(colour: str = "", where: str = "") -> str:
    """Draw the combo box arrow, and give back a path QSS can point at.

    Generated rather than bundled. The alternative is an SVG in the repo that
    quietly keeps its own colour when the palette changes, plus one more file
    for PyInstaller to forget.

    Returns "" when there is no QApplication, which is the case in the tests
    that only read the sheet. A `url("")` is ignored by Qt, so the sheet still
    builds and the only thing missing is a picture nothing is looking at.
    """
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QGuiApplication, QPainter, QPen, QPixmap

    if QGuiApplication.instance() is None:
        return ""

    colour = colour or TOKENS["text_muted"]
    # Drawn at 2x so it is not soft on a Retina display, which is every Mac
    # this runs on.
    size, scale = 12, 2
    image = QPixmap(size * scale, size * scale)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(_colour(colour))
    pen.setWidth(2 * scale)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline([
        QPointF(3.5 * scale, 5 * scale),
        QPointF(6 * scale, 7.5 * scale),
        QPointF(8.5 * scale, 5 * scale),
    ])
    painter.end()

    where = where or os.path.join(
        tempfile.gettempdir(), f"podcastnotes-chevron-{colour.lstrip('#')}.png"
    )
    image.save(where, "PNG")
    # Forward slashes, because a QSS url() on Windows reads a backslash as an
    # escape. Harmless on macOS and one less thing to find out later.
    return where.replace("\\", "/")


def _colour(value: str):
    from PyQt6.QtGui import QColor

    return QColor(value)


def stylesheet() -> str:
    return STYLESHEET.format(chevron=chevron(), **TOKENS)


def document_stylesheet() -> str:
    """The sheet for a QTextBrowser's document, from the same tokens."""
    return DOCUMENT_STYLESHEET.format(**TOKENS)


def apply(app):
    """Apply the theme to a QApplication."""
    app.setStyleSheet(stylesheet())
