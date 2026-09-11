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
"""

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
    "text": "#171b22",
    "text_muted": "#5b6472",
    "text_faint": "#8b94a3",

    "accent": "#2f6bff",
    "accent_hover": "#255ae0",
    "accent_pressed": "#1c49bb",
    "accent_soft": "#eaf0ff",

    "success": "#137a3a",
    "danger": "#c02a20",
    "warning": "#8a5a00",

    "radius": "8px",
    "radius_lg": "12px",

    "font": '-apple-system, "SF Pro Text", "Helvetica Neue", sans-serif',
    "size_body": "13px",
    "size_small": "12px",
    "size_h1": "24px",
    "size_h2": "16px",
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
QLabel[role="faint"] {{ color: {text_faint}; font-size: {size_small}; }}
QLabel[role="success"] {{ color: {success}; }}
QLabel[role="danger"] {{ color: {danger}; }}

/* Fields ------------------------------------------------------------- */

QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {{
    background: {surface};
    border: 1px solid {border};
    border-radius: {radius};
    padding: 8px 10px;
    selection-background-color: {accent_soft};
    selection-color: {text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{
    border: 1px solid {accent};
}}
QLineEdit:disabled, QPlainTextEdit:disabled {{
    background: {sunken};
    color: {text_faint};
}}
QLineEdit[role="title"] {{
    font-size: {size_h2};
    padding: 11px 12px;
}}

QComboBox::drop-down {{ border: none; width: 22px; }}
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
    border: 1px solid {border_strong};
    border-radius: {radius};
    padding: 7px 14px;
    color: {text};
}}
QPushButton:hover {{ background: {sunken}; }}
QPushButton:pressed {{ background: {border}; }}
QPushButton:disabled {{ color: {text_faint}; border-color: {border}; }}

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
    border: 1px solid {accent};
    color: #ffffff;
    font-weight: 600;
    padding: 9px 20px;
}}
QPushButton[role="primary"]:hover {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton[role="primary"]:pressed {{ background: {accent_pressed}; }}
QPushButton[role="primary"]:disabled {{
    background: {border};
    border-color: {border};
    color: {text_faint};
}}

QPushButton[role="quiet"] {{
    background: transparent;
    border: none;
    color: {accent};
    padding: 6px 8px;
}}
QPushButton[role="quiet"]:hover {{ background: {accent_soft}; }}

/* Containers --------------------------------------------------------- */

QFrame[role="card"] {{
    background: {surface};
    border: 1px solid {border};
    border-radius: {radius_lg};
}}
QFrame[role="rule"] {{
    background: {border};
    max-height: 1px;
    border: none;
}}

QListWidget {{
    background: {surface};
    border: 1px solid {border};
    border-radius: {radius};
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 7px 8px;
    border-radius: 6px;
    color: {text};
}}
QListWidget::item:selected {{ background: {accent_soft}; color: {text}; }}
QListWidget::item:hover {{ background: {sunken}; }}

QGroupBox {{
    background: {surface};
    border: 1px solid {border};
    border-radius: {radius_lg};
    margin-top: 14px;
    padding: 14px 14px 12px 14px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    background: {canvas};
}}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 11px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {border_strong}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {text_faint}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QFrame[role="dropzone"] {{
    border: 1px dashed {border_strong};
    border-radius: {radius_lg};
    background: {sunken};
}}
QFrame[role="dropzone"][hover="true"] {{
    border: 1px solid {accent};
    background: {accent_soft};
}}

/* The indicator is deliberately left to macOS. Styling the box without
   supplying a tick image produces a filled square that reads as
   indeterminate, and the native control is better than anything drawn
   from QSS here. */
QCheckBox {{ spacing: 8px; background: transparent; }}

QProgressBar {{
    background: {sunken};
    border: none;
    border-radius: 5px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {accent}; border-radius: 5px; }}

QMenuBar, QMenu {{ background: {surface}; }}
QMenu::item:selected {{ background: {accent_soft}; }}
"""


def stylesheet() -> str:
    return STYLESHEET.format(**TOKENS)


def apply(app):
    """Apply the theme to a QApplication."""
    app.setStyleSheet(stylesheet())
