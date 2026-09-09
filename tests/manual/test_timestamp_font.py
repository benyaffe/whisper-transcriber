#!/usr/bin/env python3
"""Test script to verify timestamp monospace rendering in QTextBrowser."""

import sys
from PyQt6.QtWidgets import QApplication, QTextBrowser, QVBoxLayout, QWidget, QLabel

def main():
    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("Timestamp Font Test - Document Stylesheet")
    window.setMinimumSize(700, 300)
    layout = QVBoxLayout(window)

    # THE CORRECT APPROACH: Document stylesheet with class
    label = QLabel("<b>Document stylesheet with class (CORRECT APPROACH):</b>")
    layout.addWidget(label)

    browser = QTextBrowser()
    browser.setMinimumHeight(100)
    browser.setOpenLinks(False)

    # Set document stylesheet - THIS IS THE KEY
    # Use Menlo as primary - guaranteed on macOS
    browser.document().setDefaultStyleSheet("""
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 13px;
        }
        a.ts {
            font-family: Menlo, Monaco, Courier;
            color: #2962ff;
            text-decoration: none;
        }
    """)

    # Add multiple lines to verify
    browser.append("<a href='#' class='ts'>[0:00]</a> First line of text")
    browser.append("<a href='#' class='ts'>[0:15]</a> Second line with different numbers")
    browser.append("<a href='#' class='ts'>[1:23]</a> Third line - check alignment")
    browser.append("<a href='#' class='ts'>[10:05]</a> Fourth line with two digit minutes")
    browser.append("<a href='#' class='ts'>[123:59]</a> Long timestamp for comparison")

    layout.addWidget(browser)

    # Reference: what it SHOULD look like
    label2 = QLabel("<b>Reference - what monospace looks like:</b>")
    layout.addWidget(label2)

    ref_browser = QTextBrowser()
    ref_browser.setMinimumHeight(60)
    ref_browser.setStyleSheet("font-family: Menlo; font-size: 13px;")
    ref_browser.setPlainText("[0:00] [0:15] [1:23] [10:05]\nAll characters same width ^^^^")
    layout.addWidget(ref_browser)

    window.show()

    print("\n" + "="*60)
    print("VALIDATION CHECK:")
    print("="*60)
    print("1. Look at the timestamps in the top box")
    print("2. The brackets [ ] and numbers should all be EQUAL WIDTH")
    print("3. Compare with the reference at the bottom")
    print("4. In monospace: [0:00] and [1:23] take the same width")
    print("="*60 + "\n")

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
