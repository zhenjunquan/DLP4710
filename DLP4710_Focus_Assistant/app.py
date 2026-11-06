"""
Application entry point for DLP4710 Focus Assistant.
"""

import os
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from DLP4710_Focus_Assistant.ui.main_window import FocusMainWindow


def main():
    # Enable high DPI scaling
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setStyle("Fusion")
    window = FocusMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

