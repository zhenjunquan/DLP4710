"""
3D Measurement application launcher.
"""

import sys
from PyQt6.QtWidgets import QApplication
from .ui.main_window import MeasurementMainWindow


def main():
    app = QApplication(sys.argv)             
    window = MeasurementMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

