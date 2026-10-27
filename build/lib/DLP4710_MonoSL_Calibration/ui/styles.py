"""
Modern industrial dark/neutral style sheet for DLP4710 PyQt6 GUI applications.
"""

MODERN_STYLE = """
QMainWindow, QDialog {
    background-color: #1e1e24;
    color: #e0e0e0;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

QWidget {
    color: #e0e0e0;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}

QGroupBox {
    border: 1px solid #3a3a48;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 14px;
    padding-left: 8px;
    padding-right: 8px;
    padding-bottom: 8px;
    font-weight: bold;
    color: #4da3ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    background-color: #1e1e24;
}

QPushButton {
    background-color: #2b2b36;
    color: #ffffff;
    border: 1px solid #454558;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #38384a;
    border-color: #4da3ff;
}

QPushButton:pressed {
    background-color: #1e6acc;
    border-color: #1e6acc;
}

QPushButton:disabled {
    background-color: #25252d;
    color: #666677;
    border-color: #30303a;
}

QPushButton#primaryBtn {
    background-color: #0d6efd;
    border-color: #0d6efd;
    font-weight: bold;
}

QPushButton#primaryBtn:hover {
    background-color: #0b5ed7;
    border-color: #0a58ca;
}

QPushButton#successBtn {
    background-color: #198754;
    border-color: #198754;
    font-weight: bold;
}

QPushButton#successBtn:hover {
    background-color: #157347;
    border-color: #146c43;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #262630;
    border: 1px solid #3e3e50;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ffffff;
    selection-background-color: #0d6efd;
    min-height: 20px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #4da3ff;
}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background-color: #1d1d23;
    color: #555566;
}

QTableWidget {
    background-color: #23232c;
    border: 1px solid #3a3a48;
    border-radius: 4px;
    gridline-color: #323240;
    selection-background-color: #0d6efd;
    selection-color: #ffffff;
    alternate-background-color: #272733;
}

QHeaderView::section {
    background-color: #2b2b36;
    color: #b0b0cc;
    padding: 5px;
    border: 1px solid #363644;
    font-weight: bold;
}

QTextEdit {
    background-color: #16161b;
    border: 1px solid #363644;
    border-radius: 4px;
    color: #d0d0d8;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}

QProgressBar {
    border: 1px solid #363644;
    border-radius: 4px;
    text-align: center;
    background-color: #22222b;
    color: #ffffff;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #0d6efd;
    border-radius: 3px;
}

QScrollBar:vertical {
    background: #1e1e24;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #3e3e50;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #505068;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QLabel#statusLabel {
    color: #8da4c4;
    font-weight: 500;
}
"""

