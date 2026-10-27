"""
Modern industrial dark style sheet for DLP4710 3D Measurement Software.
"""

MEASUREMENT_STYLE = """
QMainWindow, QDialog {
    background-color: #1a1a20;
    color: #e2e2ea;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

QWidget {
    color: #e2e2ea;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}

QTabWidget::pane {
    border: 1px solid #333342;
    background-color: #1e1e26;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #262633;
    color: #9999aa;
    padding: 8px 18px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #353547;
    color: #ffffff;
    border-bottom: 2px solid #0d6efd;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #2e2e3d;
    color: #d0d0dd;
}

QGroupBox {
    border: 1px solid #363646;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 14px;
    padding-left: 8px;
    padding-right: 8px;
    padding-bottom: 8px;
    font-weight: bold;
    color: #58a6ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    background-color: #1a1a20;
}

QPushButton {
    background-color: #292936;
    color: #ffffff;
    border: 1px solid #404052;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 500;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #38384a;
    border-color: #58a6ff;
}

QPushButton:pressed {
    background-color: #1a56a6;
    border-color: #1a56a6;
}

QPushButton:disabled {
    background-color: #202028;
    color: #585866;
    border-color: #2c2c38;
}

QPushButton#measureBtn {
    background-color: #0d6efd;
    border-color: #0d6efd;
    font-size: 14px;
    font-weight: bold;
    min-height: 28px;
}

QPushButton#measureBtn:hover {
    background-color: #0b5ed7;
}

QPushButton#saveBtn {
    background-color: #198754;
    border-color: #198754;
    font-size: 13px;
    font-weight: bold;
}

QPushButton#saveBtn:hover {
    background-color: #157347;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #22222c;
    border: 1px solid #39394a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ffffff;
    selection-background-color: #0d6efd;
    min-height: 20px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #58a6ff;
}

QTextEdit {
    background-color: #141418;
    border: 1px solid #2d2d3a;
    border-radius: 4px;
    color: #c8c8d4;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}

QProgressBar {
    border: 1px solid #333342;
    border-radius: 4px;
    text-align: center;
    background-color: #202028;
    color: #ffffff;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #0d6efd;
    border-radius: 3px;
}
"""

