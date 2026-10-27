"""
Modern industrial dark style sheet for DLP4710 3D Measurement Software.
"""

MEASUREMENT_STYLE = """
QMainWindow, QDialog {
    background-color: #1a1a20;
    color: #e2e2ea;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 12px;
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
    padding: 6px 14px;
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
    margin-top: 11px;
    padding-top: 10px;
    padding-left: 6px;
    padding-right: 6px;
    padding-bottom: 6px;
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
    padding: 4px 9px;
    font-weight: 500;
    min-height: 20px;
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
    font-size: 12px;
    font-weight: bold;
    min-height: 24px;
}

QPushButton#measureBtn:hover {
    background-color: #0b5ed7;
}

QPushButton#saveBtn {
    background-color: #198754;
    border-color: #198754;
    font-size: 12px;
    font-weight: bold;
}

QPushButton#saveBtn:hover {
    background-color: #157347;
}

QLineEdit, QComboBox {
    background-color: #22222c;
    border: 1px solid #39394a;
    border-radius: 4px;
    padding: 3px 6px;
    color: #ffffff;
    selection-background-color: #0d6efd;
    min-height: 18px;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #58a6ff;
}

QLineEdit:disabled, QComboBox:disabled {
    background-color: #1a1a20;
    color: #555566;
}

QSpinBox, QDoubleSpinBox {
    background-color: #22222c;
    border: 1px solid #39394a;
    border-radius: 4px;
    padding: 3px 22px 3px 6px;
    color: #ffffff;
    selection-background-color: #0d6efd;
    min-height: 18px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #58a6ff;
}

QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #1a1a20;
    color: #555566;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #39394a;
    border-bottom: 1px solid #39394a;
    background-color: #292936;
    border-top-right-radius: 3px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #38384a;
}

QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
    background-color: #1a56a6;
}

QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled {
    background-color: #202028;
    border-color: #2c2c38;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0px;
    height: 0px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #b0b0cc;
}

QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {
    border-bottom-color: #555566;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid #39394a;
    background-color: #292936;
    border-bottom-right-radius: 3px;
}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #38384a;
}

QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background-color: #1a56a6;
}

QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {
    background-color: #202028;
    border-color: #2c2c38;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0px;
    height: 0px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #b0b0cc;
}

QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {
    border-top-color: #555566;
}

QTextEdit {
    background-color: #141418;
    border: 1px solid #2d2d3a;
    border-radius: 4px;
    color: #c8c8d4;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
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

QScrollArea#controlScroll {
    border: none;
    background: transparent;
}

QScrollArea#controlScroll > QWidget > QWidget {
    background: transparent;
}
"""

