"""
Modern industrial dark style sheet for DLP4710 Focus Assistant.
"""

FOCUS_STYLE = """
QMainWindow, QDialog {
    background-color: #1a1a22;
    color: #e2e2ea;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 12px;
}

QWidget {
    color: #e2e2ea;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
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
    background-color: #1a1a22;
}

QPushButton {
    background-color: #292936;
    color: #ffffff;
    border: 1px solid #404052;
    border-radius: 4px;
    padding: 5px 10px;
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

QPushButton#primaryBtn {
    background-color: #0d6efd;
    border-color: #0d6efd;
    font-weight: bold;
}

QPushButton#primaryBtn:hover {
    background-color: #0b5ed7;
}

QPushButton#successBtn {
    background-color: #198754;
    border-color: #198754;
    font-weight: bold;
}

QPushButton#successBtn:hover {
    background-color: #157347;
}

QPushButton#dangerBtn {
    background-color: #dc3545;
    border-color: #dc3545;
}

QPushButton#dangerBtn:hover {
    background-color: #bb2d3b;
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

QCheckBox {
    spacing: 6px;
    color: #d0d0dd;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #404052;
    background-color: #22222c;
}

QCheckBox::indicator:checked {
    background-color: #0d6efd;
    border-color: #0d6efd;
}

QScrollArea#controlScroll {
    border: none;
    background: transparent;
}

QScrollArea#controlScroll > QWidget > QWidget {
    background: transparent;
}
"""

