"""
Main Window for DLP4710 Focus Assistant.
"""

from typing import Optional
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from DLP4710_Focus_Assistant.core.focus_controller import FocusController
from DLP4710_Focus_Assistant.core.workers import FocusStreamWorker
from DLP4710_Focus_Assistant.ui.styles import FOCUS_STYLE
from DLP4710_Focus_Assistant.ui.widgets import (
    FocusCurveWidget,
    FocusStatusCard,
    InteractiveVideoCanvas,
)
from dlp_common.focus.focus_metric import FocusMetricType
from dlp_common.focus.focus_tracker import FocusState


class FocusMainWindow(QMainWindow):
    """
    Main user interface for DLP4710 Focus Assistant.
    """

    def __init__(self, controller: Optional[FocusController] = None):
        super().__init__()
        self.controller = controller if controller is not None else FocusController()
        self.worker: Optional[FocusStreamWorker] = None

        self.setWindowTitle("DLP4710 结构光微距对焦助手 (Focus Assistant) - 景深3mm辅助系统")
        self.resize(1320, 840)
        self.setStyleSheet(FOCUS_STYLE)

        self._setup_ui()
        self._refresh_camera_list()
        self._connect_signals()

        # Start stream worker immediately (auto-starts in simulation or real hardware)
        self._start_preview()

    def _setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ----------------- Left Control Panel -----------------
        scroll_area = QScrollArea()
        scroll_area.setObjectName("controlScroll")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(340)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(10)

        # 1. Camera Control Group
        grp_cam = QGroupBox("相机连接与采集参数")
        cam_layout = QVBoxLayout(grp_cam)
        cam_layout.setSpacing(6)

        cam_row1 = QHBoxLayout()
        self.combo_cameras = QComboBox()
        self.btn_refresh_cams = QPushButton("刷新")
        self.btn_refresh_cams.setFixedWidth(50)
        cam_row1.addWidget(self.combo_cameras, 1)
        cam_row1.addWidget(self.btn_refresh_cams)
        cam_layout.addLayout(cam_row1)

        self.btn_toggle_cam = QPushButton("重新连接相机")
        self.btn_toggle_cam.setObjectName("primaryBtn")
        cam_layout.addWidget(self.btn_toggle_cam)

        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("曝光 (μs):"))
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(500.0, 1000000.0)
        self.spin_exposure.setSingleStep(5000.0)
        self.spin_exposure.setValue(self.controller.config.default_exposure_us)
        exp_layout.addWidget(self.spin_exposure)
        cam_layout.addLayout(exp_layout)

        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("增益 (dB):"))
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 24.0)
        self.spin_gain.setSingleStep(0.5)
        self.spin_gain.setValue(self.controller.config.default_gain_db)
        gain_layout.addWidget(self.spin_gain)
        cam_layout.addLayout(gain_layout)

        left_layout.addWidget(grp_cam)

        # 2. Focus Algorithm Group
        grp_algo = QGroupBox("对焦评价与景深容限配置")
        algo_layout = QVBoxLayout(grp_algo)
        algo_layout.setSpacing(6)

        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("清晰度算法:"))
        self.combo_metric = QComboBox()
        for m in FocusMetricType:
            self.combo_metric.addItem(m.value, m)
        algo_row.addWidget(self.combo_metric)
        algo_layout.addLayout(algo_row)

        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("景深容限阈值:"))
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(50.0, 99.0)
        self.spin_ratio.setSingleStep(1.0)
        self.spin_ratio.setValue(92.0)
        self.spin_ratio.setSuffix(" %")
        ratio_row.addWidget(self.spin_ratio)
        algo_layout.addLayout(ratio_row)

        self.chk_peaking = QCheckBox("启用峰值边缘染色 (Focus Peaking)")
        self.chk_peaking.setChecked(True)
        algo_layout.addWidget(self.chk_peaking)

        peaking_th_row = QHBoxLayout()
        peaking_th_row.addWidget(QLabel("边缘敏感度:"))
        self.spin_peaking_th = QDoubleSpinBox()
        self.spin_peaking_th.setRange(5.0, 150.0)
        self.spin_peaking_th.setSingleStep(5.0)
        self.spin_peaking_th.setValue(35.0)
        peaking_th_row.addWidget(self.spin_peaking_th)
        algo_layout.addLayout(peaking_th_row)

        self.chk_sound = QCheckBox("进入最佳景深时蜂鸣提示")
        self.chk_sound.setChecked(True)
        algo_layout.addWidget(self.chk_sound)

        left_layout.addWidget(grp_algo)

        # 3. Target ROI Group
        grp_roi = QGroupBox("局部特征区域 (ROI)")
        roi_layout = QVBoxLayout(grp_roi)
        roi_layout.setSpacing(6)

        self.label_roi_info = QLabel("当前范围: 全画面")
        self.label_roi_info.setStyleSheet("color: #58a6ff;")
        roi_layout.addWidget(self.label_roi_info)

        self.btn_clear_roi = QPushButton("清除 ROI (恢复全图)")
        roi_layout.addWidget(self.btn_clear_roi)

        tip_roi = QLabel("提示：在右侧相机画面按住鼠标左键拖拽即可绘制对焦框；右键直接清除。")
        tip_roi.setWordWrap(True)
        tip_roi.setStyleSheet("color: #8da4c4; font-size: 11px;")
        roi_layout.addWidget(tip_roi)

        left_layout.addWidget(grp_roi)

        # 4. Action Group
        grp_action = QGroupBox("对焦操作控制")
        act_layout = QVBoxLayout(grp_action)
        act_layout.setSpacing(8)

        self.btn_reset_peak = QPushButton("重新寻焦 / 重置历史峰值")
        self.btn_reset_peak.setObjectName("successBtn")
        self.btn_reset_peak.setMinimumHeight(32)
        act_layout.addWidget(self.btn_reset_peak)

        guide_card = QFrame()
        guide_card.setStyleSheet(
            "background-color: #1e1e28; border: 1px solid #38384a; border-radius: 4px; padding: 6px;"
        )
        guide_layout = QVBoxLayout(guide_card)
        guide_title = QLabel("微距上下移动对焦指引:")
        guide_title.setStyleSheet("font-weight: bold; color: #ffd43b;")
        guide_text = QLabel(
            "1. 慢速单向上下移动 Z 轴升降台；\n"
            "2. 观察右下方波形，清晰度持续上升为接近中；\n"
            "3. 当状态卡片变为绿色并发出提示音时，即处于最佳 3mm 景深中心；\n"
            "4. 若变为橙色提示“已越过”，请微调反向回退到波峰位置。"
        )
        guide_text.setStyleSheet("color: #c0c0d0; font-size: 11px;")
        guide_layout.addWidget(guide_title)
        guide_layout.addWidget(guide_text)
        act_layout.addWidget(guide_card)

        left_layout.addWidget(grp_action)
        left_layout.addStretch(1)

        scroll_area.setWidget(left_container)
        main_layout.addWidget(scroll_area)

        # ----------------- Right Display Splitter -----------------
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Video Canvas
        self.canvas = InteractiveVideoCanvas()
        splitter.addWidget(self.canvas)

        # Bottom: Status & Waveform Curve
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        # Left: Large Status Badge
        self.card_status = FocusStatusCard()
        self.card_status.setFixedWidth(360)
        bottom_layout.addWidget(self.card_status)

        # Right: Real-time curve
        self.curve_widget = FocusCurveWidget()
        bottom_layout.addWidget(self.curve_widget, 1)

        splitter.addWidget(bottom_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.label_fps = QLabel("FPS: 0.0")
        self.label_fps.setStyleSheet("padding-right: 12px; color: #58a6ff; font-weight: bold;")
        self.status_bar.addPermanentWidget(self.label_fps)
        self.status_bar.showMessage("对焦助手就绪")

    def _connect_signals(self):
        self.btn_refresh_cams.clicked.connect(self._refresh_camera_list)
        self.btn_toggle_cam.clicked.connect(self._on_toggle_camera)
        self.spin_exposure.valueChanged.connect(self._on_exposure_changed)
        self.spin_gain.valueChanged.connect(self._on_gain_changed)

        self.combo_metric.currentIndexChanged.connect(self._on_metric_changed)
        self.spin_ratio.valueChanged.connect(self._on_ratio_changed)
        self.chk_peaking.toggled.connect(self._on_peaking_toggled)
        self.spin_peaking_th.valueChanged.connect(self._on_peaking_th_changed)
        self.chk_sound.toggled.connect(self._on_sound_toggled)

        self.canvas.roi_selected.connect(self._on_roi_selected)
        self.canvas.roi_cleared.connect(self._on_roi_cleared)
        self.btn_clear_roi.clicked.connect(self.canvas.clear_roi)

        self.btn_reset_peak.clicked.connect(self._on_reset_peak)

    def _refresh_camera_list(self):
        self.combo_cameras.clear()
        devs = self.controller.detect_cameras()
        if devs:
            for d in devs:
                self.combo_cameras.addItem(d["name"], d.get("index", 0))
        else:
            self.combo_cameras.addItem("未检测到大恒物理相机 (自动启用模拟测试源)", -1)

    def _start_preview(self):
        if self.worker is not None:
            self.worker.stop()

        idx = self.combo_cameras.currentData()
        if idx is None or idx < 0:
            idx = 0
        self.controller.connect_camera(idx)

        self.worker = FocusStreamWorker(self.controller)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.fps_updated.connect(self._on_fps_updated)
        self.worker.start()

    def _on_toggle_camera(self):
        self.controller.disconnect_camera()
        self._start_preview()
        self.status_bar.showMessage("已重新启动相机采集流", 3000)

    def _on_exposure_changed(self, val: float):
        self.controller.set_exposure(val)

    def _on_gain_changed(self, val: float):
        self.controller.set_gain(val)

    def _on_metric_changed(self, index: int):
        metric = self.combo_metric.currentData()
        if metric is not None:
            self.controller.metric_type = metric
            self.controller.reset_peak()
            self.status_bar.showMessage(f"已切换评价算法: {metric.value}", 3000)

    def _on_ratio_changed(self, val: float):
        self.controller.tracker.in_focus_ratio = val / 100.0
        self.curve_widget.in_focus_ratio = val / 100.0

    def _on_peaking_toggled(self, checked: bool):
        self.controller.peaking_enabled = checked

    def _on_peaking_th_changed(self, val: float):
        self.controller.peaking_threshold = float(val)

    def _on_sound_toggled(self, checked: bool):
        self.controller.sound_enabled = checked

    def _on_roi_selected(self, x: int, y: int, w: int, h: int):
        self.controller.set_roi(x, y, w, h)
        self.label_roi_info.setText(f"已框选: {w}x{h} @ ({x}, {y})")
        self.status_bar.showMessage(f"已锁定局部对焦区域: {w}x{h}", 3000)

    def _on_roi_cleared(self):
        self.controller.clear_roi()
        self.label_roi_info.setText("当前范围: 全画面")
        self.status_bar.showMessage("已清除 ROI，恢复全画面对焦", 3000)

    def _on_reset_peak(self):
        self.controller.reset_peak()
        self.status_bar.showMessage("已重置对焦历史峰值", 3000)

    def _on_frame_ready(
        self,
        frame_bgr: np.ndarray,
        score: float,
        state_str: str,
        ratio: float,
        peak: float,
        history: list,
    ):
        self.canvas.set_frame(frame_bgr)
        self.card_status.set_status(state_str, ratio, score, peak)
        self.curve_widget.update_curve(history, peak, self.controller.tracker.in_focus_ratio)

    def _on_fps_updated(self, fps: float):
        self.label_fps.setText(f"FPS: {fps:.1f}")

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.controller.disconnect_camera()
        event.accept()

