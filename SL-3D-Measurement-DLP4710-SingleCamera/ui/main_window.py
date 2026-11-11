"""
Main Application Window for DLP4710 Mono Structured Light 3D Measurement.
Built with PyQt6 and PyQtGraph OpenGL.
"""

from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dlp_common.reconstruction.cuda_reconstructor import (
    get_cuda_device_name,
    get_cuda_info,
    is_cuda_available,
)

from ..core.controller import MeasurementController
from ..core.workers import (
    BurnMeasurementBinWorker,
    CaptureMeasurementWorker,
    LivePreviewWorker,
    MeasureWorker,
    MultiExposureCaptureWorker,
    MultiExposureMeasureWorker,
)
from .gl_view import GLPointCloudWidget
from .styles import MEASUREMENT_STYLE
from .widgets import MultiImageViewerWidget, ScaledImageLabel


class MeasurementMainWindow(QMainWindow):
    """
    Main application window for 3D measurement and point cloud visualization.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("DLP4710 单目结构光三维测量软件 (Python 工业工程版)")
        self.resize(1600, 960)
        self.setMinimumSize(1180, 720)

        self.controller = MeasurementController()
        self.preview_worker: Optional[LivePreviewWorker] = None
        self.capture_worker: Optional[CaptureMeasurementWorker] = None
        self.burn_worker: Optional[BurnMeasurementBinWorker] = None
        self.measure_worker: Optional[MeasureWorker] = None
        self.multi_capture_worker: Optional[MultiExposureCaptureWorker] = None
        self.multi_measure_worker: Optional[MultiExposureMeasureWorker] = None

        self._init_ui()
        self.setStyleSheet(MEASUREMENT_STYLE)
        self.log("三维测量系统已就绪，运行环境: Conda dlp (Python 3.13)")

        # Auto-load template calibration if present
        default_calib = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "bin_and_templates",
                "MonoSLCalibration.yml",
            )
        )
        if os.path.isfile(default_calib):
            self._load_calib_file(default_calib)

    def _create_scrollable_container(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("controlScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left Container: Modern Tabbed Control Panel + Fixed Action Card
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_container.setMinimumWidth(360)
        left_container.setMaximumWidth(420)

        self.control_tabs = QTabWidget()
        self.control_tabs.setObjectName("controlTabs")
        self.control_tabs.addTab(self._build_hardware_tab(), "硬件标定")
        self.control_tabs.addTab(self._build_measurement_tab(), "测量采集")
        self.control_tabs.addTab(self._build_reconstruction_tab(), "重构加速")
        self.control_tabs.addTab(self._build_export_tab(), "结果导出")
        left_layout.addWidget(self.control_tabs, 1)

        # Fixed bottom progress & status card on left panel
        card_progress = QWidget()
        card_lay = QVBoxLayout(card_progress)
        card_lay.setContentsMargins(4, 2, 4, 4)
        card_lay.setSpacing(4)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        card_lay.addWidget(self.progress_bar)

        self.lbl_status = QLabel("系统就绪")
        self.lbl_status.setStyleSheet("color: #a0a0ba; font-size: 11px;")
        card_lay.addWidget(self.lbl_status)
        left_layout.addWidget(card_progress)

        main_layout.addWidget(left_container)

        # Right display area (Tabs: 3D Point Cloud, 2D Process Images)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.tab_widget = QTabWidget()
        self.view_3d = GLPointCloudWidget()
        self.view_2d = MultiImageViewerWidget()

        self.tab_widget.addTab(self.view_3d, "三维点云")
        self.tab_widget.addTab(self.view_2d, "二维过程图")
        right_layout.addWidget(self.tab_widget, 7)

        # Bottom stats and log splitter
        bot_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Statistics box
        self.text_stats = QTextEdit()
        self.text_stats.setReadOnly(True)
        self.text_stats.setPlaceholderText("点云三维度量与几何统计...")
        bot_splitter.addWidget(self.text_stats)

        # System log
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setPlaceholderText("系统运行日志...")
        bot_splitter.addWidget(self.text_log)

        bot_splitter.setStretchFactor(0, 5)
        bot_splitter.setStretchFactor(1, 5)
        right_layout.addWidget(bot_splitter, 3)

        main_layout.addWidget(right_container, 1)

        # Bottom application status bar with status badges
        self.status_bar = self.statusBar()
        self.status_bar.setObjectName("statusBar")
        self.lbl_pill_gpu = QLabel()
        self.lbl_pill_gpu.setObjectName("pillGpu")
        self.lbl_pill_camera = QLabel()
        self.lbl_pill_camera.setObjectName("pillCamera")
        self.lbl_pill_dlp = QLabel()
        self.lbl_pill_dlp.setObjectName("pillDlp")
        self.lbl_pill_calib = QLabel()
        self.lbl_pill_calib.setObjectName("pillCalib")
        self.status_bar.addPermanentWidget(self.lbl_pill_gpu)
        self.status_bar.addPermanentWidget(self.lbl_pill_camera)
        self.status_bar.addPermanentWidget(self.lbl_pill_dlp)
        self.status_bar.addPermanentWidget(self.lbl_pill_calib)

        # Connect signals and update badges
        self._connect_signals()
        self._update_status_bar_badges()

    def _build_hardware_tab(self) -> QWidget:
        tab_w = QWidget()
        lay = QVBoxLayout(tab_w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        # 1. Calibration File Group
        grp_calib = QGroupBox("1. 系统标定参数")
        lay_cal = QGridLayout(grp_calib)
        self.edit_calib_path = QLineEdit()
        self.edit_calib_path.setPlaceholderText("选择 MonoSLCalibration.yml...")
        self.btn_browse_calib = QPushButton("浏览...")
        self.btn_load_calib = QPushButton("加载参数")
        lay_cal.addWidget(self.edit_calib_path, 0, 0, 1, 2)
        lay_cal.addWidget(self.btn_browse_calib, 0, 2)
        lay_cal.addWidget(self.btn_load_calib, 1, 0, 1, 3)

        self.lbl_calib_status = QLabel("标定状态: 未加载")
        self.lbl_calib_status.setStyleSheet("color: #ff9966; font-size: 12px;")
        lay_cal.addWidget(self.lbl_calib_status, 2, 0, 1, 3)
        lay.addWidget(grp_calib)

        # 2. Devices Group
        grp_devices = QGroupBox("2. 硬件连接管理")
        lay_dev = QGridLayout(grp_devices)
        self.btn_detect_devices = QPushButton("检测大恒相机与DLP")
        self.btn_disconnect_devices = QPushButton("断开设备")
        lay_dev.addWidget(self.btn_detect_devices, 0, 0, 1, 2)
        lay_dev.addWidget(self.btn_disconnect_devices, 0, 2)

        self.lbl_device_status = QLabel("设备状态: 未检测")
        self.lbl_device_status.setWordWrap(True)
        lay_dev.addWidget(self.lbl_device_status, 1, 0, 1, 3)
        lay.addWidget(grp_devices)

        # 3. DLP Pattern BIN Group
        grp_bin = QGroupBox("3. DLP测量条纹固件")
        lay_bin = QGridLayout(grp_bin)
        self.edit_bin_path = QLineEdit()
        self.edit_bin_path.setPlaceholderText("选择12帧水平条纹BIN...")
        default_bin = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "bin_and_templates",
                "mySL3x4_hor.bin",
            )
        )
        if os.path.isfile(default_bin):
            self.edit_bin_path.setText(default_bin)
        self.btn_browse_bin = QPushButton("选择BIN")
        self.btn_burn_bin = QPushButton("烧录测量条纹BIN")
        self.btn_gen_color_bin = QPushButton("生成对应颜色条纹BIN")
        lay_bin.addWidget(self.edit_bin_path, 0, 0, 1, 2)
        lay_bin.addWidget(self.btn_browse_bin, 0, 2)
        lay_bin.addWidget(self.btn_burn_bin, 1, 0, 1, 2)
        lay_bin.addWidget(self.btn_gen_color_bin, 1, 2)
        lay.addWidget(grp_bin)

        # 4. Camera & DLP Parameter Settings
        grp_camera = QGroupBox("4. 相机与 DLP 投影参数设置 (光强/颜色/读取)")
        lay_cam = QGridLayout(grp_camera)
        lay_cam.addWidget(QLabel("基准灰度曝光 (μs):"), 0, 0)
        self.spin_gray_exposure = QDoubleSpinBox()
        self.spin_gray_exposure.setRange(100.0, 1000000.0)
        self.spin_gray_exposure.setSingleStep(1000.0)
        self.spin_gray_exposure.setValue(getattr(self.controller.config, "board_exposure_us", 150000.0))
        lay_cam.addWidget(self.spin_gray_exposure, 0, 1)

        lay_cam.addWidget(QLabel("条纹投影曝光 (μs):"), 1, 0)
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(100.0, 1000000.0)
        self.spin_exposure.setSingleStep(500.0)
        pattern_default = getattr(self.controller.config, "calibration_pattern_exposure_us", 3000.0)
        self.spin_exposure.setValue(pattern_default)
        lay_cam.addWidget(self.spin_exposure, 1, 1)

        lay_cam.addWidget(QLabel("增益 (dB):"), 2, 0)
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 24.0)
        self.spin_gain.setValue(self.controller.config.default_gain_db)
        lay_cam.addWidget(self.spin_gain, 2, 1)

        lay_cam.addWidget(QLabel("投影仪光强 (%):"), 3, 0)
        bright_box = QWidget()
        bright_layout = QHBoxLayout(bright_box)
        bright_layout.setContentsMargins(0, 0, 0, 0)
        bright_layout.setSpacing(4)
        self.slider_projector_brightness = QSlider(Qt.Orientation.Horizontal)
        self.slider_projector_brightness.setRange(10, 100)
        self.slider_projector_brightness.setValue(100)
        self.slider_projector_brightness.setSingleStep(5)
        self.spin_projector_brightness = QSpinBox()
        self.spin_projector_brightness.setRange(10, 100)
        self.spin_projector_brightness.setValue(100)
        self.spin_projector_brightness.setSuffix(" %")
        bright_layout.addWidget(self.slider_projector_brightness, 1)
        bright_layout.addWidget(self.spin_projector_brightness)
        lay_cam.addWidget(bright_box, 3, 1)

        lay_cam.addWidget(QLabel("投影仪颜色模式:"), 4, 0)
        self.combo_dlp_color = QComboBox()
        self.combo_dlp_color.addItem("💙 蓝光 (Blue - 推荐结构光)", "blue")
        self.combo_dlp_color.addItem("💚 绿光 (Green)", "green")
        self.combo_dlp_color.addItem("❤️ 红光 (Red)", "red")
        self.combo_dlp_color.addItem("🤍 白光 (White - RGB全亮)", "white")
        self.combo_dlp_color.addItem("⚙️ 自定义 RGB 通道", "custom")
        lay_cam.addWidget(self.combo_dlp_color, 4, 1)

        lay_cam.addWidget(QLabel("物理通道开关:"), 5, 0)
        channel_box = QWidget()
        channel_layout = QHBoxLayout(channel_box)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.setSpacing(6)
        self.chk_led_r = QCheckBox("R (红)")
        self.chk_led_g = QCheckBox("G (绿)")
        self.chk_led_b = QCheckBox("B (蓝)")
        self.chk_led_r.setChecked(False)
        self.chk_led_g.setChecked(False)
        self.chk_led_b.setChecked(True)
        self.chk_led_r.setEnabled(False)
        self.chk_led_g.setEnabled(False)
        self.chk_led_b.setEnabled(False)
        channel_layout.addWidget(self.chk_led_r)
        channel_layout.addWidget(self.chk_led_g)
        channel_layout.addWidget(self.chk_led_b)
        lay_cam.addWidget(channel_box, 5, 1)

        self.lbl_brightness_status = QLabel("LED编码: R:0 G:0 B:1000 / 1023 (100% 蓝光)")
        self.lbl_brightness_status.setStyleSheet("color: #8888aa; font-size: 11px;")
        lay_cam.addWidget(self.lbl_brightness_status, 6, 0, 1, 2)

        dlp_action_box = QWidget()
        dlp_action_layout = QHBoxLayout(dlp_action_box)
        dlp_action_layout.setContentsMargins(0, 0, 0, 0)
        dlp_action_layout.setSpacing(6)
        self.btn_read_dlp_params = QPushButton("🔄 读取当前 DLP 参数")
        self.btn_apply_dlp_params = QPushButton("⚡ 应用颜色与光强")
        dlp_action_layout.addWidget(self.btn_read_dlp_params)
        dlp_action_layout.addWidget(self.btn_apply_dlp_params)
        lay_cam.addWidget(dlp_action_box, 7, 0, 1, 2)

        self.btn_toggle_preview = QPushButton("开启实时预览")
        lay_cam.addWidget(self.btn_toggle_preview, 8, 0, 1, 2)
        lay.addWidget(grp_camera)

        lay.addStretch()
        return self._create_scrollable_container(tab_w)

    def _build_measurement_tab(self) -> QWidget:
        tab_w = QWidget()
        lay = QVBoxLayout(tab_w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        # 1. Single-exposure standard measurement
        grp_single = QGroupBox("1. 标准单曝光测量 (12帧)")
        lay_s = QVBoxLayout(grp_single)
        lay_s.setSpacing(6)
        self.btn_measure_online = QPushButton("单曝光采集并测量 (12帧)")
        self.btn_measure_online.setObjectName("measureBtn")
        self.btn_measure_offline = QPushButton("导入离线单曝光12帧并测量")
        lay_s.addWidget(self.btn_measure_online)
        lay_s.addWidget(self.btn_measure_offline)
        lbl_s_tip = QLabel("适合均匀漫反射表面，单次12帧高速重构")
        lbl_s_tip.setStyleSheet("color: #7a7a92; font-size: 11px;")
        lay_s.addWidget(lbl_s_tip)
        lay.addWidget(grp_single)

        # 2. Multi-exposure HDR measurement
        grp_hdr = QGroupBox("2. 多曝光 HDR 测量 (抗高反光/暗场)")
        lay_hdr = QVBoxLayout(grp_hdr)
        lay_hdr.setSpacing(4)

        row_preset = QHBoxLayout()
        row_preset.addWidget(QLabel("方案预设:"))
        self.combo_hdr_preset = QComboBox()
        self.combo_hdr_preset.addItems([
            "预设: 快速 3 档 (30%, 70%, 100%)",
            "预设: 快速 5 档梯度 (20% ~ 100%)",
            "预设: 论文 10 档光强 (44% ~ 62%)",
            "预设: 论文 13 档高精基准 (13 Exposures GT)",
            "自定义配置",
        ])
        row_preset.addWidget(self.combo_hdr_preset, 1)
        lay_hdr.addLayout(row_preset)

        row_m = QHBoxLayout()
        row_m.addWidget(QLabel("曝光次数 M:"))
        self.spin_hdr_count = QSpinBox()
        self.spin_hdr_count.setRange(2, 20)
        self.spin_hdr_count.setValue(3)
        row_m.addWidget(self.spin_hdr_count)

        self.btn_hdr_gen_gradient = QPushButton("生成等间距梯度")
        row_m.addWidget(self.btn_hdr_gen_gradient)
        lay_hdr.addLayout(row_m)

        row_range = QHBoxLayout()
        row_range.addWidget(QLabel("光强区间:"))
        self.spin_hdr_min_led = QSpinBox()
        self.spin_hdr_min_led.setRange(10, 100)
        self.spin_hdr_min_led.setValue(20)
        self.spin_hdr_min_led.setSuffix("%")
        row_range.addWidget(self.spin_hdr_min_led)
        row_range.addWidget(QLabel("~"))
        self.spin_hdr_max_led = QSpinBox()
        self.spin_hdr_max_led.setRange(10, 100)
        self.spin_hdr_max_led.setValue(100)
        self.spin_hdr_max_led.setSuffix("%")
        row_range.addWidget(self.spin_hdr_max_led)
        lay_hdr.addLayout(row_range)

        self.table_hdr_brackets = QTableWidget()
        self.table_hdr_brackets.setColumnCount(3)
        self.table_hdr_brackets.setHorizontalHeaderLabels(["档位", "光强 (%)", "曝光 (μs)"])
        self.table_hdr_brackets.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_hdr_brackets.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_hdr_brackets.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_hdr_brackets.verticalHeader().setVisible(False)
        self.table_hdr_brackets.setMinimumHeight(115)
        self.table_hdr_brackets.setMaximumHeight(150)
        lay_hdr.addWidget(self.table_hdr_brackets)

        row_tbl_btns = QHBoxLayout()
        self.btn_hdr_add_bracket = QPushButton("+ 添加")
        self.btn_hdr_del_bracket = QPushButton("- 删除")
        self.btn_hdr_reset_brackets = QPushButton("重置")
        row_tbl_btns.addWidget(self.btn_hdr_add_bracket)
        row_tbl_btns.addWidget(self.btn_hdr_del_bracket)
        row_tbl_btns.addWidget(self.btn_hdr_reset_brackets)
        lay_hdr.addLayout(row_tbl_btns)

        self.btn_measure_hdr_online = QPushButton("多曝光采集并测量 (HDR)")
        self.btn_measure_hdr_online.setObjectName("hdrMeasureBtn")
        self.btn_measure_hdr_offline = QPushButton("导入离线多曝光并测量")
        lay_hdr.addWidget(self.btn_measure_hdr_online)
        lay_hdr.addWidget(self.btn_measure_hdr_offline)

        # 默认加载 3 档
        self._load_preset_brackets(0)
        lay.addWidget(grp_hdr)

        lay.addStretch()
        return self._create_scrollable_container(tab_w)

    def _build_reconstruction_tab(self) -> QWidget:
        tab_w = QWidget()
        lay = QVBoxLayout(tab_w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        # 1. GPU Acceleration Card
        grp_gpu = QGroupBox("1. NVIDIA GPU 硬件加速 (CUDA)")
        grp_gpu.setObjectName("gpuCard")
        lay_gpu = QVBoxLayout(grp_gpu)
        lay_gpu.setSpacing(5)

        cuda_info = get_cuda_info()
        if cuda_info["available"]:
            self.lbl_gpu_name = QLabel(f"GPU 设备: {cuda_info['device_name']}")
            self.lbl_gpu_spec = QLabel(
                f"显存: {cuda_info['total_memory_gb']} GB | CUDA: {cuda_info['cuda_version']} | 状态: 🟢 已就绪"
            )
        else:
            self.lbl_gpu_name = QLabel("GPU 设备: 未检测到 NVIDIA CUDA 显卡")
            self.lbl_gpu_spec = QLabel("状态: ⚪ 使用 CPU 多线程向量化计算")

        self.lbl_gpu_name.setStyleSheet("font-weight: bold; color: #39d353; font-size: 12px;")
        self.lbl_gpu_spec.setStyleSheet("color: #7ee787; font-size: 11px;")
        lay_gpu.addWidget(self.lbl_gpu_name)
        lay_gpu.addWidget(self.lbl_gpu_spec)

        self.chk_enable_gpu = QCheckBox("启用 NVIDIA GPU 硬件加速 (CUDA 并行求解)")
        self.chk_enable_gpu.setChecked(cuda_info["available"])
        self.controller.reconstructor.enable_gpu = cuda_info["available"]
        if hasattr(self.controller, "phase_processor"):
            self.controller.phase_processor.enable_gpu = cuda_info["available"]
        lay_gpu.addWidget(self.chk_enable_gpu)

        lbl_gpu_desc = QLabel(
            "采用 PyTorch CUDA 全流程硬件加速（条纹高斯滤波、四步相移与三频解包、多曝光HDR融合、远心-Scheimpflug光线相交求解），单帧测量重构极速一气呵成。"
        )
        lbl_gpu_desc.setWordWrap(True)
        lbl_gpu_desc.setStyleSheet("color: #8b949e; font-size: 11px;")
        lay_gpu.addWidget(lbl_gpu_desc)
        lay.addWidget(grp_gpu)

        # 2. Reconstruction Parameters
        grp_params = QGroupBox("2. 重构采样与质量阈值")
        lay_par = QGridLayout(grp_params)
        lay_par.addWidget(QLabel("重构采样步长 (Step):"), 0, 0)
        self.combo_step = QComboBox()
        self.combo_step.addItems(["1 (全分辨率 1:1)", "2 (降采样 1/2)", "4 (平衡 1/4 推荐)", "8 (高速 1/8)"])
        self.combo_step.setCurrentIndex(2)  # default step 4
        lay_par.addWidget(self.combo_step, 0, 1)

        lay_par.addWidget(QLabel("最小调制对比度:"), 1, 0)
        self.spin_mod = QDoubleSpinBox()
        self.spin_mod.setRange(0.0, 100.0)
        self.spin_mod.setValue(5.0)
        lay_par.addWidget(self.spin_mod, 1, 1)
        lay.addWidget(grp_params)

        # 3. Point-cloud Processing
        grp_process = QGroupBox("3. 点云优化与滤波")
        lay_process = QGridLayout(grp_process)
        lay_process.addWidget(QLabel("连接断点半径 (px):"), 0, 0)
        self.spin_subject_radius = QSpinBox()
        self.spin_subject_radius.setRange(0, 100)
        self.spin_subject_radius.setValue(2)
        lay_process.addWidget(self.spin_subject_radius, 0, 1)

        self.btn_keep_subject = QPushButton("去除杂点并保留最大主体")
        self.btn_restore_cloud = QPushButton("恢复原始点云")
        lay_process.addWidget(self.btn_keep_subject, 1, 0, 1, 2)
        lay_process.addWidget(self.btn_restore_cloud, 2, 0, 1, 2)
        lay.addWidget(grp_process)

        lay.addStretch()
        return self._create_scrollable_container(tab_w)

    def _build_export_tab(self) -> QWidget:
        tab_w = QWidget()
        lay = QVBoxLayout(tab_w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        # 1. Export
        grp_export = QGroupBox("1. 测量数据极速导出")
        lay_exp = QVBoxLayout(grp_export)
        lay_exp.setSpacing(6)
        self.btn_save_intermediate = QPushButton("⚡ 极速保存完整测量集 (条纹/灰度/参数/PLY)")
        self.btn_save_intermediate.setObjectName("saveBtn")
        self.btn_save_intermediate.setToolTip("仅保存原始条纹BMP、灰度图、采集参数及高速二进制PLY点云，耗时 < 0.2秒")
        self.btn_save_ply = QPushButton("导出独立二进制 PLY 点云 (<0.1秒)")
        self.btn_save_txt = QPushButton("导出独立 TXT (X Y Z) 坐标")
        self.btn_ply_to_txt = QPushButton("🔄 PLY 点云转 TXT 坐标文件")
        lay_exp.addWidget(self.btn_save_intermediate)
        lay_exp.addWidget(self.btn_save_ply)
        lay_exp.addWidget(self.btn_save_txt)
        lay_exp.addWidget(self.btn_ply_to_txt)
        lay.addWidget(grp_export)

        # 2. Summary
        grp_summary = QGroupBox("2. 快速测量统计")
        lay_sum = QVBoxLayout(grp_summary)
        lay_sum.setSpacing(4)
        self.lbl_summary_mode = QLabel("测量模式: 尚未测量")
        self.lbl_summary_points = QLabel("有效点数: 0 点")
        self.lbl_summary_time = QLabel("算法耗时: 0.00 秒")
        self.lbl_summary_bounds = QLabel("点云跨度: 暂无数据")
        for lbl in (
            self.lbl_summary_mode,
            self.lbl_summary_points,
            self.lbl_summary_time,
            self.lbl_summary_bounds,
        ):
            lbl.setStyleSheet("color: #b0b0c8; font-size: 11px;")
            lay_sum.addWidget(lbl)
        lay.addWidget(grp_summary)

        lay.addStretch()
        return self._create_scrollable_container(tab_w)

    def _connect_signals(self):
        self.btn_browse_calib.clicked.connect(self._on_browse_calib)
        self.btn_load_calib.clicked.connect(self._on_load_calib)
        self.btn_detect_devices.clicked.connect(self._on_detect_devices)
        self.btn_disconnect_devices.clicked.connect(self._on_disconnect_devices)
        self.btn_browse_bin.clicked.connect(self._on_browse_bin)
        self.btn_burn_bin.clicked.connect(self._on_burn_bin)
        self.btn_gen_color_bin.clicked.connect(self._on_gen_color_bin)
        self.btn_toggle_preview.clicked.connect(self._on_toggle_preview)

        self.btn_measure_online.clicked.connect(self._on_measure_online)
        self.btn_measure_offline.clicked.connect(self._on_measure_offline)

        self.slider_projector_brightness.valueChanged.connect(self._on_slider_brightness_changed)
        self.spin_projector_brightness.valueChanged.connect(self._on_spin_brightness_changed)
        self.combo_dlp_color.currentIndexChanged.connect(self._on_dlp_color_changed)
        self.chk_led_r.toggled.connect(self._on_custom_led_toggled)
        self.chk_led_g.toggled.connect(self._on_custom_led_toggled)
        self.chk_led_b.toggled.connect(self._on_custom_led_toggled)
        self.btn_read_dlp_params.clicked.connect(self._on_read_dlp_params)
        self.btn_apply_dlp_params.clicked.connect(self._on_apply_dlp_params)

        self.combo_hdr_preset.currentIndexChanged.connect(self._on_hdr_preset_changed)
        self.btn_hdr_gen_gradient.clicked.connect(self._on_hdr_gen_gradient)
        self.btn_hdr_add_bracket.clicked.connect(self._on_hdr_add_bracket)
        self.btn_hdr_del_bracket.clicked.connect(self._on_hdr_del_bracket)
        self.btn_hdr_reset_brackets.clicked.connect(self._on_hdr_reset_brackets)
        self.btn_measure_hdr_online.clicked.connect(self._on_measure_hdr_online)
        self.btn_measure_hdr_offline.clicked.connect(self._on_measure_hdr_offline)

        self.chk_enable_gpu.toggled.connect(self._on_gpu_toggled)

        self.btn_save_ply.clicked.connect(self._on_save_ply)
        self.btn_save_txt.clicked.connect(self._on_save_txt)
        self.btn_ply_to_txt.clicked.connect(self._on_ply_to_txt)
        self.btn_save_intermediate.clicked.connect(self._on_save_intermediate)
        self.btn_keep_subject.clicked.connect(self._on_keep_subject)
        self.btn_restore_cloud.clicked.connect(self._on_restore_cloud)

    def _update_status_bar_badges(self):
        # 1. GPU Pill
        gpu_enabled = getattr(self, "chk_enable_gpu", None) and self.chk_enable_gpu.isChecked()
        cuda_ok = is_cuda_available()
        if gpu_enabled and cuda_ok:
            dev_name = get_cuda_device_name()
            short_name = "RTX 5070" if "5070" in dev_name else (dev_name[:16] or "CUDA")
            self.lbl_pill_gpu.setText(f"🟢 GPU: {short_name} (CUDA 启用)")
            self.lbl_pill_gpu.setStyleSheet(
                "background-color: #12331d; color: #39d353; border: 1px solid #1f6b3a;"
            )
        elif not gpu_enabled:
            self.lbl_pill_gpu.setText("⚪ GPU: CPU 模式 (未启用)")
            self.lbl_pill_gpu.setStyleSheet(
                "background-color: #242430; color: #888899; border: 1px solid #383848;"
            )
        else:
            self.lbl_pill_gpu.setText("🔴 GPU: CUDA 不可用")
            self.lbl_pill_gpu.setStyleSheet(
                "background-color: #331515; color: #ff6b6b; border: 1px solid #552222;"
            )

        # 2. Camera Pill
        if self.controller.camera.is_opened:
            sn_str = ""
            if hasattr(self.controller.camera, "_device_sn") and self.controller.camera._device_sn:
                sn_str = f" ({self.controller.camera._device_sn})"
            self.lbl_pill_camera.setText(f"📷 相机: 已连接{sn_str}")
            self.lbl_pill_camera.setStyleSheet(
                "background-color: #0e2942; color: #58a6ff; border: 1px solid #1f4e78;"
            )
        else:
            self.lbl_pill_camera.setText("📷 相机: 未连接")
            self.lbl_pill_camera.setStyleSheet(
                "background-color: #262024; color: #888899; border: 1px solid #3d343c;"
            )

        # 3. DLP Pill
        if self.controller.dlp.is_connected:
            color_names = {
                "blue": "蓝光",
                "green": "绿光",
                "red": "红光",
                "white": "白光",
                "custom": "自定义",
            }
            mode = getattr(self.controller.dlp, "color_mode", "blue")
            color_str = color_names.get(mode, "蓝光")
            led_pct = (
                int(round(self.controller.dlp.brightness_percent))
                if hasattr(self.controller.dlp, "brightness_percent")
                else (self.slider_projector_brightness.value() if hasattr(self, "slider_projector_brightness") else 100)
            )
            self.lbl_pill_dlp.setText(f"📽️ DLP: 已连接 ({led_pct}% {color_str})")
            self.lbl_pill_dlp.setStyleSheet(
                "background-color: #241c38; color: #c099ff; border: 1px solid #4d3678;"
            )
        else:
            self.lbl_pill_dlp.setText("📽️ DLP: 未连接")
            self.lbl_pill_dlp.setStyleSheet(
                "background-color: #262024; color: #888899; border: 1px solid #3d343c;"
            )

        # 4. Calibration Pill
        if self.controller.reconstructor.has_calibration_parameters():
            calib_name = (
                os.path.basename(self.controller.calib_file_path)
                if self.controller.calib_file_path
                else "已就绪"
            )
            self.lbl_pill_calib.setText(f"📐 标定: {calib_name}")
            self.lbl_pill_calib.setStyleSheet(
                "background-color: #12331d; color: #39d353; border: 1px solid #1f6b3a;"
            )
        else:
            self.lbl_pill_calib.setText("⚠️ 标定: 未加载")
            self.lbl_pill_calib.setStyleSheet(
                "background-color: #332312; color: #ffaa55; border: 1px solid #664422;"
            )

    def _on_gpu_toggled(self, checked: bool):
        self.controller.reconstructor.enable_gpu = checked
        if hasattr(self.controller, "phase_processor"):
            self.controller.phase_processor.enable_gpu = checked
        self._update_status_bar_badges()
        if checked:
            if is_cuda_available():
                self.log(f"已启用 GPU 全流程硬件加速 (条纹滤波/相位解包/HDR融合/点云重构): {get_cuda_device_name()}")
            else:
                self.log("警告: 尝试启用 GPU 加速，但系统未检测到可用 CUDA 设备")
        else:
            self.log("已禁用 GPU 硬件加速，切换为 CPU 解算模式")

    def log(self, text: str):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.text_log.append(f"[{time_str}] {text}")

    def _on_browse_calib(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标定文件",
            "",
            "YAML Files (*.yml *.yaml);;All Files (*.*)",
        )
        if path:
            self.edit_calib_path.setText(path)
            self._load_calib_file(path)

    def _on_load_calib(self):
        path = self.edit_calib_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "警告", "请先选择有效的标定YAML文件")
            return
        self._load_calib_file(path)

    def _on_detect_devices(self):
        devices = self.controller.camera.detect_camera_devices()
        cam_ok = False
        if devices:
            if not self.controller.camera.is_opened:
                cam_ok = self.controller.camera.open(0)
                if cam_ok:
                    camera_text = f"大恒相机: {devices[0]['name']} (已连接)"
                else:
                    camera_text = f"大恒相机: {devices[0]['name']} (打开失败: {self.controller.camera.last_error})"
            else:
                cam_ok = True
                camera_text = f"大恒相机: {devices[0]['name']} (已连接)"
        else:
            if self.controller.camera.is_opened:
                cam_ok = True
                camera_text = "大恒相机: 已连接"
            else:
                camera_text = "大恒相机: 未检测到 (请检查网线与12V供电)"

        dlp_ok = self.controller.dlp.is_connected or self.controller.dlp.connect()
        dlp_text = "DLP: 已连接" if dlp_ok else f"DLP: {self.controller.dlp.last_error}"
        if dlp_ok and not self.controller.dlp.pattern_ready:
            mode_ok, mode_message = self.controller.dlp.prepare_pattern_mode()
            dlp_text += f"，{mode_message}"
            dlp_ok = mode_ok

        all_ok = cam_ok and dlp_ok
        self.lbl_device_status.setText(f"设备状态: {camera_text}；{dlp_text}")
        self.lbl_device_status.setStyleSheet(
            f"color: {'#28a745' if all_ok else '#ff9966'}; font-size: 11px;"
        )
        self.log(f"设备检测: {camera_text}；{dlp_text}")
        self._update_status_bar_badges()

    def _on_disconnect_devices(self):
        if self.burn_worker is not None and self.burn_worker.isRunning():
            QMessageBox.warning(self, "设备忙", "DLP BIN正在烧录，请等待烧录结束后再断开设备。")
            return
        if self.capture_worker is not None and self.capture_worker.isRunning():
            QMessageBox.warning(self, "设备忙", "12帧条纹正在采集，请等待采集结束后再断开设备。")
            return

        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker.deleteLater()
            self.preview_worker = None
            self.btn_toggle_preview.setText("开启实时预览")

        ok, message = self.controller.disconnect_devices()
        self.log(message)
        if ok:
            self.lbl_device_status.setText("设备状态: 大恒相机已断开；DLP已断开")
            self.lbl_device_status.setStyleSheet("color: #ff9966; font-size: 11px;")
            self.lbl_status.setText("设备已断开")
        else:
            self.lbl_device_status.setText(f"设备状态: 断开失败；{message}")
            self.lbl_device_status.setStyleSheet("color: #dc3545; font-size: 11px;")
            QMessageBox.critical(self, "断开失败", message)
        self._update_status_bar_badges()

    def _on_browse_bin(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择DLP测量条纹BIN",
            self.edit_bin_path.text().strip(),
            "DLP BIN Files (*.bin);;All Files (*.*)",
        )
        if path:
            self.edit_bin_path.setText(path)

    def _on_burn_bin(self):
        bin_path = self.edit_bin_path.text().strip()
        if not bin_path or not os.path.isfile(bin_path):
            QMessageBox.warning(self, "BIN无效", "请选择有效的12帧水平测量条纹BIN文件。")
            return
        if self.burn_worker is not None and self.burn_worker.isRunning():
            return

        self._set_measure_controls_enabled(False)
        self.progress_bar.setRange(0, 0)
        self.lbl_status.setText("正在烧录DLP测量条纹BIN...")
        self.log(f"开始烧录测量BIN: {bin_path}")

        self.burn_worker = BurnMeasurementBinWorker(self.controller.dlp, bin_path)
        self.burn_worker.completed.connect(self._on_burn_finished)
        self.burn_worker.finished.connect(self._on_burn_thread_stopped)
        self.burn_worker.start()

    def _on_burn_finished(self, ok: bool, message: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if ok else 0)
        self.lbl_status.setText("测量BIN烧录完成" if ok else "测量BIN烧录失败")
        self.log(message)
        if ok:
            self.lbl_device_status.setText("设备状态: DLP测量条纹已就绪")
            self.lbl_device_status.setStyleSheet("color: #28a745; font-size: 11px;")
            QMessageBox.information(self, "烧录成功", message)
        else:
            QMessageBox.critical(self, "烧录失败", message)

    def _on_burn_thread_stopped(self):
        worker = self.burn_worker
        self.burn_worker = None
        self.btn_detect_devices.setEnabled(True)
        self.btn_browse_bin.setEnabled(True)
        self.btn_burn_bin.setEnabled(True)
        self._set_measure_controls_enabled(True)
        if worker is not None:
            worker.deleteLater()

    def _load_calib_file(self, path: str):
        self.edit_calib_path.setText(path)
        ok, msg = self.controller.load_calibration(path)
        if ok:
            self.lbl_calib_status.setText(f"标定状态: 已就绪 ({os.path.basename(path)})")
            self.lbl_calib_status.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")
            self.log(msg)
        else:
            self.lbl_calib_status.setText("标定状态: 加载失败")
            self.lbl_calib_status.setStyleSheet("color: #dc3545; font-size: 12px;")
            self.log(f"标定文件错误: {msg}")
            QMessageBox.critical(self, "加载失败", msg)
        self._update_status_bar_badges()

    def _on_toggle_preview(self):
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker = None
            self.btn_toggle_preview.setText("开启实时预览")
            self.log("实时相机预览已停止")
            self.lbl_status.setText("预览已停止")
        else:
            if not self.controller.camera.open(0):
                self._update_status_bar_badges()
                QMessageBox.critical(self, "大恒相机连接失败", self.controller.camera.last_error)
                return
            self._update_status_bar_badges()
            self.controller.camera.set_exposure(self.spin_exposure.value())
            self.controller.camera.set_gain(self.spin_gain.value())
            if not self.controller.camera.start_grabbing():
                QMessageBox.critical(self, "大恒相机采集失败", self.controller.camera.last_error)
                return
            self.preview_worker = LivePreviewWorker(self.controller.camera)
            self.preview_worker.frame_received.connect(self._on_preview_frame)
            self.preview_worker.start()
            self.btn_toggle_preview.setText("停止实时预览")
            self.log("开启实时相机预览...")
            self.lbl_status.setText("实时预览中")

    def _on_preview_frame(self, frame: np.ndarray):
        self.view_2d.update_views(raw_frame=frame)

    def _get_selected_step(self) -> int:
        idx = self.combo_step.currentIndex()
        steps = [1, 2, 4, 8]
        return steps[idx]

    def _on_measure_online(self):
        if not self.controller.reconstructor.has_calibration_parameters():
            QMessageBox.warning(self, "警告", "请先加载有效的标定文件 (MonoSLCalibration.yml)")
            return

        # Stop preview if running
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker = None
            self.btn_toggle_preview.setText("开启实时预览")

        if self.capture_worker is not None and self.capture_worker.isRunning():
            return

        self.lbl_status.setText("正在执行在线同步条纹采集...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.log("启动在线条纹采集 (12帧)...")

        if not self.controller.camera.is_opened:
            if not self.controller.camera.open(0):
                self._update_status_bar_badges()
                self.progress_bar.setValue(0)
                QMessageBox.critical(self, "大恒相机连接失败", self.controller.camera.last_error)
                return
            self._update_status_bar_badges()

        if not self.controller.dlp.is_connected and not self.controller.dlp.connect():
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "DLP连接错误", self.controller.dlp.last_error)
            return
        if not self.controller.dlp.pattern_ready:
            mode_ok, mode_message = self.controller.dlp.prepare_pattern_mode()
            self.log(mode_message)
            if not mode_ok:
                self.progress_bar.setValue(0)
                QMessageBox.critical(self, "DLP模式初始化失败", mode_message)
                return

        self._set_measure_controls_enabled(False)
        self.capture_worker = CaptureMeasurementWorker(
            camera=self.controller.camera,
            dlp=self.controller.dlp,
            exposure_us=self.spin_exposure.value(),
            gain_db=self.spin_gain.value(),
            gray_exposure_us=self.spin_gray_exposure.value(),
        )
        self.capture_worker.progress.connect(self._on_capture_progress)
        self.capture_worker.completed.connect(self._on_capture_completed)
        self.capture_worker.gray_frame_captured.connect(self._on_gray_frame_captured)
        self.capture_worker.finished.connect(self._on_capture_thread_stopped)
        self.capture_worker.start()

    def _on_gray_frame_captured(self, gray_frame: np.ndarray):
        self.view_2d.update_views(raw_frame=gray_frame)
        self.log(f"已采集基准灰度图 (曝光 {int(self.spin_gray_exposure.value())} μs)")

    def _set_measure_controls_enabled(self, enabled: bool):
        for widget in (
            self.btn_toggle_preview,
            self.btn_measure_online,
            self.btn_measure_offline,
            self.btn_measure_hdr_online,
            self.btn_measure_hdr_offline,
            self.spin_gray_exposure,
            self.spin_exposure,
            self.spin_gain,
            self.slider_projector_brightness,
            self.spin_projector_brightness,
            self.combo_dlp_color,
            self.chk_led_r,
            self.chk_led_g,
            self.chk_led_b,
            self.btn_read_dlp_params,
            self.btn_apply_dlp_params,
            self.combo_hdr_preset,
            self.spin_hdr_count,
            self.btn_hdr_gen_gradient,
            self.spin_hdr_min_led,
            self.spin_hdr_max_led,
            self.table_hdr_brackets,
            self.btn_hdr_add_bracket,
            self.btn_hdr_del_bracket,
            self.btn_hdr_reset_brackets,
            self.btn_detect_devices,
            self.btn_disconnect_devices,
            self.btn_browse_bin,
            self.btn_burn_bin,
            self.btn_gen_color_bin,
            self.edit_bin_path,
            self.chk_enable_gpu,
            self.combo_step,
            self.spin_mod,
            self.btn_ply_to_txt,
        ):
            widget.setEnabled(enabled)

    def _on_capture_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.lbl_status.setText(message)
        self.log(message)

    def _on_capture_completed(self, ok: bool, frames, message: str):
        self.log(message)
        if not ok or len(frames) != 12:
            self.progress_bar.setValue(0)
            self.lbl_status.setText("在线采集失败")
            self._set_measure_controls_enabled(True)
            QMessageBox.critical(self, "采集错误", message)
            return
        gray_frame = getattr(self.capture_worker, "gray_frame", None)
        self._start_reconstruction(frames, gray_frame=gray_frame)

    def _on_capture_thread_stopped(self):
        worker = self.capture_worker
        self.capture_worker = None
        if worker is not None:
            worker.deleteLater()

    def _on_measure_offline(self):
        if not self.controller.reconstructor.has_calibration_parameters():
            QMessageBox.warning(self, "警告", "请先加载有效的标定文件 (MonoSLCalibration.yml)")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择包含12张水平条纹图像的目录")
        if not folder:
            return

        ok, frames, msg = self.controller.load_offline_frames(folder)
        if not ok or len(frames) != 12:
            QMessageBox.warning(self, "加载失败", msg)
            return

        self.log(msg)
        self._start_reconstruction(frames)

    def _start_reconstruction(
        self,
        frames: List[np.ndarray],
        gray_frame: Optional[np.ndarray] = None,
    ):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(10)
        self.lbl_status.setText("开始三维点云重构计算...")

        step = self._get_selected_step()
        min_mod = self.spin_mod.value()

        self.measure_worker = MeasureWorker(
            reconstructor=self.controller.reconstructor,
            processor=self.controller.phase_processor,
            frames=frames,
            sample_step=step,
            min_modulation=min_mod,
            gray_frame=gray_frame,
        )
        self.measure_worker.progress.connect(lambda p, m: (
            self.progress_bar.setValue(p),
            self.lbl_status.setText(m),
            self.log(m),
        ))
        self.measure_worker.finished.connect(self._on_measure_finished)
        self.measure_worker.start()

    def _on_measure_finished(self, ok: bool, result: dict, msg: str):
        self._set_measure_controls_enabled(True)
        self.progress_bar.setValue(100 if ok else 0)
        self.lbl_status.setText("测量完成" if ok else "重构失败")
        self.log(msg)

        if not ok:
            QMessageBox.critical(self, "测量错误", msg)
            return

        self.controller.latest_result = result

        # 1. Update 2D Process Views
        is_hdr = bool(result.get("is_multi_exposure", False))
        fused_frames = result.get("fused_frames")
        fused_frame = fused_frames[0] if fused_frames else None
        exp_indices = result.get("exposure_indices")
        num_brackets = len(result.get("brackets_frames", [])) if is_hdr else 1

        self.view_2d.update_views(
            raw_frame=result.get("raw_frame"),
            wrapped_phase=result.get("wrapped_phase"),
            unwrapped_phase=result.get("unwrapped_phase"),
            mask=result.get("mask"),
            fused_frame=fused_frame,
            exposure_indices=exp_indices,
            num_brackets=num_brackets,
        )

        # 2. Update 3D Point Cloud View
        pts = result["valid_points"]
        self.view_3d.set_point_cloud(pts)

        # 3. Update Summary in Tab 4
        mode_name = "多曝光 (HDR) 融合测量" if is_hdr else "单曝光标准测量"
        self.lbl_summary_mode.setText(f"测量模式: {mode_name}")
        self.lbl_summary_points.setText(f"有效点数: {len(pts):,} 点")
        self.lbl_summary_time.setText(f"算法耗时: {result['elapsed_seconds']:.2f} 秒")
        if len(pts) > 0:
            min_pt = np.min(pts, axis=0)
            max_pt = np.max(pts, axis=0)
            extent = max_pt - min_pt
            self.lbl_summary_bounds.setText(f"X:{extent[0]:.1f} Y:{extent[1]:.1f} Z:{extent[2]:.1f} mm")

        self._update_status_bar_badges()

        # 4. Update Statistics Report
        if len(pts) > 0:
            min_pt = np.min(pts, axis=0)
            max_pt = np.max(pts, axis=0)
            extent = max_pt - min_pt
            mean_pt = np.mean(pts, axis=0)

            hdr_section = ""
            if result.get("is_multi_exposure") and "bracket_counts" in result:
                b_counts = result["bracket_counts"]
                tot_p = sum(b_counts.values()) or 1
                lines = ["[多曝光 HDR 像素级融合分布 (全幅像素选档统计)]"]
                for b_i, cnt in b_counts.items():
                    pct = (cnt / tot_p) * 100.0
                    lines.append(f"  - 档位 {b_i + 1}: {cnt:,} 像素 ({pct:.1f}%)")
                hdr_section = "\n".join(lines) + "\n\n"

            step_val = self._get_selected_step()
            step_str = f" (重构步长 Step={step_val})" if step_val > 1 else " (全点云 1:1)"
            mode_name = "多曝光 (HDR) 融合测量" if result.get("is_multi_exposure") else "单曝光标准测量"
            stats = (
                "================ 三维测量统计报告 ================\n"
                f"测量模式: {mode_name}\n"
                f"有效重构点数: {len(pts):,}{step_str}\n"
                f"算法总耗时: {result['elapsed_seconds']:.2f} 秒\n\n"
                f"{hdr_section}"
                f"X 坐标范围: [{min_pt[0]:.2f}, {max_pt[0]:.2f}] mm (跨度: {extent[0]:.2f} mm)\n"
                f"Y 坐标范围: [{min_pt[1]:.2f}, {max_pt[1]:.2f}] mm (跨度: {extent[1]:.2f} mm)\n"
                f"Z 深度范围: [{min_pt[2]:.2f}, {max_pt[2]:.2f}] mm (跨度: {extent[2]:.2f} mm)\n\n"
                f"点云质心坐标: ({mean_pt[0]:.2f}, {mean_pt[1]:.2f}, {mean_pt[2]:.2f}) mm\n"
            )
            self.text_stats.setText(stats)

        # Switch to 3D View tab
        self.tab_widget.setCurrentIndex(0)
        QMessageBox.information(self, "测量成功", f"三维重构顺利完成！\n提取点数: {len(pts):,}\n耗时: {result['elapsed_seconds']:.2f} s")

    def _on_slider_brightness_changed(self, value: int):
        self.spin_projector_brightness.blockSignals(True)
        self.spin_projector_brightness.setValue(value)
        self.spin_projector_brightness.blockSignals(False)
        self._apply_projector_brightness(value)

    def _on_spin_brightness_changed(self, value: int):
        self.slider_projector_brightness.blockSignals(True)
        self.slider_projector_brightness.setValue(value)
        self.slider_projector_brightness.blockSignals(False)
        self._apply_projector_brightness(value)

    def _apply_projector_brightness(self, percent: int):
        ok, msg = self.controller.dlp.set_brightness_percent(percent)
        self._update_brightness_status_label()
        if self.controller.dlp.is_connected:
            self.log(msg)
        self._update_status_bar_badges()

    def _on_dlp_color_changed(self, index: int):
        mode = self.combo_dlp_color.currentData() or "blue"
        if mode == "blue":
            self.chk_led_r.setChecked(False)
            self.chk_led_g.setChecked(False)
            self.chk_led_b.setChecked(True)
            self.chk_led_r.setEnabled(False)
            self.chk_led_g.setEnabled(False)
            self.chk_led_b.setEnabled(False)
        elif mode == "green":
            self.chk_led_r.setChecked(False)
            self.chk_led_g.setChecked(True)
            self.chk_led_b.setChecked(False)
            self.chk_led_r.setEnabled(False)
            self.chk_led_g.setEnabled(False)
            self.chk_led_b.setEnabled(False)
        elif mode == "red":
            self.chk_led_r.setChecked(True)
            self.chk_led_g.setChecked(False)
            self.chk_led_b.setChecked(False)
            self.chk_led_r.setEnabled(False)
            self.chk_led_g.setEnabled(False)
            self.chk_led_b.setEnabled(False)
        elif mode == "white":
            self.chk_led_r.setChecked(True)
            self.chk_led_g.setChecked(True)
            self.chk_led_b.setChecked(True)
            self.chk_led_r.setEnabled(False)
            self.chk_led_g.setEnabled(False)
            self.chk_led_b.setEnabled(False)
        elif mode == "custom":
            self.chk_led_r.setEnabled(True)
            self.chk_led_g.setEnabled(True)
            self.chk_led_b.setEnabled(True)
        self._update_brightness_status_label()

    def _on_custom_led_toggled(self, checked: bool = False):
        if self.combo_dlp_color.currentData() == "custom":
            self._update_brightness_status_label()

    def _update_brightness_status_label(self):
        pct = self.spin_projector_brightness.value()
        mode = self.combo_dlp_color.currentData() or "blue"
        r_en = self.chk_led_r.isChecked()
        g_en = self.chk_led_g.isChecked()
        b_en = self.chk_led_b.isChecked()
        code = int(round(51 + (pct / 100.0) * (1000 - 51))) if pct > 0 else 0
        r_code = code if r_en else 0
        g_code = code if g_en else 0
        b_code = code if b_en else 0
        mode_names = {
            "blue": "蓝光",
            "green": "绿光",
            "red": "红光",
            "white": "白光",
            "custom": "自定义",
        }
        mode_desc = mode_names.get(mode, mode)
        self.lbl_brightness_status.setText(
            f"LED编码: R:{r_code} G:{g_code} B:{b_code} / 1023 ({pct}% {mode_desc})"
        )

    def _on_apply_dlp_params(self):
        pct = self.spin_projector_brightness.value()
        mode = self.combo_dlp_color.currentData() or "blue"
        if mode == "custom":
            r_en = self.chk_led_r.isChecked()
            g_en = self.chk_led_g.isChecked()
            b_en = self.chk_led_b.isChecked()
            self.controller.dlp.set_led_enables(r_en, g_en, b_en)
            ok, msg = self.controller.dlp.set_brightness_percent(pct)
        else:
            ok, msg = self.controller.dlp.set_color_mode(mode, pct)

        self._update_brightness_status_label()
        self._update_status_bar_badges()
        self.log(msg)
        if not ok:
            QMessageBox.warning(self, "DLP参数应用提示", msg)
        else:
            self.lbl_status.setText(f"DLP参数已应用: {pct}% {mode}")

    def _on_read_dlp_params(self):
        ok, params, msg = self.controller.dlp.read_all_parameters()
        r, g, b = params.get("led_currents", (0, 0, 0))
        enables = params.get("led_enables", {})
        r_en = enables.get("red", False)
        g_en = enables.get("green", False)
        b_en = enables.get("blue", False)
        method = params.get("control_method", 0)
        method_desc = "手动恒流模式 (Manual - 推荐)" if method == 0 else "自动 CAIC 动态模式 (Auto)"
        pct = int(round(params.get("brightness_percent", 100.0)))
        color_mode = params.get("color_mode", "blue")
        connected = params.get("connected", False)

        # Update UI controls without re-triggering recursive loops
        self.spin_projector_brightness.blockSignals(True)
        self.slider_projector_brightness.blockSignals(True)
        self.spin_projector_brightness.setValue(pct)
        self.slider_projector_brightness.setValue(pct)
        self.spin_projector_brightness.blockSignals(False)
        self.slider_projector_brightness.blockSignals(False)

        # Match color dropdown
        self.combo_dlp_color.blockSignals(True)
        idx = self.combo_dlp_color.findData(color_mode)
        if idx >= 0:
            self.combo_dlp_color.setCurrentIndex(idx)
        else:
            idx_custom = self.combo_dlp_color.findData("custom")
            if idx_custom >= 0:
                self.combo_dlp_color.setCurrentIndex(idx_custom)
        self.combo_dlp_color.blockSignals(False)

        # Update checkboxes
        self.chk_led_r.blockSignals(True)
        self.chk_led_g.blockSignals(True)
        self.chk_led_b.blockSignals(True)
        self.chk_led_r.setChecked(r_en)
        self.chk_led_g.setChecked(g_en)
        self.chk_led_b.setChecked(b_en)
        is_custom = (color_mode == "custom")
        self.chk_led_r.setEnabled(is_custom)
        self.chk_led_g.setEnabled(is_custom)
        self.chk_led_b.setEnabled(is_custom)
        self.chk_led_r.blockSignals(False)
        self.chk_led_g.blockSignals(False)
        self.chk_led_b.blockSignals(False)

        self._update_brightness_status_label()
        self._update_status_bar_badges()

        conn_str = "🟢 已连接 (I2C 物理应答正常)" if connected else "⚪ 未连接物理设备 (离线模拟/缓存状态)"
        report = (
            "================ DLP4710 实时参数报告 ================\n"
            f"设备通信状态: {conn_str}\n"
            f"控制模式 (Cmd 0x51): {method_desc}\n"
            f"等效换算光强: {pct}%\n"
            f"颜色工作模式: {color_mode.upper()}\n\n"
            f"LED 驱动电流编码 (Cmd 0x55, 0~1023):\n"
            f"  - 红光 (Red):   {r:4d} / 1023  ({(r/1023.0)*100.0:.1f}%)\n"
            f"  - 绿光 (Green): {g:4d} / 1023  ({(g/1023.0)*100.0:.1f}%)\n"
            f"  - 蓝光 (Blue):  {b:4d} / 1023  ({(b/1023.0)*100.0:.1f}%)\n\n"
            f"物理 LED 通道使能开关 (Cmd 0x53):\n"
            f"  - 红光通道 (Red):   {'[开启 ON]' if r_en else '[关闭 OFF]'}\n"
            f"  - 绿光通道 (Green): {'[开启 ON]' if g_en else '[关闭 OFF]'}\n"
            f"  - 蓝光通道 (Blue):  {'[开启 ON]' if b_en else '[关闭 OFF]'}\n"
            "====================================================="
        )
        self.log(f"已读取 DLP4710 实时参数: 光强 {pct}%, 模式 {color_mode}, R={r} G={g} B={b}")
        QMessageBox.information(self, "DLP 硬件实时参数报告", report)

    def _on_gen_color_bin(self):
        mode = self.combo_dlp_color.currentData() or "blue"
        bin_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "bin_and_templates",
            )
        )
        os.makedirs(bin_dir, exist_ok=True)
        out_bin = os.path.join(bin_dir, f"mySL3x4_hor_{mode}.bin")
        from dlp_common.hardware.pattern_bin_generator import generate_pattern_bin
        ok, msg = generate_pattern_bin(out_bin, led_color=mode)
        if ok:
            self.edit_bin_path.setText(out_bin)
            self.log(f"成功生成对应颜色测量条纹固件: {os.path.basename(out_bin)}")
            QMessageBox.information(
                self,
                "固件生成成功",
                f"已为【{mode.upper()}】颜色模式生成专用 12 帧条纹固件：\n{out_bin}\n\n已自动填入固件路径，点击【烧录测量条纹BIN】即可写入光机！",
            )
        else:
            self.log(f"生成条纹固件失败: {msg}")
            QMessageBox.critical(self, "生成失败", msg)

    def _load_preset_brackets(self, index: int):
        curr_exp = self.spin_exposure.value() if hasattr(self, "spin_exposure") else 3000.0
        if index == 0:  # 快速 3 档
            brackets = [(30.0, curr_exp), (70.0, curr_exp), (100.0, curr_exp)]
        elif index == 1:  # 快速 5 档梯度
            brackets = [(20.0, curr_exp), (40.0, curr_exp), (60.0, curr_exp), (80.0, curr_exp), (100.0, curr_exp)]
        elif index == 2:  # 论文 10 档光强训练集 (44, 46, 48, 50, 52, 54, 56, 58, 60, 62)
            brackets = [(float(v), curr_exp) for v in (44, 46, 48, 50, 52, 54, 56, 58, 60, 62)]
        elif index == 3:  # 论文 13 档高精基准 (13 Exposures GT)
            vals = np.linspace(20, 100, 13)
            brackets = [(round(float(v), 0), curr_exp) for v in vals]
        else:
            return
        self._set_table_brackets(brackets)

    def _set_table_brackets(self, brackets: List[Tuple[float, float]]):
        self.table_hdr_brackets.blockSignals(True)
        self.table_hdr_brackets.setRowCount(len(brackets))
        for row, (led_pct, exp_us) in enumerate(brackets):
            idx_item = QTableWidgetItem(f"#{row + 1}")
            idx_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            led_item = QTableWidgetItem(f"{float(led_pct):.0f}")
            led_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            exp_item = QTableWidgetItem(f"{float(exp_us):.0f}")
            exp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table_hdr_brackets.setItem(row, 0, idx_item)
            self.table_hdr_brackets.setItem(row, 1, led_item)
            self.table_hdr_brackets.setItem(row, 2, exp_item)

        self.table_hdr_brackets.blockSignals(False)
        self.spin_hdr_count.blockSignals(True)
        self.spin_hdr_count.setValue(len(brackets))
        self.spin_hdr_count.blockSignals(False)

    def _on_hdr_preset_changed(self, index: int):
        if index < 4:
            self._load_preset_brackets(index)

    def _on_hdr_gen_gradient(self):
        m = self.spin_hdr_count.value()
        min_led = self.spin_hdr_min_led.value()
        max_led = self.spin_hdr_max_led.value()
        curr_exp = self.spin_exposure.value()
        if m < 2:
            m = 2
        led_values = np.linspace(min_led, max_led, m)
        brackets = [(round(float(v), 0), curr_exp) for v in led_values]
        self._set_table_brackets(brackets)
        self.combo_hdr_preset.blockSignals(True)
        self.combo_hdr_preset.setCurrentIndex(4)  # 自定义
        self.combo_hdr_preset.blockSignals(False)
        self.log(f"已根据设定生成 {m} 档等间距光强梯度 ({min_led}% ~ {max_led}%)")

    def _on_hdr_add_bracket(self):
        row = self.table_hdr_brackets.rowCount()
        curr_exp = self.spin_exposure.value()
        self.table_hdr_brackets.insertRow(row)

        idx_item = QTableWidgetItem(f"#{row + 1}")
        idx_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        led_item = QTableWidgetItem("100")
        led_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        exp_item = QTableWidgetItem(f"{int(curr_exp)}")
        exp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        self.table_hdr_brackets.setItem(row, 0, idx_item)
        self.table_hdr_brackets.setItem(row, 1, led_item)
        self.table_hdr_brackets.setItem(row, 2, exp_item)

        self.spin_hdr_count.blockSignals(True)
        self.spin_hdr_count.setValue(self.table_hdr_brackets.rowCount())
        self.spin_hdr_count.blockSignals(False)
        self.combo_hdr_preset.setCurrentIndex(4)

    def _on_hdr_del_bracket(self):
        count = self.table_hdr_brackets.rowCount()
        if count <= 2:
            QMessageBox.information(self, "提示", "多曝光至少需要 2 档曝光")
            return
        selected = self.table_hdr_brackets.selectedIndexes()
        row_to_del = selected[0].row() if selected else count - 1
        self.table_hdr_brackets.removeRow(row_to_del)
        for r in range(self.table_hdr_brackets.rowCount()):
            item = self.table_hdr_brackets.item(r, 0)
            if item:
                item.setText(f"#{r + 1}")
        self.spin_hdr_count.blockSignals(True)
        self.spin_hdr_count.setValue(self.table_hdr_brackets.rowCount())
        self.spin_hdr_count.blockSignals(False)
        self.combo_hdr_preset.setCurrentIndex(4)

    def _on_hdr_reset_brackets(self):
        self.combo_hdr_preset.setCurrentIndex(0)
        self._load_preset_brackets(0)

    def _get_hdr_brackets(self) -> List[dict]:
        gain = self.spin_gain.value()
        brackets = []
        for row in range(self.table_hdr_brackets.rowCount()):
            led_item = self.table_hdr_brackets.item(row, 1)
            exp_item = self.table_hdr_brackets.item(row, 2)
            try:
                led_val = float(led_item.text()) if led_item else 100.0
            except ValueError:
                led_val = 100.0
            try:
                exp_val = float(exp_item.text()) if exp_item else self.spin_exposure.value()
            except ValueError:
                exp_val = self.spin_exposure.value()
            brackets.append({
                "exposure_us": exp_val,
                "led_percent": led_val,
                "gain_db": gain,
            })
        if len(brackets) < 2:
            brackets = [
                {"exposure_us": self.spin_exposure.value(), "led_percent": 30.0, "gain_db": gain},
                {"exposure_us": self.spin_exposure.value(), "led_percent": 100.0, "gain_db": gain},
            ]
        return brackets

    def _on_measure_hdr_online(self):
        if not self.controller.reconstructor.has_calibration_parameters():
            QMessageBox.warning(self, "警告", "请先加载有效的标定文件 (MonoSLCalibration.yml)")
            return

        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker = None
            self.btn_toggle_preview.setText("开启实时预览")

        if self.multi_capture_worker is not None and self.multi_capture_worker.isRunning():
            return

        brackets = self._get_hdr_brackets()
        self.lbl_status.setText(f"正在准备多曝光采集 ({len(brackets)} 档)...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.log(f"启动在线多曝光 (HDR) 条纹采集 ({len(brackets)} 档，共 {len(brackets) * 12} 帧)...")

        if not self.controller.camera.is_opened:
            if not self.controller.camera.open(0):
                self._update_status_bar_badges()
                self.progress_bar.setValue(0)
                QMessageBox.critical(self, "大恒相机连接失败", self.controller.camera.last_error)
                return
            self._update_status_bar_badges()

        if not self.controller.dlp.is_connected and not self.controller.dlp.connect():
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "DLP连接错误", self.controller.dlp.last_error)
            return
        if not self.controller.dlp.pattern_ready:
            mode_ok, mode_message = self.controller.dlp.prepare_pattern_mode()
            self.log(mode_message)
            if not mode_ok:
                self.progress_bar.setValue(0)
                QMessageBox.critical(self, "DLP模式初始化失败", mode_message)
                return

        self._set_measure_controls_enabled(False)
        self.multi_capture_worker = MultiExposureCaptureWorker(
            camera=self.controller.camera,
            dlp=self.controller.dlp,
            brackets=brackets,
            gain_db=self.spin_gain.value(),
            gray_exposure_us=self.spin_gray_exposure.value(),
        )
        self.multi_capture_worker.progress.connect(self._on_capture_progress)
        self.multi_capture_worker.completed.connect(self._on_hdr_capture_completed)
        self.multi_capture_worker.gray_frame_captured.connect(self._on_gray_frame_captured)
        self.multi_capture_worker.finished.connect(self._on_hdr_capture_thread_stopped)
        self.multi_capture_worker.start()

    def _on_hdr_capture_completed(self, ok: bool, brackets_frames, message: str):
        self.log(message)
        if not ok or len(brackets_frames) == 0:
            self.progress_bar.setValue(0)
            self.lbl_status.setText("多曝光采集失败")
            self._set_measure_controls_enabled(True)
            QMessageBox.critical(self, "采集错误", message)
            return
        gray_frame = getattr(self.multi_capture_worker, "gray_frame", None)
        brackets_info = self._get_hdr_brackets()
        self._start_multi_reconstruction(brackets_frames, gray_frame=gray_frame, brackets_info=brackets_info)

    def _on_hdr_capture_thread_stopped(self):
        worker = self.multi_capture_worker
        self.multi_capture_worker = None
        if worker is not None:
            worker.deleteLater()

    def _on_measure_hdr_offline(self):
        if not self.controller.reconstructor.has_calibration_parameters():
            QMessageBox.warning(self, "警告", "请先加载有效的标定文件 (MonoSLCalibration.yml)")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择包含多曝光条纹序列的目录")
        if not folder:
            return

        ok, brackets, msg = self.controller.load_offline_multi_exposure_frames(folder)
        if not ok or len(brackets) == 0:
            QMessageBox.warning(self, "加载失败", msg)
            return

        self.log(msg)
        self._start_multi_reconstruction(brackets)

    def _start_multi_reconstruction(
        self,
        brackets_frames: List[List[np.ndarray]],
        gray_frame: Optional[np.ndarray] = None,
        brackets_info: Optional[List[dict]] = None,
    ):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(10)
        self.lbl_status.setText(f"开始多曝光 ({len(brackets_frames)}档) HDR三维重构...")

        step = self._get_selected_step()
        min_mod = self.spin_mod.value()

        self.multi_measure_worker = MultiExposureMeasureWorker(
            reconstructor=self.controller.reconstructor,
            processor=self.controller.phase_processor,
            brackets_frames=brackets_frames,
            sample_step=step,
            min_modulation=min_mod,
            gray_frame=gray_frame,
            brackets_info=brackets_info,
        )
        self.multi_measure_worker.progress.connect(lambda p, m: (
            self.progress_bar.setValue(p),
            self.lbl_status.setText(m),
            self.log(m),
        ))
        self.multi_measure_worker.finished.connect(self._on_measure_finished)
        self.multi_measure_worker.start()

    def _refresh_point_cloud_result(self):
        result = self.controller.latest_result
        if result is None:
            return
        points = result.get("valid_points")
        if points is None:
            return

        self.view_3d.set_point_cloud(points)
        if len(points) > 0:
            min_pt = np.min(points, axis=0)
            max_pt = np.max(points, axis=0)
            extent = max_pt - min_pt
            mean_pt = np.mean(points, axis=0)
            state = "已提取测量主体" if result.get("point_cloud_processed") else "原始点云"
            self.text_stats.setText(
                "================ 三维测量统计报告 ================\n"
                f"点云状态: {state}\n"
                f"有效重构点数: {len(points):,}\n"
                f"X 坐标范围: [{min_pt[0]:.2f}, {max_pt[0]:.2f}] mm (跨度: {extent[0]:.2f} mm)\n"
                f"Y 坐标范围: [{min_pt[1]:.2f}, {max_pt[1]:.2f}] mm (跨度: {extent[1]:.2f} mm)\n"
                f"Z 深度范围: [{min_pt[2]:.2f}, {max_pt[2]:.2f}] mm (跨度: {extent[2]:.2f} mm)\n"
                f"点云质心坐标: ({mean_pt[0]:.2f}, {mean_pt[1]:.2f}, {mean_pt[2]:.2f}) mm\n"
            )
        self.tab_widget.setCurrentIndex(0)

    def _on_keep_subject(self):
        ok, message = self.controller.keep_largest_subject(
            close_radius=self.spin_subject_radius.value()
        )
        self.log(message)
        if not ok:
            QMessageBox.warning(self, "点云处理失败", message)
            return
        self._refresh_point_cloud_result()
        self.lbl_status.setText("已保留最大测量主体")
        QMessageBox.information(self, "点云处理完成", message)

    def _on_restore_cloud(self):
        ok, message = self.controller.restore_original_point_cloud()
        self.log(message)
        if not ok:
            QMessageBox.information(self, "恢复点云", message)
            return
        self._refresh_point_cloud_result()
        self.lbl_status.setText("已恢复原始点云")

    def _on_save_ply(self):
        if self.controller.latest_result is None:
            QMessageBox.warning(self, "提示", "尚无测量结果，请先执行三维重构")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出PLY点云文件 (高速二进制)", "measured_cloud.ply", "PLY Files (*.ply);;All Files (*.*)"
        )
        if not path:
            return

        ok, msg = self.controller.save_ply(path, binary=True)
        if ok:
            self.log(msg)
            QMessageBox.information(self, "导出成功", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)

    def _on_save_intermediate(self):
        if self.controller.latest_result is None:
            QMessageBox.warning(self, "提示", "尚无测量结果，请先执行三维重构")
            return

        parent = QFileDialog.getExistingDirectory(self, "选择测量数据保存位置")
        if not parent:
            return

        default_name = datetime.now().strftime("measurement_%Y%m%d_%H%M%S")
        folder_name, accepted = QInputDialog.getText(
            self,
            "新建测量数据集文件夹",
            "输入文件夹名称 (将保存条纹BMP/灰度图/参数/二进制PLY):",
            text=default_name,
        )
        if not accepted:
            return

        folder_name = folder_name.strip()
        invalid_chars = '<>:"/\\|?*'
        if (
            not folder_name
            or any(char in folder_name for char in invalid_chars)
            or folder_name.endswith((".", " "))
            or folder_name in (".", "..")
        ):
            QMessageBox.warning(
                self,
                "文件夹名称无效",
                '名称不能为空，不能以点或空格结尾，也不能包含 < > : " / \\ | ? *。',
            )
            return

        output_folder = os.path.join(parent, folder_name)
        ok, msg = self.controller.save_intermediate_results(output_folder)
        self.log(msg)
        if ok:
            QMessageBox.information(self, "保存成功", f"测量数据已极速保存！\n\n{msg}")
        else:
            QMessageBox.critical(self, "保存失败", msg)

    def _on_ply_to_txt(self):
        ply_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择要转换为 TXT 的 PLY 点云文件",
            "",
            "PLY Files (*.ply);;All Files (*.*)",
        )
        if not ply_path:
            return
        default_txt = os.path.splitext(ply_path)[0] + ".txt"
        txt_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存转换后的 TXT 坐标文件",
            default_txt,
            "Text Files (*.txt);;All Files (*.*)",
        )
        if not txt_path:
            return
        from dlp_common.reconstruction.point_cloud_io import ply_to_txt
        ok, count, msg = ply_to_txt(ply_path, txt_path)
        self.log(msg)
        if ok:
            QMessageBox.information(self, "转换成功", f"成功将 {count:,} 个三维点转换为 TXT 格式：\n{txt_path}")
        else:
            QMessageBox.critical(self, "转换失败", msg)

    def _on_save_txt(self):
        if self.controller.latest_result is None:
            QMessageBox.warning(self, "提示", "尚无测量结果，请先执行三维重构")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出TXT三维坐标", "measured_points.txt", "Text Files (*.txt);;All Files (*.*)"
        )
        if not path:
            return

        ok, msg = self.controller.save_txt(path)
        if ok:
            self.log(msg)
            QMessageBox.information(self, "导出成功", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)

    def closeEvent(self, event):
        if self.burn_worker is not None and self.burn_worker.isRunning():
            QMessageBox.warning(self, "烧录尚未结束", "DLP BIN仍在烧录，请等待完成后再关闭软件。")
            event.ignore()
            return
        if self.capture_worker is not None and self.capture_worker.isRunning():
            QMessageBox.warning(self, "采集尚未结束", "在线条纹采集仍在进行，请等待采集结束后再关闭软件。")
            event.ignore()
            return
        if self.multi_capture_worker is not None and self.multi_capture_worker.isRunning():
            QMessageBox.warning(self, "采集尚未结束", "多曝光条纹采集仍在进行，请等待采集结束后再关闭软件。")
            event.ignore()
            return
        if self.measure_worker is not None and self.measure_worker.isRunning():
            QMessageBox.warning(self, "重构尚未结束", "三维重构仍在进行，请等待计算结束后再关闭软件。")
            event.ignore()
            return
        if self.multi_measure_worker is not None and self.multi_measure_worker.isRunning():
            QMessageBox.warning(self, "重构尚未结束", "多曝光三维重构仍在进行，请等待计算结束后再关闭软件。")
            event.ignore()
            return
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
        self.controller.camera.close()
        self.controller.dlp.disconnect()
        event.accept()

