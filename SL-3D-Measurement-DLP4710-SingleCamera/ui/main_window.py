"""
Main Application Window for DLP4710 Mono Structured Light 3D Measurement.
Built with PyQt6 and PyQtGraph OpenGL.
"""

from datetime import datetime
import os
from typing import List, Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.controller import MeasurementController
from ..core.workers import (
    BurnMeasurementBinWorker,
    CaptureMeasurementWorker,
    LivePreviewWorker,
    MeasureWorker,
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

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Compact, scrollable control panel.
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 2, 4, 4)
        left_layout.setSpacing(5)

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

        self.btn_detect_devices = QPushButton("检测大恒相机与DLP")
        self.btn_disconnect_devices = QPushButton("断开设备")
        lay_cal.addWidget(self.btn_detect_devices, 3, 0, 1, 2)
        lay_cal.addWidget(self.btn_disconnect_devices, 3, 2)

        self.lbl_device_status = QLabel("设备状态: 未检测")
        self.lbl_device_status.setWordWrap(True)
        lay_cal.addWidget(self.lbl_device_status, 4, 0, 1, 3)

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
        lay_cal.addWidget(QLabel("测量条纹BIN:"), 5, 0)
        lay_cal.addWidget(self.edit_bin_path, 5, 1)
        lay_cal.addWidget(self.btn_browse_bin, 5, 2)
        lay_cal.addWidget(self.btn_burn_bin, 6, 0, 1, 3)
        left_layout.addWidget(grp_calib)

        # 2. Camera & Preview Group
        grp_camera = QGroupBox("2. 相机与曝光设置")
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

        self.btn_toggle_preview = QPushButton("开启实时预览")
        lay_cam.addWidget(self.btn_toggle_preview, 3, 0, 1, 2)
        left_layout.addWidget(grp_camera)

        # 3. Measurement Execution Group
        grp_measure = QGroupBox("3. 测量与采集控制")
        lay_meas = QGridLayout(grp_measure)

        self.btn_measure_online = QPushButton("采集并执行3D测量 (12帧)")
        self.btn_measure_online.setObjectName("measureBtn")
        self.btn_measure_offline = QPushButton("导入离线12帧图像并测量")

        lay_meas.addWidget(self.btn_measure_online, 0, 0)
        lay_meas.addWidget(self.btn_measure_offline, 1, 0)
        left_layout.addWidget(grp_measure)

        # 4. Reconstruction Settings Group
        grp_params = QGroupBox("4. 重构与滤波设置")
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

        left_layout.addWidget(grp_params)

        # 5. Point-cloud Processing Group
        grp_process = QGroupBox("5. 点云处理")
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
        left_layout.addWidget(grp_process)

        # 6. Export Group
        grp_export = QGroupBox("6. 结果保存与导出")
        lay_exp = QGridLayout(grp_export)
        self.btn_save_intermediate = QPushButton("保存本次中间结果")
        self.btn_save_intermediate.setObjectName("saveBtn")
        self.btn_save_ply = QPushButton("导出标准 PLY 点云")
        self.btn_save_txt = QPushButton("导出 TXT (X Y Z) 坐标")

        lay_exp.addWidget(self.btn_save_intermediate, 0, 0)
        lay_exp.addWidget(self.btn_save_ply, 1, 0)
        lay_exp.addWidget(self.btn_save_txt, 2, 0)
        left_layout.addWidget(grp_export)

        # Progress & Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("就绪")
        left_layout.addWidget(self.lbl_status)

        left_layout.addStretch()
        control_scroll = QScrollArea()
        control_scroll.setObjectName("controlScroll")
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setWidget(left_panel)
        control_scroll.setMinimumWidth(330)
        control_scroll.setMaximumWidth(380)
        main_layout.addWidget(control_scroll)

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

        # Connect signals
        self._connect_signals()

    def _connect_signals(self):
        self.btn_browse_calib.clicked.connect(self._on_browse_calib)
        self.btn_load_calib.clicked.connect(self._on_load_calib)
        self.btn_detect_devices.clicked.connect(self._on_detect_devices)
        self.btn_disconnect_devices.clicked.connect(self._on_disconnect_devices)
        self.btn_browse_bin.clicked.connect(self._on_browse_bin)
        self.btn_burn_bin.clicked.connect(self._on_burn_bin)
        self.btn_toggle_preview.clicked.connect(self._on_toggle_preview)

        self.btn_measure_online.clicked.connect(self._on_measure_online)
        self.btn_measure_offline.clicked.connect(self._on_measure_offline)

        self.btn_save_ply.clicked.connect(self._on_save_ply)
        self.btn_save_txt.clicked.connect(self._on_save_txt)
        self.btn_save_intermediate.clicked.connect(self._on_save_intermediate)
        self.btn_keep_subject.clicked.connect(self._on_keep_subject)
        self.btn_restore_cloud.clicked.connect(self._on_restore_cloud)

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
        camera_text = (
            f"大恒相机: {devices[0]['name']}" if devices else "大恒相机: 未检测到"
        )

        dlp_ok = self.controller.dlp.is_connected or self.controller.dlp.connect()
        dlp_text = "DLP: 已连接" if dlp_ok else f"DLP: {self.controller.dlp.last_error}"
        if dlp_ok and not self.controller.dlp.pattern_ready:
            mode_ok, mode_message = self.controller.dlp.prepare_pattern_mode()
            dlp_text += f"，{mode_message}"
            dlp_ok = mode_ok

        all_ok = bool(devices) and dlp_ok
        self.lbl_device_status.setText(f"设备状态: {camera_text}；{dlp_text}")
        self.lbl_device_status.setStyleSheet(
            f"color: {'#28a745' if all_ok else '#ff9966'}; font-size: 11px;"
        )
        self.log(f"设备检测: {camera_text}；{dlp_text}")

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

    def _on_toggle_preview(self):
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker = None
            self.btn_toggle_preview.setText("开启实时预览")
            self.log("实时相机预览已停止")
            self.lbl_status.setText("预览已停止")
        else:
            if not self.controller.camera.open(0):
                QMessageBox.critical(self, "大恒相机连接失败", self.controller.camera.last_error)
                return
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
                self.progress_bar.setValue(0)
                QMessageBox.critical(self, "大恒相机连接失败", self.controller.camera.last_error)
                return

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
            self.spin_gray_exposure,
            self.spin_exposure,
            self.spin_gain,
            self.btn_detect_devices,
            self.btn_disconnect_devices,
            self.btn_browse_bin,
            self.btn_burn_bin,
            self.edit_bin_path,
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
        self.view_2d.update_views(
            raw_frame=result["raw_frame"],
            wrapped_phase=result["wrapped_phase"],
            unwrapped_phase=result["unwrapped_phase"],
            mask=result["mask"],
        )

        # 2. Update 3D Point Cloud View
        pts = result["valid_points"]
        self.view_3d.set_point_cloud(pts)

        # 3. Update Statistics Report
        if len(pts) > 0:
            min_pt = np.min(pts, axis=0)
            max_pt = np.max(pts, axis=0)
            extent = max_pt - min_pt
            mean_pt = np.mean(pts, axis=0)

            stats = (
                "================ 三维测量统计报告 ================\n"
                f"有效重构点数: {len(pts):,}\n"
                f"算法总耗时: {result['elapsed_seconds']:.2f} 秒\n\n"
                f"X 坐标范围: [{min_pt[0]:.2f}, {max_pt[0]:.2f}] mm (跨度: {extent[0]:.2f} mm)\n"
                f"Y 坐标范围: [{min_pt[1]:.2f}, {max_pt[1]:.2f}] mm (跨度: {extent[1]:.2f} mm)\n"
                f"Z 深度范围: [{min_pt[2]:.2f}, {max_pt[2]:.2f}] mm (跨度: {extent[2]:.2f} mm)\n\n"
                f"点云质心坐标: ({mean_pt[0]:.2f}, {mean_pt[1]:.2f}, {mean_pt[2]:.2f}) mm\n"
            )
            self.text_stats.setText(stats)

        # Switch to 3D View tab
        self.tab_widget.setCurrentIndex(0)
        QMessageBox.information(self, "测量成功", f"三维重构顺利完成！\n提取点数: {len(pts):,}\n耗时: {result['elapsed_seconds']:.2f} s")

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
            self, "导出PLY点云文件", "measured_cloud.ply", "PLY Files (*.ply);;All Files (*.*)"
        )
        if not path:
            return

        ok, msg = self.controller.save_ply(path)
        if ok:
            self.log(msg)
            QMessageBox.information(self, "导出成功", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)

    def _on_save_intermediate(self):
        if self.controller.latest_result is None:
            QMessageBox.warning(self, "提示", "尚无测量结果，请先执行三维重构")
            return

        parent = QFileDialog.getExistingDirectory(self, "选择中间结果保存位置")
        if not parent:
            return

        default_name = datetime.now().strftime("measurement_%Y%m%d_%H%M%S")
        folder_name, accepted = QInputDialog.getText(
            self,
            "新建结果文件夹",
            "输入新的文件夹名称:",
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
            QMessageBox.information(self, "保存成功", msg)
        else:
            QMessageBox.critical(self, "保存失败", msg)

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
        if self.measure_worker is not None and self.measure_worker.isRunning():
            QMessageBox.warning(self, "重构尚未结束", "三维重构仍在进行，请等待计算结束后再关闭软件。")
            event.ignore()
            return
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
        self.controller.camera.close()
        self.controller.dlp.disconnect()
        event.accept()

