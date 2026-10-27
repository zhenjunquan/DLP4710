"""
Main Window for DLP4710 Mono Structured Light Calibration Software.
Features full real-time live camera feed, device selection, parameter controls,
and synchronized 24-frame acquisition with corner/phase previews.
"""

from datetime import datetime
import os
import re
import time
from typing import Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
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

from dlp_common.fringe.generator import FringeGenerator
from ..core.controller import CalibrationController
from ..core.workers import (
    BurnBinWorker,
    CalibrateWorker,
    CapturePoseWorker,
    DecodePhasesWorker,
    DetectCornersWorker,
)
from .styles import MODERN_STYLE
from .widgets import ImageDisplayWidget, MultiFolderDialog, PoseTableWidget


class CalibrationMainWindow(QMainWindow):
    """
    Main application window for system calibration.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("DLP4710 单目结构光标定软件 (Python 工业工程版)")
        self.resize(1600, 960)
        self.setMinimumSize(1180, 720)

        self.controller = CalibrationController()
        self._current_worker = None
        self._auto_decode_worker = None
        self._auto_decode_pose = None

        # Live preview state & FPS timer
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(30)  # ~33 FPS
        self._preview_timer.timeout.connect(self._on_preview_tick)
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        self._current_fps = 0.0

        self._init_ui()
        self.setStyleSheet(MODERN_STYLE)
        self.log("标定系统已就绪，运行环境: Conda dlp (Python 3.13)")

        # Initial device scan and auto-select
        self._refresh_camera_devices()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ---------------- Compact scrollable control panel ----------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 2, 4, 4)
        left_layout.setSpacing(5)

        # 1. Device Control Group
        grp_device = QGroupBox("1. 相机与设备设置")
        lay_dev = QGridLayout(grp_device)

        lay_dev.addWidget(QLabel("选择相机:"), 0, 0)
        self.combo_camera = QComboBox()
        lay_dev.addWidget(self.combo_camera, 0, 1, 1, 2)

        self.btn_detect = QPushButton("刷新设备")
        self.btn_connect = QPushButton("连接并开启相机")
        self.btn_connect.setObjectName("primaryBtn")
        self.btn_disconnect = QPushButton("断开")
        self.btn_disconnect.setEnabled(False)

        lay_dev.addWidget(self.btn_detect, 1, 0)
        lay_dev.addWidget(self.btn_connect, 1, 1)
        lay_dev.addWidget(self.btn_disconnect, 1, 2)

        self.btn_toggle_preview = QPushButton("开启/暂停实时预览")
        self.btn_toggle_preview.setObjectName("primaryBtn")
        lay_dev.addWidget(self.btn_toggle_preview, 2, 0, 1, 3)

        self.chk_crosshair = QCheckBox("显示中心准星与标定参考十字线")
        self.chk_crosshair.setChecked(True)
        lay_dev.addWidget(self.chk_crosshair, 3, 0, 1, 3)

        lay_dev.addWidget(QLabel("条纹曝光 (μs):"), 4, 0)
        self.spin_pattern_exp = QDoubleSpinBox()
        self.spin_pattern_exp.setRange(100.0, 1000000.0)
        self.spin_pattern_exp.setValue(self.controller.config.calibration_pattern_exposure_us)
        lay_dev.addWidget(self.spin_pattern_exp, 4, 1, 1, 2)

        lay_dev.addWidget(QLabel("基底曝光 (μs):"), 5, 0)
        self.spin_board_exp = QDoubleSpinBox()
        self.spin_board_exp.setRange(100.0, 500000.0)
        self.spin_board_exp.setValue(self.controller.config.board_exposure_us)
        lay_dev.addWidget(self.spin_board_exp, 5, 1, 1, 2)

        left_layout.addWidget(grp_device)

        # 2. DLP Pattern & BIN Group
        grp_bin = QGroupBox("2. DLP投影与BIN固件")
        lay_bin = QGridLayout(grp_bin)
        self.edit_bin_path = QLineEdit()
        self.edit_bin_path.setPlaceholderText("选择或输入24帧标定BIN路径...")
        # Auto fill default BIN if found
        default_bin = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "bin_and_templates", "mySL_calib_12horAnd12vec.bin")
        )
        if os.path.isfile(default_bin):
            self.edit_bin_path.setText(default_bin)
            self.controller.bin_path = default_bin

        self.btn_browse_bin = QPushButton("选择BIN")
        self.btn_burn_bin = QPushButton("烧录BIN")

        lay_bin.addWidget(self.edit_bin_path, 0, 0, 1, 2)
        lay_bin.addWidget(self.btn_browse_bin, 0, 2)
        lay_bin.addWidget(self.btn_burn_bin, 1, 0, 1, 3)

        self.btn_gen_h = QPushButton("生成12张水平条纹")
        self.btn_gen_v = QPushButton("生成12张竖直条纹")
        lay_bin.addWidget(self.btn_gen_h, 2, 0, 1, 1)
        lay_bin.addWidget(self.btn_gen_v, 2, 1, 1, 2)

        left_layout.addWidget(grp_bin)

        # 3. Calibration Pose Management Group
        grp_poses = QGroupBox("3. 姿态采集与管理")
        lay_poses = QGridLayout(grp_poses)

        self.btn_capture_pose = QPushButton("采集当前姿态 (1+24)")
        self.btn_capture_pose.setObjectName("primaryBtn")
        self.btn_new_session = QPushButton("新建标定保存会话")
        self.btn_import_pose = QPushButton("导入离线姿态")
        self.btn_import_pose.setToolTip("批量导入离线姿态文件夹 (按住 Ctrl/Shift 可多选，也可直接选择包含姿态的父目录)")
        self.btn_delete_pose = QPushButton("删除选中姿态")

        lay_poses.addWidget(self.btn_new_session, 0, 0, 1, 2)
        lay_poses.addWidget(self.btn_capture_pose, 1, 0, 1, 2)
        lay_poses.addWidget(self.btn_import_pose, 2, 0)
        lay_poses.addWidget(self.btn_delete_pose, 2, 1)

        left_layout.addWidget(grp_poses)

        # 4. Calibration Parameters & Solve Group
        grp_calib = QGroupBox("4. 联合标定计算")
        lay_calib = QGridLayout(grp_calib)

        lay_calib.addWidget(QLabel("角点列数:"), 0, 0)
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(3, 50)
        self.spin_cols.setValue(self.controller.config.board_cols)
        lay_calib.addWidget(self.spin_cols, 0, 1)

        lay_calib.addWidget(QLabel("角点行数:"), 0, 2)
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(3, 50)
        self.spin_rows.setValue(self.controller.config.board_rows)
        lay_calib.addWidget(self.spin_rows, 0, 3)

        lay_calib.addWidget(QLabel("棋盘格间距 (mm):"), 1, 0, 1, 2)
        self.spin_square = QDoubleSpinBox()
        self.spin_square.setRange(0.1, 100.0)
        self.spin_square.setValue(self.controller.config.square_size_mm)
        lay_calib.addWidget(self.spin_square, 1, 2, 1, 2)

        self.btn_detect_all = QPushButton("批量检测角点")
        self.btn_decode_all = QPushButton("批量相位解算")
        self.btn_calibrate = QPushButton("系统联合标定")
        self.btn_calibrate.setObjectName("primaryBtn")
        self.btn_save_calib = QPushButton("保存YML与TXT")
        self.btn_save_calib.setObjectName("successBtn")
        self.btn_save_complete = QPushButton("保存完整标定结果")
        self.btn_save_complete.setObjectName("successBtn")

        lay_calib.addWidget(self.btn_detect_all, 2, 0, 1, 2)
        lay_calib.addWidget(self.btn_decode_all, 2, 2, 1, 2)
        lay_calib.addWidget(self.btn_calibrate, 3, 0, 1, 4)
        lay_calib.addWidget(self.btn_save_calib, 4, 0, 1, 4)
        lay_calib.addWidget(self.btn_save_complete, 5, 0, 1, 4)

        left_layout.addWidget(grp_calib)

        # Progress bar & status
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("状态: 待机")
        self.lbl_status.setObjectName("statusLabel")
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

        # ---------------- Right Display Area ----------------
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: two large image views with equal space.
        img_container = QWidget()
        lay_imgs = QHBoxLayout(img_container)
        lay_imgs.setContentsMargins(0, 0, 0, 0)
        lay_imgs.setSpacing(6)

        # 1. Dedicated Live Camera View
        self.view_live = ImageDisplayWidget("相机实时画面")
        self.view_live.show_crosshair = True
        lay_imgs.addWidget(self.view_live, 5)

        # 2. Captured Board & Phase Tabs
        self.tab_analysis = QTabWidget()
        self.view_board = ImageDisplayWidget("标定板角点检测")
        self.view_phase = ImageDisplayWidget("绝对相位展开")
        self.tab_analysis.addTab(self.view_board, "标定板角点标注")
        self.tab_analysis.addTab(self.view_phase, "绝对展开相位")
        lay_imgs.addWidget(self.tab_analysis, 5)

        right_splitter.addWidget(img_container)

        # Middle: pose table spans the full workspace width.
        self.table_poses = PoseTableWidget()
        self.table_poses.setMinimumHeight(150)
        right_splitter.addWidget(self.table_poses)

        # Bottom area: Logs & Calibration Result text
        bot_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setPlaceholderText("系统运行日志...")

        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setPlaceholderText("标定结果与精度评估报告...")

        bot_splitter.addWidget(self.text_log)
        bot_splitter.addWidget(self.text_result)
        bot_splitter.setStretchFactor(0, 5)
        bot_splitter.setStretchFactor(1, 5)

        right_splitter.addWidget(bot_splitter)
        right_splitter.setStretchFactor(0, 6)
        right_splitter.setStretchFactor(1, 2)
        right_splitter.setStretchFactor(2, 2)
        right_splitter.setSizes([540, 180, 200])

        main_layout.addWidget(right_splitter, 1)

        # Connect signals
        self._connect_signals()

    def _connect_signals(self):
        self.btn_detect.clicked.connect(self._refresh_camera_devices)
        self.btn_connect.clicked.connect(self._on_connect_devices)
        self.btn_disconnect.clicked.connect(self._on_disconnect_devices)
        self.btn_toggle_preview.clicked.connect(self._on_toggle_preview)
        self.chk_crosshair.toggled.connect(self._on_crosshair_toggled)

        self.btn_browse_bin.clicked.connect(self._on_browse_bin)
        self.btn_burn_bin.clicked.connect(self._on_burn_bin)
        self.btn_gen_h.clicked.connect(lambda: self._on_generate_fringes(0))
        self.btn_gen_v.clicked.connect(lambda: self._on_generate_fringes(1))

        self.btn_capture_pose.clicked.connect(self._on_capture_live_pose)
        self.btn_new_session.clicked.connect(self._on_new_incremental_session)
        self.btn_import_pose.clicked.connect(self._on_import_pose)
        self.btn_delete_pose.clicked.connect(self._on_delete_pose)
        self.table_poses.itemSelectionChanged.connect(self._on_pose_selection_changed)

        self.btn_detect_all.clicked.connect(self._on_detect_all_corners)
        self.btn_decode_all.clicked.connect(self._on_decode_all_phases)
        self.btn_calibrate.clicked.connect(self._on_calibrate_system)
        self.btn_save_calib.clicked.connect(self._on_save_calibration)
        self.btn_save_complete.clicked.connect(self._on_save_complete_session)

        self.spin_cols.valueChanged.connect(self._on_board_param_changed)
        self.spin_rows.valueChanged.connect(self._on_board_param_changed)
        self.spin_square.valueChanged.connect(self._on_board_param_changed)
        self.spin_pattern_exp.valueChanged.connect(self._on_exposure_changed)
        self.spin_board_exp.valueChanged.connect(self._on_exposure_changed)

    def log(self, text: str):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.text_log.append(f"[{time_str}] {text}")

    def _refresh_camera_devices(self):
        devices = self.controller.camera.detect_camera_devices()
        self.combo_camera.clear()
        for dev in devices:
            self.combo_camera.addItem(dev["name"], dev["index"])
        self.log(f"已扫描大恒Galaxy相机，发现 {len(devices)} 台设备")

    def _on_crosshair_toggled(self, checked: bool):
        self.view_live.show_crosshair = checked
        self.view_live.update()

    def _on_connect_devices(self):
        # Open selected camera
        dev_idx = self.combo_camera.currentData()
        if dev_idx is None:
            QMessageBox.warning(self, "未发现相机", "未检测到大恒Galaxy相机，请检查电源、USB连接和Galaxy驱动。")
            return

        ok_cam = self.controller.camera.open(dev_idx)
        ok_dlp = self.controller.dlp.is_connected or self.controller.dlp.connect()

        if ok_cam:
            self.controller.camera.set_exposure(self.spin_board_exp.value())
            self.controller.camera.start_grabbing()
            self._start_preview()
            res = self.controller.camera.resolution
            self.log(f"大恒相机连接成功，工作分辨率: {res[0]}x{res[1]}")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.lbl_status.setText("状态: 相机采集中 (实时预览)")
            if ok_dlp:
                self.log("DLP连接成功；采集前必须完成24帧标定BIN烧录")
            else:
                self.log(f"DLP连接失败: {self.controller.dlp.last_error}")
                QMessageBox.warning(self, "DLP连接失败", self.controller.dlp.last_error)
        else:
            QMessageBox.warning(self, "连接警告", f"相机连接失败: {self.controller.camera.last_error}")

    def _on_disconnect_devices(self):
        self._stop_preview()
        self.controller.camera.close()
        self.controller.dlp.stop_work()
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.view_live.set_mat_image(None, badge="已断开")
        self.log("相机与设备已安全断开")
        self.lbl_status.setText("状态: 设备已断开")

    def _start_preview(self):
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        self._preview_timer.start()

    def _stop_preview(self):
        self._preview_timer.stop()

    def _on_toggle_preview(self):
        if self._preview_timer.isActive():
            self._stop_preview()
            self.log("实时预览已暂停")
            self.lbl_status.setText("状态: 预览已暂停")
            self.view_live.badge_text = "PAUSED"
            self.view_live.update()
        else:
            if not self.controller.camera.is_opened:
                self._on_connect_devices()
            else:
                self._start_preview()
                self.log("恢复实时预览")
                self.lbl_status.setText("状态: 实时采集中")

    def _on_preview_tick(self):
        ok, frame = self.controller.camera.grab_frame()
        if ok and frame is not None:
            self._fps_frame_count += 1
            now = time.time()
            elapsed = now - self._fps_start_time
            if elapsed >= 1.0:
                self._current_fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._fps_start_time = now

            res = frame.shape[:2]
            badge = (
                f"LIVE | 基底 {self.spin_board_exp.value():.0f} μs | "
                f"{self._current_fps:.1f} FPS | {res[1]}x{res[0]}"
            )
            self.view_live.set_mat_image(frame, badge=badge)

    def _on_exposure_changed(self):
        self.log(
            f"曝光参数已更新：条纹 {self.spin_pattern_exp.value():.0f} μs，"
            f"基底 {self.spin_board_exp.value():.0f} μs"
        )
        capture_running = self._current_worker is not None and self._current_worker.isRunning()
        if self.controller.camera.is_opened and not capture_running:
            if not self.controller.camera.set_exposure(self.spin_board_exp.value()):
                self.log(f"实时预览基底曝光设置失败: {self.controller.camera.last_error}")

    def _on_capture_live_pose(self):
        """
        Captures the current live board frame, triggers DLP sequence, and creates a calibration pose.
        """
        if not self.controller.camera.is_opened:
            QMessageBox.warning(self, "提示", "请先连接相机并开启预览")
            return
        if not self.controller.dlp.is_connected:
            QMessageBox.warning(self, "采集失败", "DLP尚未连接，请先点击“连接并开启相机”并确认DLP连接成功")
            return
        if not self.controller.dlp.pattern_ready:
            QMessageBox.warning(self, "采集失败", "DLP条纹序列尚未就绪，请先选择并烧录24帧标定BIN")
            return

        if self._current_worker is not None and self._current_worker.isRunning():
            return
        if not self._ensure_incremental_session():
            return

        pose_index = len(self.controller.poses) + 1
        pose_name = f"Pose_{pose_index:02d}"
        storage_ok, pose_folder, storage_message = self.controller.prepare_pose_storage(
            pose_index, pose_name
        )
        if not storage_ok:
            QMessageBox.critical(self, "创建姿态目录失败", storage_message)
            return
        self._pending_pose_name = pose_name

        self._capture_preview_was_active = self._preview_timer.isActive()
        self._stop_preview()
        self._set_capture_controls_enabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("状态: 正在采集当前姿态...")
        self.log("开始采集当前姿态，界面可继续响应")

        self._current_worker = CapturePoseWorker(
            camera=self.controller.camera,
            dlp=self.controller.dlp,
            detector=self.controller.detector,
            board_exposure_us=self.spin_board_exp.value(),
            pattern_exposure_us=self.spin_pattern_exp.value(),
            pose_folder=pose_folder,
        )
        self._current_worker.progress.connect(self._on_capture_progress)
        self._current_worker.completed.connect(self._on_capture_finished)
        self._current_worker.finished.connect(self._on_capture_thread_stopped)
        self._current_worker.start()

    def _set_capture_controls_enabled(self, enabled: bool):
        for widget in (
            self.btn_capture_pose,
            self.btn_new_session,
            self.btn_connect,
            self.btn_disconnect,
            self.btn_toggle_preview,
            self.btn_burn_bin,
            self.spin_pattern_exp,
            self.spin_board_exp,
            self.spin_cols,
            self.spin_rows,
            self.spin_square,
        ):
            widget.setEnabled(enabled)

    def _on_capture_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.lbl_status.setText(f"状态: {message}")
        self.log(message)

    def _on_capture_finished(self, ok: bool, pose, message: str):
        self._set_capture_controls_enabled(True)
        self.btn_connect.setEnabled(not self.controller.camera.is_opened)
        self.btn_disconnect.setEnabled(self.controller.camera.is_opened)
        self.progress_bar.setValue(100 if ok else 0)
        if getattr(self, "_capture_preview_was_active", False) and self.controller.camera.is_opened:
            self._start_preview()

        if not ok or pose is None:
            self.lbl_status.setText("状态: 当前姿态采集失败")
            self.log(message)
            QMessageBox.critical(self, "采集失败", message)
            return

        pose.name = getattr(self, "_pending_pose_name", "") or f"Pose_{len(self.controller.poses) + 1:02d}"
        self.controller.poses.append(pose)
        self.controller.calibration_result = None
        self._refresh_pose_table()
        self.table_poses.selectRow(len(self.controller.poses) - 1)
        self._on_pose_selection_changed()
        self.tab_analysis.setCurrentIndex(0)
        self.lbl_status.setText(f"状态: 已采集 {pose.name} 的基底图与24张条纹")
        self.log(f"{pose.name}: {message}")
        if pose.corner_valid:
            self._start_auto_phase_decode(pose)
        else:
            self.view_phase.set_mat_image(None)
            self.log(f"{pose.name}: 缺少有效角点，暂不执行自动相位解算")

    def _on_capture_thread_stopped(self):
        worker = self._current_worker
        self._current_worker = None
        if worker is not None:
            worker.deleteLater()
        self._pending_pose_name = ""

    def _start_auto_phase_decode(self, pose):
        self._auto_decode_pose = pose
        self.btn_capture_pose.setEnabled(False)
        self.btn_delete_pose.setEnabled(False)
        self.btn_decode_all.setEnabled(False)
        self.btn_calibrate.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.lbl_status.setText(f"状态: 正在自动解算 {pose.name} 的绝对相位...")
        self.log(f"开始自动解算 {pose.name} 的水平、竖直绝对相位")

        self._auto_decode_worker = DecodePhasesWorker(
            poses=[pose],
            processor=self.controller.phase_processor,
            projector_size=(
                self.controller.config.projector_width,
                self.controller.config.projector_height,
            ),
            roi_margin=self.controller.config.phase_roi_margin,
            result_saver=self.controller.persist_decoded_pose,
        )
        self._auto_decode_worker.finished.connect(self._on_auto_phase_finished)
        self._auto_decode_worker.start()

    def _on_auto_phase_finished(self, ok: bool, message: str):
        pose = self._auto_decode_pose
        worker = self._auto_decode_worker
        if worker is not None:
            worker.wait(1000)
        self._auto_decode_worker = None
        self._auto_decode_pose = None
        if worker is not None:
            worker.deleteLater()

        if pose is None:
            return

        phase_saved = os.path.isfile(self.controller.pose_phase_path(pose))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if pose.horizontal_phase is not None or phase_saved else 0)
        self.btn_capture_pose.setEnabled(self.controller.camera.is_opened)
        self.btn_delete_pose.setEnabled(True)
        self.btn_decode_all.setEnabled(True)
        self.btn_calibrate.setEnabled(True)
        try:
            pose_row = self.controller.poses.index(pose)
        except ValueError:
            pose_row = -1
        self._refresh_pose_table()
        if pose_row >= 0:
            self.table_poses.selectRow(pose_row)
        self._on_pose_selection_changed()
        self.log(message)

        if pose.horizontal_phase is not None or phase_saved:
            self.tab_analysis.setCurrentIndex(1)
            if pose.phase_valid:
                self.lbl_status.setText(f"状态: {pose.name} 绝对相位及DLP坐标解算完成")
            else:
                self.lbl_status.setText(f"状态: {pose.name} 已生成绝对相位，DLP坐标存在无效点")
        else:
            self.lbl_status.setText(f"状态: {pose.name} 绝对相位解算失败")
            QMessageBox.warning(self, "相位解算失败", message)

    def _on_browse_bin(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择DLP标定BIN文件", "", "DLP BIN Files (*.bin);;All Files (*.*)"
        )
        if path:
            self.edit_bin_path.setText(path)
            self.controller.bin_path = path
            self.log(f"已选择BIN文件: {path}")

    def _on_burn_bin(self):
        bin_path = self.edit_bin_path.text().strip()
        if not bin_path or not os.path.isfile(bin_path):
            QMessageBox.warning(self, "警告", "请先选择有效的BIN文件")
            return

        self.lbl_status.setText("正在烧录BIN文件到DLP Flash...")
        self.progress_bar.setRange(0, 0)
        self.log(f"开始烧录BIN到DLP: {bin_path}")

        self.worker = BurnBinWorker(self.controller.dlp, bin_path)
        self.worker.finished.connect(self._on_burn_finished)
        self.worker.start()

    def _on_burn_finished(self, ok: bool, msg: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if ok else 0)
        self.lbl_status.setText(msg)
        self.log(f"BIN烧录结果: {msg}")
        if ok:
            QMessageBox.information(self, "烧录成功", msg)
        else:
            QMessageBox.critical(self, "烧录失败", msg)

    def _on_generate_fringes(self, direction: int):
        dir_name = "水平" if direction == 0 else "竖直"
        out_dir = QFileDialog.getExistingDirectory(self, f"选择保存{dir_name}条纹的目标目录")
        if not out_dir:
            return

        gen = FringeGenerator(
            width=self.controller.config.projector_width,
            height=self.controller.config.projector_height,
            periods=self.controller.config.fringe_periods,
            step_num=self.controller.config.phase_step_count,
        )
        try:
            paths = gen.save_fringes(direction, out_dir)
            self.log(f"已成功生成并在目录导出 12 张{dir_name}条纹图: {out_dir}")
            QMessageBox.information(self, "条纹生成完成", f"已成功保存 12 张条纹到:\n{out_dir}")
        except Exception as e:
            QMessageBox.critical(self, "生成失败", f"错误: {e}")

    def _on_import_pose(self):
        dialog = MultiFolderDialog(
            self,
            caption="选择姿态文件夹 (按住 Ctrl/Shift 可多选，也可直接选择包含多个姿态的根目录)",
            directory="",
        )
        if not dialog.exec():
            return

        chosen_paths = dialog._chosen_dirs or dialog.selectedFolders()
        if not chosen_paths:
            return

        pose_folders = self.controller.find_pose_folders(chosen_paths)
        if not pose_folders:
            QMessageBox.warning(
                self,
                "未找到有效姿态",
                f"在所选路径中未找到包含棋盘格基底图像 (0.bmp 或 board.bmp) 的姿态文件夹。\n所选路径:\n"
                + "\n".join(chosen_paths[:3]),
            )
            return

        old_count = len(self.controller.poses)
        imported_count, poses, messages = self.controller.add_poses_from_folders(pose_folders)

        if imported_count > 0:
            for msg in messages:
                self.log(msg)
            self._refresh_pose_table()
            self.table_poses.selectRow(old_count)
            self._on_pose_selection_changed()

            names = [p.name for p in poses]
            if imported_count == 1:
                self.log(f"成功导入姿态: {names[0]}")
            else:
                summary = f"成功批量导入 {imported_count} 个离线姿态: {', '.join(names[:6])}"
                if imported_count > 6:
                    summary += f" 等共 {imported_count} 个"
                self.log(summary)
                QMessageBox.information(
                    self,
                    "批量导入完成",
                    f"已成功批量导入 {imported_count} 个姿态文件夹！\n\n"
                    f"列表当前总姿态数: {len(self.controller.poses)}\n"
                    f"可在姿态列表中查看，并执行【批量检测角点】与【批量解算相位】。",
                )
        else:
            msg_str = "\n".join(messages[:5])
            QMessageBox.warning(self, "导入完成 (无新增)", f"未新增姿态，可能所选姿态已全部导入过:\n{msg_str}")

    def _on_delete_pose(self):
        row = self.table_poses.currentRow()
        if row < 0:
            return
        self.controller.delete_pose(row)
        self._refresh_pose_table()
        self.log(f"已删除第 {row + 1} 个姿态")

    def _refresh_pose_table(self):
        selected_row = self.table_poses.currentRow()
        self.table_poses.setRowCount(0)
        for i, p in enumerate(self.controller.poses):
            self.table_poses.set_pose_row(
                row=i,
                index_num=i + 1,
                name=p.name,
                corner_ok=p.corner_valid,
                phase_ok=p.phase_valid,
                cam_err=p.camera_view_error,
                proj_err=p.projector_view_error,
            )
        if 0 <= selected_row < self.table_poses.rowCount():
            self.table_poses.selectRow(selected_row)
        geometry_locked = bool(self.controller.poses)
        self.spin_cols.setEnabled(not geometry_locked)
        self.spin_rows.setEnabled(not geometry_locked)
        self.spin_square.setEnabled(not geometry_locked)

    def _on_pose_selection_changed(self):
        row = self.table_poses.currentRow()
        if 0 <= row < len(self.controller.poses):
            pose = self.controller.poses[row]
            board = pose.board_image
            if board is None and pose.folder_path:
                board = cv2.imread(
                    os.path.join(pose.folder_path, "board.bmp"), cv2.IMREAD_GRAYSCALE
                )
            self.view_board.set_mat_image(board, pose.camera_corners)
            phase = pose.horizontal_phase
            phase_path = self.controller.pose_phase_path(pose)
            if phase is None and phase_path and os.path.isfile(phase_path):
                phase = np.load(phase_path, mmap_mode="r")
            phase_badge = ""
            if phase is not None:
                phase_badge = f"水平相位 | {phase.shape[1]}x{phase.shape[0]} | 硬盘按需加载"
            self.view_phase.set_mat_image(phase, badge=phase_badge)

    def _on_board_param_changed(self):
        self.controller.update_board_params(
            cols=self.spin_cols.value(),
            rows=self.spin_rows.value(),
            square_size_mm=self.spin_square.value(),
        )

    def _on_detect_all_corners(self):
        if not self.controller.poses:
            QMessageBox.warning(self, "提示", "请先导入或采集标定姿态")
            return

        self.progress_bar.setRange(0, len(self.controller.poses))
        self.progress_bar.setValue(0)
        self.lbl_status.setText("正在批量检测棋盘格角点...")

        self.worker = DetectCornersWorker(self.controller.poses, self.controller.detector)
        self.worker.progress.connect(lambda cur, tot, name: (
            self.progress_bar.setValue(cur),
            self.lbl_status.setText(name),
        ))
        self.worker.finished.connect(self._on_detect_finished)
        self.worker.start()

    def _on_detect_finished(self, ok: bool, msg: str):
        self.controller.calibration_result = None
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.lbl_status.setText("角点检测完成")
        self.log(msg)
        self._refresh_pose_table()
        self._on_pose_selection_changed()

    def _on_decode_all_phases(self):
        if not self.controller.poses:
            QMessageBox.warning(self, "提示", "请先导入或采集标定姿态")
            return

        self.progress_bar.setRange(0, len(self.controller.poses))
        self.progress_bar.setValue(0)
        self.lbl_status.setText("正在批量解算相位及对应DLP坐标...")

        self.worker = DecodePhasesWorker(
            poses=self.controller.poses,
            processor=self.controller.phase_processor,
            projector_size=(self.controller.config.projector_width, self.controller.config.projector_height),
            roi_margin=self.controller.config.phase_roi_margin,
            result_saver=self.controller.persist_decoded_pose,
        )
        self.worker.progress.connect(lambda cur, tot, name: (
            self.progress_bar.setValue(cur),
            self.lbl_status.setText(name),
        ))
        self.worker.finished.connect(self._on_decode_finished)
        self.worker.start()

    def _on_decode_finished(self, ok: bool, msg: str):
        self.controller.calibration_result = None
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.lbl_status.setText("相位解算完成")
        self.log(msg)
        self._refresh_pose_table()
        self._on_pose_selection_changed()

    def _on_calibrate_system(self):
        views = self.controller.get_valid_views()
        minimum_views = self.controller.config.minimum_view_count
        if len(views) < minimum_views:
            QMessageBox.warning(
                self,
                "标定姿态不足",
                f"有效标定姿态不足 ({len(views)} 组)。\n"
                f"需要至少完成 {minimum_views} 组角点检测和相位解算。",
            )
            return

        self.lbl_status.setText("正在执行系统联合非线性优化标定...")
        self.progress_bar.setRange(0, 0)
        self.log(f"开始使用 {len(views)} 组有效姿态进行联合优化标定...")

        self.worker = CalibrateWorker(self.controller.joint_engine, views)
        self.worker.finished.connect(self._on_calibrate_finished)
        self.worker.start()

    def _on_calibrate_finished(self, ok: bool, res, msg: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if ok else 0)
        self.lbl_status.setText("联合标定结束")
        self.log(msg)

        if not ok:
            QMessageBox.critical(self, "标定失败", msg)
            return

        self.controller.calibration_result = res
        if self.controller.session_output_folder:
            save_ok, save_message = self.controller.finalize_incremental_session()
            self.log(save_message)
            if not save_ok:
                QMessageBox.warning(self, "自动保存标定参数失败", save_message)

        # Update pose errors in table
        for i, name in enumerate(res.view_names):
            for p in self.controller.poses:
                if p.name == name:
                    p.camera_view_error = res.camera_view_rms[i]
                    p.projector_view_error = res.projector_view_rms[i]

        self._refresh_pose_table()

        report = (
            "================ 系统联合标定结果 ================\n"
            f"标定模型: 远心仿射相机 + Scheimpflug 倾斜投影仪\n"
            f"有效位姿数量: {len(res.view_names)}\n\n"
            f"相机综合 RMS 重投影误差: {res.camera_rms:.4f} px\n"
            f"投影仪综合 RMS 重投影误差: {res.projector_rms:.4f} px\n"
            f"系统联合 RMS 重投影误差: {res.joint_rms:.4f} px\n\n"
            f"远心缩放比 X (水平): {res.telecentric_scale_x:.4f} px/mm\n"
            f"远心缩放比 Y (竖直): {res.telecentric_scale_y:.4f} px/mm\n\n"
            f"远心相机 2x4 投影矩阵 (DLP XYZ -> Camera UV):\n{res.telecentric_projection_2x4}\n\n"
            f"远心相机 5 参数畸变 [k1, k2, p1, p2, k3]:\n{res.telecentric_dist_5.flatten()}\n\n"
            f"DLP 投影仪内参矩阵 3x3:\n{res.projector_matrix}\n\n"
            f"DLP 投影仪 14 参数畸变 (含Scheimpflug倾斜角 tauX, tauY):\n{res.projector_dist_14.flatten()}\n"
        )
        if res.warnings:
            report += "\n标定警告:\n" + "\n".join(f"- {w}" for w in res.warnings)

        self.text_result.setText(report)
        QMessageBox.information(
            self,
            "联合标定成功",
            f"系统联合标定成功完成！\n相机 RMS: {res.camera_rms:.4f} px\n投影仪 RMS: {res.projector_rms:.4f} px\n联合 RMS: {res.joint_rms:.4f} px",
        )

    def _on_save_calibration(self):
        if self.controller.calibration_result is None:
            QMessageBox.warning(self, "提示", "尚未完成标定，无可保存参数")
            return

        yml_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存标定YML文件",
            "MonoSLCalibration.yml",
            "YAML Files (*.yml *.yaml);;All Files (*.*)",
        )
        if not yml_path:
            return

        txt_path = os.path.splitext(yml_path)[0] + ".txt"
        ok, msg = self.controller.save_calibration(yml_path, txt_path)
        if ok:
            self.log(f"标定参数已保存到:\n- {yml_path}\n- {txt_path}")
            QMessageBox.information(self, "保存成功", f"标定参数已成功保存！\n{yml_path}")
        else:
            QMessageBox.critical(self, "保存失败", msg)

    def _on_save_complete_session(self):
        if self.controller.session_output_folder:
            ok, message = self.controller.finalize_incremental_session()
            self.log(message)
            if ok:
                QMessageBox.information(self, "保存状态", message)
            else:
                QMessageBox.critical(self, "保存失败", message)
            return
        if not self.controller.poses:
            QMessageBox.warning(self, "提示", "当前没有标定姿态，无法保存完整结果")
            return

        parent = QFileDialog.getExistingDirectory(self, "选择完整标定结果的父目录")
        if not parent:
            return
        default_name = datetime.now().strftime("Calibration_%Y%m%d_%H%M%S")
        folder_name, accepted = QInputDialog.getText(
            self,
            "新建结果文件夹",
            "请输入新文件夹名称（不会覆盖已有目录）:",
            QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not accepted:
            return
        folder_name = folder_name.strip()
        if (
            not folder_name
            or re.search(r'[<>:"/\\|?*]', folder_name)
            or folder_name.endswith((".", " "))
        ):
            QMessageBox.warning(self, "文件夹名称无效", "名称不能为空，且不能包含Windows非法字符或以点/空格结尾")
            return

        output_folder = os.path.join(parent, folder_name)
        ok, message = self.controller.save_complete_session(output_folder)
        self.log(message)
        if ok:
            QMessageBox.information(self, "保存成功", message)
        else:
            QMessageBox.critical(self, "保存失败", message)

    def _on_new_incremental_session(self):
        if self.controller.poses:
            QMessageBox.warning(self, "无法新建", "当前已有姿态，请重新启动软件后再建立新的标定会话")
            return
        self._ensure_incremental_session(force_dialog=True)

    def _ensure_incremental_session(self, force_dialog: bool = False) -> bool:
        if self.controller.session_output_folder and not force_dialog:
            return True
        parent = QFileDialog.getExistingDirectory(self, "选择标定结果的父目录")
        if not parent:
            return False
        default_name = datetime.now().strftime("Calibration_%Y%m%d_%H%M%S")
        folder_name, accepted = QInputDialog.getText(
            self,
            "新建标定会话",
            "请输入新文件夹名称，采集结果将立即写入该目录:",
            QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not accepted:
            return False
        folder_name = folder_name.strip()
        if (
            not folder_name
            or re.search(r'[<>:"/\\|?*]', folder_name)
            or folder_name.endswith((".", " "))
        ):
            QMessageBox.warning(self, "文件夹名称无效", "名称不能为空，且不能包含Windows非法字符或以点/空格结尾")
            return False
        ok, message = self.controller.start_incremental_session(os.path.join(parent, folder_name))
        self.log(message)
        if not ok:
            QMessageBox.critical(self, "新建会话失败", message)
            return False
        self.lbl_status.setText("状态: 标定保存会话已建立")
        return True

    def closeEvent(self, event):
        capture_running = self._current_worker is not None and self._current_worker.isRunning()
        decode_running = self._auto_decode_worker is not None and self._auto_decode_worker.isRunning()
        if capture_running or decode_running:
            QMessageBox.warning(self, "任务尚未结束", "当前采集或相位解算仍在进行，请等待任务结束后再关闭软件。")
            event.ignore()
            return
        self._stop_preview()
        self.controller.camera.close()
        self.controller.dlp.stop_work()
        event.accept()
