"""
Main Window for DLP4710 Mono Structured Light Calibration Software.
Features full real-time live camera feed, device selection, parameter controls,
and synchronized 24-frame acquisition with corner/phase previews.
"""

from datetime import datetime
import os
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
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
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
    DecodePhasesWorker,
    DetectCornersWorker,
)
from .styles import MODERN_STYLE
from .widgets import ImageDisplayWidget, PoseTableWidget


class CalibrationMainWindow(QMainWindow):
    """
    Main application window for system calibration.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("DLP4710 单目结构光标定软件 (Python 工业工程版)")
        self.resize(1520, 960)

        self.controller = CalibrationController()
        self._current_worker = None

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
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---------------- Left Control Panel (~430px) ----------------
        left_panel = QWidget()
        left_panel.setFixedWidth(430)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

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
        self.spin_pattern_exp.setValue(self.controller.config.default_exposure_us)
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
        self.btn_import_pose = QPushButton("导入离线姿态")
        self.btn_delete_pose = QPushButton("删除选中姿态")

        lay_poses.addWidget(self.btn_capture_pose, 0, 0, 1, 2)
        lay_poses.addWidget(self.btn_import_pose, 1, 0)
        lay_poses.addWidget(self.btn_delete_pose, 1, 1)

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

        lay_calib.addWidget(self.btn_detect_all, 2, 0, 1, 2)
        lay_calib.addWidget(self.btn_decode_all, 2, 2, 1, 2)
        lay_calib.addWidget(self.btn_calibrate, 3, 0, 1, 4)
        lay_calib.addWidget(self.btn_save_calib, 4, 0, 1, 4)

        left_layout.addWidget(grp_calib)

        # Progress bar & status
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("状态: 待机")
        self.lbl_status.setObjectName("statusLabel")
        left_layout.addWidget(self.lbl_status)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # ---------------- Right Display Area ----------------
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top area: Pose Table + Image previews
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table widget for poses
        self.table_poses = PoseTableWidget()
        top_splitter.addWidget(self.table_poses)

        # Image view container: Live Camera View + Captured / Phase Analysis Tabs
        img_container = QWidget()
        lay_imgs = QHBoxLayout(img_container)
        lay_imgs.setContentsMargins(0, 0, 0, 0)
        lay_imgs.setSpacing(6)

        # 1. Dedicated Live Camera View
        self.view_live = ImageDisplayWidget("📸 相机实时画面 / 实时预览 (Live Camera)")
        self.view_live.show_crosshair = True
        lay_imgs.addWidget(self.view_live, 5)

        # 2. Captured Board & Phase Tabs
        self.tab_analysis = QTabWidget()
        self.view_board = ImageDisplayWidget("🏁 棋盘格角点检测视图 (Corners View)")
        self.view_phase = ImageDisplayWidget("🌊 绝对相位展开视图 (Phase View)")
        self.tab_analysis.addTab(self.view_board, "标定板角点标注")
        self.tab_analysis.addTab(self.view_phase, "绝对展开相位")
        lay_imgs.addWidget(self.tab_analysis, 5)

        top_splitter.addWidget(img_container)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 7)

        right_splitter.addWidget(top_splitter)

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
        right_splitter.setStretchFactor(0, 7)
        right_splitter.setStretchFactor(1, 3)

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
        self.btn_import_pose.clicked.connect(self._on_import_pose)
        self.btn_delete_pose.clicked.connect(self._on_delete_pose)
        self.table_poses.itemSelectionChanged.connect(self._on_pose_selection_changed)

        self.btn_detect_all.clicked.connect(self._on_detect_all_corners)
        self.btn_decode_all.clicked.connect(self._on_decode_all_phases)
        self.btn_calibrate.clicked.connect(self._on_calibrate_system)
        self.btn_save_calib.clicked.connect(self._on_save_calibration)

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
        self.log(f"已扫描可用视频设备，发现 {len(devices)} 个输入通道")

    def _on_crosshair_toggled(self, checked: bool):
        self.view_live.show_crosshair = checked
        self.view_live.update()

    def _on_connect_devices(self):
        # Open selected camera
        dev_idx = self.combo_camera.currentData()
        if dev_idx is None:
            dev_idx = 0

        ok_cam = self.controller.camera.open(dev_idx)
        ok_dlp = self.controller.dlp.connect()

        if ok_cam:
            self.controller.camera.start_grabbing()
            self._start_preview()
            res = self.controller.camera.resolution
            self.log(f"相机连接成功，工作分辨率: {res[0]}x{res[1]}，模式: {'模拟仿真' if self.controller.camera.is_mock else '物理硬件'}")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.lbl_status.setText("状态: 相机采集中 (实时预览)")
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
            mode_tag = "MOCK" if self.controller.camera.is_mock else "LIVE"
            badge = f"● {mode_tag} | {self._current_fps:.1f} FPS | {res[1]}x{res[0]}"
            self.view_live.set_mat_image(frame, badge=badge)

    def _on_exposure_changed(self):
        exp = self.spin_pattern_exp.value()
        self.controller.camera.set_exposure(exp)
        self.log(f"已更新相机曝光: {exp:.0f} μs")

    def _on_capture_live_pose(self):
        """
        Captures the current live board frame, triggers DLP sequence, and creates a calibration pose.
        """
        if not self.controller.camera.is_opened:
            QMessageBox.warning(self, "提示", "请先连接相机并开启预览")
            return

        preview_was_active = self._preview_timer.isActive()
        self._stop_preview()

        board_frame = None
        horizontal_frames = []
        vertical_frames = []
        try:
            self.controller.camera.set_trigger_mode(False)
            ok, board_frame = self.controller.camera.grab_frame()
            if not ok or board_frame is None:
                QMessageBox.critical(self, "采集失败", "无法读取相机当前基底图像")
                return

            self.lbl_status.setText("正在采集24张硬触发标定条纹...")
            self.controller.camera.stop_grabbing()

            if self.controller.camera.is_mock:
                generator = FringeGenerator(
                    width=board_frame.shape[1],
                    height=board_frame.shape[0],
                    periods=self.controller.config.fringe_periods,
                    step_num=self.controller.config.phase_step_count,
                )
                horizontal_frames = generator.generate_directional_fringes(0)
                vertical_frames = generator.generate_directional_fringes(1)
            else:
                if not self.controller.dlp.is_connected and not self.controller.dlp.connect():
                    QMessageBox.critical(self, "采集失败", self.controller.dlp.last_error)
                    return
                if not self.controller.camera.set_trigger_mode(True, source="Line2"):
                    QMessageBox.critical(self, "采集失败", self.controller.camera.last_error)
                    return
                if not self.controller.camera.start_grabbing():
                    QMessageBox.critical(self, "采集失败", self.controller.camera.last_error)
                    return

                time.sleep(0.02)
                self.controller.dlp.start_work(0)
                ok, pattern_frames = self.controller.camera.grab_sequence(24, timeout_ms=5000)
                if not ok or len(pattern_frames) != 24:
                    QMessageBox.critical(
                        self,
                        "采集失败",
                        f"硬触发条纹图像不足24张，实际收到 {len(pattern_frames)} 张",
                    )
                    return
                horizontal_frames = pattern_frames[:12]
                vertical_frames = pattern_frames[12:]
        finally:
            self.controller.camera.stop_grabbing()
            self.controller.camera.set_trigger_mode(False)
            self.controller.camera.start_grabbing()
            if preview_was_active:
                self._start_preview()

        if board_frame is None or len(horizontal_frames) != 12 or len(vertical_frames) != 12:
            return

        if board_frame.ndim == 3:
            board_gray = cv2.cvtColor(board_frame, cv2.COLOR_BGR2GRAY)
        else:
            board_gray = board_frame

        pose_index = len(self.controller.poses) + 1
        pose_name = f"Pose_{pose_index:02d}"

        from dlp_common.calibration.models import CalibrationPose
        pose = CalibrationPose(
            name=pose_name,
            board_image=board_gray,
            camera_image_size=(board_gray.shape[1], board_gray.shape[0]),
            horizontal_frames=horizontal_frames,
            vertical_frames=vertical_frames,
        )

        # Detect corners immediately on the captured frame
        ok_corner, corners, msg = self.controller.detector.detect_corners(board_gray)
        if ok_corner and corners is not None:
            pose.camera_corners = corners
            pose.corner_valid = True
            self.log(f"姿态 {pose_name} 基底角点检测成功 ({len(corners)} 点)")
        else:
            pose.corner_valid = False
            self.log(f"姿态 {pose_name} 未检测到角点: {msg}")

        self.controller.poses.append(pose)
        self._refresh_pose_table()

        # Select the newly added row
        self.table_poses.selectRow(len(self.controller.poses) - 1)
        self._on_pose_selection_changed()
        self.tab_analysis.setCurrentIndex(0)
        self.lbl_status.setText(f"状态: 已采集 {pose_name} 的基底图与24张条纹")
        self.log(f"已成功捕获姿态 {pose_name}（1张基底图 + 24张条纹图）")

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
        folder = QFileDialog.getExistingDirectory(self, "选择姿态图像目录 (包含0.bmp和条纹图像)")
        if not folder:
            return

        ok, pose, msg = self.controller.add_pose_from_folder(folder)
        if ok and pose is not None:
            self.log(msg)
            self._refresh_pose_table()
            self.table_poses.selectRow(len(self.controller.poses) - 1)
            self._on_pose_selection_changed()
        else:
            QMessageBox.warning(self, "导入失败", msg)

    def _on_delete_pose(self):
        row = self.table_poses.currentRow()
        if row < 0:
            return
        self.controller.delete_pose(row)
        self._refresh_pose_table()
        self.log(f"已删除第 {row + 1} 个姿态")

    def _refresh_pose_table(self):
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

    def _on_pose_selection_changed(self):
        row = self.table_poses.currentRow()
        if 0 <= row < len(self.controller.poses):
            pose = self.controller.poses[row]
            self.view_board.set_mat_image(pose.board_image, pose.camera_corners)
            self.view_phase.set_mat_image(pose.horizontal_phase)

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
        )
        self.worker.progress.connect(lambda cur, tot, name: (
            self.progress_bar.setValue(cur),
            self.lbl_status.setText(name),
        ))
        self.worker.finished.connect(self._on_decode_finished)
        self.worker.start()

    def _on_decode_finished(self, ok: bool, msg: str):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.lbl_status.setText("相位解算完成")
        self.log(msg)
        self._refresh_pose_table()
        self._on_pose_selection_changed()

    def _on_calibrate_system(self):
        views = self.controller.get_valid_views()
        if len(views) < 3:
            QMessageBox.warning(
                self,
                "标定姿态不足",
                f"有效标定姿态不足 ({len(views)} 组)。\n需要至少完成 3 组（推荐 8 组以上）角点检测和相位解算。",
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

    def closeEvent(self, event):
        self._stop_preview()
        self.controller.camera.close()
        self.controller.dlp.stop_work()
        event.accept()
