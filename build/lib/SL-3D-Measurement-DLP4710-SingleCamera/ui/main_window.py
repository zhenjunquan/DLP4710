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

from ..core.controller import MeasurementController
from ..core.workers import LivePreviewWorker, MeasureWorker
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
        self.resize(1500, 960)

        self.controller = MeasurementController()
        self.preview_worker: Optional[LivePreviewWorker] = None
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
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left control panel (width ~ 430px)
        left_panel = QWidget()
        left_panel.setFixedWidth(430)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

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
        left_layout.addWidget(grp_calib)

        # 2. Camera & Preview Group
        grp_camera = QGroupBox("2. 相机与实时预览")
        lay_cam = QGridLayout(grp_camera)

        lay_cam.addWidget(QLabel("曝光时间 (μs):"), 0, 0)
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(100.0, 1000000.0)
        self.spin_exposure.setValue(self.controller.config.default_exposure_us)
        lay_cam.addWidget(self.spin_exposure, 0, 1)

        lay_cam.addWidget(QLabel("增益 (dB):"), 1, 0)
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 24.0)
        self.spin_gain.setValue(self.controller.config.default_gain_db)
        lay_cam.addWidget(self.spin_gain, 1, 1)

        self.btn_toggle_preview = QPushButton("开启实时预览")
        lay_cam.addWidget(self.btn_toggle_preview, 2, 0, 1, 2)
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

        # 5. Export Group
        grp_export = QGroupBox("5. 点云导出")
        lay_exp = QGridLayout(grp_export)
        self.btn_save_ply = QPushButton("导出标准 PLY 点云")
        self.btn_save_ply.setObjectName("saveBtn")
        self.btn_save_txt = QPushButton("导出 TXT (X Y Z) 坐标")

        lay_exp.addWidget(self.btn_save_ply, 0, 0)
        lay_exp.addWidget(self.btn_save_txt, 1, 0)
        left_layout.addWidget(grp_export)

        # Progress & Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("就绪")
        left_layout.addWidget(self.lbl_status)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Right display area (Tabs: 3D Point Cloud, 2D Process Images)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.tab_widget = QTabWidget()
        self.view_3d = GLPointCloudWidget()
        self.view_2d = MultiImageViewerWidget()

        self.tab_widget.addTab(self.view_3d, "🌐 交互式三维点云视图 (3D Cloud View)")
        self.tab_widget.addTab(self.view_2d, "🖼️ 二维过程视图 (2D Image View)")
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
        self.btn_toggle_preview.clicked.connect(self._on_toggle_preview)

        self.btn_measure_online.clicked.connect(self._on_measure_online)
        self.btn_measure_offline.clicked.connect(self._on_measure_offline)

        self.btn_save_ply.clicked.connect(self._on_save_ply)
        self.btn_save_txt.clicked.connect(self._on_save_txt)

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
            self.controller.camera.open()
            self.controller.camera.start_grabbing()
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

        self.lbl_status.setText("正在执行在线同步条纹采集...")
        self.progress_bar.setRange(0, 0)
        self.log("启动在线条纹采集 (12帧)...")

        if not self.controller.camera.is_opened:
            self.controller.camera.open()

        if self.controller.camera.is_mock:
            ok, frames = self.controller.camera.grab_sequence(12)
        else:
            if not self.controller.dlp.is_connected and not self.controller.dlp.connect():
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                QMessageBox.critical(self, "DLP连接错误", self.controller.dlp.last_error)
                return

            frames = []
            ok = False
            try:
                self.controller.camera.stop_grabbing()
                if not self.controller.camera.set_trigger_mode(True, source="Line2"):
                    raise RuntimeError(self.controller.camera.last_error or "无法启用Line2硬触发")
                if not self.controller.camera.start_grabbing():
                    raise RuntimeError(self.controller.camera.last_error or "无法启动相机采集")

                self.controller.dlp.start_work(0)
                ok, frames = self.controller.camera.grab_sequence(12, timeout_ms=5000)
            except Exception as exc:
                QMessageBox.critical(self, "同步采集错误", str(exc))
            finally:
                self.controller.camera.stop_grabbing()
                self.controller.camera.set_trigger_mode(False)

        if not ok or len(frames) != 12:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "采集错误", "相机条纹序列采集失败")
            return

        self._start_reconstruction(frames)

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

    def _start_reconstruction(self, frames: List[np.ndarray]):
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
        )
        self.measure_worker.progress.connect(lambda p, m: (
            self.progress_bar.setValue(p),
            self.lbl_status.setText(m),
            self.log(m),
        ))
        self.measure_worker.finished.connect(self._on_measure_finished)
        self.measure_worker.start()

    def _on_measure_finished(self, ok: bool, result: dict, msg: str):
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
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.stop()
        self.controller.camera.close()
        self.controller.dlp.stop_work()
        event.accept()

