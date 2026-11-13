"""
DLP4710 Pattern Binary Building, Inspection, and Hardware Flash Verification Dialog.
Provides a comprehensive UI for:
1. Generating PATN binary files for DLP4710LC / DLPC3479 dual-controller.
2. Inspecting and validating binary pattern files (static byte-level & roundtrip verification).
3. Burning binary to DLP flash and performing multi-stage hardware verification.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dlp_common.hardware.dlp_controller import DLP4710Controller
from dlp_common.hardware.patn_toolkit import (
    build_gray_and_phase_shift_bin,
    build_standard_phase_shift_bin,
    inspect_patn,
    validate_patn,
)
from ..core.workers import BurnMeasurementBinWorker


class PatternBinToolDialog(QDialog):
    """Integrated Pattern Binary Builder, Validator & Burner Dialog for DLP4710."""

    def __init__(self, dlp_controller: DLP4710Controller, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.dlp = dlp_controller
        self._burn_worker: Optional[BurnMeasurementBinWorker] = None
        self._last_generated_bin: Optional[str] = None

        self.setWindowTitle("DLP4710 条纹图案构建、校验与烧录验证工具")
        self.resize(880, 680)
        self.setMinimumSize(780, 560)

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_build = QWidget()
        self.tab_inspect = QWidget()
        self.tab_burn = QWidget()

        self.tab_widget.addTab(self.tab_build, "1. 图案构建 (Pattern Builder)")
        self.tab_widget.addTab(self.tab_inspect, "2. 文件分析与校验 (Validator)")
        self.tab_widget.addTab(self.tab_burn, "3. 烧录与全流程验证 (Flash & Verify)")

        main_layout.addWidget(self.tab_widget)

        self._build_tab_build()
        self._build_tab_inspect()
        self._build_tab_burn()

        # Bottom buttons
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        self.btn_close = QPushButton("关闭")
        self.btn_close.setFixedWidth(100)
        self.btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(self.btn_close)
        main_layout.addLayout(btn_bar)

    # -------------------------------------------------------------------------
    # TAB 1: Pattern Builder
    # -------------------------------------------------------------------------
    def _build_tab_build(self):
        layout = QVBoxLayout(self.tab_build)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Preset selection
        grp_preset = QGroupBox("预设图案模板")
        preset_layout = QHBoxLayout(grp_preset)
        preset_layout.addWidget(QLabel("选择模板:"))
        self.cbo_preset = QComboBox()
        self.cbo_preset.addItems([
            "标准三频四步相移 (70, 64, 59 周期, 12帧, 推荐)",
            "格雷码 + 单频相移 (4位格雷码 + 64周期4步, 12帧)",
            "自定义多频多步相移",
        ])
        self.cbo_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.cbo_preset, stretch=1)
        layout.addWidget(grp_preset)

        # Parameter configuration
        grp_params = QGroupBox("图案序列与时序参数")
        params_layout = QVBoxLayout(grp_params)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("相移频率 (周期数, 逗号分隔):"))
        self.txt_frequencies = QLineEdit("70, 64, 59")
        row1.addWidget(self.txt_frequencies, stretch=1)

        row1.addWidget(QLabel("相移步数:"))
        self.spn_steps = QSpinBox()
        self.spn_steps.setRange(3, 16)
        self.spn_steps.setValue(4)
        row1.addWidget(self.spn_steps)
        params_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("单幅曝光时间 (μs):"))
        self.spn_exposure = QSpinBox()
        self.spn_exposure.setRange(100, 1000000)
        self.spn_exposure.setSingleStep(1000)
        self.spn_exposure.setValue(20000)
        row2.addWidget(self.spn_exposure)

        row2.addWidget(QLabel("暗场间隔时间 (μs):"))
        self.spn_dark = QSpinBox()
        self.spn_dark.setRange(100, 1000000)
        self.spn_dark.setSingleStep(1000)
        self.spn_dark.setValue(5000)
        row2.addWidget(self.spn_dark)

        row2.addWidget(QLabel("光源色彩:"))
        self.cbo_led = QComboBox()
        self.cbo_led.addItems(["blue", "green", "red", "white"])
        row2.addWidget(self.cbo_led)
        params_layout.addLayout(row2)

        layout.addWidget(grp_params)

        # Output Path
        grp_output = QGroupBox("输出文件")
        out_layout = QHBoxLayout(grp_output)
        self.txt_out_bin = QLineEdit()
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "patterns"))
        os.makedirs(default_dir, exist_ok=True)
        default_path = os.path.join(default_dir, "pattern_dlp4710_custom.bin")
        self.txt_out_bin.setText(default_path)
        out_layout.addWidget(self.txt_out_bin, stretch=1)

        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_save_bin)
        out_layout.addWidget(btn_browse)
        layout.addWidget(grp_output)

        # Build button & Jump button
        btn_box = QHBoxLayout()
        self.btn_build_bin = QPushButton("🚀 构建 PATN BIN 文件")
        self.btn_build_bin.setStyleSheet("font-weight: bold; height: 32px; background-color: #007acc; color: white;")
        self.btn_build_bin.clicked.connect(self._on_build_bin)
        btn_box.addWidget(self.btn_build_bin, stretch=2)

        self.btn_jump_burn = QPushButton("👉 前往烧录与验证")
        self.btn_jump_burn.setEnabled(False)
        self.btn_jump_burn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        btn_box.addWidget(self.btn_jump_burn, stretch=1)
        layout.addLayout(btn_box)

        # Build log
        self.txt_build_log = QTextEdit()
        self.txt_build_log.setReadOnly(True)
        self.txt_build_log.setPlaceholderText("构建状态与图案参数报告...")
        layout.addWidget(self.txt_build_log, stretch=1)

    def _on_preset_changed(self, idx: int):
        if idx == 0:
            self.txt_frequencies.setEnabled(True)
            self.txt_frequencies.setText("70, 64, 59")
            self.spn_steps.setEnabled(True)
            self.spn_steps.setValue(4)
        elif idx == 1:
            self.txt_frequencies.setEnabled(True)
            self.txt_frequencies.setText("64")
            self.spn_steps.setEnabled(True)
            self.spn_steps.setValue(4)
        else:
            self.txt_frequencies.setEnabled(True)
            self.spn_steps.setEnabled(True)

    def _browse_save_bin(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存 PATN BIN 文件", self.txt_out_bin.text(), "DLP BIN (*.bin)")
        if path:
            self.txt_out_bin.setText(path)

    def _on_build_bin(self):
        out_path = self.txt_out_bin.text().strip()
        if not out_path:
            QMessageBox.warning(self, "提示", "请指定输出 BIN 文件路径")
            return

        preset = self.cbo_preset.currentIndex()
        exp_us = self.spn_exposure.value()
        dark_us = self.spn_dark.value()
        led = self.cbo_led.currentText()

        self.txt_build_log.append(f"开始构建 PATN 二进制文件: {out_path}")
        try:
            if preset == 1:
                # Gray + Phase shift
                freq_text = self.txt_frequencies.text().strip()
                freqs = [float(f.strip()) for f in freq_text.split(",") if f.strip()]
                steps = self.spn_steps.value()
                bin_bytes, info = build_gray_and_phase_shift_bin(
                    gray_bits=4,
                    frequencies=tuple(freqs),
                    steps=steps,
                    exposure_us=exp_us,
                    dark_us=dark_us,
                    led=led,
                )
            else:
                # Standard or custom phase shift
                freq_text = self.txt_frequencies.text().strip()
                freqs = [float(f.strip()) for f in freq_text.split(",") if f.strip()]
                steps = self.spn_steps.value()
                bin_bytes, info = build_standard_phase_shift_bin(
                    frequencies=tuple(freqs),
                    steps=steps,
                    exposure_us=exp_us,
                    dark_us=dark_us,
                    led=led,
                )

            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(bin_bytes)

            self._last_generated_bin = out_path
            self.txt_inspect_path.setText(out_path)
            self.txt_burn_path.setText(out_path)
            self.btn_jump_burn.setEnabled(True)

            # Auto validate the generated file
            val_ok, issues, report = validate_patn(bin_bytes)

            log_msg = (
                f"✅ 构建成功！\n"
                f"文件大小: {len(bin_bytes):,} 字节\n"
                f"输出路径: {out_path}\n"
                f"包含 Pattern Sets: {len(info)} 个\n"
                f"总条纹图案数: {report.get('total_patterns', 0)} 张\n"
                f"SHA256: {report.get('sha256', '')}\n"
                f"静态回环验证: {'通过' if val_ok else '未通过'}\n"
            )
            self.txt_build_log.append(log_msg)
            QMessageBox.information(self, "成功", f"PATN 文件构建成功！共 {report.get('total_patterns', 0)} 张图案。")
        except Exception as exc:
            self.txt_build_log.append(f"❌ 构建失败: {exc}")
            QMessageBox.critical(self, "构建错误", f"构建 BIN 失败: {exc}")

    # -------------------------------------------------------------------------
    # TAB 2: File Inspect & Validate
    # -------------------------------------------------------------------------
    def _build_tab_inspect(self):
        layout = QVBoxLayout(self.tab_inspect)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # File selection
        file_box = QHBoxLayout()
        file_box.addWidget(QLabel("BIN 文件:"))
        self.txt_inspect_path = QLineEdit()
        file_box.addWidget(self.txt_inspect_path, stretch=1)

        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_inspect_bin)
        file_box.addWidget(btn_browse)

        self.btn_inspect = QPushButton("🔍 校验与解析")
        self.btn_inspect.setStyleSheet("font-weight: bold; background-color: #2b5b84; color: white;")
        self.btn_inspect.clicked.connect(self._on_inspect_file)
        file_box.addWidget(self.btn_inspect)
        layout.addLayout(file_box)

        # Summary Info Label
        self.lbl_inspect_summary = QLabel("请选择并校验 PATN BIN 文件")
        self.lbl_inspect_summary.setStyleSheet("background-color: #252526; padding: 8px; border-radius: 4px;")
        layout.addWidget(self.lbl_inspect_summary)

        # Tables splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Table 1: Pattern Sets
        grp_sets = QGroupBox("图案集列表 (Pattern Sets)")
        layout_sets = QVBoxLayout(grp_sets)
        self.tbl_sets = QTableWidget()
        self.tbl_sets.setColumnCount(5)
        self.tbl_sets.setHorizontalHeaderLabels(["Set 编号", "排布方向", "位深度", "图案数量", "载荷大小 (Bytes)"])
        self.tbl_sets.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout_sets.addWidget(self.tbl_sets)
        splitter.addWidget(grp_sets)

        # Table 2: Pattern Order Table
        grp_order = QGroupBox("播放顺序表 (Pattern Order Table)")
        layout_order = QVBoxLayout(grp_order)
        self.tbl_order = QTableWidget()
        self.tbl_order.setColumnCount(7)
        self.tbl_order.setHorizontalHeaderLabels([
            "条目索引", "Pattern Set", "起始图案", "图案张数", "曝光时间 (μs)", "暗场时间 (μs)", "硬触发2"
        ])
        self.tbl_order.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout_order.addWidget(self.tbl_order)
        splitter.addWidget(grp_order)

        layout.addWidget(splitter, stretch=1)

    def _browse_inspect_bin(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 PATN BIN 文件", "", "DLP BIN (*.bin);;All Files (*.*)")
        if path:
            self.txt_inspect_path.setText(path)

    def _on_inspect_file(self):
        path = self.txt_inspect_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "提示", "请选择有效的 BIN 文件")
            return

        try:
            val_ok, issues, report = validate_patn(path)
            sha = report.get("sha256", "")[:16] + "..."
            bytes_len = report.get("file_bytes", 0)
            sets_cnt = report.get("set_count", 0)
            total_patns = report.get("total_patterns", 0)
            order_cnt = report.get("order_count", 0)

            status_badge = "✅ 校验通过 (Valid PATN)" if val_ok else f"❌ 校验失败: {', '.join(issues)}"
            status_color = "#4CAF50" if val_ok else "#F44336"

            self.lbl_inspect_summary.setText(
                f"<b>状态:</b> <span style='color: {status_color}; font-weight: bold;'>{status_badge}</span> | "
                f"<b>文件大小:</b> {bytes_len:,} 字节 | "
                f"<b>SHA256:</b> {sha} | "
                f"<b>图案集:</b> {sets_cnt} 个 | "
                f"<b>总图案:</b> {total_patns} 张 | "
                f"<b>播放条目:</b> {order_cnt} 条"
            )

            # Fill Pattern Sets Table
            sets_info = report.get("pattern_sets", [])
            self.tbl_sets.setRowCount(len(sets_info))
            for row, s in enumerate(sets_info):
                self.tbl_sets.setItem(row, 0, QTableWidgetItem(f"Set #{s.get('set_index', row)}"))
                self.tbl_sets.setItem(row, 1, QTableWidgetItem(str(s.get("orientation", "horizontal"))))
                self.tbl_sets.setItem(row, 2, QTableWidgetItem(f"{s.get('bit_depth', 8)}-bit"))
                self.tbl_sets.setItem(row, 3, QTableWidgetItem(str(s.get("pattern_count", 0))))
                self.tbl_sets.setItem(row, 4, QTableWidgetItem(f"{s.get('payload_bytes', 0):,}"))

            # Fill Pattern Order Table
            order_entries = report.get("order", [])
            self.tbl_order.setRowCount(len(order_entries))
            for row, e in enumerate(order_entries):
                self.tbl_order.setItem(row, 0, QTableWidgetItem(str(row)))
                self.tbl_order.setItem(row, 1, QTableWidgetItem(f"Set #{e.get('set', 0)}"))
                self.tbl_order.setItem(row, 2, QTableWidgetItem(str(e.get("start", 0))))
                self.tbl_order.setItem(row, 3, QTableWidgetItem(str(e.get("count", 0))))
                self.tbl_order.setItem(row, 4, QTableWidgetItem(str(e.get("exposure_us", 0))))
                self.tbl_order.setItem(row, 5, QTableWidgetItem(str(e.get("dark_time_us", 0))))
                trig_str = "上升沿反转" if e.get("trigger_out_2", 0) == 1 else "高电平维持"
                self.tbl_order.setItem(row, 6, QTableWidgetItem(trig_str))

        except Exception as exc:
            QMessageBox.critical(self, "解析错误", f"解析 BIN 文件异常: {exc}")

    # -------------------------------------------------------------------------
    # TAB 3: Flash & Verify
    # -------------------------------------------------------------------------
    def _build_tab_burn(self):
        layout = QVBoxLayout(self.tab_burn)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # File selection
        file_box = QHBoxLayout()
        file_box.addWidget(QLabel("待烧录 BIN 文件:"))
        self.txt_burn_path = QLineEdit()
        file_box.addWidget(self.txt_burn_path, stretch=1)

        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_burn_bin)
        file_box.addWidget(btn_browse)
        layout.addLayout(file_box)

        # Options
        opt_box = QHBoxLayout()
        self.chk_verify_projection = QCheckBox("执行物理投影空跑校验 (Dry-Run Projection Verification)")
        self.chk_verify_projection.setChecked(True)
        opt_box.addWidget(self.chk_verify_projection)
        opt_box.addStretch()
        layout.addLayout(opt_box)

        # Burn button
        self.btn_burn_verify = QPushButton("🔥 开始烧录与全流程验证")
        self.btn_burn_verify.setStyleSheet(
            "font-size: 14px; font-weight: bold; height: 38px; background-color: #d32f2f; color: white;"
        )
        self.btn_burn_verify.clicked.connect(self._on_start_burn_and_verify)
        layout.addWidget(self.btn_burn_verify)

        # Progress bar
        self.bar_burn = QProgressBar()
        self.bar_burn.setRange(0, 100)
        self.bar_burn.setValue(0)
        self.bar_burn.setTextVisible(True)
        layout.addWidget(self.bar_burn)

        # Status cards
        grp_stages = QGroupBox("多阶段硬件验证清单")
        stages_layout = QGridLayout(grp_stages)
        self.lbl_stage1 = QLabel("⚪ 1. 静态格式与校验和校验 (Static PATN Validation)")
        self.lbl_stage2 = QLabel("⚪ 2. 闪存擦除与固件写入 (Flash Write)")
        self.lbl_stage3 = QLabel("⚪ 3. 图案顺序表与触发配置 (Pattern Order & Trigger Out 2)")
        self.lbl_stage4 = QLabel("⚪ 4. 控制器模式状态回读 (Readback DLPC3479 Mode)")
        self.lbl_stage5 = QLabel("⚪ 5. 投影空跑与硬同步验证 (Dry-Run Projection Test)")

        stages_layout.addWidget(self.lbl_stage1, 0, 0)
        stages_layout.addWidget(self.lbl_stage2, 0, 1)
        stages_layout.addWidget(self.lbl_stage3, 1, 0)
        stages_layout.addWidget(self.lbl_stage4, 1, 1)
        stages_layout.addWidget(self.lbl_stage5, 2, 0, 1, 2)
        layout.addWidget(grp_stages)

        # Detailed Log
        self.txt_burn_log = QTextEdit()
        self.txt_burn_log.setReadOnly(True)
        self.txt_burn_log.setPlaceholderText("烧录与硬件验证详细日志...")
        layout.addWidget(self.txt_burn_log, stretch=1)

    def _browse_burn_bin(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择待烧录 BIN 文件", "", "DLP BIN (*.bin);;All Files (*.*)")
        if path:
            self.txt_burn_path.setText(path)

    def _reset_stage_indicators(self):
        self.lbl_stage1.setText("⚪ 1. 静态格式与校验和校验")
        self.lbl_stage2.setText("⚪ 2. 闪存擦除与固件写入")
        self.lbl_stage3.setText("⚪ 3. 图案顺序表与触发配置")
        self.lbl_stage4.setText("⚪ 4. 控制器模式状态回读")
        self.lbl_stage5.setText("⚪ 5. 投影空跑与硬同步验证")

    def _on_start_burn_and_verify(self):
        bin_path = self.txt_burn_path.text().strip()
        if not bin_path or not os.path.isfile(bin_path):
            QMessageBox.warning(self, "提示", "请指定有效的待烧录 BIN 文件")
            return

        confirm = QMessageBox.question(
            self,
            "确认烧录",
            f"确定要将以下 BIN 文件烧录至 DLP4710 投影仪 Flash 吗？\n{bin_path}\n\n此操作将更新板载图案固件。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.btn_burn_verify.setEnabled(False)
        self.bar_burn.setValue(0)
        self._reset_stage_indicators()
        self.txt_burn_log.clear()
        self.txt_burn_log.append(f"开始烧录流程: {bin_path}")

        verify_proj = self.chk_verify_projection.isChecked()
        self._burn_worker = BurnMeasurementBinWorker(
            dlp=self.dlp,
            bin_path=bin_path,
            verify_projection=verify_proj,
        )
        self._burn_worker.progress.connect(self._on_burn_progress)
        self._burn_worker.verified.connect(self._on_burn_verified)
        self._burn_worker.completed.connect(self._on_burn_completed)
        self._burn_worker.start()

    def _on_burn_progress(self, pct: int, msg: str):
        self.bar_burn.setValue(pct)
        self.txt_burn_log.append(f"[{pct}%] {msg}")

        if pct >= 30:
            self.lbl_stage1.setText("🟢 1. 静态格式与校验和校验 (通过)")
        if pct >= 60:
            self.lbl_stage2.setText("🟢 2. 闪存擦除与固件写入 (完成)")
        if pct >= 80:
            self.lbl_stage3.setText("🟢 3. 图案顺序表与触发配置 (完成)")
        if pct >= 90:
            self.lbl_stage4.setText("🟢 4. 控制器模式状态回读 (正常)")
        if pct >= 98:
            self.lbl_stage5.setText("🟢 5. 投影空跑与硬同步验证 (成功)")

    def _on_burn_verified(self, ok: bool, report: dict, msg: str):
        if not ok:
            stage = report.get("error_stage", "unknown")
            if stage == "pre_burn_validation":
                self.lbl_stage1.setText("🔴 1. 静态格式校验 (失败)")
            elif stage in ("device_connection", "flash_write"):
                self.lbl_stage2.setText("🔴 2. 闪存固件写入 (失败)")
            elif stage == "order_table_setup":
                self.lbl_stage3.setText("🔴 3. 图案顺序表配置 (失败)")
            elif stage == "readback":
                self.lbl_stage4.setText("🔴 4. 控制器回读校验 (失败)")
            elif stage in ("dry_run_start", "dry_run_stop"):
                self.lbl_stage5.setText("🔴 5. 投影空跑验证 (失败)")

    def _on_burn_completed(self, ok: bool, msg: str):
        self.btn_burn_verify.setEnabled(True)
        if ok:
            self.bar_burn.setValue(100)
            self.txt_burn_log.append(f"🎉 烧录与全流程验证成功: {msg}")
            QMessageBox.information(self, "烧录成功", f"DLP4710 固件烧录与硬件验证顺利完成！\n{msg}")
        else:
            self.txt_burn_log.append(f"❌ 烧录流程失败: {msg}")
            QMessageBox.critical(self, "烧录失败", f"烧录与硬件验证未能完成:\n{msg}")
