"""
Controller managing hardware, calibration parameters, and measurement pipeline.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from dlp_common.config import SystemConfig
from dlp_common.fringe.phase_unwrapping import PhaseProcessor
from dlp_common.hardware.daheng_camera import DahengCameraDriver
from dlp_common.hardware.dlp_controller import DLP4710Controller
from dlp_common.reconstruction.point_cloud_io import (
    extract_valid_points,
    keep_largest_image_component,
    save_point_cloud_ply,
    save_point_cloud_txt,
)
from dlp_common.reconstruction.reconstructor import MonoSLReconstructor


class MeasurementController:
    """
    Business logic and state management for 3D measurement application.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config if config is not None else SystemConfig()

        self.reconstructor = MonoSLReconstructor()
        self.camera = DahengCameraDriver(
            width=self.config.camera_width,
            height=self.config.camera_height,
        )
        self.dlp = DLP4710Controller()
        self.phase_processor = PhaseProcessor(
            periods=self.config.fringe_periods,
            step_num=self.config.phase_step_count,
        )

        self.calib_file_path: str = ""
        self.latest_result: Optional[Dict] = None

    def load_calibration(self, yml_path: str) -> Tuple[bool, str]:
        """
        Loads the system calibration parameters from YAML.
        """
        ok = self.reconstructor.load_parameters(yml_path)
        if ok:
            self.calib_file_path = yml_path
            # Update camera and projector specs from calibration
            cw, ch = self.reconstructor.camera_image_size
            self.camera.target_width = cw
            self.camera.target_height = ch
            if not self.camera.is_opened:
                self.camera.actual_width = cw
                self.camera.actual_height = ch
            self.phase_processor.periods = self.reconstructor.fringe_periods
            return True, f"成功加载标定文件: {yml_path}\n相机分辨率: {cw}x{ch}, 投影分辨率: {self.reconstructor.projector_image_size[0]}x{self.reconstructor.projector_image_size[1]}"
        else:
            return False, self.reconstructor.last_error

    def load_offline_frames(self, folder_path: str) -> Tuple[bool, List[np.ndarray], str]:
        """
        Loads 12 horizontal fringe frames from an offline directory.
        """
        if not os.path.isdir(folder_path):
            return False, [], f"目录不存在: {folder_path}"

        frames = []
        for i in range(1, 13):
            candidates = [
                os.path.join(folder_path, f"H_Fringe_{i}.bmp"),
                os.path.join(folder_path, f"Fringe_{i}.bmp"),
                os.path.join(folder_path, f"{i}.bmp"),
            ]
            found = False
            for c in candidates:
                if os.path.isfile(c):
                    img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        frames.append(img)
                        found = True
                        break
            if not found:
                break

        if len(frames) != 12:
            return False, [], f"在目录中未集齐12张条纹图像 (需 1..12 或 H_Fringe_1..12)，当前找到 {len(frames)} 张"

        return True, frames, f"成功载入12张条纹图像: {folder_path}"

    def load_offline_multi_exposure_frames(
        self, folder_path: str
    ) -> Tuple[bool, List[List[np.ndarray]], str]:
        """
        Loads multi-exposure fringe sequences from a folder structure.
        Supports:
          - Subdirectories bracket_1, bracket_2, ... or exp_1, exp_2, ... each containing 12 BMPs.
          - 01_fringes/ containing bracket subdirectories.
          - Fallback to a single directory with 12 BMPs as a 1-bracket sequence.
        """
        if not os.path.isdir(folder_path):
            return False, [], f"目录不存在: {folder_path}"

        search_root = folder_path
        fringe_sub = os.path.join(folder_path, "01_fringes")
        if os.path.isdir(fringe_sub):
            search_root = fringe_sub

        subdirs = [
            os.path.join(search_root, d)
            for d in sorted(os.listdir(search_root))
            if os.path.isdir(os.path.join(search_root, d))
        ]

        brackets = []
        for sdir in subdirs:
            ok, frames, _ = self.load_offline_frames(sdir)
            if ok and len(frames) == 12:
                brackets.append(frames)

        if len(brackets) >= 2:
            return True, brackets, f"成功载入 {len(brackets)} 档多曝光条纹序列 (共 {len(brackets) * 12} 帧)"

        # Fallback to single 12-frame folder
        ok, frames, msg = self.load_offline_frames(folder_path)
        if ok and len(frames) == 12:
            return True, [frames], f"成功载入单档条纹序列 (12帧): {folder_path}"

        return False, [], f"未在目录或子目录中找到多曝光条纹序列 (每档需12张条纹图像): {folder_path}"

    def disconnect_devices(self) -> Tuple[bool, str]:
        """Release the Daheng camera and stop/disconnect the DLP controller."""
        camera_was_open = self.camera.is_opened
        dlp_was_connected = self.dlp.is_connected
        try:
            self.camera.close()
        except Exception as exc:
            return False, f"关闭大恒相机失败: {exc}"

        dlp_ok = self.dlp.disconnect()
        if not dlp_ok:
            return False, self.dlp.last_error or "停止并断开DLP失败"

        if not camera_was_open and not dlp_was_connected:
            return True, "相机和DLP当前均未连接"
        return True, "大恒相机和DLP已断开"

    def keep_largest_subject(self, close_radius: int = 2) -> Tuple[bool, str]:
        if self.latest_result is None or "xyz_map" not in self.latest_result:
            return False, "当前尚无测量点云，请先执行三维重构"

        result = self.latest_result
        source_xyz = result.get("original_xyz_map", result["xyz_map"])
        try:
            filtered, subject_mask, original_count, retained_count = (
                keep_largest_image_component(source_xyz, close_radius=close_radius)
            )
        except Exception as exc:
            return False, f"点云主体提取失败: {exc}"

        if "original_xyz_map" not in result:
            result["original_xyz_map"] = source_xyz
            result["original_valid_points"] = result.get("valid_points")

        valid_points, _ = extract_valid_points(filtered)
        result["xyz_map"] = filtered
        result["valid_points"] = valid_points
        result["point_count"] = retained_count
        result["subject_mask"] = subject_mask
        result["point_cloud_processed"] = True
        removed_count = original_count - retained_count
        return (
            True,
            f"主体提取完成：保留 {retained_count:,} 点，去除 {removed_count:,} 个杂点",
        )

    def restore_original_point_cloud(self) -> Tuple[bool, str]:
        if self.latest_result is None:
            return False, "当前尚无测量点云"
        result = self.latest_result
        original_xyz = result.get("original_xyz_map")
        if original_xyz is None:
            return False, "当前点云尚未执行主体提取，无需恢复"

        original_points = result.get("original_valid_points")
        if original_points is None:
            original_points, _ = extract_valid_points(original_xyz)
        result["xyz_map"] = original_xyz
        result["valid_points"] = original_points
        result["point_count"] = len(original_points)
        result.pop("subject_mask", None)
        result["point_cloud_processed"] = False
        return True, f"已恢复原始点云，共 {len(original_points):,} 点"

    def save_ply(self, file_path: str, binary: bool = True) -> Tuple[bool, str]:
        if self.latest_result is None or "xyz_map" not in self.latest_result:
            return False, "当前尚无测量结果，无法导出"

        xyz_map = self.latest_result["xyz_map"]
        ok, count, msg = save_point_cloud_ply(file_path, xyz_map, binary=binary)
        return ok, msg

    def save_txt(self, file_path: str) -> Tuple[bool, str]:
        if self.latest_result is None or "xyz_map" not in self.latest_result:
            return False, "当前尚无测量结果，无法导出"

        xyz_map = self.latest_result["xyz_map"]
        ok, count, msg = save_point_cloud_txt(file_path, xyz_map)
        return ok, msg

    @staticmethod
    def _write_image(file_path: str, image: np.ndarray):
        if not cv2.imwrite(file_path, image):
            raise OSError(f"图像写入失败: {file_path}")

    @staticmethod
    def _phase_preview(phase: np.ndarray) -> np.ndarray:
        values = np.asarray(phase, dtype=np.float32)
        finite = np.isfinite(values)
        normalized = np.zeros(values.shape, dtype=np.uint8)
        if np.any(finite):
            low = float(np.min(values[finite]))
            high = float(np.max(values[finite]))
            if high > low:
                normalized[finite] = np.clip(
                    (values[finite] - low) * 255.0 / (high - low),
                    0,
                    255,
                ).astype(np.uint8)
        preview = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        preview[~finite] = 0
        return preview

    def _save_float_result(self, folder: str, name: str, values: np.ndarray):
        float_values = np.asarray(values, dtype=np.float32)
        np.save(os.path.join(folder, f"{name}.npy"), float_values)
        self._write_image(os.path.join(folder, f"{name}.tiff"), float_values)
        self._write_image(
            os.path.join(folder, f"{name}_preview.png"),
            self._phase_preview(float_values),
        )

    def save_intermediate_results(self, output_folder: str) -> Tuple[bool, str]:
        """
        Fast export of measurement data:
        Only saves raw captured fringes (BMP), optional gray frame (BMP),
        complete capture parameters (JSON + TXT), and the final raw point cloud (Binary PLY).
        Skips all intermediate wrapped/unwrapped phase/modulation files and duplicate TXT points.
        """
        result = self.latest_result
        if result is None:
            return False, "当前尚无测量结果，无法保存数据"

        if "frames" not in result or "xyz_map" not in result:
            return False, "当前测量结果不完整，缺少原始条纹或点云数据"
        if len(result["frames"]) != 12:
            return False, f"条纹图像应为12帧，当前为 {len(result['frames'])} 帧"
        if os.path.exists(output_folder):
            return False, f"目标文件夹已存在，请使用新的文件夹名称: {output_folder}"

        parent = os.path.dirname(os.path.abspath(output_folder))
        if not os.path.isdir(parent):
            return False, f"父目录不存在: {parent}"

        created = False
        try:
            os.mkdir(output_folder)
            created = True
            fringe_dir = os.path.join(output_folder, "01_fringes")
            point_cloud_dir = os.path.join(output_folder, "02_point_cloud")
            os.mkdir(fringe_dir)
            os.mkdir(point_cloud_dir)

            # 1. Save raw fringe frames (BMP, uncompressed for fastest disk I/O)
            for index, frame in enumerate(result["frames"], start=1):
                self._write_image(
                    os.path.join(fringe_dir, f"H_Fringe_{index}.bmp"),
                    np.asarray(frame),
                )

            # If multi-exposure, save all individual bracket fringe sequences
            if result.get("is_multi_exposure") and "brackets_frames" in result:
                for b_idx, b_frames in enumerate(result["brackets_frames"], start=1):
                    b_fringe_dir = os.path.join(fringe_dir, f"bracket_{b_idx}")
                    os.mkdir(b_fringe_dir)
                    for f_idx, f_img in enumerate(b_frames, start=1):
                        self._write_image(
                            os.path.join(b_fringe_dir, f"H_Fringe_{f_idx}.bmp"),
                            np.asarray(f_img),
                        )

            # 2. Save gray frame if present
            if result.get("gray_frame") is not None:
                self._write_image(
                    os.path.join(output_folder, "gray_frame.bmp"),
                    np.asarray(result["gray_frame"]),
                )

            # 3. Save final raw point cloud in high-speed Binary PLY format
            original_xyz = result.get("original_xyz_map", result["xyz_map"])
            ply_path = os.path.join(point_cloud_dir, "original_cloud.ply")
            ply_ok, ply_count, ply_message = save_point_cloud_ply(
                ply_path,
                original_xyz,
                binary=True,
            )
            if not ply_ok:
                raise OSError(ply_message)

            # If user processed cloud (e.g. extracted subject), also save processed_cloud.ply
            if result.get("point_cloud_processed"):
                proc_ply_path = os.path.join(point_cloud_dir, "processed_cloud.ply")
                save_point_cloud_ply(proc_ply_path, result["xyz_map"], binary=True)

            # 4. Save structured capture & measurement parameters JSON
            valid_pts, _ = extract_valid_points(original_xyz)
            bounds = {}
            if len(valid_pts) > 0:
                min_pt = np.min(valid_pts, axis=0)
                max_pt = np.max(valid_pts, axis=0)
                extent = max_pt - min_pt
                bounds = {
                    "min_x": float(min_pt[0]), "max_x": float(max_pt[0]), "extent_x": float(extent[0]),
                    "min_y": float(min_pt[1]), "max_y": float(max_pt[1]), "extent_y": float(extent[1]),
                    "min_z": float(min_pt[2]), "max_z": float(max_pt[2]), "extent_z": float(extent[2]),
                }

            params_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "measurement_mode": "多曝光 (HDR) 融合测量" if result.get("is_multi_exposure") else "单曝光标准测量",
                "calibration_file": self.calib_file_path or "未记录",
                "point_cloud": {
                    "total_points": ply_count,
                    "reconstruction_elapsed_seconds": float(result.get("elapsed_seconds", 0.0)),
                    "bounds_mm": bounds,
                    "format": "binary_little_endian 1.0 (PLY)",
                    "file_path": "02_point_cloud/original_cloud.ply",
                },
                "camera": {
                    "exposure_us": float(getattr(self.config, "calibration_pattern_exposure_us", 3000.0)),
                    "gain_db": float(getattr(self.config, "default_gain_db", 0.0)),
                    "gray_exposure_us": float(getattr(self.config, "board_exposure_us", 150000.0)),
                },
                "dlp": {
                    "brightness_percent": float(self.dlp.brightness_percent),
                    "color_mode": self.dlp.color_mode,
                    "led_currents": list(self.dlp.led_currents),
                    "led_enables": self.dlp.led_enables,
                    "control_method": "manual" if self.dlp.led_control_method == 0 else "auto",
                },
            }
            if result.get("is_multi_exposure"):
                params_data["multi_exposure"] = {
                    "bracket_count": len(result.get("brackets_frames", [])),
                    "brackets_info": result.get("brackets_info", []),
                    "bracket_distribution": result.get("bracket_counts", {}),
                }

            with open(os.path.join(output_folder, "capture_parameters.json"), "w", encoding="utf-8") as jf:
                json.dump(params_data, jf, ensure_ascii=False, indent=2)

            # 5. Human-readable measurement_info.txt
            info_lines = [
                "DLP4710 单目结构光三维测量数据",
                f"保存时间: {params_data['timestamp']}",
                f"测量模式: {params_data['measurement_mode']}",
                f"条纹帧数: {len(result['frames'])} 帧",
                f"有效三维点数: {ply_count:,} 点",
                f"重构算法耗时: {float(result.get('elapsed_seconds', 0.0)):.3f} 秒",
                f"标定文件: {self.calib_file_path or '未记录'}",
                f"点云格式: 二进制 PLY (高速格式，可用 MeshLab / CloudCompare 打开或一键转 TXT)",
                f"点云路径: 02_point_cloud/original_cloud.ply",
                f"条纹路径: 01_fringes/ (H_Fringe_01.bmp ~ H_Fringe_12.bmp)",
            ]
            if bounds:
                info_lines.append(f"点云跨度: X={bounds['extent_x']:.2f} mm, Y={bounds['extent_y']:.2f} mm, Z={bounds['extent_z']:.2f} mm")
            if result.get("gray_frame") is not None:
                info_lines.append("灰度图像: gray_frame.bmp")
            if result.get("is_multi_exposure"):
                num_b = len(result.get("brackets_frames", []))
                info_lines.append(f"多曝光档位数: {num_b}")
                if "bracket_counts" in result:
                    info_lines.append("各曝光档位贡献点数:")
                    tot_pts = sum(result["bracket_counts"].values()) or 1
                    for b_i, cnt in result["bracket_counts"].items():
                        info_lines.append(f"  - 档位 {b_i + 1}: {cnt:,} 点 ({cnt * 100.0 / tot_pts:.1f}%)")

            with open(
                os.path.join(output_folder, "measurement_info.txt"),
                "w",
                encoding="utf-8",
            ) as info_file:
                info_file.write("\n".join(info_lines) + "\n")

        except Exception as exc:
            if created:
                shutil.rmtree(output_folder, ignore_errors=True)
            return False, f"保存测量数据失败: {exc}"

        return True, f"测量数据已极速保存 (含条纹/灰度/参数/二进制PLY点云): {output_folder}"

