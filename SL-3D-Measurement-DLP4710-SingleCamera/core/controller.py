"""
Controller managing hardware, calibration parameters, and measurement pipeline.
"""

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

    def save_ply(self, file_path: str) -> Tuple[bool, str]:
        if self.latest_result is None or "xyz_map" not in self.latest_result:
            return False, "当前尚无测量结果，无法导出"

        xyz_map = self.latest_result["xyz_map"]
        ok, count, msg = save_point_cloud_ply(file_path, xyz_map)
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
        """Save the complete measurement package, including PLY and TXT clouds."""
        result = self.latest_result
        if result is None:
            return False, "当前尚无测量结果，无法保存中间结果"

        required = (
            "frames",
            "wrapped_phases",
            "unwrapped_phase",
            "modulation",
            "mask",
            "xyz_map",
        )
        missing = [key for key in required if key not in result]
        if missing:
            return False, f"当前测量结果不完整，缺少: {', '.join(missing)}"
        if len(result["frames"]) != 12:
            return False, f"条纹图像应为12帧，当前为 {len(result['frames'])} 帧"
        if len(result["wrapped_phases"]) != 3:
            return False, f"包裹相位应为3组，当前为 {len(result['wrapped_phases'])} 组"
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
            wrapped_dir = os.path.join(output_folder, "02_wrapped_phase")
            unwrapped_dir = os.path.join(output_folder, "03_unwrapped_phase")
            auxiliary_dir = os.path.join(output_folder, "04_auxiliary")
            point_cloud_dir = os.path.join(output_folder, "05_point_cloud")
            for folder in (
                fringe_dir,
                wrapped_dir,
                unwrapped_dir,
                auxiliary_dir,
                point_cloud_dir,
            ):
                os.mkdir(folder)

            for index, frame in enumerate(result["frames"], start=1):
                self._write_image(
                    os.path.join(fringe_dir, f"H_Fringe_{index}.bmp"),
                    np.asarray(frame),
                )

            for index, phase in enumerate(result["wrapped_phases"], start=1):
                self._save_float_result(
                    wrapped_dir,
                    f"wrapped_frequency_{index:02d}",
                    phase,
                )

            self._save_float_result(
                unwrapped_dir,
                "unwrapped_phase",
                result["unwrapped_phase"],
            )
            self._save_float_result(auxiliary_dir, "modulation", result["modulation"])
            mask = (np.asarray(result["mask"]) > 0).astype(np.uint8) * 255
            self._write_image(os.path.join(auxiliary_dir, "mask.png"), mask)
            if result.get("gray_frame") is not None:
                self._write_image(
                    os.path.join(auxiliary_dir, "gray_frame.bmp"),
                    np.asarray(result["gray_frame"]),
                )

            def save_cloud_pair(prefix: str, xyz_values: np.ndarray) -> int:
                ply_path = os.path.join(point_cloud_dir, f"{prefix}_cloud.ply")
                ply_ok, ply_count, ply_message = save_point_cloud_ply(
                    ply_path,
                    xyz_values,
                )
                if not ply_ok:
                    raise OSError(ply_message)

                txt_path = os.path.join(point_cloud_dir, f"{prefix}_points.txt")
                txt_ok, txt_count, txt_message = save_point_cloud_txt(
                    txt_path,
                    xyz_values,
                )
                if not txt_ok:
                    raise OSError(txt_message)
                if txt_count != ply_count:
                    raise OSError(
                        f"{prefix}点云PLY与TXT点数不一致: "
                        f"PLY={ply_count}, TXT={txt_count}"
                    )
                return ply_count

            original_xyz = result.get("original_xyz_map", result["xyz_map"])
            original_count = save_cloud_pair("original", original_xyz)
            processed_count = save_cloud_pair("processed", result["xyz_map"])

            info_lines = [
                "DLP4710 单目结构光完整测量结果",
                f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"条纹帧数: {len(result['frames'])}",
                f"包裹相位组数: {len(result['wrapped_phases'])}",
                f"点云状态: {'已提取测量主体' if result.get('point_cloud_processed') else '尚未处理，处理后点云与原始点云相同'}",
                f"原始点云有效点数: {original_count}",
                f"处理后点云有效点数: {processed_count}",
                f"重构耗时: {float(result.get('elapsed_seconds', 0.0)):.3f} 秒",
                f"标定文件: {self.calib_file_path or '未记录'}",
                "原始点云: 05_point_cloud/original_cloud.ply",
                "原始坐标: 05_point_cloud/original_points.txt",
                "处理后点云: 05_point_cloud/processed_cloud.ply",
                "处理后坐标: 05_point_cloud/processed_points.txt",
                "说明: NPY为原始浮点数值，TIFF为浮点图，PNG为归一化伪彩预览。",
            ]
            with open(
                os.path.join(output_folder, "measurement_info.txt"),
                "w",
                encoding="utf-8",
            ) as info_file:
                info_file.write("\n".join(info_lines) + "\n")
        except Exception as exc:
            if created:
                shutil.rmtree(output_folder, ignore_errors=True)
            return False, f"保存中间结果失败: {exc}"

        return True, f"完整结果（含原始/处理后PLY和TXT点云）已保存到: {output_folder}"

