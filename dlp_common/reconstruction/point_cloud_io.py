"""
Point cloud export utilities (ASCII PLY and tab-separated TXT).
Compatible with MeshLab, CloudCompare, Open3D, and C++ StructuredLightMeasurement.
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np


def extract_valid_points(
    xyz_map: np.ndarray,
    colors_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extracts valid non-zero (X, Y, Z) points and optional colors from an (H, W, 3) map.
    Returns:
      valid_pts: (N, 3) float64
      valid_colors: (N, 3) uint8 or None
    """
    pts = xyz_map.reshape(-1, 3)
    finite_mask = np.all(np.isfinite(pts), axis=1)
    non_zero_mask = (
        (np.abs(pts[:, 0]) > 1e-9)
        | (np.abs(pts[:, 1]) > 1e-9)
        | (np.abs(pts[:, 2]) > 1e-9)
    )
    valid_mask = finite_mask & non_zero_mask
    valid_pts = pts[valid_mask]

    valid_colors = None
    if colors_map is not None:
        c = colors_map.reshape(-1, colors_map.shape[-1])
        valid_colors = c[valid_mask]

    return valid_pts, valid_colors


def keep_largest_image_component(
    xyz_map: np.ndarray,
    close_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Keep the largest subject component in the organized camera point map.

    A small morphological closing connects short acquisition gaps before the
    component search. Only originally valid 3D samples are copied to output.
    """
    if xyz_map is None or xyz_map.ndim != 3 or xyz_map.shape[2] != 3:
        raise ValueError("点云必须为 HxWx3 的规则XYZ矩阵")

    finite = np.all(np.isfinite(xyz_map), axis=2)
    non_zero = np.any(np.abs(xyz_map) > 1e-9, axis=2)
    valid = finite & non_zero
    original_count = int(np.count_nonzero(valid))
    if original_count == 0:
        raise ValueError("当前点云没有有效三维点")

    topology = valid.astype(np.uint8)
    radius = max(0, int(close_radius))
    if radius > 0:
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        topology = cv2.morphologyEx(topology, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        topology,
        connectivity=8,
    )
    if component_count <= 1:
        raise ValueError("未检测到可保留的点云主体")

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    subject_mask = valid & (labels == largest_label)
    retained_count = int(np.count_nonzero(subject_mask))
    if retained_count == 0:
        raise ValueError("最大连通主体中没有有效三维点")

    filtered = np.zeros_like(xyz_map)
    filtered[subject_mask] = xyz_map[subject_mask]
    return filtered, subject_mask, original_count, retained_count


def save_point_cloud_ply(
    file_path: str,
    xyz_map: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> Tuple[bool, int, str]:
    """
    Saves point cloud to standard ASCII PLY file.
    Returns: (success: bool, point_count: int, message: str)
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    valid_pts, valid_colors = extract_valid_points(xyz_map, colors)
    count = len(valid_pts)
    if count == 0:
        return False, 0, "点云数据为空，未导出"

    try:
        has_color = valid_colors is not None and len(valid_colors) == count
        with open(file_path, "w", encoding="ascii") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {count}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")

            chunk_size = 50000
            if has_color:
                for i in range(0, count, chunk_size):
                    chunk_pts = valid_pts[i : i + chunk_size]
                    chunk_clr = valid_colors[i : i + chunk_size]
                    lines = [
                        f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(clr[0])} {int(clr[1])} {int(clr[2])}"
                        for pt, clr in zip(chunk_pts, chunk_clr)
                    ]
                    f.write("\n".join(lines) + "\n")
            else:
                for i in range(0, count, chunk_size):
                    chunk_pts = valid_pts[i : i + chunk_size]
                    lines = [
                        f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}"
                        for pt in chunk_pts
                    ]
                    f.write("\n".join(lines) + "\n")

        return True, count, f"成功导出 {count} 个点到 {file_path}"
    except Exception as e:
        return False, 0, f"保存PLY文件异常: {e}"


def save_point_cloud_txt(
    file_path: str,
    xyz_map: np.ndarray,
) -> Tuple[bool, int, str]:
    """
    Saves point cloud to tab-separated TXT file (X \\t Y \\t Z).
    Returns: (success: bool, point_count: int, message: str)
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    valid_pts, _ = extract_valid_points(xyz_map)
    count = len(valid_pts)
    if count == 0:
        return False, 0, "点云数据为空，未导出"

    try:
        chunk_size = 50000
        with open(file_path, "w", encoding="ascii") as f:
            for i in range(0, count, chunk_size):
                chunk_pts = valid_pts[i : i + chunk_size]
                lines = [
                    f"{pt[0]:.6f}\t{pt[1]:.6f}\t{pt[2]:.6f}"
                    for pt in chunk_pts
                ]
                f.write("\n".join(lines) + "\n")
        return True, count, f"成功导出 {count} 个点到 {file_path}"
    except Exception as e:
        return False, 0, f"保存TXT文件异常: {e}"

