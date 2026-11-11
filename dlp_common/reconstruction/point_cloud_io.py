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
    binary: bool = True,
) -> Tuple[bool, int, str]:
    """
    Saves point cloud to standard PLY file.
    binary=True (default): Ultra-fast Binary PLY (format binary_little_endian 1.0, ~50ms).
    binary=False: Standard ASCII PLY.
    Returns: (success: bool, point_count: int, message: str)
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    valid_pts, valid_colors = extract_valid_points(xyz_map, colors)
    count = len(valid_pts)
    if count == 0:
        return False, 0, "点云数据为空，未导出"

    try:
        has_color = valid_colors is not None and len(valid_colors) == count
        if binary:
            header = [
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {count}",
                "property float x",
                "property float y",
                "property float z",
            ]
            if has_color:
                header.extend([
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                ])
            header.append("end_header\n")
            header_bytes = "\n".join(header).encode("ascii")

            with open(file_path, "wb") as f:
                f.write(header_bytes)
                if has_color:
                    dt = np.dtype([
                        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                    ])
                    structured = np.empty(count, dtype=dt)
                    structured["x"] = valid_pts[:, 0]
                    structured["y"] = valid_pts[:, 1]
                    structured["z"] = valid_pts[:, 2]
                    structured["red"] = valid_colors[:, 0]
                    structured["green"] = valid_colors[:, 1]
                    structured["blue"] = valid_colors[:, 2]
                    f.write(structured.tobytes())
                else:
                    pts_f32 = np.ascontiguousarray(valid_pts, dtype="<f4")
                    f.write(pts_f32.tobytes())

            return True, count, f"成功快速导出 {count} 个点到 {file_path} (二进制PLY格式)"
        else:
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

            return True, count, f"成功导出 {count} 个点到 {file_path} (ASCII PLY)"
    except Exception as e:
        return False, 0, f"保存PLY文件异常: {e}"


def read_point_cloud_ply(
    file_path: str,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Reads point cloud from PLY file (supports both binary_little_endian 1.0 and ascii 1.0).
    Returns: (success: bool, points: (N, 3) float32, colors: (N, 3) uint8 or None, message: str)
    """
    if not os.path.isfile(file_path):
        return False, None, None, f"文件不存在: {file_path}"

    try:
        with open(file_path, "rb") as f:
            is_binary = False
            vertex_count = 0
            has_color = False
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("format binary_little_endian"):
                    is_binary = True
                elif line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif "property uchar red" in line or "property uint8 red" in line:
                    has_color = True
                elif line == "end_header":
                    break

            if vertex_count == 0:
                return True, np.empty((0, 3), dtype=np.float32), None, "点云为空 (0个顶点)"

            if is_binary:
                if has_color:
                    dt = np.dtype([
                        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                    ])
                    raw = f.read(vertex_count * dt.itemsize)
                    arr = np.frombuffer(raw, dtype=dt)
                    pts = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
                    clrs = np.column_stack([arr["red"], arr["green"], arr["blue"]]).astype(np.uint8)
                    return True, pts, clrs, f"成功读取 {len(pts)} 个二进制点"
                else:
                    raw = f.read(vertex_count * 12)
                    pts = np.frombuffer(raw, dtype="<f4").reshape(-1, 3).astype(np.float32)
                    return True, pts, None, f"成功读取 {len(pts)} 个二进制点"
            else:
                text = f.read().decode("ascii", errors="ignore")
                lines = [l for l in text.strip().splitlines() if l.strip()]
                data = np.loadtxt(lines, dtype=np.float32)
                pts = data[:, :3]
                clrs = data[:, 3:6].astype(np.uint8) if data.shape[1] >= 6 else None
                return True, pts, clrs, f"成功读取 {len(pts)} 个ASCII点"
    except Exception as e:
        return False, None, None, f"读取PLY文件异常: {e}"


def ply_to_txt(
    ply_path: str,
    txt_path: str,
) -> Tuple[bool, int, str]:
    """
    Converts a PLY point cloud file (binary or ASCII) to a tab-separated TXT file (X \\t Y \\t Z).
    Returns: (success: bool, point_count: int, message: str)
    """
    ok, pts, clrs, msg = read_point_cloud_ply(ply_path)
    if not ok or pts is None:
        return False, 0, f"读取PLY失败: {msg}"

    count = len(pts)
    if count == 0:
        return False, 0, "PLY文件点云为空，未转换"

    os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
    try:
        np.savetxt(txt_path, pts, fmt="%.4f", delimiter="\t")
        return True, count, f"成功将 {count} 个点由 PLY 转换为 TXT 格式: {txt_path}"
    except Exception as e:
        return False, 0, f"转换为TXT文件异常: {e}"


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
        np.savetxt(file_path, valid_pts, fmt="%.4f", delimiter="\t")
        return True, count, f"成功导出 {count} 个点到 {file_path}"
    except Exception as e:
        return False, 0, f"保存TXT文件异常: {e}"

