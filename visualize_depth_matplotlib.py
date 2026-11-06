"""
DLP4710 原生 Matplotlib 3D 点云深度图桌面交互窗口
支持鼠标旋转、缩放、平移，并在窗口中直接点击保存按钮保存图片。
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_points(file_path: str, sample_step: int = 15) -> np.ndarray:
    print(f"正在读取点云: {file_path} ...")
    pts_list = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if idx % sample_step == 0:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        pts_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
    return np.array(pts_list, dtype=np.float64)


def compute_relative_height(pts: np.ndarray) -> np.ndarray:
    centroid = np.mean(pts, axis=0)
    cov = (pts - centroid).T @ (pts - centroid)
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, 0]
    if normal[2] < 0:
        normal = -normal
    h_proj = (pts - centroid) @ normal
    base_val = np.percentile(h_proj, 2)
    return np.clip(h_proj - base_val, 0.0, 3.25)


def main():
    parser = argparse.ArgumentParser(description="Matplotlib 3D 点云深度图")
    parser.add_argument("--file", "-f", default=r"D:\ZhenWork\DLP4710\original_points.asc")
    parser.add_argument("--step", "-s", type=int, default=15, help="Matplotlib推荐抽样步长 15~20")
    args = parser.parse_args()

    pts = load_points(args.file, sample_step=args.step)
    rel_h = compute_relative_height(pts)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")

    scatter = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=rel_h,
        cmap="turbo",
        s=1.2,
        alpha=0.9
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label("台阶相对高度 Height (mm)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_xlabel("X (mm)", color="#aaaaaa")
    ax.set_ylabel("Y (mm)", color="#aaaaaa")
    ax.set_zlabel("Z (mm)", color="#aaaaaa")
    ax.set_title("DLP4710 3D 点云深度图 (鼠标拖拽旋转，滚轮缩放，点击保存图标导出图片)", color="white", fontsize=13)

    ax.view_init(elev=35, azim=45)
    print("窗口已打开：可使用鼠标左键拖拽旋转视角，右键缩放，点击左下角磁盘图标保存当前视角图片。")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

