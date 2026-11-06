"""
DLP4710 3D 点云深度图交互可视化与截图下载工具
支持加载 .asc / .txt / .ply 文件，生成带有深度/高度 Colorbar 的可360°旋转交互3D视图，并支持一键下载任意视角的高清截图。
"""

import os
import sys
import argparse
import webbrowser
import numpy as np
import plotly.graph_objects as go


def load_point_cloud(file_path: str, sample_step: int = 6) -> np.ndarray:
    """
    加载点云数据并进行适度抽样（保证浏览器WebGL以60FPS流畅旋转）。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到点云文件: {file_path}")

    print(f"正在读取点云文件: {file_path} ...")
    
    # 支持 .asc / .txt / .csv
    pts_list = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 按抽样步长取点
            if idx % sample_step == 0:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        pts_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue

    pts = np.array(pts_list, dtype=np.float64)
    print(f"成功加载点云！抽样步长: {sample_step}，显示点数: {len(pts):,} 点")
    return pts


def compute_relative_step_height(pts: np.ndarray) -> np.ndarray:
    """
    基于主成分分析（PCA）自动对齐台阶面法向量，
    计算垂直于台阶表面的相对物理高度（0 ~ 3.2 mm），使阶梯色差最清晰。
    """
    centroid = np.mean(pts, axis=0)
    cov = (pts - centroid).T @ (pts - centroid)
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, 0]
    if normal[2] < 0:
        normal = -normal

    h_proj = (pts - centroid) @ normal
    base_val = np.percentile(h_proj, 2)
    h_rel = h_proj - base_val
    return np.clip(h_rel, 0.0, 3.25)


def generate_interactive_viewer(
    pts: np.ndarray,
    output_html: str,
    point_size: float = 1.6,
    auto_open: bool = True
):
    """
    构建带有深色专业UI、深度Colorbar、视角预设和一键下载高清截图的交互式HTML。
    """
    raw_z = pts[:, 2]
    rel_height = compute_relative_step_height(pts)

    print("正在构建交互式三维可视化场景...")

    # Trace 1: 相对阶梯高度 (Turbo 色标, 0 ~ 3.1mm 阶跃色彩分明)
    trace_rel = go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        name="相对阶梯高度",
        visible=True,
        marker=dict(
            size=point_size,
            color=rel_height,
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="台阶相对高度 (mm)",
                    font=dict(color="#ffffff", size=13, family="Segoe UI, sans-serif")
                ),
                thickness=20,
                len=0.75,
                tickfont=dict(color="#ffffff", size=11),
                bgcolor="rgba(25, 30, 45, 0.7)",
                bordercolor="rgba(100, 120, 160, 0.4)",
                borderwidth=1
            ),
            opacity=0.92
        ),
        hovertemplate="X: %{x:.3f} mm<br>Y: %{y:.3f} mm<br>Z: %{z:.3f} mm<br>高度: %{marker.color:.3f} mm<extra></extra>"
    )

    # Trace 2: 空间绝对深度 Z (Viridis 色标, 182.7 ~ 191.9mm)
    trace_raw = go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        name="绝对坐标 Z",
        visible=False,
        marker=dict(
            size=point_size,
            color=raw_z,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="空间绝对 Z 坐标 (mm)",
                    font=dict(color="#ffffff", size=13, family="Segoe UI, sans-serif")
                ),
                thickness=20,
                len=0.75,
                tickfont=dict(color="#ffffff", size=11),
                bgcolor="rgba(25, 30, 45, 0.7)",
                bordercolor="rgba(100, 120, 160, 0.4)",
                borderwidth=1
            ),
            opacity=0.92
        ),
        hovertemplate="X: %{x:.3f} mm<br>Y: %{y:.3f} mm<br>Z: %{z:.3f} mm<extra></extra>"
    )

    fig = go.Figure(data=[trace_rel, trace_raw])

    # 布局配置
    fig.update_layout(
        title=dict(
            text="<b>DLP4710 3D 点云深度图 (支持自由旋转 / 滚轮缩放 / 高清下载)</b>",
            font=dict(color="#ffffff", size=17, family="Segoe UI, sans-serif"),
            x=0.03,
            y=0.97
        ),
        paper_bgcolor="#0c0e14",
        plot_bgcolor="#0c0e14",
        scene=dict(
            xaxis=dict(
                title=dict(text="X (mm)", font=dict(color="#aaaaaa")),
                tickfont=dict(color="#888888"),
                gridcolor="#252b3b",
                zerolinecolor="#3a445d",
                backgroundcolor="#111520",
                showbackground=True
            ),
            yaxis=dict(
                title=dict(text="Y (mm)", font=dict(color="#aaaaaa")),
                tickfont=dict(color="#888888"),
                gridcolor="#252b3b",
                zerolinecolor="#3a445d",
                backgroundcolor="#111520",
                showbackground=True
            ),
            zaxis=dict(
                title=dict(text="Z (mm)", font=dict(color="#aaaaaa")),
                tickfont=dict(color="#888888"),
                gridcolor="#252b3b",
                zerolinecolor="#3a445d",
                backgroundcolor="#111520",
                showbackground=True
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2)  # 经典轴测立体视角
            )
        ),
        margin=dict(l=0, r=0, b=0, t=55),
        legend=dict(
            font=dict(color="#ffffff"),
            bgcolor="rgba(20, 25, 35, 0.8)",
            x=0.02,
            y=0.88
        ),
        updatemenus=[
            # 色标模式切换下拉框
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=0.93,
                xanchor="left",
                yanchor="top",
                bgcolor="#1e2333",
                font=dict(color="#ffffff", size=12),
                bordercolor="#3f4a68",
                borderwidth=1,
                buttons=[
                    dict(
                        label="🎨 色彩: 相对阶梯高度 (0~3.1mm)",
                        method="update",
                        args=[{"visible": [True, False]}]
                    ),
                    dict(
                        label="🎨 色彩: 绝对坐标 Z (182~192mm)",
                        method="update",
                        args=[{"visible": [False, True]}]
                    )
                ]
            ),
            # 常用预设视角切换按钮组
            dict(
                type="buttons",
                direction="right",
                x=0.35,
                y=0.97,
                xanchor="left",
                yanchor="top",
                bgcolor="#1e2333",
                font=dict(color="#ffffff", size=11),
                bordercolor="#3f4a68",
                borderwidth=1,
                buttons=[
                    dict(
                        label="📐 轴测立体视角",
                        method="relayout",
                        args=[{"scene.camera.eye": dict(x=1.6, y=1.6, z=1.2)}]
                    ),
                    dict(
                        label="👁️ 正对俯视",
                        method="relayout",
                        args=[{"scene.camera.eye": dict(x=0.01, y=0.01, z=2.4)}]
                    ),
                    dict(
                        label="🔍 台阶侧面剖视",
                        method="relayout",
                        args=[{"scene.camera.eye": dict(x=2.2, y=0.2, z=0.3)}]
                    ),
                    dict(
                        label="🔄 视角重置",
                        method="relayout",
                        args=[{"scene.camera.eye": dict(x=1.6, y=1.6, z=1.2)}]
                    )
                ]
            )
        ]
    )

    # 导出并嵌入一键下载高清截图的浮动UI
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "DLP4710_3D_PointCloud",
            "height": 1440,
            "width": 2560,
            "scale": 2
        }
    }

    # 渲染基础 HTML
    html_content = fig.to_html(include_plotlyjs=True, config=config)

    # 注入定制的下载控制按钮与操作提示栏
    custom_ui = """
    <div style="position: absolute; top: 12px; right: 20px; z-index: 9999; display: flex; gap: 10px; align-items: center;">
        <button onclick="downloadCustomPNG(2560, 1440)" style="
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: #ffffff;
            border: 1px solid #3b82f6;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;">
            📸 下载当前旋转视角图片 (2.5K高清)
        </button>
        <button onclick="downloadCustomPNG(3840, 2160)" style="
            background: linear-gradient(135deg, #059669, #047857);
            color: #ffffff;
            border: 1px solid #10b981;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.4);
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;">
            🖼️ 4K 超高清导出
        </button>
    </div>

    <div style="position: absolute; bottom: 15px; left: 20px; z-index: 9999;
        background: rgba(18, 22, 34, 0.85);
        color: #94a3b8;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 12px;
        border: 1px solid rgba(71, 85, 105, 0.4);
        pointer-events: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);">
        🖱️ <b>操作提示</b>：鼠标左键拖拽旋转视角 | 滚轮前后滑动缩放 | 鼠标右键平移画面
    </div>

    <script>
    function downloadCustomPNG(width, height) {
        var plotElement = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotElement) return;
        var dateStr = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '_');
        var fileName = 'DLP4710_3D_Depth_' + width + 'x' + height + '_' + dateStr;
        Plotly.downloadImage(plotElement, {
            format: 'png',
            width: width,
            height: height,
            filename: fileName
        });
    }
    </script>
    """

    final_html = html_content.replace("</body>", f"{custom_ui}\n</body>")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"\n✅ 3D 可视化页面生成成功！")
    print(f"📁 文件保存路径: {output_html}")

    if auto_open:
        print("🌐 正在使用默认浏览器打开可视化窗口...")
        webbrowser.open(os.path.abspath(output_html))


def main():
    parser = argparse.ArgumentParser(description="DLP4710 3D 点云深度图交互查看与截图工具")
    parser.add_argument(
        "--file", "-f",
        default=r"D:\ZhenWork\DLP4710\original_points.asc",
        help="点云文件路径 (.asc / .txt / .ply)"
    )
    parser.add_argument(
        "--output", "-o",
        default=r"D:\ZhenWork\DLP4710\point_cloud_3d_viewer.html",
        help="输出 HTML 交互文件路径"
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        default=6,
        help="抽样步长 (默认为 6，显示约 30 万点，兼顾画质与 60FPS 流畅度；设为 1 则加载全部点)"
    )
    parser.add_argument(
        "--point_size", "-p",
        type=float,
        default=1.6,
        help="点大小 (默认 1.6)"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="生成 HTML 后不自动打开浏览器"
    )

    args = parser.parse_args()

    pts = load_point_cloud(args.file, sample_step=args.step)
    generate_interactive_viewer(
        pts=pts,
        output_html=args.output,
        point_size=args.point_size,
        auto_open=not args.no_open
    )


if __name__ == "__main__":
    main()

