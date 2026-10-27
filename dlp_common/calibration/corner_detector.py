"""
Chessboard corner detection and sub-pixel refinement.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class CornerDetector:
    """
    Detects inner chessboard corners with multi-scale coarse-to-fine search
    and sub-pixel corner refinement.
    """

    def __init__(
        self,
        board_cols: int = 11,
        board_rows: int = 8,
        square_size_mm: float = 3.0,
    ):
        self.board_cols = board_cols
        self.board_rows = board_rows
        self.square_size_mm = square_size_mm
        self.pattern_size = (board_cols, board_rows)

    def create_object_points(self) -> np.ndarray:
        """
        Creates ideal 3D coordinates of chessboard corners in millimeters:
        (col * square_size, row * square_size, 0.0).
        Returns shape (N, 3), float32.
        """
        objp = np.zeros((self.board_rows * self.board_cols, 3), dtype=np.float32)
        # OpenCV returns corners row by row. Keep the same ordering as the C++
        # implementation: (column * square, row * square, 0).
        grid = np.mgrid[0 : self.board_cols, 0 : self.board_rows].T.reshape(-1, 2)
        objp[:, :2] = grid.astype(np.float32) * self.square_size_mm
        return objp

    def detect_corners(
        self,
        image: np.ndarray,
        subpix_win: Tuple[int, int] = (11, 11),
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Detects inner corners of chessboard in `image`.
        Returns:
          success: bool
          corners: (N, 2) float32 array or None
          message: status/error string
        """
        if image is None or image.size == 0:
            return False, None, "输入图像为空"

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape[:2]
        expected_count = self.board_cols * self.board_rows

        # Multi-scale attempts: scale 0.25 -> 0.20 -> 1.0 (full res)
        scales = [0.25, 0.20, 0.50, 1.0]
        found = False
        corners = None

        for scale in scales:
            if scale < 1.0:
                small_w = int(w * scale)
                small_h = int(h * scale)
                small_gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
            else:
                small_gray = gray

            # 1. Try findChessboardCornersSB (Sector Based, modern & accurate)
            found, small_corners = cv2.findChessboardCornersSB(
                small_gray,
                self.pattern_size,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )

            # 2. Fallback to classic findChessboardCorners
            if not found or small_corners is None:
                flags = (
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    | cv2.CALIB_CB_NORMALIZE_IMAGE
                    | cv2.CALIB_CB_FAST_CHECK
                )
                found, small_corners = cv2.findChessboardCorners(
                    small_gray, self.pattern_size, flags=flags
                )

            if found and small_corners is not None and len(small_corners) == expected_count:
                # Rescale corners back to original image size
                corners = small_corners.reshape(-1, 2) / scale
                break

        if not found or corners is None:
            return False, None, f"未检测到完整棋盘格内角点 ({self.board_cols}x{self.board_rows})，请调整姿态、曝光或焦距"

        # Sub-pixel refinement on original full-resolution grayscale image
        corners_f32 = corners.astype(np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        refined = cv2.cornerSubPix(gray, corners_f32, subpix_win, (-1, -1), criteria)
        refined_corners = refined.reshape(-1, 2)

        return True, refined_corners, "角点检测成功"

