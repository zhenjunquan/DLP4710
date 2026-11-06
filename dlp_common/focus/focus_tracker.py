"""
Real-time focus tracking and dynamic state machine for stage vertical movement.
"""

from collections import deque
from enum import Enum
from typing import Deque, List, Optional, Tuple
import numpy as np


class FocusState(str, Enum):
    IDLE = "空闲"
    OUT_OF_FOCUS = "未对焦 / 寻找中"
    APPROACHING = "正在接近焦平面 (清晰度上升)"
    IN_FOCUS = "★ 对焦成功 (景深范围内) ★"
    OVERSHOT = "⚠️ 已越过焦点 (请反向微调回拨)"


class FocusTracker:
    """
    Maintains a rolling window of focus scores, tracks the dynamic peak,
    evaluates trend / rate-of-change, and determines focus state during stage movement.
    """

    def __init__(
        self,
        history_len: int = 150,
        in_focus_ratio: float = 0.92,
        overshot_threshold: float = 0.85,
        min_peak_valid: float = 5.0,
    ):
        self.history_len = history_len
        self.in_focus_ratio = in_focus_ratio          # Score >= ratio * peak is considered in-focus
        self.overshot_threshold = overshot_threshold  # Score < threshold * peak after having reached peak is overshot
        self.min_peak_valid = min_peak_valid          # Minimum baseline score to avoid false triggers on black/blank image

        self.scores: Deque[float] = deque(maxlen=history_len)
        self.timestamps: Deque[float] = deque(maxlen=history_len)
        self.peak_score: float = 0.0
        self.peak_index: int = 0
        self.current_score: float = 0.0
        self.current_ratio: float = 0.0  # Percentage of current score relative to peak (0 ~ 100%)
        self.state: FocusState = FocusState.IDLE
        self._has_entered_focus: bool = False

    def reset_peak(self):
        """Resets tracked peak and state history."""
        self.scores.clear()
        self.timestamps.clear()
        self.peak_score = 0.0
        self.peak_index = 0
        self.current_score = 0.0
        self.current_ratio = 0.0
        self.state = FocusState.IDLE
        self._has_entered_focus = False

    def update(self, score: float, timestamp: float) -> Tuple[FocusState, float, float]:
        """
        Updates tracker with a new focus score.
        Returns:
            (state, current_ratio, peak_score)
        """
        self.current_score = float(score)
        self.scores.append(self.current_score)
        self.timestamps.append(timestamp)

        # Update dynamic peak
        if self.current_score > self.peak_score:
            self.peak_score = self.current_score
            self.peak_index = len(self.scores) - 1

        if self.peak_score > max(self.min_peak_valid, 1e-6):
            self.current_ratio = min(1.0, max(0.0, self.current_score / self.peak_score))
        else:
            self.current_ratio = 0.0

        # Calculate local trend (smoothed derivative over last 5 frames)
        trend = self._compute_trend()

        # State transition logic:
        # 1. Below minimal threshold or very first frames: OUT_OF_FOCUS
        if self.peak_score < self.min_peak_valid or len(self.scores) < 3:
            self.state = FocusState.OUT_OF_FOCUS
        # 2. Currently overshot: was previously in focus or reached a peak, but now dropped significantly and trend <= 0
        elif (self._has_entered_focus or self.peak_score > self.min_peak_valid * 1.5) and self.current_ratio < self.overshot_threshold and trend <= 0:
            self.state = FocusState.OVERSHOT
        # 3. Actively climbing rapidly toward a new higher peak: APPROACHING
        elif trend > 0.015 and self.current_score >= self.peak_score * 0.99:
            self.state = FocusState.APPROACHING
        # 4. Within in-focus tolerance band (score >= in_focus_ratio * peak)
        elif self.current_ratio >= self.in_focus_ratio:
            self.state = FocusState.IN_FOCUS
            self._has_entered_focus = True
        # 5. Rising from below: APPROACHING
        elif trend > 0.005:
            self.state = FocusState.APPROACHING
        else:
            self.state = FocusState.OUT_OF_FOCUS

        return self.state, self.current_ratio, self.peak_score

    def _compute_trend(self) -> float:
        """Computes rate of change of recent scores."""
        if len(self.scores) < 4:
            return 0.0
        recent = list(self.scores)[-5:]
        n = len(recent)
        # Linear regression slope of recent points
        x = np.arange(n)
        y = np.array(recent)
        denom = n * np.sum(x * x) - np.sum(x) ** 2
        if denom == 0:
            return 0.0
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / denom
        # Normalize slope relative to peak
        ref = max(self.peak_score, 1e-3)
        return float(slope / ref)

    def get_history(self) -> List[float]:
        """Returns list of historical focus scores for plotting."""
        return list(self.scores)
