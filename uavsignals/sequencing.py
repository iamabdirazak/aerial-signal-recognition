"""Sequencing: the keypoint stream is read as short overlapping windows rather than single frames.

Every 0.2 s the most recent 30 frames (1.2 s at 25 Hz) form one window, and each window yields one prediction.
"""
import numpy as np

from .constants import WINDOW


def pad_to_window(x, visible, length=WINDOW):
    """Repeat the last frame so that clips shorter than one window can still be classified."""
    if len(x) >= length:
        return x, visible
    extra = length - len(x)
    return (np.concatenate([x, np.repeat(x[-1:], extra, 0)]),
            np.concatenate([visible, np.repeat(visible[-1:], extra, 0)]))


def sliding_windows(x, visible, stride, length=WINDOW):
    """List of (keypoints, visible) windows taken every `stride` frames."""
    x, visible = pad_to_window(x, visible, length)
    return [(x[i:i + length], visible[i:i + length]) for i in range(0, len(x) - length + 1, stride)]
