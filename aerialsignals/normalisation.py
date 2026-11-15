"""Normalisation: each pose is described relative to the body, so the features depend on the signal
rather than on where the person stands in the frame or how far away the camera is.

A window of raw keypoints (T, 14, 2) becomes a (T, 56) feature matrix:
missing joints are interpolated over time, positions are centred on the neck and divided by the
median torso length, and frame-to-frame velocities are appended.
"""
import numpy as np

from .constants import L_HIP, NECK, R_HIP


def fill_missing(x, visible):
    """Linearly interpolate missing joints over time; a joint never seen is placed at the neck."""
    x = x.copy()
    t = np.arange(len(x))
    for j in range(x.shape[1]):
        seen = visible[:, j]
        if not seen.any():
            x[:, j] = x[:, NECK] if visible[:, NECK].any() else 0
        elif not seen.all():
            for d in range(2):
                x[~seen, j, d] = np.interp(t[~seen], t[seen], x[seen, j, d])
    return x


def torso_length(x):
    """Median neck to mid-hip distance over the window."""
    mid_hip = (x[:, R_HIP] + x[:, L_HIP]) / 2
    return np.median(np.linalg.norm(mid_hip - x[:, NECK], axis=-1))


def normalise(x, visible, mode='body'):
    """'body': centre on the neck and scale by torso length (used by the proposed model).
    'image': divide by the 1920x1080 frame size (the original final-year-project input)."""
    if mode == 'image':
        return x / np.array([1920., 1080.], dtype=np.float32)
    x = fill_missing(x, visible)
    return (x - x[:, NECK:NECK + 1]) / (torso_length(x) + 1e-6)


def to_features(x_norm, velocity=True):
    """(T, 14, 2) normalised keypoints -> (T, 28) positions, plus (T, 28) velocities if requested."""
    T = len(x_norm)
    parts = [x_norm.reshape(T, -1)]
    if velocity:
        parts.append(np.diff(x_norm, axis=0, prepend=x_norm[:1]).reshape(T, -1))
    return np.concatenate(parts, 1).astype(np.float32)


def window_features(x, visible, norm='body', velocity=True):
    """Raw window -> model input. This is the exact transform used in training and on the live stream."""
    if norm == 'image':
        x = fill_missing(x, visible)
    return to_features(normalise(x, visible, norm), velocity)
