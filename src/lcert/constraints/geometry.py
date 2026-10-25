from __future__ import annotations

import numpy as np


def world_to_body(relative_positions, headings) -> np.ndarray:
    positions = np.asarray(relative_positions, float)
    headings = np.asarray(headings, float)
    if positions.ndim == 0 or positions.shape[-1] != 2 or not np.isfinite(positions).all() or not np.isfinite(headings).all():
        raise ValueError('Body-frame transform requires finite 2D vectors and headings')
    cosine, sine = np.cos(headings), np.sin(headings)
    return np.stack((cosine * positions[..., 0] + sine * positions[..., 1],
                     -sine * positions[..., 0] + cosine * positions[..., 1]), axis=-1)


def rotate_vectors(vectors, angle: float) -> np.ndarray:
    vectors = np.asarray(vectors, float)
    if vectors.ndim == 0 or vectors.shape[-1] != 2 or not np.isfinite(vectors).all() or not np.isfinite(angle):
        raise ValueError('Rotation requires finite 2D vectors and angle')
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return vectors @ rotation.T
