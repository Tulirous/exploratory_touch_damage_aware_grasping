from __future__ import annotations

import numpy as np


def safe_norm(vector: np.ndarray, eps: float = 1e-8) -> float:
    return float(max(np.linalg.norm(vector), eps))


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    denominator = safe_norm(first) * safe_norm(second)
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.arccos(cosine))


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def low_pass(previous: np.ndarray | None, current: np.ndarray, alpha: float) -> np.ndarray:
    if previous is None:
        return current.copy()
    return alpha * current + (1.0 - alpha) * previous


def limit_step(previous: np.ndarray | None, current: np.ndarray, max_step: float) -> np.ndarray:
    if previous is None:
        return current.copy()
    return previous + np.clip(current - previous, -max_step, max_step)
