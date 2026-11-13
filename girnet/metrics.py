from __future__ import annotations

import numpy as np


def si_sdr(target: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    target = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    projection = np.dot(estimate, target) / (np.dot(target, target) + eps) * target
    error = estimate - projection
    return float(10.0 * np.log10((np.dot(projection, projection) + eps) / (np.dot(error, error) + eps)))


def bss_projection_metrics(sources: np.ndarray, estimate: np.ndarray, target_index: int, eps: float = 1e-8) -> tuple[float, float]:
    """Ordered target/interference/artifact projection for one output channel."""
    basis = np.asarray(sources, dtype=np.float64).T
    estimate = np.asarray(estimate, dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(basis, estimate, rcond=None)
    components = coefficients[:, None] * np.asarray(sources, dtype=np.float64)
    target_component = components[target_index]
    interference = components.sum(axis=0) - target_component
    artifact = estimate - components.sum(axis=0)
    sir = 10.0 * np.log10((np.sum(target_component ** 2) + eps) / (np.sum(interference ** 2) + eps))
    sar = 10.0 * np.log10((np.sum((target_component + interference) ** 2) + eps) / (np.sum(artifact ** 2) + eps))
    return float(sir), float(sar)


def adjacency_metrics(target: np.ndarray, prediction: np.ndarray, floor_db: float = -60.0) -> dict[str, float]:
    channels = target.shape[-1]
    mask = ~np.eye(channels, dtype=bool)
    target_edges = target[..., mask]
    prediction_edges = prediction[..., mask]
    target_db = 20.0 * np.log10(np.maximum(target_edges, 10.0 ** (floor_db / 20.0)))
    prediction_db = 20.0 * np.log10(np.maximum(prediction_edges, 10.0 ** (floor_db / 20.0)))
    correlation = float(np.corrcoef(target_db.ravel(), prediction_db.ravel())[0, 1]) if np.std(prediction_db) > 0 else float("nan")
    return {
        "mae_db": float(np.mean(np.abs(target_db - prediction_db))),
        "rmse_db": float(np.sqrt(np.mean((target_db - prediction_db) ** 2))),
        "correlation_db": correlation,
    }

