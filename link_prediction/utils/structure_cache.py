from pathlib import Path
from typing import Optional

import numpy as np


def build_shared_structure_cache_root(
    base_root: Optional[Path], dataset_name: str, cache_key: str
) -> Optional[Path]:
    if base_root is None:
        return None
    return Path(base_root).resolve() / dataset_name / cache_key


def stage_raw_cache_path(shared_root: Optional[Path], mode: str) -> Optional[Path]:
    if shared_root is None:
        return None
    return shared_root / f"{mode.lower()}_raw.pkl"


def advance_uniform_rng(
    rng: np.random.RandomState,
    score_shape,
    low: float = 0.0,
    high: float = 1e-8,
    chunk_size: int = 1_000_000,
) -> None:
    total = int(np.prod(score_shape))
    offset = 0
    while offset < total:
        step = min(chunk_size, total - offset)
        rng.uniform(low, high, size=step)
        offset += step


def materialize_seeded_scores(
    raw_scores: np.ndarray,
    rng: np.random.RandomState,
    low: float = 0.0,
    high: float = 1e-8,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    scores = np.array(raw_scores, copy=True)
    flat_scores = scores.reshape(-1)
    total = flat_scores.size
    offset = 0
    while offset < total:
        step = min(chunk_size, total - offset)
        noise = rng.uniform(low, high, size=step)
        score_chunk = flat_scores[offset : offset + step]
        zero_mask = score_chunk == 0.0
        if np.any(zero_mask):
            score_chunk[zero_mask] += noise[zero_mask]
        offset += step
    return scores


def materialize_tppr_payload(
    raw_payload,
    rng: np.random.RandomState,
    low: float = 0.0,
    high: float = 1e-8,
    chunk_size: int = 1_000_000,
):
    source_nodes, timestamps_all, raw_scores, wall_seconds, memory_mb = raw_payload
    scores = materialize_seeded_scores(
        raw_scores,
        rng,
        low=low,
        high=high,
        chunk_size=chunk_size,
    )
    return source_nodes, timestamps_all, scores, wall_seconds, memory_mb
