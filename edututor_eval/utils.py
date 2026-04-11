"""Shared utilities for data I/O, logging, and agreement metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from edututor_eval.datatypes import TutorResponse, EvalResult


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_responses(path: str | Path) -> list[TutorResponse]:
    """Load tutor responses from JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "responses" in data:
        data = data["responses"]

    return [TutorResponse(**item) for item in data]


def save_results(results: list[EvalResult], path: str | Path) -> None:
    """Save evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)


def results_to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Convert results to a flat DataFrame for analysis."""
    rows = []
    for r in results:
        row = {
            "response_id": r.response_id,
            "evaluator": r.evaluator,
            "overall_score": r.overall_score,
            "n_flags": len(r.flags),
            "flags": ",".join(r.flags) if r.flags else "",
        }
        for ds in r.dimension_scores:
            row[f"score_{ds.dimension}"] = ds.score
            row[f"conf_{ds.dimension}"] = ds.confidence
        rows.append(row)
    return pd.DataFrame(rows)


def _pearsonr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson correlation (fallback when scipy unavailable)."""
    n = len(a)
    if n < 3:
        return 0.0, 1.0
    a_mean, b_mean = a.mean(), b.mean()
    a_dev, b_dev = a - a_mean, b - b_mean
    r = float(np.sum(a_dev * b_dev) / (np.sqrt(np.sum(a_dev**2)) * np.sqrt(np.sum(b_dev**2)) + 1e-12))
    return r, 0.0  # p-value omitted in fallback


def _spearmanr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation (fallback when scipy unavailable)."""
    from numpy import argsort
    rank_a = np.empty_like(a)
    rank_b = np.empty_like(b)
    rank_a[argsort(a)] = np.arange(len(a), dtype=float)
    rank_b[argsort(b)] = np.arange(len(b), dtype=float)
    return _pearsonr(rank_a, rank_b)


def compute_agreement(
    scores_a: list[float],
    scores_b: list[float],
    tolerance: float = 0.5,
) -> dict[str, float]:
    """Compute agreement metrics between two sets of scores.

    Returns exact agreement, adjacent agreement (within tolerance),
    Pearson correlation, and Spearman rank correlation.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)

    exact = float(np.mean(np.abs(a - b) < 0.01))
    adjacent = float(np.mean(np.abs(a - b) <= tolerance))

    try:
        from scipy import stats
        pearson_r, pearson_p = stats.pearsonr(a, b)
        spearman_r, spearman_p = stats.spearmanr(a, b)
    except ImportError:
        pearson_r, pearson_p = _pearsonr(a, b)
        spearman_r, spearman_p = _spearmanr(a, b)

    return {
        "exact_agreement": round(exact, 4),
        "adjacent_agreement": round(adjacent, 4),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 6),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 6),
        "mean_absolute_error": round(float(np.mean(np.abs(a - b))), 4),
    }


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's kappa for inter-annotator agreement."""
    try:
        from sklearn.metrics import cohen_kappa_score
        return round(cohen_kappa_score(labels_a, labels_b, weights="quadratic"), 4)
    except ImportError:
        # Simple kappa fallback
        a, b = np.array(labels_a), np.array(labels_b)
        po = float(np.mean(a == b))
        pe = sum((np.sum(a == c) * np.sum(b == c)) for c in np.unique(np.concatenate([a, b]))) / (len(a) ** 2)
        return round((po - pe) / (1 - pe + 1e-12), 4)
