import numpy as np
from sklearn.metrics import ndcg_score
from typing import List, Dict

def cal_mrr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num_samples = y_true.shape[0]
    rr_sum = 0.0

    for i in range(num_samples):
        if not np.any(y_true[i] > 0):
            continue

        pred_idx = int(np.argmax(y_pred[i])) # rank 1
        true_sorted = np.argsort(-y_true[i])
        rank = int(np.where(true_sorted == pred_idx)[0][0])
        rr_sum += 1.0 / (rank + 1)

    return rr_sum / num_samples


def cal_hr(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    num_samples = y_true.shape[0]
    hr_sum = 0.0

    for i in range(num_samples):
        m = int(np.sum(y_true[i] > 0))
        if m == 0:
            continue

        k_eff = min(k, m)
        true_top = set(np.argsort(-y_true[i])[:k_eff])
        pred_top = set(np.argsort(-y_pred[i])[:k_eff])
        hits = len(true_top & pred_top)
        hr_sum += hits / k_eff

    return hr_sum / num_samples


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str], k: int = 10) -> Dict[str, float]:
    """
    y_true: (N, num_classes), one-hot or multi-hot
    y_pred: (N, num_classes)
    """
    result = {}

    ndcg = ndcg_score(y_true, y_pred, k=k)
    mrr = cal_mrr(y_true, y_pred)
    hr = cal_hr(y_true, y_pred, k=k)

    if "ndcg" in metrics:
        result["ndcg"] = ndcg
    if "mrr" in metrics:
        result["mrr"] = mrr
    if "hit_ratio" in metrics:
        result["hit_ratio"] = hr

    return result
