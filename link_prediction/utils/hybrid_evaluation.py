import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from utils.util import compute_metrics


YITA_VALUES = tuple(
    float(value)
    for value in np.concatenate(
        [np.array([1e-6, 2e-6, 3e-6, 5e-6, 8e-6]) * 10**exponent for exponent in range(0, 7)]
    )
)


def prepare_hybrid_batches(
    data,
    batch_size: int,
    time_pos_score,
    time_neg_score,
    structure_score,
    delta_times_list,
    all_inds_list,
    num_neg_per_pos: int,
    time_topk: int,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    num_pos_edge = data.n_interactions
    effective_batch_size = batch_size if batch_size != -1 else num_pos_edge
    num_batch = math.ceil(num_pos_edge / effective_batch_size)
    batches: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            batch_time_pos_score = time_pos_score[batch_idx].squeeze(1).to(device)
            batch_time_neg_score = time_neg_score[batch_idx].squeeze(1).to(device)

            start_idx = batch_idx * effective_batch_size
            end_idx = min(num_pos_edge, start_idx + effective_batch_size)
            pos_ids = np.arange(start_idx, end_idx)
            cur_batch_size = end_idx - start_idx
            neg_ids = np.concatenate(
                [pos_ids + offset * num_pos_edge for offset in range(1, 1 + num_neg_per_pos)]
            )

            batch_structure_pos = torch.tensor(
                structure_score[pos_ids], dtype=torch.float32, device=device
            )
            batch_structure_neg = torch.tensor(
                structure_score[neg_ids], dtype=torch.float32, device=device
            )

            delta_times = delta_times_list[batch_idx].squeeze(1).to(device)
            all_inds = all_inds_list[batch_idx].to(device)
            total_groups = (2 + num_neg_per_pos) * cur_batch_size

            group_ids = torch.div(all_inds, time_topk, rounding_mode="floor")
            counts = torch.bincount(group_ids, minlength=total_groups)
            max_delta_value = delta_times.max()
            sums = torch.bincount(group_ids, weights=delta_times, minlength=total_groups)
            padded_sums = sums + (time_topk - counts.to(delta_times.dtype)) * max_delta_value
            average_delta = padded_sums / time_topk
            average_delta = average_delta / average_delta.mean() - 1

            src_delta = average_delta[:cur_batch_size]
            pos_dst_delta = average_delta[cur_batch_size : 2 * cur_batch_size]
            neg_dst_delta = average_delta[2 * cur_batch_size :]

            pos_time_component = (
                (torch.exp(-src_delta) + torch.exp(-pos_dst_delta)) / 2
            ) * batch_time_pos_score
            neg_time_component = (
                (torch.exp(-src_delta.repeat(num_neg_per_pos)) + torch.exp(-neg_dst_delta)) / 2
            ) * batch_time_neg_score

            batches.append(
                {
                    "pos_structure": batch_structure_pos.cpu(),
                    "neg_structure": batch_structure_neg.cpu(),
                    "pos_time_component": pos_time_component.cpu(),
                    "neg_time_component": neg_time_component.cpu(),
                }
            )

    return batches


def evaluate_val_ap(prepared_batches: Sequence[Dict[str, torch.Tensor]], yita: float) -> float:
    ap_list = []
    for batch in prepared_batches:
        pos_score = batch["pos_structure"] + yita * batch["pos_time_component"]
        neg_score = batch["neg_structure"] + yita * batch["neg_time_component"]
        prediction = torch.cat([pos_score, neg_score], dim=0).numpy()
        target = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0).numpy()
        ap_list.append(average_precision_score(target, prediction))
    return float(np.mean(ap_list))


def search_best_yita(
    prepared_val_batches: Sequence[Dict[str, torch.Tensor]],
) -> Tuple[float, float]:
    best_yita = 0.0
    best_val_ap = float("-inf")
    for yita in YITA_VALUES:
        val_ap = evaluate_val_ap(prepared_val_batches, yita)
        print(f"yita: {yita:.0e} -- Val ap: {val_ap}")
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_yita = yita
    return best_yita, best_val_ap


def evaluate_test_metrics(
    prepared_batches: Sequence[Dict[str, torch.Tensor]],
    yita: float,
    k_list: Sequence[int],
):
    ap_list, mrr_list, hit_list = [], [], []
    cpu_device = torch.device("cpu")
    for batch in prepared_batches:
        pos_score = batch["pos_structure"] + yita * batch["pos_time_component"]
        neg_score = batch["neg_structure"] + yita * batch["neg_time_component"]
        ap, mrr, hr_list = compute_metrics(pos_score, neg_score, cpu_device, k_list=list(k_list))
        ap_list.append(ap)
        mrr_list.append(mrr)
        hit_list.append(hr_list)

    ap = float(np.mean(ap_list))
    mrr = float(np.mean(mrr_list))
    all_hr = np.mean(np.array(hit_list), axis=0)
    print(
        f"Test: ap: {ap:.4f}, mrr: {mrr:.4f}, "
        + ", ".join([f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, all_hr)])
        + "\n"
    )
    return ap, mrr, all_hr
