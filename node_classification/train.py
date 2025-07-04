import argparse
import timeit
from typing import List, Dict

import numpy as np
import torch
from numba import types
from numba.typed import Dict as NDict, List as NList
from numba.experimental import jitclass
from torch_geometric.loader import TemporalDataLoader

from utils import compute_metrics
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

class MovingAverage:
    def __init__(self, num_class, window=7):
        self.dict = {}
        self.num_class = num_class
        self.window = window

    def update_dict(self, node_id, label):
        if node_id in self.dict:
            total = self.dict[node_id] * (self.window - 1) + label
            self.dict[node_id] = total / self.window
        else:
            self.dict[node_id] = label

    def query_dict(self, node_id):
        if node_id in self.dict:
            return self.dict[node_id]
        else:
            return np.zeros(self.num_class)


nb_dict_type = types.DictType(types.int64, types.float64)

spec_tppr_node_finder = [
    ("num_nodes", types.int64),
    ("k", types.int64),
    ("alpha", types.float64), # restart/self weight
    ("beta", types.float64), # temporal decay factor
    ("norm_list", types.float64[:]),
    ("PPR_list", types.ListType(types.ListType(nb_dict_type))),
]

@jitclass(spec_tppr_node_finder)
class TPPRNodeFinder:
    """Maintain top-k TPPR dictionaries per node in an interaction stream."""

    def __init__(self, num_nodes: int, k: int, alpha: float, beta: float):
        self.num_nodes = num_nodes
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.reset_tppr()


    def reset_tppr(self):
        outer = NList()
        inner = NList()
        for _ in range(self.num_nodes):
            inner.append(NDict.empty(key_type=types.int64, value_type=types.float64))
        outer.append(inner)
        self.PPR_list = outer
        self.norm_list = np.zeros(self.num_nodes, dtype=np.float64)

    def update_tppr(self, source: int, target: int):
        """Update TPPR for `source` after seeing an edge source→target."""
        temp = self.PPR_list[0]
        src_ppr = temp[source]
        tgt_ppr = temp[target]
        norm = self.norm_list[source]

        if norm == 0:
            t_dict = NDict.empty(key_type=types.int64, value_type=types.float64)
            scale_target = 1.0 - self.alpha
        else:
            t_dict = src_ppr.copy()
            last_norm = norm
            new_norm = last_norm * self.beta + self.beta
            scale_source = last_norm / new_norm * self.beta
            scale_target = self.beta / new_norm * (1.0 - self.alpha)
            for k_ in t_dict.keys():
                t_dict[k_] = t_dict[k_] * scale_source

        if self.norm_list[target] == 0:
            t_dict[target] = scale_target * self.alpha if self.alpha != 0 else scale_target
        else:
            for k_, v in tgt_ppr.items():
                t_dict[k_] = t_dict.get(k_, 0.0) + v * scale_target
            t_dict[target] = t_dict.get(target, 0.0) + scale_target * (self.alpha if self.alpha != 0 else 1.0)

        if len(t_dict) > self.k:
            keys = list(t_dict.keys())
            vals = np.array([t_dict[x] for x in keys])
            idx = np.argsort(vals)[-self.k:]
            new_d = NDict.empty(key_type=types.int64, value_type=types.float64)
            for i in idx:
                new_d[keys[i]] = vals[i]
            t_dict = new_d

        temp[source] = t_dict
        self.norm_list[source] = self.norm_list[source] * self.beta + self.beta


def get_topk_neighbours(tppr_finder: TPPRNodeFinder, node_id: int, k: int) -> List[int]:
    # Return recent top-k neighbours of node_id
    tppr_d = tppr_finder.PPR_list[0][node_id]
    if len(tppr_d) == 0:
        return []
    keys = list(tppr_d.keys())
    vals = np.array([tppr_d[k_] for k_ in keys])
    order = np.argsort(vals)[::-1]
    order = order[:k]
    return [int(keys[i]) for i in order]


def run_epoch(loader: TemporalDataLoader,
              forecaster: MovingAverage,
              tppr_finder: TPPRNodeFinder,
              k: int,
              blend_alpha: float,
              metrics: List[str],
              dataset: PyGNodePropPredDataset,
              device: str = "cpu") -> Dict[str, float]:

    label_t = dataset.get_label_time()
    num_label_ts = 0
    total_scores = {m: 0.0 for m in metrics}

    for batch in loader:
        batch = batch.to(device)

        src_np = batch.src.cpu().numpy()
        dst_np = batch.dst.cpu().numpy()

        for s, d in zip(src_np, dst_np):
            tppr_finder.update_tppr(int(s), int(d))

        query_t = int(batch.t[-1].cpu().item())
        if query_t > label_t:
            lbl_tuple = dataset.get_node_label(query_t)
            if lbl_tuple is None:
                break
            _, label_nodes, labels = lbl_tuple
            label_nodes = label_nodes.cpu().numpy()
            labels = labels.cpu().numpy()
            label_t = dataset.get_label_time()

            preds = []
            for node_id in label_nodes:
                ma_self = forecaster.query_dict(node_id)
                neigh_ids = get_topk_neighbours(tppr_finder, int(node_id), k)
                if len(neigh_ids) == 0:
                    ma_neigh = np.zeros_like(ma_self)
                else:
                    neigh_vecs = [forecaster.query_dict(nid) for nid in neigh_ids]
                    ma_neigh = np.mean(neigh_vecs, axis=0)
                pred_vec = blend_alpha * ma_self + (1.0 - blend_alpha) * ma_neigh
                preds.append(pred_vec)

                lbl_idx = np.where(label_nodes == node_id)[0][0]
                forecaster.update_dict(int(node_id), labels[lbl_idx])

            y_pred = np.stack(preds, axis=0)
            y_true = labels
            score_dict = compute_metrics(y_true, y_pred, metrics)
            for m in metrics:
                total_scores[m] += score_dict[m]
            num_label_ts += 1

    if num_label_ts == 0:
        return {m: 0.0 for m in metrics}
    return {m: total_scores[m] / num_label_ts for m in metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="tgbn-trade")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.5, help="Weight for self vs neighbour")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tppr_alpha", type=float, default=0.2, help="TPPR restart weight α")
    parser.add_argument("--tppr_beta", type=float, default=0.9, help="Temporal decay β for TPPR")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = "cpu" if not torch.cuda.is_available() else f"cuda:{args.gpu}"

    dataset = PyGNodePropPredDataset(name=args.dataset_name, root=args.root)
    num_classes = dataset.num_classes
    data = dataset.get_TemporalData().to(device)

    train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    loaders = {
        "train": TemporalDataLoader(train_data, batch_size=args.batch_size),
        "val":   TemporalDataLoader(val_data, batch_size=args.batch_size),
        "test":  TemporalDataLoader(test_data, batch_size=args.batch_size),
    }

    forecaster = MovingAverage(num_classes, window=args.window)
    num_nodes = int(max(data.src.max(), data.dst.max())) + 1
    tppr_finder = TPPRNodeFinder(num_nodes=num_nodes, k=args.k, alpha=args.tppr_alpha, beta=args.tppr_beta)

    metrics_train = ["ndcg"]
    metrics_test = ["ndcg", "mrr", "hit_ratio"]

    for split in ["train", "val", "test"]:
        start = timeit.default_timer()
        metrics = metrics_train if split != "test" else metrics_test
        result = run_epoch(loaders[split], forecaster, tppr_finder, args.k, args.gamma, metrics, dataset, device)
        elapsed = timeit.default_timer() - start
        metric_str = ", ".join(f"{k}: {v:.4f}" for k,v in result.items())
        print(f"{split.title():5s} | {metric_str} | {elapsed:.1f}s")

    dataset.reset_label_time()


if __name__ == "__main__":
    main()
