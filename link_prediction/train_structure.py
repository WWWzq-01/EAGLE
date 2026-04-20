import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from utils.data_processing import get_data_transductive
from utils.structure_cache import (
    advance_uniform_rng,
    build_shared_structure_cache_root,
    materialize_tppr_payload,
    stage_raw_cache_path,
)
from utils.util import NegEdgeSampler, compute_metrics, set_random_seed, tppr_node_finder

import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser("Training EAGLE-Structure.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="wikipedia",
    choices=["Contacts", "lastfm", "wikipedia", "reddit", "superuser", "askubuntu", "wikitalk"],
)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--topk", type=int, default=100)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--beta", type=float, default=0.8)
parser.add_argument("--sim", type=str, default="mul_wo_norm")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--workspace_root", type=str, default=".")
parser.add_argument("--report_dir", type=str, default="")
parser.add_argument("--shared_raw_cache_base", type=str, default="")

args = parser.parse_args()

WORKSPACE_ROOT = Path(args.workspace_root).resolve()
REPORT_DIR = Path(args.report_dir).resolve() if args.report_dir else None
SHARED_RAW_CACHE_BASE = Path(args.shared_raw_cache_base).resolve() if args.shared_raw_cache_base else None


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def workspace_path(*parts: str) -> Path:
    return WORKSPACE_ROOT.joinpath(*parts)


device = torch.device(f"cuda:{args.gpu}")
set_random_seed(args.seed)
DATA = args.dataset_name

train_bs = args.batch_size
val_bs = args.batch_size
test_bs = args.batch_size

filename = "topk_" + str(args.topk) + "_alpha_" + str(args.alpha) + "_beta_" + str(args.beta)
shared_cache_key = filename + "_" + args.sim + "_bs_" + str(args.batch_size)

log_dir = workspace_path("log_learn_structure", DATA)
log_dir.mkdir(parents=True, exist_ok=True)
filepath = log_dir / filename
structure_score_dir = workspace_path("structure_score_cache", DATA)
structure_score_dir.mkdir(parents=True, exist_ok=True)
shared_structure_raw_root = build_shared_structure_cache_root(SHARED_RAW_CACHE_BASE, DATA, shared_cache_key)


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    if isinstance(obj, dict):
        return {key: move_to_cpu(value) for key, value in obj.items()}
    return obj


def compute_raw_tppr_stats(mode, tppr_finder, data, neg_edge_sampler, dataset_name, bs, num_neg):
    sources_all = data.sources
    destinations_all = data.destinations
    timestamps_all = data.timestamps

    num_instance = data.n_interactions
    batch_size = bs if bs != -1 else num_instance
    num_batch = math.ceil(num_instance / batch_size)

    negatives_all = None
    torch.cuda.reset_max_memory_allocated()
    batchneg_dir = workspace_path("data", "batchneg", dataset_name)
    batchneg_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx in tqdm(range(0, num_batch)):
        start_idx = batch_idx * batch_size
        end_idx = min(num_instance, start_idx + batch_size)
        sample_inds = np.array(list(range(start_idx, end_idx)))

        destinations_batch = data.destinations[sample_inds]
        neg_filepath = batchneg_dir / f"{mode}_neg{num_neg}_bs{bs}_batch{batch_idx}.pkl"
        if neg_filepath.exists():
            with neg_filepath.open("rb") as f:
                negatives_batch = pickle.load(f)
        else:
            negatives_batch = neg_edge_sampler.sample(destinations_batch)
            negatives_batch = move_to_cpu(negatives_batch)
            with neg_filepath.open("wb") as f:
                pickle.dump(negatives_batch, f)

        if negatives_all is None:
            negatives_all = negatives_batch
        else:
            negatives_all = torch.cat((negatives_all, negatives_batch), dim=0)

    negatives_all = negatives_all.t().flatten()
    if isinstance(negatives_all, torch.Tensor):
        negatives_all = negatives_all.cpu().numpy()

    source_nodes = np.concatenate([sources_all, destinations_all, negatives_all], dtype=np.int32)

    t1 = time.time()
    scores = tppr_finder.precompute_link_prediction(source_nodes, num_neg)
    t_cal_tppr_score = time.time() - t1
    allocated_memory = torch.cuda.max_memory_allocated() / (1024**2)

    payload = move_to_cpu((source_nodes, timestamps_all, scores, t_cal_tppr_score, allocated_memory))
    return payload, t_cal_tppr_score, allocated_memory


def load_or_materialize_tppr_stats(
    mode,
    local_cache_path: Path,
    shared_raw_path: Path,
    tppr_finder,
    data,
    neg_edge_sampler,
    dataset_name,
    bs,
    num_neg,
    rng: np.random.RandomState,
):
    if local_cache_path.exists():
        payload = load_pickle(local_cache_path)
        advance_uniform_rng(rng, payload[2].shape)
        print(f"Loading {mode} TPPR data from {local_cache_path}")
        return payload, payload[3], payload[4]

    raw_payload = None
    if shared_raw_path is not None and shared_raw_path.exists():
        raw_payload = load_pickle(shared_raw_path)
        print(f"Loading {mode} raw TPPR data from {shared_raw_path}")
    else:
        raw_payload, _, _ = compute_raw_tppr_stats(mode, tppr_finder, data, neg_edge_sampler, dataset_name, bs, num_neg)
        if shared_raw_path is not None:
            save_pickle(shared_raw_path, raw_payload)
            print(f"{mode} raw TPPR data has been saved at {shared_raw_path}")

    payload = materialize_tppr_payload(raw_payload, rng)
    save_pickle(local_cache_path, move_to_cpu(payload))
    print(f"{mode} TPPR data has been saved at {local_cache_path}")
    return payload, payload[3], payload[4]


def get_cached_tppr_status(
    finder,
    dataset_name,
    train_data,
    val_data,
    test_data,
    train_neg_edge_sampler,
    val_neg_edge_sampler,
    test_neg_edge_sampler,
    cache_filename,
):
    train_file = structure_score_dir / f"train_{cache_filename}"
    rng = np.random.RandomState(args.seed)
    train_stats, t_cal_train_tppr, mem_train = load_or_materialize_tppr_stats(
        "Train",
        train_file,
        stage_raw_cache_path(shared_structure_raw_root, "train"),
        finder,
        train_data,
        train_neg_edge_sampler,
        dataset_name,
        train_bs,
        num_neg=1,
        rng=rng,
    )

    val_file = structure_score_dir / f"val_{cache_filename}"
    val_stats, t_cal_val_tppr, mem_val = load_or_materialize_tppr_stats(
        "Val",
        val_file,
        stage_raw_cache_path(shared_structure_raw_root, "val"),
        finder,
        val_data,
        val_neg_edge_sampler,
        dataset_name,
        val_bs,
        num_neg=1,
        rng=rng,
    )

    test_file = structure_score_dir / f"test_{cache_filename}"
    test_stats, t_cal_test_tppr, mem_test = load_or_materialize_tppr_stats(
        "Test",
        test_file,
        stage_raw_cache_path(shared_structure_raw_root, "test"),
        finder,
        test_data,
        test_neg_edge_sampler,
        dataset_name,
        test_bs,
        num_neg=99,
        rng=rng,
    )

    return val_stats, test_stats, t_cal_train_tppr, t_cal_val_tppr, t_cal_test_tppr, mem_train, mem_val, mem_test


def get_scores(data, tppr_stats, cached_neg_samples):
    tppr_scores = tppr_stats[2]
    num_instance = data.n_interactions
    sample_inds = np.array(list(range(0, num_instance)))

    neg_sample_inds = np.concatenate([sample_inds + i * num_instance for i in range(1, 1 + cached_neg_samples)])
    pos_score_structure = tppr_scores[sample_inds]
    neg_score_structure = tppr_scores[neg_sample_inds]
    pos_score_structure = torch.from_numpy(pos_score_structure).unsqueeze(-1).type(torch.float)
    neg_score_structure = torch.from_numpy(neg_score_structure).unsqueeze(-1).type(torch.float)

    return pos_score_structure, neg_score_structure


full_data, train_data, val_data, test_data, n_nodes, n_edges = get_data_transductive(DATA, use_validation=True)
n_train = train_data.n_interactions
n_val = val_data.n_interactions
n_test = test_data.n_interactions
print(f"#Edge: train {n_train}, val {n_val}, test {n_test}")

train_neg_edge_sampler = NegEdgeSampler(
    destinations=train_data.destinations,
    full_destinations=train_data.destinations,
    num_neg=1,
    device=device,
    seed=2024,
)
val_neg_edge_sampler = NegEdgeSampler(
    destinations=val_data.destinations,
    full_destinations=full_data.destinations,
    num_neg=1,
    device=device,
    seed=2025,
)
test_neg_edge_sampler = NegEdgeSampler(
    destinations=test_data.destinations,
    full_destinations=full_data.destinations,
    num_neg=99,
    device=device,
    seed=2026,
)

tppr_finder = tppr_node_finder(n_nodes + 1, args.topk, args.alpha, args.beta, args.sim)
tppr_finder.reset_tppr()

cache_filename = "topk_" + str(args.topk) + "_alpha_" + str(args.alpha) + "_beta_" + str(args.beta) + "_" + args.sim

val_stats, test_stats, t_train, t_val, t_test, mem_train, mem_val, mem_test = get_cached_tppr_status(
    tppr_finder,
    DATA,
    train_data,
    val_data,
    test_data,
    train_neg_edge_sampler,
    val_neg_edge_sampler,
    test_neg_edge_sampler,
    cache_filename,
)

with torch.no_grad():
    pos_score_val, neg_score_val = get_scores(val_data, val_stats, cached_neg_samples=1)
    y_pred = torch.cat([pos_score_val, neg_score_val], dim=0).cpu().detach()
    y_true = torch.cat([torch.ones_like(pos_score_val), torch.zeros_like(neg_score_val)], dim=0).cpu().detach()
    val_ap = average_precision_score(y_true, y_pred)
    print(f"Val: ap: {val_ap:.4f}")
    sys.stdout.flush()

    pos_score_test, neg_score_test = get_scores(test_data, test_stats, cached_neg_samples=99)
    k_list = [10]
    test_ap, test_mrr, test_hr_list = compute_metrics(pos_score_test, neg_score_test, device, k_list=k_list)

    print(
        f"Test: ap: {test_ap:.4f}, mrr: {test_mrr:.4f}, "
        + ", ".join([f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, test_hr_list)])
    )

    with filepath.open("a", encoding="utf-8") as log_file:
        log_file.write(f"Val ap = {val_ap:.4f}\n")
        log_file.write(
            f"Test ap = {test_ap:.4f}, Test mrr = {test_mrr:.4f}, "
            + ", ".join([f"hr@{k} = {hr:.4f}" for k, hr in zip(k_list, test_hr_list)])
        )

    print(f"Results have been save at {filename}!")
    sys.stdout.flush()

if REPORT_DIR is not None:
    summary_payload = {
        "system": "EAGLE-Structure",
        "model": "EAGLE",
        "stage": "structure",
        "dataset": DATA,
        "run_type": "tta",
        "config": {
            "dataset_name": DATA,
            "batch_size": args.batch_size,
            "topk": args.topk,
            "alpha": args.alpha,
            "beta": args.beta,
            "sim": args.sim,
            "gpu": args.gpu,
            "seed": args.seed,
            "workspace_root": str(WORKSPACE_ROOT),
            "cache_filename": cache_filename,
            "shared_cache_key": shared_cache_key,
        },
        "best_val_ap": float(val_ap),
        "best_val_auc": None,
        "final_test_metrics": {
            "ap": float(test_ap),
            "auc_or_mrr": float(test_mrr),
            "hit_at_10": float(test_hr_list[0]) if len(test_hr_list) > 0 else None,
        },
        "paths": {
            "log_path": str(filepath),
            "train_cache_path": str(structure_score_dir / f"train_{cache_filename}"),
            "val_cache_path": str(structure_score_dir / f"val_{cache_filename}"),
            "test_cache_path": str(structure_score_dir / f"test_{cache_filename}"),
            "shared_raw_cache_root": str(shared_structure_raw_root) if shared_structure_raw_root is not None else "",
        },
        "wall_seconds": {
            "train": round(float(t_train), 6),
            "val": round(float(t_val), 6),
            "test": round(float(t_test), 6),
            "total": round(float(t_train + t_val + t_test), 6),
        },
        "memory_mb": {
            "train": float(mem_train),
            "val": float(mem_val),
            "test": float(mem_test),
        },
    }
    dump_json(REPORT_DIR / "summary.json", summary_payload)
