import argparse
import os
import sys
import time
import math
import pickle
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from utils.util import set_random_seed, compute_metrics, tppr_node_finder, NegEdgeSampler
from utils.data_processing import get_data_transductive

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Training EAGLE-Structure.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="wikipedia",
    choices=["Contacts", "lastfm", "wikipedia", "reddit", "superuser", "askubuntu", "wikitalk"]
)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--topk", type=int, default=100)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.9
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.8,
)
parser.add_argument(
    "--sim",
    type=str,
    default="mul_wo_norm",
)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=2024)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}")
set_random_seed(args.seed)
DATA = args.dataset_name

train_bs = args.batch_size
val_bs = args.batch_size
test_bs = args.batch_size

filename = (
    "topk_"
    + str(args.topk)
    + "_alpha_"
    + str(args.alpha)
    + "_beta_"
    + str(args.beta)
)

Path(f"log_learn_structure/{DATA}").mkdir(parents=True, exist_ok=True)
filepath = f"log_learn_structure/{DATA}/{filename}"
Path(f"structure_score_cache/{DATA}").mkdir(parents=True, exist_ok=True)


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    else:
        return obj


def cal_tppr_stats(mode, tppr_finder, data, neg_edge_sampler, filepath, DATA, bs, num_neg):
    sources_all = data.sources
    destinations_all = data.destinations
    timestamps_all = data.timestamps

    num_instance = data.n_interactions
    BATCH_SIZE = bs if bs != -1 else num_instance
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    negatives_all = None
    torch.cuda.reset_max_memory_allocated()

    for batch_idx in tqdm(range(0, num_batch)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sample_inds = np.array(list(range(start_idx, end_idx)))

        destinations_batch = data.destinations[sample_inds]
        
        neg_filepath = (
            f"./data/batchneg/{DATA}/{mode}_neg{num_neg}_bs{bs}_batch{batch_idx}.pkl"
        )
        if os.path.exists(neg_filepath):
            with open(neg_filepath, "rb") as f:
                negatives_batch = pickle.load(f)
        else:
            negatives_batch = neg_edge_sampler.sample(destinations_batch)

            negatives_batch = move_to_cpu(negatives_batch)

            os.makedirs(os.path.dirname(neg_filepath), exist_ok=True)
            with open(neg_filepath, "wb") as f:
                pickle.dump(negatives_batch, f)
        
        if negatives_all is None:
            negatives_all = negatives_batch
        else:
            negatives_all = torch.cat((negatives_all, negatives_batch), dim=0)


    negatives_all = negatives_all.t().flatten() # arr[dst(1)_neg1, dst(2)_neg1, ..., dst(n_edge)_neg1, dst(1)_neg2, ...]
    
    if isinstance(negatives_all, torch.Tensor):
        negatives_all = negatives_all.cpu().numpy()

    source_nodes = np.concatenate(
        [sources_all, destinations_all, negatives_all], dtype=np.int32
    )

    t1 = time.time()
    scores = tppr_finder.precompute_link_prediction(
        source_nodes, num_neg
    ) # concat[pos_score*SIZE, neg_score_1*SIZE, ..., neg_score_num_neg*SIZE]
    t_cal_tppr_score = time.time() - t1
    allocated_memory = torch.cuda.max_memory_allocated() / (1024**2) # /MB

    noise4zeros = np.random.uniform(0, 1e-8, scores.shape)
    scores[scores == 0.0] += noise4zeros[scores == 0.0]

    data = source_nodes, timestamps_all, scores, t_cal_tppr_score, allocated_memory

    data = move_to_cpu(data)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"{mode} TPPR data has been saved at {filepath}")

    return data, t_cal_tppr_score, allocated_memory


def get_cached_tppr_status(
    finder,
    DATA,
    train_data,
    val_data,
    test_data,
    train_neg_edge_sampler,
    val_neg_edge_sampler,
    test_neg_edge_sampler,
    filename,
):
    train_file = os.path.join(f"structure_score_cache/{DATA}/train_{filename}")
    if not os.path.exists(train_file):
        train_stats, t_cal_train_tppr, mem_train = cal_tppr_stats(
            "Train",
            finder,
            train_data,
            train_neg_edge_sampler,
            train_file,
            DATA,
            train_bs,
            num_neg=1,
        )
    else:
        f = open(train_file, "rb")
        train_stats = pickle.load(f)
        t_cal_train_tppr = train_stats[3]
        mem_train = train_stats[4]
        print(f"Loading Train TPPR data from {train_file}")

    val_file = os.path.join(f"structure_score_cache/{DATA}/val_{filename}")
    if not os.path.exists(val_file):
        val_stats, t_cal_val_tppr, mem_val = cal_tppr_stats(
            "Val",
            finder,
            val_data,
            val_neg_edge_sampler,
            val_file,
            DATA,
            val_bs,
            num_neg=1,
        )
    else:
        f = open(val_file, "rb")
        val_stats = pickle.load(f)
        t_cal_val_tppr = val_stats[3]
        mem_val = val_stats[4]
        print(f"Loading Val TPPR data from {val_file}")

    test_file = os.path.join(f"structure_score_cache/{DATA}/test_{filename}")
    if not os.path.exists(test_file):
        test_stats, t_cal_test_tppr, mem_test = cal_tppr_stats(
            "Test",
            finder,
            test_data,
            test_neg_edge_sampler,
            test_file,
            DATA,
            test_bs,
            num_neg=99,
        )
    else:
        f = open(test_file, "rb")
        test_stats = pickle.load(f)
        t_cal_test_tppr = test_stats[3]
        mem_test = test_stats[4]
        print(f"Loading Test TPPR data from {test_file}")

    return val_stats, test_stats, t_cal_train_tppr, t_cal_val_tppr, t_cal_test_tppr, mem_train, mem_val, mem_test


def get_scores(data, tppr_stats, cached_neg_samples):
    tppr_scores = tppr_stats[2]
    num_instance = data.n_interactions
    sample_inds = np.array(list(range(0, num_instance)))

    neg_sample_inds = np.concatenate(
        [sample_inds + i * num_instance for i in range(1, 1 + cached_neg_samples)]
    )
    pos_score_structure = tppr_scores[sample_inds]
    neg_score_structure = tppr_scores[neg_sample_inds]
    pos_score_structure = (
        torch.from_numpy(pos_score_structure).unsqueeze(-1).type(torch.float)
    )
    neg_score_structure = (
        torch.from_numpy(neg_score_structure).unsqueeze(-1).type(torch.float)
    )

    return pos_score_structure, neg_score_structure


full_data, train_data, val_data, test_data, n_nodes, n_edges = get_data_transductive(
    DATA, use_validation=True
)
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

tppr_finder = tppr_node_finder(
    n_nodes + 1, args.topk, args.alpha, args.beta, args.sim
)
tppr_finder.reset_tppr()

cache_filename = (
    "topk_"
    + str(args.topk)
    + "_alpha_"
    + str(args.alpha)
    + "_beta_"
    + str(args.beta)
    + "_"
    + args.sim
)

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
    pos_score_val, neg_score_val = get_scores(
        val_data, val_stats, cached_neg_samples=1
    )
    with torch.no_grad():
        y_pred = (
            torch.cat([pos_score_val, neg_score_val], dim=0)
            .cpu()
            .detach()
        )
        y_true = (
            torch.cat(
                [
                    torch.ones_like(pos_score_val),
                    torch.zeros_like(neg_score_val),
                ],
                dim=0,
            )
            .cpu()
            .detach()
        )
        val_ap = average_precision_score(y_true, y_pred)
    print(f"Val: ap: {val_ap:.4f}")
    sys.stdout.flush()

    pos_score_test, neg_score_test = get_scores(
        test_data, test_stats, cached_neg_samples=99
    )
    
    k_list = [10]
    test_ap, test_mrr, test_hr_list = compute_metrics(
        pos_score_test, neg_score_test, device, k_list=k_list
    )

    print(
        f"Test: ap: {test_ap:.4f}, mrr: {test_mrr:.4f}, "
        + ", ".join(
            [f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, test_hr_list)]
        )
    )

    with open(filepath, "a") as log_file:
        log_file.write(
            f"Val ap = {val_ap:.4f}\n"
        )
        log_file.write(
            f"Test ap = {test_ap:.4f}, Test mrr = {test_mrr:.4f}, "
            + ", ".join(
                [f"hr@{k} = {hr:.4f}" for k, hr in zip(k_list, test_hr_list)]
            )
        )
    
    print(f"Results have been save at {filename}!")
        
    sys.stdout.flush()
