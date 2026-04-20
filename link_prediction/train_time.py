import argparse
import csv
import json
import math
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numba as nb
import numpy as np
import torch
from numba import typed
from numba.experimental import jitclass
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from utils.data_processing import get_data_transductive
from utils.model import Mixer_per_node
from utils.util import EarlyStopMonitor, NegEdgeSampler, compute_metrics, print_model_info, set_random_seed

import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser("Training EAGLE-Time.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="wikipedia",
    choices=["Contacts", "lastfm", "wikipedia", "reddit", "superuser", "askubuntu", "wikitalk"],
)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--topk", type=int, default=15)
parser.add_argument("--topk_sample_flag", type=str, default="last")

parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=5e-5)

parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--hidden_dims", type=int, default=100)

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--workspace_root", type=str, default=".")
parser.add_argument("--report_dir", type=str, default="")
parser.add_argument("--save_epoch_score_cache", action="store_true")
parser.add_argument("--min_epoch_before_stop", type=int, default=0)
parser.add_argument("--early_stop_min_delta", type=float, default=0.0)

args = parser.parse_args()

WORKSPACE_ROOT = Path(args.workspace_root).resolve()
REPORT_DIR = Path(args.report_dir).resolve() if args.report_dir else None


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def workspace_path(*parts: str) -> Path:
    return WORKSPACE_ROOT.joinpath(*parts)


device = torch.device(f"cuda:{args.gpu}")
set_random_seed(args.seed)
DATA = args.dataset_name
args.ignore_zero = True

filename = (
    "topk_"
    + str(args.topk)
    + "_flag_"
    + args.topk_sample_flag
    + "_lr_"
    + str(args.lr)
    + "_wd_"
    + str(args.weight_decay)
    + "_bs_"
    + str(args.batch_size)
)


def get_neighbor_finder(data):
    max_node_idx = max(data.sources.max(), data.destinations.max())
    adj_list = [[] for _ in range(max_node_idx + 1)]

    for source, destination, edge_idx, timestamp in zip(
        data.sources, data.destinations, data.edge_idxs, data.timestamps
    ):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    node_to_neighbors = typed.List()
    node_to_edge_idxs = typed.List()
    node_to_edge_timestamps = typed.List()

    for neighbors in adj_list:
        sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
        node_to_neighbors.append(np.array([x[0] for x in sorted_neighbors], dtype=np.int32))
        node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbors], dtype=np.int32))
        node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighbors], dtype=np.float64))

    return NeighborFinder(node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps)


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    if isinstance(obj, dict):
        return {key: move_to_cpu(value) for key, value in obj.items()}
    return obj


def save_score_cache(path: Path, pos_scores, neg_scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(move_to_cpu((pos_scores, neg_scores)), f)


l_int = typed.List()
l_float = typed.List()
a_int = np.array([1, 2], dtype=np.int32)
a_float = np.array([1, 2], dtype=np.float64)
l_int.append(a_int)
l_float.append(a_float)
spec = [
    ("node_to_neighbors", nb.typeof(l_int)),
    ("node_to_edge_idxs", nb.typeof(l_int)),
    ("node_to_edge_timestamps", nb.typeof(l_float)),
]


@jitclass(spec)
class NeighborFinder:
    def __init__(self, node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps):
        self.node_to_neighbors = node_to_neighbors
        self.node_to_edge_idxs = node_to_edge_idxs
        self.node_to_edge_timestamps = node_to_edge_timestamps

    def find_before(self, src_idx, cut_time):
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return (
            self.node_to_neighbors[src_idx][:i],
            self.node_to_edge_idxs[src_idx][:i],
            self.node_to_edge_timestamps[src_idx][:i],
        )

    def get_clean_delta_times(self, source_nodes, timestamps, n_neighbors, topk_sample_flag="last"):
        if topk_sample_flag not in ["last", "early", "random"]:
            raise ValueError("TopK sample flag must be in ['last', 'early', 'random']")

        if topk_sample_flag == "random":
            np.random.seed(2024)

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        delta_times = np.zeros(len(source_nodes) * tmp_n_neighbors, dtype=np.float32)
        n_edges = np.zeros(len(source_nodes), dtype=np.int32)
        cum_sum = 0
        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            _, _, edge_times = self.find_before(source_node, timestamp)
            n_ngh = len(edge_times)
            if n_ngh > 0:
                if topk_sample_flag == "last":
                    selected_times = edge_times[-n_neighbors:][::-1]
                elif topk_sample_flag == "early":
                    selected_times = edge_times[:n_neighbors]
                else:
                    if n_ngh <= n_neighbors:
                        selected_times = edge_times
                    else:
                        selected_indices = np.random.choice(n_ngh, n_neighbors, replace=False)
                        selected_times = np.sort(edge_times[selected_indices])

                n_ngh = len(selected_times)
                delta_times[cum_sum : cum_sum + n_ngh] = timestamp - selected_times

            n_edges[i] = n_ngh
            cum_sum += n_ngh
        return delta_times, n_edges, cum_sum


log_dir = workspace_path("log_learn_time", DATA)
time_score_dir = workspace_path("time_score_cache", DATA)
checkpoint_dir = workspace_path("saved_checkpoints")
saved_model_dir = workspace_path("saved_time_models", "learn_time", DATA)
processed_data_dir = workspace_path("time_processed_data", DATA)
epoch_score_dir = workspace_path("time_epoch_scores", DATA, filename)

log_dir.mkdir(parents=True, exist_ok=True)
time_score_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
saved_model_dir.mkdir(parents=True, exist_ok=True)
processed_data_dir.mkdir(parents=True, exist_ok=True)
epoch_score_dir.mkdir(parents=True, exist_ok=True)

print(f"Time training result will be written at {log_dir / filename}.\n")

best_checkpoint_path = checkpoint_dir / f"{filename}.pth"
best_model_path = saved_model_dir / f"{filename}.pth"


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

finder = get_neighbor_finder(full_data)

edge_predictor_configs = {
    "dim": 100,
}

mixer_configs = {
    "per_graph_size": args.topk,
    "time_channels": 100,
    "num_layers": args.num_layers,
    "use_single_layer": False,
    "device": device,
}

model = Mixer_per_node(mixer_configs, edge_predictor_configs).to(device)

print_model_info(model)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
early_stopper = EarlyStopMonitor(
    max_round=args.patience,
    min_epoch_before_stop=args.min_epoch_before_stop,
    min_delta=args.early_stop_min_delta,
)


def process_data(mode, finder, data, bs, num_neg=1, filepath: Optional[Path] = None):
    print(f"Processing {mode} data...")
    num_instance = data.n_interactions
    batch_size = bs if bs != -1 else num_instance
    num_batch = math.ceil(num_instance / batch_size)

    delta_times_list = []
    all_inds_list = []
    batch_size_list = []

    batchneg_dir = workspace_path("data", "batchneg", DATA)
    batchneg_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx in tqdm(range(0, num_batch)):
        start_idx = batch_idx * batch_size
        end_idx = min(num_instance, start_idx + batch_size)
        sample_inds = np.array(list(range(start_idx, end_idx)))

        sources_batch = data.sources[sample_inds]
        destinations_batch = data.destinations[sample_inds]
        timestamps_batch = data.timestamps[sample_inds]

        if mode in {"Train", "Val"}:
            neg_filepath = batchneg_dir / f"{mode}_neg1_bs{bs}_batch{batch_idx}.pkl"
            if neg_filepath.exists():
                with neg_filepath.open("rb") as f:
                    negatives_batch = pickle.load(f).t().flatten()
            else:
                if mode == "Train":
                    negatives_batch = train_neg_edge_sampler.sample(destinations_batch).t().flatten()
                else:
                    negatives_batch = val_neg_edge_sampler.sample(destinations_batch).t().flatten()
                negatives_batch = move_to_cpu(negatives_batch)
                with neg_filepath.open("wb") as f:
                    pickle.dump(negatives_batch, f)
        else:
            neg_filepath = batchneg_dir / f"{mode}_neg99_bs{bs}_batch{batch_idx}.pkl"
            if neg_filepath.exists():
                with neg_filepath.open("rb") as f:
                    negatives_batch = pickle.load(f).t().flatten()
            else:
                negatives_batch = test_neg_edge_sampler.sample(destinations_batch).t().flatten()
                negatives_batch = move_to_cpu(negatives_batch)
                with neg_filepath.open("wb") as f:
                    pickle.dump(negatives_batch, f)

        if isinstance(negatives_batch, torch.Tensor):
            negatives_batch = negatives_batch.cpu().numpy()

        source_nodes = np.concatenate([sources_batch, destinations_batch, negatives_batch], dtype=np.int32)
        timestamps = np.tile(timestamps_batch, num_neg + 2)

        delta_times, n_neighbors, total_edges = finder.get_clean_delta_times(
            source_nodes, timestamps, args.topk, args.topk_sample_flag
        )
        delta_times = torch.from_numpy(delta_times[:total_edges]).to(device).unsqueeze(-1)

        all_inds = []
        for i, n_ngh in enumerate(n_neighbors):
            all_inds.extend([(args.topk * i + j) for j in range(n_ngh)])

        all_inds = torch.tensor(all_inds, device=device)
        cur_batch_size = len(n_neighbors)

        delta_times_list.append(move_to_cpu(delta_times))
        all_inds_list.append(move_to_cpu(all_inds))
        batch_size_list.append(cur_batch_size)

    if filepath:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f:
            pickle.dump((num_batch, delta_times_list, all_inds_list, batch_size_list), f)
    print(f"Processed data has been saved at {filepath}.")

    return num_batch, delta_times_list, all_inds_list, batch_size_list


def process_time_data(mode, finder, data, batch_size, num_neg=1, filepath: Optional[Path] = None):
    if filepath and filepath.exists():
        print(f"Loading cached {mode} data from {filepath}.")
        with filepath.open("rb") as f:
            num_batch, delta_times_list, all_inds_list, batch_size_list = pickle.load(f)
    else:
        num_batch, delta_times_list, all_inds_list, batch_size_list = process_data(
            mode, finder, data, batch_size, num_neg, filepath
        )

    all_inds_list = [tensor.to(device) for tensor in all_inds_list]
    delta_times_list = [tensor.to(device) for tensor in delta_times_list]

    return num_batch, delta_times_list, all_inds_list, batch_size_list


def run(
    model,
    mode,
    epoch,
    optimizer,
    criterion,
    log_path: Path,
    num_neg,
    num_batch,
    delta_times_list=None,
    all_inds_list=None,
    batch_size_list=None,
    k_list=[10],
    record_log=True,
):
    if mode == "Train":
        model = model.train()
    else:
        model = model.eval()

    ap_list, mrr_list, hit_list = [], [], []
    t_epo = 0.0
    allocated_memory = 0.0
    loss_value = None

    all_pos_score = []
    all_neg_score = []

    for batch_idx in tqdm(range(num_batch)):
        t1 = time.time()
        torch.cuda.reset_max_memory_allocated(device)
        no_neighbor_flag = False

        if delta_times_list[batch_idx].numel() == 0:
            no_neighbor_flag = True
            num_pos_sc = batch_size_list[batch_idx] // (num_neg + 2)
            num_neg_sc = num_pos_sc * num_neg
            pos_score = torch.zeros(num_pos_sc, 1)
            neg_score = torch.zeros(num_neg_sc, 1)
        else:
            pos_score, neg_score = model(
                delta_times_list[batch_idx],
                all_inds_list[batch_idx],
                batch_size_list[batch_idx],
                num_neg,
            )

        mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        allocated_memory = max(allocated_memory, mem)
        t_epo += time.time() - t1

        if mode == "Train" and not no_neighbor_flag:
            t2 = time.time()
            optimizer.zero_grad()
            predicts = torch.cat([pos_score, neg_score], dim=0).to(device)
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0).to(device)
            loss = criterion(input=predicts, target=labels)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            t_epo += time.time() - t2
        elif mode == "Val":
            with torch.no_grad():
                all_pos_score.append(pos_score.sigmoid().cpu())
                all_neg_score.append(neg_score.sigmoid().cpu())
                y_pred = torch.cat([pos_score, neg_score], dim=0).sigmoid().cpu().detach()
                y_true = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0).cpu().detach()
                ap = average_precision_score(y_true, y_pred)
                ap_list.append(ap)
        elif mode == "Test":
            with torch.no_grad():
                all_pos_score.append(pos_score.sigmoid().cpu())
                all_neg_score.append(neg_score.sigmoid().cpu())
                ap, mrr, hr_list = compute_metrics(pos_score.sigmoid(), neg_score.sigmoid(), device, k_list=k_list)
                ap_list.append(ap)
                mrr_list.append(mrr)
                hit_list.append(hr_list)

    if mode == "Train":
        print(
            f"Epoch{epoch}-{mode}: loss: {loss_value if loss_value is not None else float('nan'):.5f}, "
            f"time: {t_epo}, memory used: {allocated_memory}"
        )
        return {
            "loss": loss_value,
            "t_epo": t_epo,
            "allocated_memory": allocated_memory,
            "ap": None,
            "mrr": None,
            "hit_at_10": None,
            "pos_scores": None,
            "neg_scores": None,
        }

    if mode == "Val":
        ap = float(np.mean(ap_list))
        print(f"Epoch{epoch}-{mode}: ap: {ap:.4f}, time: {t_epo}, memory used: {allocated_memory}")
        return {
            "loss": None,
            "t_epo": t_epo,
            "allocated_memory": allocated_memory,
            "ap": ap,
            "mrr": None,
            "hit_at_10": None,
            "pos_scores": all_pos_score,
            "neg_scores": all_neg_score,
        }

    ap = float(np.mean(ap_list))
    mrr = float(np.mean(mrr_list))
    hit_array = np.array(hit_list)
    mean_hr = np.mean(hit_array, axis=0)
    print(
        f"Epoch{epoch}-{mode}-Neg_sam{num_neg}: time: {t_epo}, ap: {ap:.4f}, mrr: {mrr:.4f}, "
        + ", ".join([f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, mean_hr)])
        + f", memory used: {allocated_memory}"
    )
    if record_log:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"Final Test with Neg_sam {num_neg}: ap: {ap:.4f}, mrr: {mrr:.4f}, "
                + ", ".join([f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, mean_hr)])
                + "\n"
            )
    return {
        "loss": None,
        "t_epo": t_epo,
        "allocated_memory": allocated_memory,
        "ap": ap,
        "mrr": mrr,
        "hit_at_10": float(mean_hr[0]) if len(mean_hr) > 0 else None,
        "pos_scores": all_pos_score,
        "neg_scores": all_neg_score,
    }


train_bs = args.batch_size
val_bs = args.batch_size
test_bs = args.batch_size

train_filepath = processed_data_dir / f"Train_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{train_bs}_numneg_1.pkl"
val_filepath = processed_data_dir / f"Val_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{val_bs}_numneg_1.pkl"
test_filepath = processed_data_dir / f"Test_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{test_bs}_numneg_99.pkl"

prepare_started = time.perf_counter()
train_num_batch, train_delta_times_list, train_all_inds_list, train_batch_size_list = process_time_data(
    "Train",
    finder,
    train_data,
    batch_size=train_bs,
    num_neg=1,
    filepath=train_filepath,
)
val_num_batch, val_delta_times_list, val_all_inds_list, val_batch_size_list = process_time_data(
    "Val",
    finder,
    val_data,
    batch_size=val_bs,
    num_neg=1,
    filepath=val_filepath,
)
test_num_batch, test_delta_times_list, test_all_inds_list, test_batch_size_list = process_time_data(
    "Test",
    finder,
    test_data,
    batch_size=test_bs,
    num_neg=99,
    filepath=test_filepath,
)
prepare_wall_s = time.perf_counter() - prepare_started

num_epo = 0
t_train = 0.0
t_val = 0.0
t_epoch_test = 0.0
log_path = log_dir / filename
val_time_score_filepath = time_score_dir / f"val_{filename}"
test_time_score_filepath = time_score_dir / f"test_{filename}"

save_epoch_scores = args.save_epoch_score_cache or REPORT_DIR is not None
epoch_rows: List[Dict[str, object]] = []
best_epoch_idx = 0
best_epoch_test_metrics: Optional[Dict[str, object]] = None
best_val_score_path: Optional[Path] = None
best_test_score_path: Optional[Path] = None
best_val_scores = None
cumulative_train_wall_s = 0.0
cumulative_epoch_wall_s = 0.0
peak_train_memory_mb = 0.0

for epoch_idx in range(args.num_epochs):
    num_epo += 1
    epoch_number = epoch_idx + 1

    train_result = run(
        model,
        "Train",
        epoch_number,
        optimizer,
        criterion,
        log_path,
        num_neg=1,
        num_batch=train_num_batch,
        delta_times_list=train_delta_times_list,
        all_inds_list=train_all_inds_list,
        batch_size_list=train_batch_size_list,
    )
    t_train += train_result["t_epo"]
    peak_train_memory_mb = max(peak_train_memory_mb, float(train_result["allocated_memory"]))

    with torch.no_grad():
        val_result = run(
            model,
            "Val",
            epoch_number,
            None,
            None,
            log_path,
            num_neg=1,
            num_batch=val_num_batch,
            delta_times_list=val_delta_times_list,
            all_inds_list=val_all_inds_list,
            batch_size_list=val_batch_size_list,
        )
    t_val += val_result["t_epo"]

    test_result = None
    val_score_path = None
    test_score_path = None
    if save_epoch_scores:
        with torch.no_grad():
            test_result = run(
                model,
                "Test",
                epoch_number,
                None,
                None,
                log_path,
                num_neg=99,
                num_batch=test_num_batch,
                delta_times_list=test_delta_times_list,
                all_inds_list=test_all_inds_list,
                batch_size_list=test_batch_size_list,
                record_log=False,
            )
        t_epoch_test += test_result["t_epo"]
        val_score_path = epoch_score_dir / f"val_epoch_{epoch_number}.pkl"
        test_score_path = epoch_score_dir / f"test_epoch_{epoch_number}.pkl"
        save_score_cache(val_score_path, val_result["pos_scores"], val_result["neg_scores"])
        save_score_cache(test_score_path, test_result["pos_scores"], test_result["neg_scores"])

    should_stop = early_stopper.early_stop_check(float(val_result["ap"]))
    if epoch_idx == early_stopper.best_epoch:
        torch.save(model.state_dict(), best_checkpoint_path)
        best_epoch_idx = epoch_idx
        best_epoch_test_metrics = test_result
        best_val_score_path = val_score_path
        best_test_score_path = test_score_path
        best_val_scores = (val_result["pos_scores"], val_result["neg_scores"])
        print("Saving the best model.")

    cumulative_train_wall_s += float(train_result["t_epo"])
    cumulative_epoch_wall_s += float(train_result["t_epo"]) + float(val_result["t_epo"]) + float(
        test_result["t_epo"] if test_result is not None else 0.0
    )
    epoch_rows.append(
        {
            "epoch": epoch_number,
            "train_loss": train_result["loss"],
            "val_ap": val_result["ap"],
            "val_auc": "",
            "test_ap": test_result["ap"] if test_result is not None else "",
            "test_auc": test_result["mrr"] if test_result is not None else "",
            "train_wall_s": round(float(train_result["t_epo"]), 6),
            "val_wall_s": round(float(val_result["t_epo"]), 6),
            "test_wall_s": round(float(test_result["t_epo"]), 6) if test_result is not None else "",
            "cumulative_train_wall_s": round(cumulative_train_wall_s, 6),
            "cumulative_epoch_wall_s": round(cumulative_epoch_wall_s, 6),
            "val_score_path": str(val_score_path) if val_score_path is not None else "",
            "test_score_path": str(test_score_path) if test_score_path is not None else "",
        }
    )

    if should_stop:
        if best_checkpoint_path.exists():
            model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        print("\nLoading the best model.")
        model.eval()
        break

if best_checkpoint_path.exists():
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
model.eval()

final_test_result = best_epoch_test_metrics
mem_test = final_test_result["allocated_memory"] if final_test_result is not None else 0.0
if final_test_result is None:
    with torch.no_grad():
        final_test_result = run(
            model,
            "Test",
            best_epoch_idx + 1,
            None,
            None,
            log_path,
            num_neg=99,
            num_batch=test_num_batch,
            delta_times_list=test_delta_times_list,
            all_inds_list=test_all_inds_list,
            batch_size_list=test_batch_size_list,
        )
    mem_test = final_test_result["allocated_memory"]

if best_val_score_path is not None and best_val_score_path.exists():
    shutil.copyfile(best_val_score_path, val_time_score_filepath)
else:
    assert best_val_scores is not None
    save_score_cache(val_time_score_filepath, best_val_scores[0], best_val_scores[1])
print(f"Val time score has been saved at {val_time_score_filepath}")

if best_test_score_path is not None and best_test_score_path.exists():
    shutil.copyfile(best_test_score_path, test_time_score_filepath)
else:
    save_score_cache(test_time_score_filepath, final_test_result["pos_scores"], final_test_result["neg_scores"])
print(f"Test time score has been saved at {test_time_score_filepath}")

if best_model_path.exists():
    os.remove(best_model_path)
if best_checkpoint_path.exists():
    os.replace(best_checkpoint_path, best_model_path)

best_epoch = best_epoch_idx + 1
best_val_ap = float(epoch_rows[best_epoch_idx]["val_ap"])
best_test_ap = float(final_test_result["ap"])
best_test_mrr = float(final_test_result["mrr"])
best_hit_at_10 = final_test_result["hit_at_10"]
mem_train = peak_train_memory_mb

print(
    f"\nNum_epochs: {num_epo}, total_train_time: {t_train:4f}, total_val_time: {t_val:4f}, "
    f"total_test_time_neg99: {t_epoch_test:4f}, memory_train: {mem_train}, memory_val: {val_result['allocated_memory']}, "
    f"memory_test: {mem_test}."
)

with log_path.open("a", encoding="utf-8") as log_file:
    log_file.write(
        f"Num_epochs: {num_epo}, total_train_time: {t_train:4f}, total_val_time: {t_val:4f}, "
        f"total_test_time_neg99: {t_epoch_test:4f}, memory_val: {val_result['allocated_memory']}, "
        f"memory_test: {mem_test}."
    )

if REPORT_DIR is not None:
    summary_payload = {
        "system": "EAGLE-Time",
        "model": "EAGLE",
        "stage": "time",
        "dataset": DATA,
        "run_type": "tta",
        "epochs_requested": args.num_epochs,
        "completed_epochs": num_epo,
        "best_epoch": best_epoch,
        "best_val_ap": best_val_ap,
        "best_val_auc": None,
        "final_test_metrics": {
            "ap": best_test_ap,
            "auc_or_mrr": best_test_mrr,
            "hit_at_10": best_hit_at_10,
        },
        "config": {
            "dataset_name": DATA,
            "batch_size": args.batch_size,
            "topk": args.topk,
            "topk_sample_flag": args.topk_sample_flag,
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_layers": args.num_layers,
            "hidden_dims": args.hidden_dims,
            "gpu": args.gpu,
            "seed": args.seed,
            "filename": filename,
            "workspace_root": str(WORKSPACE_ROOT),
        },
        "paths": {
            "log_path": str(log_path),
            "best_model_path": str(best_model_path),
            "val_time_score_path": str(val_time_score_filepath),
            "test_time_score_path": str(test_time_score_filepath),
            "epoch_score_dir": str(epoch_score_dir),
        },
        "wall_seconds": {
            "prepare": round(prepare_wall_s, 6),
            "train": round(t_train, 6),
            "val": round(t_val, 6),
            "test": round(t_epoch_test, 6),
            "total": round(prepare_wall_s + t_train + t_val + t_epoch_test, 6),
        },
        "stop_reason": "early_stop" if num_epo < args.num_epochs else "max_epochs",
    }
    dump_json(REPORT_DIR / "summary.json", summary_payload)
    write_csv_rows(
        REPORT_DIR / "epoch_metrics.csv",
        [
            "epoch",
            "train_loss",
            "val_ap",
            "val_auc",
            "test_ap",
            "test_auc",
            "train_wall_s",
            "val_wall_s",
            "test_wall_s",
            "cumulative_train_wall_s",
            "cumulative_epoch_wall_s",
            "val_score_path",
            "test_score_path",
        ],
        epoch_rows,
    )
