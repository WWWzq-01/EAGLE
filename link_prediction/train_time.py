import argparse
import time
import math
import os
import pickle
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numba as nb
from numba import typed
from numba.experimental import jitclass

from utils.util import set_random_seed, compute_metrics, print_model_info, EarlyStopMonitor, NegEdgeSampler
from utils.data_processing import get_data_transductive
from utils.model import Mixer_per_node

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Training EAGLE-Time.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="wikipedia",
    choices=["Contacts", "lastfm", "wikipedia", "reddit", "superuser", "askubuntu", "wikitalk"]
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

args = parser.parse_args()

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
        node_to_neighbors.append(
            np.array([x[0] for x in sorted_neighbors], dtype=np.int32)
        )
        node_to_edge_idxs.append(
            np.array([x[1] for x in sorted_neighbors], dtype=np.int32)
        )
        node_to_edge_timestamps.append(
            np.array([x[2] for x in sorted_neighbors], dtype=np.float64)
        )

    return NeighborFinder(node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps)


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    else:
        return obj


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
                    selected_times = edge_times[-n_neighbors:][
                        ::-1
                    ] # delta time from last to early
                elif topk_sample_flag == "early":
                    selected_times = edge_times[
                        :n_neighbors
                    ] # delta time from early to last
                elif topk_sample_flag == "random":
                    if n_ngh <= n_neighbors:
                        selected_times = edge_times
                    else:
                        selected_indices = np.random.choice(
                            n_ngh, n_neighbors, replace=False
                        )
                        selected_times = edge_times[selected_indices]
                        selected_times = np.sort(
                            selected_times
                        ) # delta time from early to last

                n_ngh = len(selected_times)
                delta_times[cum_sum : cum_sum + n_ngh] = timestamp - selected_times

            n_edges[i] = n_ngh
            cum_sum += n_ngh
        return delta_times, n_edges, cum_sum


Path(f"log_learn_time/{DATA}").mkdir(parents=True, exist_ok=True)
Path(f"time_score_cache/{DATA}").mkdir(parents=True, exist_ok=True)
print(
    f"Time training result will be written at log_learn_time/{DATA}/{filename}.\n"
)

best_checkpoint_path = f"saved_checkpoints/{time.time()}.pth"
Path(f"saved_checkpoints").mkdir(parents=True, exist_ok=True)
Path(f"saved_time_models/learn_time/{DATA}").mkdir(parents=True, exist_ok=True)
Path(f"time_processed_data/{DATA}").mkdir(parents=True, exist_ok=True)
best_model_name = filename
best_model_path = f"saved_time_models/learn_time/{DATA}/{best_model_name}.pth"


full_data, train_data, val_data, test_data, n_nodes, n_edges = get_data_transductive(
    DATA, use_validation=True
)
n_train = train_data.n_interactions
n_val = val_data.n_interactions
n_test = test_data.n_interactions
print(f"#Edge: train {n_train}, val {n_val}, test {n_test}")

train_neg_edge_sampler = NegEdgeSampler(destinations=train_data.destinations, full_destinations=train_data.destinations, num_neg=1, device=device, seed=2024)
val_neg_edge_sampler = NegEdgeSampler(destinations=val_data.destinations, full_destinations=full_data.destinations, num_neg=1, device=device, seed=2025)
test_neg_edge_sampler = NegEdgeSampler(destinations=test_data.destinations, full_destinations=full_data.destinations, num_neg=99, device=device, seed=2026)


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
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
early_stopper = EarlyStopMonitor(max_round=args.patience)


def process_data(mode, finder, data, bs, num_neg=1, filepath=None):
    print(f"Processing {mode} data...")
    num_instance = data.n_interactions
    BATCH_SIZE = bs if bs != -1 else num_instance
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    delta_times_list = []
    all_inds_list = []
    batch_size_list = []

    for batch_idx in tqdm(range(0, num_batch)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sample_inds = np.array(list(range(start_idx, end_idx)))

        sources_batch = data.sources[sample_inds]
        destinations_batch = data.destinations[sample_inds]
        timestamps_batch = data.timestamps[sample_inds]

        if mode == "Train" or mode == "Val":
            neg_filepath = f"./data/batchneg/{DATA}/{mode}_neg1_bs{bs}_batch{batch_idx}.pkl" # 1 neg sample for each pos sample

            if os.path.exists(neg_filepath):
                with open(neg_filepath, "rb") as f:
                    negatives_batch = (pickle.load(f)).t().flatten()
            else:
                if mode == "Train":
                    negatives_batch = train_neg_edge_sampler.sample(destinations_batch).t().flatten() # arr[dst(1)_neg1, dst(2)_neg1, ..., dst(n_edge)_neg1, dst(1)_neg2, ...]
                elif mode == "Val":
                    negatives_batch = val_neg_edge_sampler.sample(destinations_batch).t().flatten()

                negatives_batch = move_to_cpu(negatives_batch)
                
                os.makedirs(os.path.dirname(neg_filepath), exist_ok=True)
                with open(neg_filepath, 'wb') as f:
                    pickle.dump(negatives_batch, f)

        elif mode == "Test":
            neg_filepath = f"./data/batchneg/{DATA}/{mode}_neg99_bs{bs}_batch{batch_idx}.pkl" # 99 neg samples for each pos sample
            if os.path.exists(neg_filepath):
                with open(neg_filepath, "rb") as f:
                    negatives_batch = (pickle.load(f)).t().flatten()
            else:
                negatives_batch = test_neg_edge_sampler.sample(destinations_batch).t().flatten()
                
                negatives_batch = move_to_cpu(negatives_batch)

                os.makedirs(os.path.dirname(neg_filepath), exist_ok=True)
                with open(neg_filepath, 'wb') as f:
                    pickle.dump(negatives_batch, f)

        if isinstance(negatives_batch, torch.Tensor):
            negatives_batch = negatives_batch.cpu().numpy()

        source_nodes = np.concatenate(
            [sources_batch, destinations_batch, negatives_batch], dtype=np.int32
        )
        timestamps = np.tile(timestamps_batch, num_neg + 2)

        delta_times, n_neighbors, total_edges = finder.get_clean_delta_times(
            source_nodes, timestamps, args.topk, args.topk_sample_flag
        )
        delta_times = delta_times[:total_edges]
        delta_times = torch.from_numpy(delta_times).to(device).unsqueeze(-1)

        all_inds = []
        for i, n_ngh in enumerate(n_neighbors):
            all_inds.extend([(args.topk * i + j) for j in range(n_ngh)])

        all_inds = torch.tensor(all_inds, device=device)
        batch_size = len(n_neighbors)

        delta_times = move_to_cpu(delta_times)
        all_inds = move_to_cpu(all_inds)

        delta_times_list.append(delta_times)
        all_inds_list.append(all_inds)
        batch_size_list.append(batch_size)

    if filepath:
        with open(filepath, "wb") as f:
            pickle.dump(
                (num_batch, delta_times_list, all_inds_list, batch_size_list), f
            )
    print(f"Processed data has been saved at {filepath}.")

    return num_batch, delta_times_list, all_inds_list, batch_size_list


def process_time_data(mode, finder, data, batch_size, num_neg=1, filepath=None):
    if filepath and os.path.exists(filepath):
        print(f"Loading cached {mode} data from {filepath}.")
        with open(filepath, "rb") as f:
            num_batch, delta_times_list, all_inds_list, batch_size_list = pickle.load(f)

        all_inds_list = [tensor.to(device) for tensor in all_inds_list]
        delta_times_list = [tensor.to(device) for tensor in delta_times_list]

        return num_batch, delta_times_list, all_inds_list, batch_size_list

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
    f,
    num_neg,
    num_batch,
    features_list=None,
    delta_times_list=None,
    all_inds_list=None,
    batch_size_list=None,
    k_list = [10],
):
    if mode == "Train":
        model = model.train()
    else:
        model = model.eval()

    ap_list, mrr_list, hit_list = [], [], []
    t_epo = 0.0
    allocated_memory = 0.0

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

        mem = torch.cuda.max_memory_allocated(device) / (1024**2) # /MB
        allocated_memory = max(allocated_memory, mem)

        t_epo += time.time() - t1

        if mode == "Train" and no_neighbor_flag == False:
            t2 = time.time()
            optimizer.zero_grad()
            predicts = torch.cat([pos_score, neg_score], dim=0).to(device)
            labels = torch.cat(
                [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
            ).to(device)
            loss = criterion(input=predicts, target=labels)
            loss.backward()
            optimizer.step()
            t_epo += time.time() - t2
        elif mode == "Val":
            with torch.no_grad():
                all_pos_score.append(pos_score.sigmoid().cpu())
                all_neg_score.append(neg_score.sigmoid().cpu())
                y_pred = (
                    torch.cat([pos_score, neg_score], dim=0).sigmoid().cpu().detach()
                )
                y_true = (
                    torch.cat(
                        [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
                    )
                    .cpu()
                    .detach()
                )
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
            f"Epoch{epoch}-{mode}: loss: {loss.item():.5f}, time: {t_epo}, memory used: {allocated_memory}"
        )
        return t_epo, allocated_memory

    elif mode == "Val":
        ap = np.mean(ap_list)
        print(f"Epoch{epoch}-{mode}: ap: {ap:.4f}, time: {t_epo}, memory used: {allocated_memory}")

    elif mode == "Test":
        ap = np.mean(ap_list)
        mrr = np.mean(mrr_list)
        hit_array = np.array(hit_list)
        mean_hr = np.mean(hit_array, axis=0)
        print(
            f"Epoch{epoch}-{mode}-Neg_sam{num_neg}: time: {t_epo}, ap: {ap:.4f}, mrr: {mrr:.4f}, "
            + ", ".join([f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, mean_hr)])
            + f", memory used: {allocated_memory}"
        )

        with open(f, "a") as log_file:
            log_file.write(
                f"Final Test with Neg_sam {num_neg}: ap: {ap:.4f}, mrr: {mrr:.4f}, "
                + ", ".join([f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, mean_hr)])
                + "\n"
            )

    return ap, t_epo, all_pos_score, all_neg_score, allocated_memory


train_bs = args.batch_size
val_bs = args.batch_size
test_bs = args.batch_size

train_filepath = f"time_processed_data/{DATA}/Train_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{train_bs}_numneg_1.pkl"
val_filepath = f"time_processed_data/{DATA}/Val_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{val_bs}_numneg_1.pkl"
test_filepath = f"time_processed_data/{DATA}/Test_topk_{args.topk}_flag_{args.topk_sample_flag}_bs_{test_bs}_numneg_99.pkl"

train_num_batch, train_delta_times_list, train_all_inds_list, train_batch_size_list = (
    process_time_data(
        "Train",
        finder,
        train_data,
        batch_size=train_bs,
        num_neg=1,
        filepath=train_filepath,
    )
)
val_num_batch, val_delta_times_list, val_all_inds_list, val_batch_size_list = (
    process_time_data(
        "Val", finder, val_data, batch_size=val_bs, num_neg=1, filepath=val_filepath
    )
)
test_num_batch, test_delta_times_list, test_all_inds_list, test_batch_size_list = (
    process_time_data(
        "Test",
        finder,
        test_data,
        batch_size=test_bs,
        num_neg=99,
        filepath=test_filepath,
    )
)


num_epo = 0
t_train = 0.0
t_val = 0.0
f = f"log_learn_time/{DATA}/{filename}"
val_time_score_filepath = f"time_score_cache/{DATA}/val_{filename}"
test_time_score_filepath = f"time_score_cache/{DATA}/test_{filename}"
for epoch in range(args.num_epochs):
    num_epo += 1

    t_train_epo, mem_train = run(
        model,
        "Train",
        epoch,
        optimizer,
        criterion,
        f,
        num_neg=1,
        num_batch=train_num_batch,
        delta_times_list=train_delta_times_list,
        all_inds_list=train_all_inds_list,
        batch_size_list=train_batch_size_list,
    )
    t_train += t_train_epo

    with torch.no_grad():
        val_ap, t_val_epo, val_all_pos_score, val_all_neg_score, mem_val = run(
            model,
            "Val",
            epoch,
            None,
            None,
            f,
            num_neg=1,
            num_batch=val_num_batch,
            delta_times_list=val_delta_times_list,
            all_inds_list=val_all_inds_list,
            batch_size_list=val_batch_size_list,
        )
        t_val += t_val_epo

        last_best_epoch = early_stopper.best_epoch
        if early_stopper.early_stop_check(val_ap):
            model_parameters = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(model_parameters)
            print("\nLoading the best model.")
            model.eval()
            break
        else:
            if epoch == early_stopper.best_epoch:
                torch.save((model.state_dict()), best_checkpoint_path)
                print("Saving the best model.")
                val_time_score = val_all_pos_score, val_all_neg_score
                val_time_score = move_to_cpu(val_time_score)
                os.makedirs(os.path.dirname(val_time_score_filepath), exist_ok=True)
                with open(val_time_score_filepath, "wb") as vf:
                    pickle.dump(val_time_score, vf)
                print(f"Val time score has been saved at {val_time_score_filepath}")


if os.path.exists(best_model_path):
    os.remove(best_model_path)
os.rename(best_checkpoint_path, best_model_path)

with torch.no_grad():
    _, t_test, test_all_pos_score, test_all_neg_score, mem_test = run(
        model,
        "Test",
        early_stopper.best_epoch,
        None,
        None,
        f,
        num_neg=99,
        num_batch=test_num_batch,
        delta_times_list=test_delta_times_list,
        all_inds_list=test_all_inds_list,
        batch_size_list=test_batch_size_list,
    )
    test_time_score = test_all_pos_score, test_all_neg_score
    test_time_score = move_to_cpu(test_time_score)
    os.makedirs(os.path.dirname(test_time_score_filepath), exist_ok=True)
    with open(test_time_score_filepath, "wb") as tf:
        pickle.dump(test_time_score, tf)
    print(f"Test time score has been saved at {test_time_score_filepath}")

    print(
        f"\nNum_epochs: {num_epo}, total_train_time: {t_train:4f}, total_val_time: {t_val:4f}, total_test_time_neg99: {t_test:4f}, memory_train: {mem_train}, memory_val: {mem_val}, memory_test: {mem_test}."
    )

with open(f, "a") as log_file:
    log_file.write(
        f"Num_epochs: {num_epo}, total_train_time: {t_train:4f}, total_val_time: {t_val:4f}, total_test_time_neg99: {t_test:4f}, memory_train: {mem_train}, memory_val: {mem_val}, memory_test: {mem_test}."
    )
