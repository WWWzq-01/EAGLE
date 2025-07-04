import argparse
import json
import math
import pickle
import os

import torch
import numpy as np
from sklearn.metrics import average_precision_score

from utils.data_processing import get_data_transductive
from utils.util import compute_metrics

parser = argparse.ArgumentParser("Training EAGLE-Hybrid.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="wikipedia",
    choices=["Contacts", "lastfm", "wikipedia", "reddit", "superuser", "askubuntu", "wikitalk"]
)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

DATA = args.dataset_name

val_num_neg_per_pos = 1
test_num_neg_per_pos = 99


full_data, train_data, val_data, test_data, n_nodes, n_edges = get_data_transductive(DATA, use_validation=True)
n_train = train_data.n_interactions
n_val = val_data.n_interactions
n_test = test_data.n_interactions
print(f"#Edge: train {n_train}, val {n_val}, test {n_test}")


best_param_filepath = f"hybrid_best_param/{DATA}.json"
with open(best_param_filepath, 'r') as f:
    params = json.load(f)

structure_topk = params["structure"]["topk"]
structure_alpha = params["structure"]["alpha"]
structure_beta = params["structure"]["beta"]
time_topk = params["time"]["topk"]
topk_sample_flag = params["time"]["topk_sample_flag"]
lr = params["time"]["lr"]
wd = params["time"]["wd"]
bs = params["time"]["bs"]

best_time_score_filename = f'topk_{time_topk}_flag_{topk_sample_flag}_lr_{lr}_wd_{wd}_bs_{bs}'
best_structure_score_filename = f'topk_{structure_topk}_alpha_{structure_alpha}_beta_{structure_beta}_mul_wo_norm'

val_time_score_filepath = f'time_score_cache/{DATA}/val_{best_time_score_filename}'
test_time_score_filepath = f'time_score_cache/{DATA}/test_{best_time_score_filename}'
val_structure_score_filepath = f'structure_score_cache/{DATA}/val_{best_structure_score_filename}'
test_structure_score_filepath = f'structure_score_cache/{DATA}/test_{best_structure_score_filename}'

missing_msgs = []

if not (os.path.exists(val_time_score_filepath) and os.path.exists(test_time_score_filepath)):
    cmd_time = (
        f'python train_time.py '
        f'--dataset_name {DATA} '
        f'--topk {time_topk} '
        f'--topk_sample_flag {topk_sample_flag} '
        f'--lr {lr} '
        f'--weight_decay {wd} '
        f'--batch_size {bs}'
    )
    missing_msgs.append(
        f"EAGLE-Time training results not found.\n"
        f"Please train EAGLE-Time first using params in hybrid_best_param/{DATA}.json:\n{cmd_time}."
    )

if not (os.path.exists(val_structure_score_filepath) and os.path.exists(test_structure_score_filepath)):
    cmd_struct = (
        f"python train_structure.py "
        f"--dataset_name {DATA} "
        f"--topk {structure_topk} "
        f"--alpha {structure_alpha} "
        f"--beta {structure_beta} "
        f"--sim mul_wo_norm"
    )
    missing_msgs.append(
        f"EAGLE-Structure training results not found.\n"
        f"Please train EAGLE-Structure first using params in hybrid_best_param/{DATA}.json:\n{cmd_struct}."
    )

if missing_msgs:
    raise FileNotFoundError("\n".join(missing_msgs))


with open(val_time_score_filepath, 'rb') as vtf:
    val_time_score = pickle.load(vtf)
val_time_pos_score, val_time_neg_score = val_time_score

with open(test_time_score_filepath, 'rb') as ttf:
    test_time_score = pickle.load(ttf)
test_time_pos_score, test_time_neg_score = test_time_score

with open(val_structure_score_filepath, 'rb') as vsf:
    val_structure_data = pickle.load(vsf)
val_all_node = val_structure_data[0]
val_structure_score = val_structure_data[2]

with open(test_structure_score_filepath, 'rb') as tsf:
    test_structure_data = pickle.load(tsf)
test_all_node = test_structure_data[0]
test_structure_score = test_structure_data[2]


val_time_data_filepath = f'time_processed_data/{DATA}/Val_topk_{time_topk}_flag_{topk_sample_flag}_bs_{bs}_numneg_{val_num_neg_per_pos}.pkl'
with open(val_time_data_filepath, 'rb') as vtdf:
    _, val_delta_times_list, val_all_inds_list, val_batch_size_list = pickle.load(vtdf)

test_time_data_filepath = f'time_processed_data/{DATA}/Test_topk_{time_topk}_flag_{topk_sample_flag}_bs_{bs}_numneg_{test_num_neg_per_pos}.pkl'
with open(test_time_data_filepath, 'rb') as ttdf:
    _, test_delta_times_list, test_all_inds_list, test_batch_size_list = pickle.load(ttdf)


def mix_scores(mode, data, batch_size, time_pos_score, time_neg_score, structure_score, delta_times_list, all_inds_list, num_neg_per_pos, time_topk, yita, device, k_list):
    torch.set_grad_enabled(False)

    num_pos_edge = data.n_interactions
    BATCH_SIZE = batch_size if batch_size != -1 else num_pos_edge
    num_batch = math.ceil(num_pos_edge / BATCH_SIZE)

    ap_list, mrr_list, hit_list = [], [], []

    for batch_idx in range(0, num_batch):
        batch_time_pos_score = time_pos_score[batch_idx].to(device)
        batch_time_neg_score = time_neg_score[batch_idx].to(device)

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_pos_edge, start_idx + BATCH_SIZE)
        pos_ids = np.array(list(range(start_idx, end_idx)))

        cur_batch_size = min(BATCH_SIZE, num_pos_edge - start_idx)

        neg_ids = np.concatenate([pos_ids + i*num_pos_edge for i in range(1, 1 + num_neg_per_pos)])
        batch_skc_pos_score = structure_score[pos_ids]
        batch_skc_neg_score = structure_score[neg_ids]


        delta_times, all_inds = delta_times_list[batch_idx].to(device), all_inds_list[batch_idx].to(device)


        total_groups = (2 + num_neg_per_pos) * cur_batch_size
        # max_delta = delta_times.max()

        groups_avg_dts = []

        delta_times = delta_times.squeeze(1) # [2906]

        all_cur_full_ids = torch.arange(total_groups * time_topk, device=device).reshape(total_groups, time_topk)

        groups_avg_dts = []

        max_delta_value = delta_times.max()

        for group_id in range(total_groups):
            cur_full_ids = all_cur_full_ids[group_id]
            
            # check existed cur_ids in all_inds
            mask = (all_inds.unsqueeze(1) == cur_full_ids.unsqueeze(0)) # [2906, time_topk]

            matched = mask.any(dim=1) # [2906], in-True, not in-False

            cur_top_ids = torch.nonzero(matched, as_tuple=False).squeeze(1) # [num_matched]

            if cur_top_ids.numel() > 0:
                group_deltas = delta_times[cur_top_ids] # [num_matched]
            else:
                group_deltas = torch.tensor([], dtype=delta_times.dtype, device=device)

            num_matched = group_deltas.size(0)
            if num_matched < time_topk:
                padding = torch.full((time_topk - num_matched,), max_delta_value, dtype=delta_times.dtype, device=device)
                group_deltas = torch.cat([group_deltas, padding], dim=0) # [time_topk]

            avg_dt = group_deltas.mean()
            groups_avg_dts.append(avg_dt)

        avg_dts_tensor = torch.stack(groups_avg_dts) # [total_groups]
        avg_dts_tensor = avg_dts_tensor / avg_dts_tensor.mean() - 1

        src_dts = avg_dts_tensor[:cur_batch_size]
        pos_dst_dts = avg_dts_tensor[cur_batch_size : 2*cur_batch_size]
        neg_dst_dts = avg_dts_tensor[2*cur_batch_size:] # [num_neg_per_pos * BATCH_SIZE]

        batch_skc_pos_score_tensor = torch.tensor(batch_skc_pos_score, dtype=torch.float32, device=device) # [BATCH_SIZE]
        batch_skc_neg_score_tensor = torch.tensor(batch_skc_neg_score, dtype=torch.float32, device=device) # [num_neg_per_pos*BATCH_SIZE]

        batch_time_pos_score = batch_time_pos_score.squeeze(1) # [BATCH_SIZE]
        batch_time_neg_score = batch_time_neg_score.squeeze(1) # [num_neg_per_pos*BATCH_SIZE]
        
        hy_pos_weight = yita * ((torch.exp(-src_dts) + torch.exp(-pos_dst_dts))/2) # [BATCH_SIZE]
        batch_hy_pos_score = hy_pos_weight * batch_time_pos_score + batch_skc_pos_score_tensor # [BATCH_SIZE]

        hy_neg_weight = yita * ((torch.exp(-src_dts.repeat(num_neg_per_pos)) + torch.exp(-neg_dst_dts))/2) # [num_neg_per_pos*BATCH_SIZE]
        batch_hy_neg_score = hy_neg_weight * batch_time_neg_score + batch_skc_neg_score_tensor  # [num_neg_per_pos * BATCH_SIZE]


        if mode == 'Val':
            y_pred = torch.cat([batch_hy_pos_score, batch_hy_neg_score], dim=0).cpu().detach()
            y_true = torch.cat([torch.ones_like(batch_hy_pos_score), torch.zeros_like(batch_hy_neg_score)], dim=0).cpu().detach()
            ap = average_precision_score(y_true, y_pred)
            ap_list.append(ap)
        
        elif mode == 'Test':
            ap, mrr, hr_list = compute_metrics(batch_hy_pos_score, batch_hy_neg_score, device, k_list=k_list)
            ap_list.append(ap)
            mrr_list.append(mrr)
            hit_list.append(hr_list)
            
    if mode == 'Val':
        ap = np.mean(ap_list)
        print(f"yita: {yita:.0e} -- Val ap: {ap}")

        return ap
    
    elif mode == 'Test':
        ap = np.mean(ap_list)
        mrr = np.mean(mrr_list)
        hit_array = np.array(hit_list)
        all_hr = np.mean(hit_array, axis=0)
        print(f"Test: ap: {ap:.4f}, mrr: {mrr:.4f}, " + ", ".join([f"hr@{k}: {hr:.4f}" for k, hr in zip(k_list, all_hr)]) + "\n")

        return ap, mrr, all_hr


device = torch.device(f'cuda:{args.gpu}')

k_list=[10]
yita_list = np.concatenate([np.array([1e-6, 2e-6, 3e-6, 5e-6, 8e-6]) * 10**i for i in range(0, 7)])

best_yita = 0
best_val_ap = 0
for yita in yita_list:
    val_ap = mix_scores('Val', val_data, bs, val_time_pos_score, val_time_neg_score, val_structure_score, val_delta_times_list, val_all_inds_list, 1, time_topk, yita=yita, device=device, k_list=k_list)

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_yita = yita

print(f"\nBest yita: {best_yita}\n")

_ = mix_scores('Test', test_data, bs, test_time_pos_score, test_time_neg_score, test_structure_score, test_delta_times_list, test_all_inds_list, 99, time_topk, yita=best_yita, device=device, k_list=k_list)
