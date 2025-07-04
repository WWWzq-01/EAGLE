import numpy as np
from numba.experimental import jitclass
from numba import types, typed
import numba as nb
import torch
import pickle
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaTypeSafetyWarning,
)
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)
from sklearn.metrics import average_precision_score

def set_random_seed(seed):
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_info(model):
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3f million" % parameters)


def get_best_para(dataset):
    file = "cached_ap_score"
    f = open(file, "rb")
    results = pickle.load(f)
    para = results[dataset]["best_para"]
    alpha = para[0]
    beta = para[1]
    best_ap = results[dataset]["best_ap"]
    f.close()
    return alpha, beta, best_ap


def compute_metrics(
    pred_pos, pred_neg, device, k_list=[5, 10, 20, 30, 40, 50, 60, 70, 80]
):
    with torch.no_grad():
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).cpu().detach()
        y_true = (
            torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0)
            .cpu()
            .detach()
        )
        ap = average_precision_score(y_true, y_pred)

        num_edge = pred_pos.shape[0]
        num_neg = pred_neg.shape[0] // num_edge

        mrr = 0
        hr_list = [0] * len(k_list)

        for i in range(num_edge):
            neg_samples = [pred_neg[j * num_edge + i] for j in range(num_neg)]
            combined_scores = torch.tensor(
                [pred_pos[i].item()] + [s.item() for s in neg_samples]
            ).to(device)

            sorted_indices = torch.argsort(combined_scores, descending=True)
            rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1

            mrr += 1 / rank

            for idx, k in enumerate(k_list):
                if rank <= k:
                    hr_list[idx] += 1

        mrr /= num_edge
        hr_list = [hr / num_edge for hr in hr_list]

        return ap, mrr, hr_list


class EarlyStopMonitor(object):
    def __init__(self, max_round=5, higher_better=True, tolerance=1e-8):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1

        if self.last_best is None:
            self.last_best = curr_val

        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class NegEdgeSampler:
    def __init__(self,  destinations: np.ndarray, full_destinations: np.ndarray, num_neg: int, device: str = 'cuda', seed: int = 2024):
        self.seed = seed
        self.destinations = destinations
        self.full_destinations = full_destinations
        self.num_neg = num_neg
        self.device = device

        assert num_neg <= len(full_destinations) - 1, f"num_neg should be <= {len(full_destinations) - 1}"

        self.unique_destinations = torch.tensor(np.unique(full_destinations), dtype=torch.long, device=self.device)
        self.num_dst_node = len(destinations)

        torch.manual_seed(self.seed)
    
    def sample(self, batch_destinations: np.ndarray):
        batch_size = len(batch_destinations)

        batch_destinations = torch.tensor(batch_destinations, dtype=torch.long, device=self.device)

        all_choices = self.unique_destinations.unsqueeze(0).expand(batch_size, -1)

        mask = (all_choices != batch_destinations.unsqueeze(1))

        all_choices = all_choices[mask].view(batch_size, -1) # [batch_size, len(self.unique_destinations)-1]

        random_scores = torch.rand_like(all_choices, dtype=torch.float, device=self.device)
        _, topk_indices = torch.topk(random_scores, self.num_neg, dim=1)

        batch_neg_samples = torch.gather(all_choices, 1, topk_indices)

        return batch_neg_samples


nb_tppr_dict2 = nb.typed.Dict.empty(
    key_type=types.int64,
    value_type=types.float64,
)
nb_dict_type2 = nb.typeof(nb_tppr_dict2)
list_dict2 = typed.List()
list_dict2.append(nb_tppr_dict2)
list_list_dict2 = typed.List()
list_list_dict2.append(list_dict2)

spec_tppr_node_finder = [
    ("num_nodes", types.int64),
    ("k", types.int64),
    ("n_tppr", types.int64),
    ("alpha_list", types.List(types.float64)),
    ("beta_list", types.List(types.float64)),
    ("norm_list", types.Array(types.float64, 2, "C")),
    ("PPR_list", nb.typeof(list_list_dict2)),
    ("sim", nb.typeof("123")),
]


@jitclass(spec_tppr_node_finder)
class tppr_node_finder:
    def __init__(self, num_nodes, k, alpha, beta, sim, n_tppr=1):
        self.num_nodes = num_nodes
        self.k = k
        self.n_tppr = n_tppr
        self.alpha_list = [alpha]
        self.beta_list = [beta]
        self.sim = sim
        self.reset_tppr()

    def reset_tppr(self):
        PPR_list = typed.List()
        for _ in range(self.n_tppr):
            temp_PPR_list = typed.List()
            for _ in range(self.num_nodes):
                tppr_dict = nb.typed.Dict.empty(
                    key_type=types.int64,
                    value_type=types.float64,
                )
                temp_PPR_list.append(tppr_dict)
            PPR_list.append(temp_PPR_list)
        self.norm_list = np.zeros((self.n_tppr, self.num_nodes), dtype=np.float64)
        self.PPR_list = PPR_list

    def get_similarity(self, tppr_index, source, target):
        PPR_list = self.PPR_list[tppr_index]
        source_tppr = PPR_list[source]
        target_tppr = PPR_list[target]
        similarity = 0

        for key, weight in source_tppr.items():
            if self.sim == "mul_wo_norm":
                if key in target_tppr:
                    similarity += weight * target_tppr[key]
        return similarity

    def precompute_link_prediction(self, source_nodes, num_neg):
        n_edges = len(source_nodes) // (2 + num_neg)
        scores = np.zeros(n_edges * (num_neg + 1))

        for index0, alpha in enumerate(self.alpha_list):
            beta = self.beta_list[index0]
            norm_list = self.norm_list[index0]
            PPR_list = self.PPR_list[index0]

            for i in range(n_edges):
                source = source_nodes[i]
                target = source_nodes[i + n_edges]
                sim = self.get_similarity(index0, source, target)
                scores[i] = sim

                for neg_index in range(num_neg):
                    fake = source_nodes[i + (neg_index + 2) * n_edges]
                    sim = self.get_similarity(index0, source, fake)
                    scores[i + (neg_index + 1) * n_edges] = sim

                pairs = (
                    [(source, target), (target, source)]
                    if source != target
                    else [(source, target)]
                )

                for index, pair in enumerate(pairs):
                    s1 = pair[0] # node 1
                    s2 = pair[1] # node 2

                    if norm_list[s1] == 0:
                        t_s1_PPR = nb.typed.Dict.empty(
                            key_type=types.int64,
                            value_type=types.float64,
                        )
                        scale_s2 = 1 - alpha
                    else:
                        t_s1_PPR = PPR_list[s1].copy()
                        last_norm = norm_list[s1]
                        new_norm = last_norm * beta + beta
                        scale_s1 = last_norm / new_norm * beta
                        scale_s2 = beta / new_norm * (1 - alpha)
                        for key, value in t_s1_PPR.items():
                            t_s1_PPR[key] = value * scale_s1

                    if norm_list[s2] == 0:
                        t_s1_PPR[s2] = scale_s2 * alpha if alpha != 0 else scale_s2
                    else:

                        s2_PPR = PPR_list[s2]
                        for key, value in s2_PPR.items():
                            if key in t_s1_PPR:
                                t_s1_PPR[key] += value * scale_s2
                            else:
                                t_s1_PPR[key] = value * scale_s2

                        if s2 in t_s1_PPR:
                            t_s1_PPR[s2] += scale_s2 * alpha if alpha != 0 else scale_s2
                        else:
                            t_s1_PPR[s2] = scale_s2 * alpha if alpha != 0 else scale_s2

                    updated_tppr = nb.typed.Dict.empty(
                        key_type=types.int64, value_type=types.float64
                    )

                    tppr_size = len(t_s1_PPR)
                    if tppr_size <= self.k:
                        updated_tppr = t_s1_PPR
                    else:
                        keys = list(t_s1_PPR.keys())
                        values = np.array(list(t_s1_PPR.values()))
                        inds = np.argsort(values)[-self.k :]
                        for ind in inds:
                            key = keys[ind]
                            value = values[ind]
                            updated_tppr[key] = value

                    if index == 0:
                        new_s1_PPR = updated_tppr
                    else:
                        new_s2_PPR = updated_tppr

                if source != target:
                    PPR_list[source] = new_s1_PPR
                    PPR_list[target] = new_s2_PPR
                    norm_list[source] = norm_list[source] * beta + beta
                    norm_list[target] = norm_list[target] * beta + beta
                else:
                    PPR_list[source] = new_s1_PPR
                    norm_list[source] = norm_list[source] * beta + beta

        return scores
