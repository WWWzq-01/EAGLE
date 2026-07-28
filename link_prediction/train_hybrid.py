import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from utils.data_processing import get_data_transductive
from utils.hybrid_evaluation import (
    evaluate_test_metrics,
    prepare_hybrid_batches,
    search_best_yita,
)


parser = argparse.ArgumentParser("Training EAGLE-Hybrid.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="wikipedia",
    choices=["Contacts", "lastfm", "wikipedia", "reddit", "superuser", "askubuntu", "wikitalk"],
)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--workspace_root", type=str, default=".")
parser.add_argument("--report_dir", type=str, default="")
parser.add_argument("--stage_root", type=str, default="")
parser.add_argument("--epochs_requested", type=int, default=0)
parser.add_argument("--seed", type=int, default=2024)
args = parser.parse_args()

DATA = args.dataset_name
WORKSPACE_ROOT = Path(args.workspace_root).resolve()
REPORT_DIR = Path(args.report_dir).resolve() if args.report_dir else None
STAGE_ROOT = Path(args.stage_root).resolve() if args.stage_root else None

val_num_neg_per_pos = 1
test_num_neg_per_pos = 99


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


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def safe_float(value) -> Optional[float]:
    if value in {"", None}:
        return None
    return float(value)


def load_stage_summary(stage_name: str) -> Optional[Dict[str, object]]:
    if STAGE_ROOT is None:
        return None
    path = STAGE_ROOT / stage_name / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


full_data, train_data, val_data, test_data, n_nodes, n_edges = get_data_transductive(DATA, use_validation=True)
n_train = train_data.n_interactions
n_val = val_data.n_interactions
n_test = test_data.n_interactions
print(f"#Edge: train {n_train}, val {n_val}, test {n_test}")


selected_params_path = None
if REPORT_DIR is not None:
    candidate = REPORT_DIR / "selected_params.json"
    if candidate.exists():
        selected_params_path = candidate
if selected_params_path is None and STAGE_ROOT is not None:
    candidate = STAGE_ROOT.parent / "selected_params.json"
    if candidate.exists():
        selected_params_path = candidate

best_param_filepath = Path("hybrid_best_param") / f"{DATA}.json"
params_path = selected_params_path or best_param_filepath
with params_path.open("r", encoding="utf-8") as f:
    params = json.load(f)

structure_topk = params["structure"]["topk"]
structure_alpha = params["structure"]["alpha"]
structure_beta = params["structure"]["beta"]
time_topk = params["time"]["topk"]
topk_sample_flag = params["time"]["topk_sample_flag"]
lr = params["time"]["lr"]
wd = params["time"]["wd"]
bs = params["time"]["bs"]

best_time_score_filename = f"topk_{time_topk}_flag_{topk_sample_flag}_lr_{lr}_wd_{wd}_bs_{bs}"
best_structure_score_filename = f"topk_{structure_topk}_alpha_{structure_alpha}_beta_{structure_beta}_mul_wo_norm"

val_time_score_filepath = WORKSPACE_ROOT / "time_score_cache" / DATA / f"val_{best_time_score_filename}"
test_time_score_filepath = WORKSPACE_ROOT / "time_score_cache" / DATA / f"test_{best_time_score_filename}"
val_structure_score_filepath = WORKSPACE_ROOT / "structure_score_cache" / DATA / f"val_{best_structure_score_filename}"
test_structure_score_filepath = WORKSPACE_ROOT / "structure_score_cache" / DATA / f"test_{best_structure_score_filename}"

val_time_data_filepath = WORKSPACE_ROOT / "time_processed_data" / DATA / (
    f"Val_topk_{time_topk}_flag_{topk_sample_flag}_bs_{bs}_numneg_{val_num_neg_per_pos}.pkl"
)
test_time_data_filepath = WORKSPACE_ROOT / "time_processed_data" / DATA / (
    f"Test_topk_{time_topk}_flag_{topk_sample_flag}_bs_{bs}_numneg_{test_num_neg_per_pos}.pkl"
)

missing_msgs = []
if not (val_time_score_filepath.exists() and test_time_score_filepath.exists()):
    missing_msgs.append(
        f"EAGLE-Time training results not found under {WORKSPACE_ROOT}. "
        f"Expected {val_time_score_filepath} and {test_time_score_filepath}."
    )
if not (val_structure_score_filepath.exists() and test_structure_score_filepath.exists()):
    missing_msgs.append(
        f"EAGLE-Structure training results not found under {WORKSPACE_ROOT}. "
        f"Expected {val_structure_score_filepath} and {test_structure_score_filepath}."
    )
if missing_msgs:
    raise FileNotFoundError("\n".join(missing_msgs))


with val_structure_score_filepath.open("rb") as vsf:
    val_structure_data = pickle.load(vsf)
val_structure_score = val_structure_data[2]

with test_structure_score_filepath.open("rb") as tsf:
    test_structure_data = pickle.load(tsf)
test_structure_score = test_structure_data[2]

with val_time_data_filepath.open("rb") as vtdf:
    _, val_delta_times_list, val_all_inds_list, val_batch_size_list = pickle.load(vtdf)

with test_time_data_filepath.open("rb") as ttdf:
    _, test_delta_times_list, test_all_inds_list, test_batch_size_list = pickle.load(ttdf)


device = torch.device(f"cuda:{args.gpu}")
k_list = [10]

time_summary = load_stage_summary("time")
structure_summary = load_stage_summary("structure")
structure_test_summary = load_stage_summary("structure_test")
time_epoch_metrics_path = STAGE_ROOT / "time" / "epoch_metrics.csv" if STAGE_ROOT is not None else None

benchmark_mode = time_epoch_metrics_path is not None and time_epoch_metrics_path.exists()
lifecycle_mode = bool(time_summary and time_summary.get("validation_lifecycle_v1"))

if benchmark_mode and lifecycle_mode:
    time_epoch_rows = load_csv_rows(time_epoch_metrics_path)
    if not time_epoch_rows:
        raise ValueError("EAGLE lifecycle hybrid selection requires non-empty time epoch metrics")

    hybrid_rows: List[Dict[str, object]] = []
    for raw_row in time_epoch_rows:
        if raw_row.get("test_executed_this_epoch", "").strip().lower() != "false":
            raise ValueError(f"epoch {raw_row.get('epoch')} executed test before best selection")
        if raw_row.get("test_score_path"):
            raise ValueError(f"epoch {raw_row.get('epoch')} exposes a per-epoch test score cache")
        record: Dict[str, object] = dict(raw_row)
        record["val_ap"] = raw_row["validation_ap"]
        record["hybrid_eval_wall_s"] = raw_row["hybrid_validation_wall_s"]
        hybrid_rows.append(record)

    best_row = max(hybrid_rows, key=lambda row: float(row["validation_ap"]))
    selected_epoch = int(best_row["epoch"])
    selected_yita = float(best_row["best_yita"])
    if time_summary is None:
        raise ValueError("EAGLE lifecycle hybrid selection requires the time-stage summary")
    if selected_epoch != int(time_summary["best_epoch"]):
        raise ValueError("time-stage best epoch disagrees with hybrid validation rows")
    if selected_yita != float(time_summary["best_yita"]):
        raise ValueError("time-stage best yita disagrees with hybrid validation rows")

    final_test_started = time.perf_counter()
    test_time_pos_score, test_time_neg_score = load_pickle(test_time_score_filepath)
    prepared_test_batches = prepare_hybrid_batches(
        test_data,
        bs,
        test_time_pos_score,
        test_time_neg_score,
        test_structure_score,
        test_delta_times_list,
        test_all_inds_list,
        test_num_neg_per_pos,
        time_topk,
        device,
    )
    test_ap, test_mrr, test_hr = evaluate_test_metrics(prepared_test_batches, selected_yita, k_list)
    final_test_wall_s = time.perf_counter() - final_test_started

    structure_total_s = (
        safe_float(structure_summary.get("wall_seconds", {}).get("total"))
        if structure_summary
        else 0.0
    )
    structure_test_total_s = (
        safe_float(structure_test_summary.get("wall_seconds", {}).get("total"))
        if structure_test_summary
        else 0.0
    )
    time_total_s = safe_float(time_summary.get("wall_seconds", {}).get("total")) or 0.0
    time_prepare_s = safe_float(time_summary.get("wall_seconds", {}).get("prepare")) or 0.0
    summary_payload = {
        "system": "EAGLE",
        "model": "EAGLE",
        "dataset": DATA,
        "run_type": "tta",
        "epochs_requested": args.epochs_requested or int(time_summary["epochs_requested"]),
        "completed_epochs": len(hybrid_rows),
        "best_epoch": selected_epoch,
        "best_val_ap": float(best_row["validation_ap"]),
        "best_val_auc": None,
        "best_yita": selected_yita,
        "val_num_neg_per_pos": val_num_neg_per_pos,
        "test_num_neg_per_pos": test_num_neg_per_pos,
        "metric_protocol_note": (
            "Validation AP uses 1 negative per positive while test AP uses 99; "
            "the two AP values are not directly comparable as a val-test gap."
        ),
        "final_test_metrics": {
            "ap": float(test_ap),
            "auc_or_mrr": float(test_mrr),
            "hit_at_10": float(test_hr[0]) if len(test_hr) > 0 else None,
        },
        "final_test_executed_count": 1,
        "final_test_wall_s": final_test_wall_s,
        "temporal_state_origin": "not_applicable",
        "validation_lifecycle_v1": True,
        "config": {
            "dataset_name": DATA,
            "seed": args.seed,
            "structure": params["structure"],
            "time": params["time"],
            "val_num_neg_per_pos": val_num_neg_per_pos,
            "test_num_neg_per_pos": test_num_neg_per_pos,
            "workspace_root": str(WORKSPACE_ROOT),
        },
        "paths": {
            "time_summary_path": str(time_epoch_metrics_path.parent / "summary.json"),
            "time_epoch_metrics_path": str(time_epoch_metrics_path),
            "structure_summary_path": str(STAGE_ROOT / "structure" / "summary.json"),
            "structure_test_summary_path": str(STAGE_ROOT / "structure_test" / "summary.json"),
            "val_structure_score_path": str(val_structure_score_filepath),
            "test_structure_score_path": str(test_structure_score_filepath),
            "val_time_data_path": str(val_time_data_filepath),
            "test_time_data_path": str(test_time_data_filepath),
            "test_time_score_path": str(test_time_score_filepath),
        },
        "wall_seconds": {
            "prepare": round(time_prepare_s, 6),
            "build": round((structure_total_s or 0.0) + (structure_test_total_s or 0.0), 6),
            "hybrid_final_test": round(final_test_wall_s, 6),
            "total": round(
                (structure_total_s or 0.0)
                + (structure_test_total_s or 0.0)
                + time_total_s
                + final_test_wall_s,
                6,
            ),
        },
    }
    if REPORT_DIR is not None:
        dump_json(REPORT_DIR / "summary.json", summary_payload)
        write_csv_rows(REPORT_DIR / "epoch_metrics.csv", list(hybrid_rows[0]), hybrid_rows)
    print(f"\nBest epoch: {selected_epoch}, best val ap: {summary_payload['best_val_ap']:.6f}")
elif benchmark_mode:
    time_epoch_rows = load_csv_rows(time_epoch_metrics_path)
    structure_total_s = safe_float(structure_summary.get("wall_seconds", {}).get("total")) if structure_summary else 0.0
    time_total_s = safe_float(time_summary.get("wall_seconds", {}).get("total")) if time_summary else 0.0
    time_prepare_s = safe_float(time_summary.get("wall_seconds", {}).get("prepare")) if time_summary else 0.0
    hybrid_rows: List[Dict[str, object]] = []
    cumulative_hybrid_wall_s = 0.0
    best_row: Optional[Dict[str, object]] = None

    for raw_row in time_epoch_rows:
        epoch = int(raw_row["epoch"])
        val_score_path = Path(raw_row["val_score_path"])
        test_score_path = Path(raw_row["test_score_path"])
        if not val_score_path.exists() or not test_score_path.exists():
            raise FileNotFoundError(f"missing per-epoch score cache for epoch {epoch}")

        val_time_pos_score, val_time_neg_score = load_pickle(val_score_path)
        test_time_pos_score, test_time_neg_score = load_pickle(test_score_path)

        hybrid_started = time.perf_counter()
        prepared_val_batches = prepare_hybrid_batches(
            val_data,
            bs,
            val_time_pos_score,
            val_time_neg_score,
            val_structure_score,
            val_delta_times_list,
            val_all_inds_list,
            1,
            time_topk,
            device,
        )
        best_yita, hybrid_val_ap = search_best_yita(prepared_val_batches)
        prepared_test_batches = prepare_hybrid_batches(
            test_data,
            bs,
            test_time_pos_score,
            test_time_neg_score,
            test_structure_score,
            test_delta_times_list,
            test_all_inds_list,
            99,
            time_topk,
            device,
        )
        hybrid_test_ap, hybrid_test_mrr, hybrid_test_hr = evaluate_test_metrics(prepared_test_batches, best_yita, k_list)
        hybrid_wall_s = time.perf_counter() - hybrid_started
        cumulative_hybrid_wall_s += hybrid_wall_s

        time_cumulative_train = safe_float(raw_row.get("cumulative_train_wall_s")) or 0.0
        time_cumulative_epoch = safe_float(raw_row.get("cumulative_epoch_wall_s")) or 0.0
        record = {
            "epoch": epoch,
            "train_loss": raw_row.get("train_loss", ""),
            "val_ap": hybrid_val_ap,
            "val_auc": "",
            "test_ap": hybrid_test_ap,
            "test_auc": hybrid_test_mrr,
            "train_wall_s": raw_row.get("train_wall_s", ""),
            "hybrid_eval_wall_s": round(hybrid_wall_s, 6),
            "cumulative_train_wall_s": round(time_cumulative_train, 6),
            "cumulative_epoch_wall_s": round(time_cumulative_epoch + cumulative_hybrid_wall_s, 6),
            "best_yita": best_yita,
            "val_score_path": str(val_score_path),
            "test_score_path": str(test_score_path),
            "hit_at_10": float(hybrid_test_hr[0]) if len(hybrid_test_hr) > 0 else "",
        }
        hybrid_rows.append(record)
        if best_row is None or hybrid_val_ap > float(best_row["val_ap"]):
            best_row = record

    assert best_row is not None
    completed_epochs = len(hybrid_rows)
    summary_payload = {
        "system": "EAGLE",
        "model": "EAGLE",
        "dataset": DATA,
        "run_type": "tta",
        "epochs_requested": args.epochs_requested or int(time_summary.get("epochs_requested", completed_epochs)),
        "completed_epochs": completed_epochs,
        "best_epoch": int(best_row["epoch"]),
        "best_val_ap": float(best_row["val_ap"]),
        "best_val_auc": None,
        "val_num_neg_per_pos": val_num_neg_per_pos,
        "test_num_neg_per_pos": test_num_neg_per_pos,
        "metric_protocol_note": (
            "Validation AP uses 1 negative per positive while test AP uses 99; "
            "the two AP values are not directly comparable as a val-test gap."
        ),
        "final_test_metrics": {
            "ap": float(best_row["test_ap"]),
            "auc_or_mrr": float(best_row["test_auc"]),
            "hit_at_10": float(best_row["hit_at_10"]) if best_row["hit_at_10"] != "" else None,
        },
        "config": {
            "dataset_name": DATA,
            "seed": args.seed,
            "structure": params["structure"],
            "time": params["time"],
            "val_num_neg_per_pos": val_num_neg_per_pos,
            "test_num_neg_per_pos": test_num_neg_per_pos,
            "workspace_root": str(WORKSPACE_ROOT),
        },
        "paths": {
            "time_summary_path": str(time_epoch_metrics_path.parent / "summary.json"),
            "time_epoch_metrics_path": str(time_epoch_metrics_path),
            "structure_summary_path": str(STAGE_ROOT / "structure" / "summary.json") if STAGE_ROOT is not None else "",
            "val_structure_score_path": str(val_structure_score_filepath),
            "test_structure_score_path": str(test_structure_score_filepath),
            "val_time_data_path": str(val_time_data_filepath),
            "test_time_data_path": str(test_time_data_filepath),
        },
        "wall_seconds": {
            "prepare": round(time_prepare_s or 0.0, 6),
            "build": round(structure_total_s or 0.0, 6),
            "hybrid": round(cumulative_hybrid_wall_s, 6),
            "total": round((structure_total_s or 0.0) + (time_total_s or 0.0) + cumulative_hybrid_wall_s, 6),
        },
    }
    if REPORT_DIR is not None:
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
                "hybrid_eval_wall_s",
                "cumulative_train_wall_s",
                "cumulative_epoch_wall_s",
                "best_yita",
                "val_score_path",
                "test_score_path",
                "hit_at_10",
            ],
            hybrid_rows,
        )
    print(f"\nBest epoch: {summary_payload['best_epoch']}, best val ap: {summary_payload['best_val_ap']:.6f}")
else:
    with val_time_score_filepath.open("rb") as vtf:
        val_time_score = pickle.load(vtf)
    val_time_pos_score, val_time_neg_score = val_time_score

    with test_time_score_filepath.open("rb") as ttf:
        test_time_score = pickle.load(ttf)
    test_time_pos_score, test_time_neg_score = test_time_score

    prepared_val_batches = prepare_hybrid_batches(
        val_data,
        bs,
        val_time_pos_score,
        val_time_neg_score,
        val_structure_score,
        val_delta_times_list,
        val_all_inds_list,
        1,
        time_topk,
        device,
    )
    best_yita, best_val_ap = search_best_yita(prepared_val_batches)
    print(f"\nBest yita: {best_yita}\n")

    prepared_test_batches = prepare_hybrid_batches(
        test_data,
        bs,
        test_time_pos_score,
        test_time_neg_score,
        test_structure_score,
        test_delta_times_list,
        test_all_inds_list,
        99,
        time_topk,
        device,
    )
    test_ap, test_mrr, test_hr = evaluate_test_metrics(prepared_test_batches, best_yita, k_list)

    if REPORT_DIR is not None:
        summary_payload = {
            "system": "EAGLE",
            "model": "EAGLE",
            "dataset": DATA,
            "run_type": "tta",
            "epochs_requested": args.epochs_requested or 1,
            "completed_epochs": 1,
            "best_epoch": 1,
            "best_val_ap": best_val_ap,
            "best_val_auc": None,
            "val_num_neg_per_pos": val_num_neg_per_pos,
            "test_num_neg_per_pos": test_num_neg_per_pos,
            "metric_protocol_note": (
                "Validation AP uses 1 negative per positive while test AP uses 99; "
                "the two AP values are not directly comparable as a val-test gap."
            ),
            "final_test_metrics": {
                "ap": float(test_ap),
                "auc_or_mrr": float(test_mrr),
                "hit_at_10": float(test_hr[0]) if len(test_hr) > 0 else None,
            },
            "config": {
                "dataset_name": DATA,
                "seed": args.seed,
                "structure": params["structure"],
                "time": params["time"],
                "val_num_neg_per_pos": val_num_neg_per_pos,
                "test_num_neg_per_pos": test_num_neg_per_pos,
                "workspace_root": str(WORKSPACE_ROOT),
            },
            "paths": {
                "val_time_score_path": str(val_time_score_filepath),
                "test_time_score_path": str(test_time_score_filepath),
                "val_structure_score_path": str(val_structure_score_filepath),
                "test_structure_score_path": str(test_structure_score_filepath),
            },
            "wall_seconds": {
                "prepare": 0.0,
                "build": 0.0,
                "hybrid": 0.0,
                "total": 0.0,
            },
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
                "hybrid_eval_wall_s",
                "cumulative_train_wall_s",
                "cumulative_epoch_wall_s",
                "best_yita",
                "val_score_path",
                "test_score_path",
                "hit_at_10",
            ],
            [
                {
                    "epoch": 1,
                    "train_loss": "",
                    "val_ap": best_val_ap,
                    "val_auc": "",
                    "test_ap": test_ap,
                    "test_auc": test_mrr,
                    "train_wall_s": "",
                    "hybrid_eval_wall_s": 0.0,
                    "cumulative_train_wall_s": 0.0,
                    "cumulative_epoch_wall_s": 0.0,
                    "best_yita": best_yita,
                    "val_score_path": str(val_time_score_filepath),
                    "test_score_path": str(test_time_score_filepath),
                    "hit_at_10": float(test_hr[0]) if len(test_hr) > 0 else "",
                }
            ],
        )
