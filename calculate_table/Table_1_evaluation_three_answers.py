"""Evaluate one or three answer streams and align paired files by fixID.

Base files provide answer 1 through ``inference_results``. ``*_n2.json``
files provide answers 2 and 3 through ``inference_results_req_1`` and
``inference_results_req_2``. Metrics use the number of instances actually
evaluated for each answer as their denominator and follow ``evaluation.py``.
Standalone base or ``_n2`` files are also supported.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

DEFAULT_EVAL_DIRS = [
    "./data/gpt-5.2",
    "./data/gpt-4o",
    "./data/claude-sonnet-4-5",
    "./data/deepseek-v3.2",
]
FULL_DATASET_N = 453


def normalize_path(value: Any) -> Optional[str]:
    """Normalize diff-style paths used by predictions, context, and ground truth."""
    if not isinstance(value, str) or not value or value == "/dev/null":
        return None
    return value[2:] if value.startswith(("a/", "b/")) else value


def score_one_instance(
    instance: Dict[str, Any],
    inference_results: Any,
    eval_line: int,
) -> Tuple[int, int, int, bool]:
    """Return TP/FP/FN and hit status using evaluation.py's logic."""
    if not isinstance(inference_results, list):
        inference_results = []

    retrieved_paths = {
        path
        for raw in instance.get("context_files_paths") or []
        if (path := normalize_path(raw)) is not None
    }
    ground_truth = {
        (path, item.get("line"))
        for item in instance.get("ground_truth_results") or []
        if isinstance(item, dict)
        and (path := normalize_path(item.get("path"))) in retrieved_paths
        and isinstance(item.get("line"), int)
    }
    scope_paths = {path for path, _ in ground_truth}
    predictions = {
        (path, item.get("line"))
        for item in inference_results
        if isinstance(item, dict)
        and (path := normalize_path(item.get("path"))) in scope_paths
        and isinstance(item.get("line"), int)
    }

    # Match unique locations one-to-one. For sorted line locations in one file,
    # this two-pointer procedure gives a maximum-cardinality tolerance match.
    tp = 0
    for path in scope_paths:
        pred_lines = sorted(line for pred_path, line in predictions if pred_path == path)
        gt_lines = sorted(line for gt_path, line in ground_truth if gt_path == path)
        pred_index = gt_index = 0
        while pred_index < len(pred_lines) and gt_index < len(gt_lines):
            pred_line = pred_lines[pred_index]
            gt_line = gt_lines[gt_index]
            if pred_line < gt_line - eval_line:
                pred_index += 1
            elif gt_line < pred_line - eval_line:
                gt_index += 1
            else:
                tp += 1
                pred_index += 1
                gt_index += 1

    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn, tp > 0


def finalize_counts(
    tp: int, fp: int, fn: int, hit_num: int, total_instances: int
) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = hit_num / total_instances if total_instances > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "hit_num": hit_num,
        "precision": precision,
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "total_instances": total_instances,
    }


def n2_counterpart(base_json_path: str) -> str:
    """..._bm25_5.json -> ..._bm25_5_n2.json"""
    if base_json_path.endswith("_n2.json"):
        return base_json_path
    if not base_json_path.endswith(".json"):
        return base_json_path + "_n2.json"
    return base_json_path[:-5] + "_n2.json"


def base_counterpart(n2_json_path: str) -> str:
    """..._bm25_5_n2.json -> ..._bm25_5.json"""
    if n2_json_path.endswith("_n2.json"):
        return n2_json_path[: -len("_n2.json")] + ".json"
    return n2_json_path


def detect_request_keys(sample_instance: Dict[str, Any]) -> List[str]:
    """Finalize metrics using the same definitions as evaluation.py."""
    request_keys = [
        "inference_results_req_1",
        "inference_results_req_2",
        "inference_results_req_3",
    ]
    found = [k for k in request_keys if k in sample_instance]
    return found if found else ["inference_results"]


def aggregate_single_preloaded(
    by_id: Dict[str, Dict[str, Any]],
    path: str,
    skipped: int,
    eval_line: int,
    allowed_fixids: Optional[Set[str]] = None,
    *,
    progress: bool = True,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    """
    Use the same logic as aggregate_single_file with preloaded fixID rows.
    """
    empty = finalize_counts(0, 0, 0, 0, 0)
    if allowed_fixids is not None:
        by_id = {k: v for k, v in by_id.items() if k in allowed_fixids}
    if not by_id:
        meta = {
            "eval_mode": "single_file",
            "source_path": os.path.normpath(path),
            "skipped_json_lines": skipped,
            "fixid_total": 0,
            "detected_inference_keys": [],
            "evaluated_answer1": 0,
            "evaluated_answer2": 0,
            "evaluated_answer3": 0,
            "evaluated_inference_results_req_3": 0,
            "skipped_base_json": skipped,
            "skipped_n2_json": 0,
            "fixid_union": 0,
            "fixid_in_both": 0,
            "fixid_only_in_base": 0,
            "fixid_only_in_n2": 0,
            "fixid_in_base_total": 0,
            "fixid_in_n2_total": 0,
        }
        return empty, empty, empty, meta, dict(empty)

    first = next(iter(by_id.values()))
    keys = detect_request_keys(first)

    sums: List[Tuple[int, int, int, int]] = [(0, 0, 0, 0) for _ in range(3)]
    totals = [0, 0, 0]
    sum_req3 = (0, 0, 0, 0)
    total_req3 = 0

    key_to_slot = {
        "inference_results": 0,
        "inference_results_req_1": 1,
        "inference_results_req_2": 2,
    }

    for inst in tqdm(
        by_id.values(),
        desc="eval single file",
        unit="inst",
        disable=not progress,
    ):
        for k in keys:
            if k == "inference_results_req_3":
                tp, fp, fn, hit = score_one_instance(inst, inst.get(k), eval_line)
                t0, t1, t2, h = sum_req3
                sum_req3 = (t0 + tp, t1 + fp, t2 + fn, h + (1 if hit else 0))
                total_req3 += 1
            elif k in key_to_slot:
                slot = key_to_slot[k]
                tp, fp, fn, hit = score_one_instance(inst, inst.get(k), eval_line)
                t0, t1, t2, h = sums[slot]
                sums[slot] = (t0 + tp, t1 + fp, t2 + fn, h + (1 if hit else 0))
                totals[slot] += 1

    accuracy_denominator = (
        len(allowed_fixids) if allowed_fixids is not None else FULL_DATASET_N
    )
    out = [
        finalize_counts(tp, fp, fn, hit, accuracy_denominator if observed else 0)
        for (tp, fp, fn, hit), observed in zip(sums, totals)
    ]
    m_req3 = finalize_counts(
        *sum_req3, accuracy_denominator if total_req3 else 0
    )

    n = len(by_id)
    meta = {
        "eval_mode": "single_file",
        "source_path": os.path.normpath(path),
        "skipped_json_lines": skipped,
        "fixid_total": n,
        "detected_inference_keys": list(keys),
        "evaluated_answer1": totals[0],
        "evaluated_answer2": totals[1],
        "evaluated_answer3": totals[2],
        "evaluated_inference_results_req_3": total_req3,
        "skipped_base_json": skipped,
        "skipped_n2_json": 0,
        "fixid_union": n,
        "fixid_in_both": 0,
        "fixid_only_in_base": 0,
        "fixid_only_in_n2": 0,
        "fixid_in_base_total": n,
        "fixid_in_n2_total": 0,
    }
    return out[0], out[1], out[2], meta, m_req3


def aggregate_single_file(
    path: str,
    eval_line: int,
    allowed_fixids: Optional[Set[str]] = None,
    *,
    progress: bool = True,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    """
    Aggregate every inference field detected in a standalone file.
    The final return value is an empty placeholder when req_3 is absent.
    """
    by_id, skipped = load_instances_by_fixid(path, progress=progress)
    return aggregate_single_preloaded(
        by_id,
        path,
        skipped,
        eval_line,
        allowed_fixids,
        progress=progress,
    )


def load_instances_by_fixid(
    path: str, *, progress: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Return fixID-to-row mapping and parse-error count; later rows win."""
    by_id: Dict[str, Dict[str, Any]] = {}
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in tqdm(
            f,
            desc=f"load {os.path.basename(path)}",
            unit="line",
            disable=not progress,
        ):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                continue
            fid = obj.get("fixID")
            if fid is None:
                skipped += 1
                continue
            by_id[str(fid)] = obj
    return by_id, skipped


def aggregate_triple_preloaded(
    base_map: Dict[str, Dict[str, Any]],
    n2_map: Dict[str, Dict[str, Any]],
    skip_b: int,
    skip_n2: int,
    eval_line: int,
    allowed_fixids: Optional[Set[str]] = None,
    *,
    progress: bool = True,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, int],
]:
    """
    Use aggregate_triple_by_fixid logic with preloaded base and n2 maps.
    """
    sums: List[Tuple[int, int, int, int]] = [(0, 0, 0, 0) for _ in range(3)]
    totals = [0, 0, 0]

    if allowed_fixids is not None:
        base_map = {k: v for k, v in base_map.items() if k in allowed_fixids}
        n2_map = {k: v for k, v in n2_map.items() if k in allowed_fixids}

    keys_b = set(base_map.keys())
    keys_n2 = set(n2_map.keys())
    common = keys_b & keys_n2
    only_base = keys_b - keys_n2
    only_n2 = keys_n2 - keys_b
    all_ids = sorted(keys_b | keys_n2)

    for fid in tqdm(
        all_ids, desc="eval fixID union", unit="inst", disable=not progress
    ):
        inst_b = base_map.get(fid)
        inst_n2 = n2_map.get(fid)
        gt_inst = inst_b if inst_b is not None else inst_n2
        assert gt_inst is not None

        if inst_b is not None:
            inf1 = inst_b.get("inference_results")
            tp, fp, fn, hit = score_one_instance(gt_inst, inf1, eval_line)
            t0, t1, t2, h = sums[0]
            sums[0] = (t0 + tp, t1 + fp, t2 + fn, h + (1 if hit else 0))
            totals[0] += 1

        if inst_n2 is not None:
            inf2 = inst_n2.get("inference_results_req_1")
            tp, fp, fn, hit = score_one_instance(gt_inst, inf2, eval_line)
            t0, t1, t2, h = sums[1]
            sums[1] = (t0 + tp, t1 + fp, t2 + fn, h + (1 if hit else 0))
            totals[1] += 1

            inf3 = inst_n2.get("inference_results_req_2")
            tp, fp, fn, hit = score_one_instance(gt_inst, inf3, eval_line)
            t0, t1, t2, h = sums[2]
            sums[2] = (t0 + tp, t1 + fp, t2 + fn, h + (1 if hit else 0))
            totals[2] += 1

    accuracy_denominator = (
        len(allowed_fixids) if allowed_fixids is not None else FULL_DATASET_N
    )
    out = [
        finalize_counts(tp, fp, fn, hit, accuracy_denominator if observed else 0)
        for (tp, fp, fn, hit), observed in zip(sums, totals)
    ]

    meta = {
        "skipped_base_json": skip_b,
        "skipped_n2_json": skip_n2,
        "eval_mode": "pair",
        "fixid_union": len(all_ids),
        "fixid_in_both": len(common),
        "fixid_only_in_base": len(only_base),
        "fixid_only_in_n2": len(only_n2),
        "fixid_in_base_total": len(base_map),
        "fixid_in_n2_total": len(n2_map),
        "evaluated_answer1": totals[0],
        "evaluated_answer2": totals[1],
        "evaluated_answer3": totals[2],
    }
    return out[0], out[1], out[2], meta


def aggregate_triple_by_fixid(
    path_base: str,
    path_n2: str,
    eval_line: int,
    allowed_fixids: Optional[Set[str]] = None,
    *,
    progress: bool = True,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, int],
]:
    """
    Evaluate answer 1 when the base row exists and answers 2/3 when the n2 row
    exists. Prefer ground truth from the base row. ``allowed_fixids`` limits
    evaluation to a selected subset.
    """
    base_map, skip_b = load_instances_by_fixid(path_base, progress=progress)
    n2_map, skip_n2 = load_instances_by_fixid(path_n2, progress=progress)
    return aggregate_triple_preloaded(
        base_map,
        n2_map,
        skip_b,
        skip_n2,
        eval_line,
        allowed_fixids,
        progress=progress,
    )


def aggregate_over_answers(
    m1: Dict[str, Any], m2: Dict[str, Any], m3: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Return mean and population stddev over nonempty answer streams."""
    active = [m for m in (m1, m2, m3) if m["total_instances"] > 0]
    if not active:
        return None
    prec = [m["precision"] for m in active]
    acc = [m["accuracy"] for m in active]
    f1s = [m["f1_score"] for m in active]
    rec = [m["recall"] for m in active]
    return {
        "answers_included": len(active),
        "precision_mean": statistics.mean(prec),
        "precision_std": statistics.pstdev(prec) if len(prec) > 1 else 0.0,
        "recall_mean": statistics.mean(rec),
        "recall_std": statistics.pstdev(rec) if len(rec) > 1 else 0.0,
        "f1_mean": statistics.mean(f1s),
        "f1_std": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
        "accuracy_mean": statistics.mean(acc),
        "accuracy_std": statistics.pstdev(acc) if len(acc) > 1 else 0.0,
    }


def build_result_record_pair(
    path_base: str,
    path_n2: str,
    eval_line: int,
    m1: Dict[str, Any],
    m2: Dict[str, Any],
    m3: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    agg = aggregate_over_answers(m1, m2, m3)
    return {
        "eval_mode": "pair",
        "eval_line": eval_line,
        "source_base": os.path.normpath(path_base),
        "source_n2": os.path.normpath(path_n2),
        "source_file": None,
        "meta": dict(meta),
        "answer_1_inference_results": dict(m1),
        "answer_2_inference_results_req_1": dict(m2),
        "answer_3_inference_results_req_2": dict(m3),
        "answer_inference_results_req_3": None,
        "aggregate_over_available_answers": agg,
    }


def build_result_record_single(
    path: str,
    eval_line: int,
    m1: Dict[str, Any],
    m2: Dict[str, Any],
    m3: Dict[str, Any],
    meta: Dict[str, Any],
    m_req3: Dict[str, Any],
) -> Dict[str, Any]:
    agg = aggregate_over_answers(m1, m2, m3)
    rec: Dict[str, Any] = {
        "eval_mode": "single_file",
        "eval_line": eval_line,
        "source_base": None,
        "source_n2": None,
        "source_file": os.path.normpath(path),
        "meta": dict(meta),
        "answer_1_inference_results": dict(m1),
        "answer_2_inference_results_req_1": dict(m2),
        "answer_3_inference_results_req_2": dict(m3),
        "aggregate_over_available_answers": agg,
    }
    if m_req3.get("total_instances", 0) > 0:
        rec["answer_inference_results_req_3"] = dict(m_req3)
    else:
        rec["answer_inference_results_req_3"] = None
    return rec


def write_result_json(path_out: str, record: Dict[str, Any]) -> None:
    parent = os.path.dirname(path_out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def print_block(
    label: str,
    m1: Dict[str, Any],
    m2: Dict[str, Any],
    m3: Dict[str, Any],
    meta: Dict[str, Any],
    m_req3: Dict[str, Any] | None = None,
) -> None:
    print("--------------------------------")
    print(label)
    if meta.get("eval_mode") == "single_file":
        print(
            f"Mode: single_file | {meta.get('source_path')} | "
            f"fixID total: {meta.get('fixid_total')} | "
            f"keys: {meta.get('detected_inference_keys')}"
        )
        print(
            f"Per-answer sample counts: ans1={meta.get('evaluated_answer1', 0)} "
            f"ans2={meta.get('evaluated_answer2', 0)} "
            f"ans3={meta.get('evaluated_answer3', 0)}"
        )
        rq3 = meta.get("evaluated_inference_results_req_3", 0)
        if rq3:
            print(f"  inference_results_req_3 samples: {rq3}")
        print(f"Skipped JSON lines: {meta.get('skipped_json_lines', meta.get('skipped_base_json', 0))}")
    else:
        print(
            f"fixID union: {meta['fixid_union']} | in both files: {meta['fixid_in_both']} | "
            f"only base: {meta['fixid_only_in_base']} | only _n2: {meta['fixid_only_in_n2']} | "
            f"base total: {meta['fixid_in_base_total']} | n2 total: {meta['fixid_in_n2_total']}"
        )
        print(
            f"Per-answer sample counts: ans1={meta['evaluated_answer1']} "
            f"ans2={meta['evaluated_answer2']} ans3={meta['evaluated_answer3']}"
        )
        print(
            f"Skipped JSON lines: base {meta['skipped_base_json']} | "
            f"n2 {meta['skipped_n2_json']}"
        )
    for tag, m in (
        ("Answer 1 — inference_results (base .json)", m1),
        ("Answer 2 — inference_results_req_1 (_n2.json)", m2),
        ("Answer 3 — inference_results_req_2 (_n2.json)", m3),
    ):
        print(f"  [{tag}]  (n={m['total_instances']})")
        print(
            f"    P/R/F1/Acc: {m['precision']:.4f} / {m['recall']:.4f} / "
            f"{m['f1_score']:.4f} / {m['accuracy']:.4f} | "
            f"TP/FP/FN: {m['TP']}/{m['FP']}/{m['FN']}"
        )

    if m_req3 is not None and m_req3.get("total_instances", 0) > 0:
        m = m_req3
        print(f"  [Answer extra — inference_results_req_3]  (n={m['total_instances']})")
        print(
            f"    P/R/F1/Acc: {m['precision']:.4f} / {m['recall']:.4f} / "
            f"{m['f1_score']:.4f} / {m['accuracy']:.4f} | "
            f"TP/FP/FN: {m['TP']}/{m['FP']}/{m['FN']}"
        )

    agg = aggregate_over_answers(m1, m2, m3)
    if agg is not None:
        label_mean = (
            "mean ± std (over 3 answers)"
            if agg["answers_included"] == 3
            else f"mean ± std (over {agg['answers_included']} answer(s) with n>0)"
        )
        print(f"  ---- {label_mean} ----")
        print(
            f"    Precision: {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}"
        )
        print(f"    Recall:    {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}")
        print(f"    F1:        {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")
        print(f"    Accuracy:  {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate paired base/_n2 files by fixID union, or detected inference keys in standalone files",
    )
    parser.add_argument(
        "-e",
        "--eval_dirs",
        nargs="*",
        default=None,
        help="One or more evaluation directories; defaults to four model directories",
    )
    parser.add_argument(
        "-n",
        "--line",
        type=int,
        default=0,
        help="Line-number tolerance window, matching evaluation.py",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="eval_result",
        help="JSON output root; results are grouped by input directory name",
    )
    args = parser.parse_args()
    dirs = args.eval_dirs if args.eval_dirs else DEFAULT_EVAL_DIRS
    out_root = os.path.normpath(args.out_dir)

    for eval_dir in dirs:
        eval_dir = os.path.normpath(eval_dir)
        if not os.path.isdir(eval_dir):
            print(f"[skip] not a directory: {eval_dir}")
            continue

        print("########################################")
        print(f"Directory: {eval_dir}")
        dir_slug = os.path.basename(eval_dir.rstrip(os.sep)) or "data"
        json_dir_out = os.path.join(out_root, dir_slug)

        json_files = sorted(
            f for f in os.listdir(eval_dir) if f.endswith(".json")
        )
        base_files = [f for f in json_files if not f.endswith("_n2.json")]

        for fname in base_files:
            path_base = os.path.join(eval_dir, fname)
            path_n2 = n2_counterpart(path_base)
            stem = os.path.splitext(fname)[0]

            if not os.path.isfile(path_n2):
                print("--------------------------------")
                print(f"Single file (no _n2 pair): {fname}")
                m1, m2, m3, meta, m_req3 = aggregate_single_file(
                    path_base, args.line
                )
                print_block(f"Single: {fname}", m1, m2, m3, meta, m_req3)
                out_name = f"{stem}_single_eval.json"
                out_path = os.path.join(json_dir_out, out_name)
                record = build_result_record_single(
                    path_base, args.line, m1, m2, m3, meta, m_req3
                )
                write_result_json(out_path, record)
                print(f"  [saved] {out_path}")
                continue

            m1, m2, m3, meta = aggregate_triple_by_fixid(
                path_base, path_n2, args.line
            )
            print_block(
                f"Pair: {fname}  +  {os.path.basename(path_n2)}",
                m1,
                m2,
                m3,
                meta,
            )

            out_name = f"{stem}_triple_eval.json"
            out_path = os.path.join(json_dir_out, out_name)
            record = build_result_record_pair(
                path_base, path_n2, args.line, m1, m2, m3, meta
            )
            write_result_json(out_path, record)
            print(f"  [saved] {out_path}")

        # Standalone _n2 file without a matching base file.
        n2_files = [f for f in json_files if f.endswith("_n2.json")]
        for fname in n2_files:
            path_n2_only = os.path.join(eval_dir, fname)
            path_base_c = base_counterpart(path_n2_only)
            if os.path.isfile(path_base_c):
                continue
            print("--------------------------------")
            print(f"Single file (orphan _n2, no base): {fname}")
            m1, m2, m3, meta, m_req3 = aggregate_single_file(
                path_n2_only, args.line
            )
            print_block(f"Single: {fname}", m1, m2, m3, meta, m_req3)
            stem = os.path.splitext(fname)[0]
            out_name = f"{stem}_single_eval.json"
            out_path = os.path.join(json_dir_out, out_name)
            record = build_result_record_single(
                path_n2_only, args.line, m1, m2, m3, meta, m_req3
            )
            write_result_json(out_path, record)
            print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
