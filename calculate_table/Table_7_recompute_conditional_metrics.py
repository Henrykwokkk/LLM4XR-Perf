"""Recompute the BM25/CodeBERT table on fixed retrieval-success subsets.

The retrieval-success set is shared across models because retrieval is
deterministic.  For each (retriever, k), this script:

1. derives the successful fixIDs from complete (453-instance) result files;
2. verifies that all complete files produce the same set and that its size
   matches the subset size stated in the paper caption;
3. evaluates each available model answer only on that exact fixID set; and
4. excludes an answer/run unless it covers every fixID in the set.

Conditional precision/recall/F1 are therefore never computed by silently
treating missing successful instances as failures. E2E accuracy uses the fixed
453-instance benchmark denominator for every answer stream; instances absent
from an incomplete stream are counted as failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from Table_1_evaluation_three_answers import finalize_counts, score_one_instance


ROOT = Path(__file__).resolve().parent.parent
FULL_N = 453
KS = (5, 10, 20, 30, 35)
EXPECTED_N = {
    "bm25": {5: 40, 10: 66, 20: 86, 30: 97, 35: 113},
    "codebert": {5: 31, 10: 53, 20: 73, 30: 85, 35: 93},
}
INFERENCE_KEYS = (
    "inference_results",
    "inference_results_req_1",
    "inference_results_req_2",
    "inference_results_req_3",
)

MODEL_DIRS = {
    "Gemini-3-Pro": {
        "bm25": ROOT / "data" / "gemini3-pro",
        "codebert": ROOT / "data" / "embedding" / "gemini-3-pro",
    },
    "Qwen3-235B-A22B": {
        "bm25": ROOT / "data" / "qwen3",
        "codebert": ROOT / "data" / "embedding" / "qwen3",
    },
    "DeepSeek-v3.2": {
        "bm25": ROOT / "data" / "deepseek-v3.2",
        "codebert": ROOT / "data" / "embedding" / "deepseek-3-2",
    },
    "Claude-Sonnet-4.5": {
        "bm25": ROOT / "data" / "claude-sonnet-4-5",
        "codebert": ROOT / "data" / "embedding" / "claude-sonnet-4-5",
    },
    "GPT-4o": {
        "bm25": ROOT / "data" / "gpt-4o",
        "codebert": ROOT / "data" / "embedding" / "gpt-4o",
    },
    "GPT-5.2": {
        "bm25": ROOT / "data" / "gpt-5.2",
        "codebert": ROOT / "data" / "embedding" / "gpt-5.2",
    },
}


def normalize_path(value: Optional[str]) -> Optional[str]:
    if not value or value == "/dev/null":
        return None
    return value[2:] if value.startswith(("a/", "b/")) else value


def ground_truth_paths(instance: Dict[str, Any]) -> Set[str]:
    paths: Set[str] = set()
    for item in instance.get("ground_truth_results") or []:
        raw = item.get("path") if isinstance(item, dict) else item
        path = normalize_path(raw)
        if path:
            paths.add(path)
    return paths


def retrieved_paths(instance: Dict[str, Any]) -> Set[str]:
    paths: Set[str] = set()
    for raw in instance.get("context_files_paths") or []:
        path = normalize_path(raw)
        if path:
            paths.add(path)
    return paths


def is_retrieval_success(instance: Dict[str, Any]) -> bool:
    gt = ground_truth_paths(instance)
    return bool(gt and (gt & retrieved_paths(instance)))


def load_jsonl(path: Path) -> Tuple[Dict[str, Dict[str, Any]], int]:
    by_id: Dict[str, Dict[str, Any]] = {}
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            fixid = item.get("fixID")
            if fixid is not None:
                by_id[str(fixid)] = item
    return by_id, malformed


def result_files(retriever: str, k: int, directory: Path) -> List[Path]:
    kind = "bm25" if retriever == "bm25" else "embedding"
    pattern = re.compile(rf"_{kind}_{k}(?:_n\d+)?\.json$")
    return sorted(
        path
        for path in directory.glob("*.json")
        if pattern.search(path.name) and not path.name.endswith(".progress.json")
    )


def answer_streams(path: Path, by_id: Dict[str, Dict[str, Any]]) -> Iterable[Tuple[str, Path, Dict[str, Dict[str, Any]]]]:
    if not by_id:
        return
    sample = next(iter(by_id.values()))
    keys = [key for key in INFERENCE_KEYS if key in sample]
    for key in keys or ["inference_results"]:
        yield key, path, by_id


def build_shared_success_set(retriever: str, k: int) -> Tuple[Set[str], List[str]]:
    candidates: List[Tuple[Path, Set[str]]] = []
    reports: List[str] = []
    seen_paths: Set[Path] = set()
    for dirs in MODEL_DIRS.values():
        for path in result_files(retriever, k, dirs[retriever]):
            # For BM25, _n2 files contain extra answers but are not independent
            # retrieval sources. CodeBERT stores all three answers in _n3.
            if retriever == "bm25" and re.search(r"_n\d+\.json$", path.name):
                continue
            if path in seen_paths:
                continue
            seen_paths.add(path)
            by_id, malformed = load_jsonl(path)
            if len(by_id) != FULL_N:
                reports.append(f"SKIP {path}: instances={len(by_id)}, malformed={malformed}")
                continue
            success = {fid for fid, inst in by_id.items() if is_retrieval_success(inst)}
            candidates.append((path, success))
            reports.append(f"USE  {path}: instances={len(by_id)}, success={len(success)}")

    if not candidates:
        raise RuntimeError(f"No complete retrieval source for {retriever} k={k}")
    reference_path, reference = candidates[0]
    for path, success in candidates[1:]:
        if success != reference:
            only_ref = len(reference - success)
            only_other = len(success - reference)
            raise RuntimeError(
                f"Retrieval-success sets disagree for {retriever} k={k}: "
                f"{reference_path.name} vs {path.name} "
                f"(only_reference={only_ref}, only_other={only_other})"
            )
    expected = EXPECTED_N[retriever][k]
    if len(reference) != expected:
        raise RuntimeError(
            f"Caption mismatch for {retriever} k={k}: expected n={expected}, "
            f"derived n={len(reference)}"
        )
    return reference, reports


def score_answer(
    by_id: Dict[str, Dict[str, Any]],
    inference_key: str,
    allowed_ids: Set[str],
    line_window: int,
    accuracy_denominator: Optional[int] = None,
) -> Dict[str, Any]:
    covered = allowed_ids & set(by_id)
    tp = fp = fn = hits = 0
    for fid in sorted(covered):
        instance = by_id[fid]
        one_tp, one_fp, one_fn, hit = score_one_instance(
            instance, instance.get(inference_key), line_window
        )
        tp += one_tp
        fp += one_fp
        fn += one_fn
        hits += int(hit)
    # Accuracy uses the declared evaluation population.  Any allowed instance
    # absent from an incomplete run therefore contributes zero hits instead of
    # being silently removed from the denominator.
    denominator = (
        len(allowed_ids) if accuracy_denominator is None else accuracy_denominator
    )
    metrics = finalize_counts(tp, fp, fn, hits, denominator)
    metrics.update(
        {
            "inference_key": inference_key,
            "covered_success_instances": len(covered),
            "expected_success_instances": len(allowed_ids),
            "complete_success_coverage": covered == allowed_ids,
        }
    )
    return metrics


def pool_counts(
    rows: List[Dict[str, Any]], accuracy_denominator_per_answer: int
) -> Optional[Dict[str, Any]]:
    """Pool integer counts across answer streams, then calculate metrics."""
    if not rows:
        return None
    return finalize_counts(
        sum(int(row["TP"]) for row in rows),
        sum(int(row["FP"]) for row in rows),
        sum(int(row["FN"]) for row in rows),
        sum(int(row["hit_num"]) for row in rows),
        accuracy_denominator_per_answer * len(rows),
    )


def evaluate_model(
    model: str,
    retriever: str,
    k: int,
    success_ids: Set[str],
    line_window: int,
) -> Dict[str, Any]:
    per_answer: List[Dict[str, Any]] = []
    conditional_accuracy_answers: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    e2e_answers: List[Dict[str, Any]] = []
    for path in result_files(retriever, k, MODEL_DIRS[model][retriever]):
        by_id, malformed = load_jsonl(path)
        for key, source, answer_map in answer_streams(path, by_id):
            metrics = score_answer(
                answer_map,
                key,
                success_ids,
                line_window,
                accuracy_denominator=len(success_ids),
            )
            record = {
                "source_file": source.name,
                "source_instances": len(answer_map),
                "malformed_lines": malformed,
                **metrics,
            }
            if metrics["complete_success_coverage"]:
                per_answer.append(record)
            else:
                record["exclusion_reason"] = "incomplete retrieval-success subset coverage"
                excluded.append(record)

            # Conditional Accuracy accepts incomplete streams but retains the
            # caption subset size as denominator; missing instances are misses.
            conditional_accuracy_answers.append(record)

            # Conditional and E2E accuracy use the same successful-localization
            # numerator from the fixed retrieval-success subset. Their only
            # difference is the denominator: n versus the full 453 instances.
            e2e_ids = success_ids
            e2e_answers.append(
                score_answer(
                    answer_map,
                    key,
                    e2e_ids,
                    line_window,
                    accuracy_denominator=FULL_N,
                )
            )

    pooled_conditional = pool_counts(per_answer, len(success_ids))
    pooled_conditional_accuracy = pool_counts(
        conditional_accuracy_answers, len(success_ids)
    )
    pooled_e2e = pool_counts(e2e_answers, FULL_N)

    return {
        "model": model,
        "retriever": retriever,
        "k": k,
        "subset_n": len(success_ids),
        "answers_included": len(per_answer),
        "three_run_equivalent_weight_per_answer": (
            3.0 / len(per_answer) if per_answer else None
        ),
        "three_run_equivalent": bool(per_answer),
        "answers_excluded": len(excluded),
        "conditional_precision": (
            pooled_conditional["precision"] if pooled_conditional else None
        ),
        "conditional_recall": (
            pooled_conditional["recall"] if pooled_conditional else None
        ),
        "conditional_f1": (
            pooled_conditional["f1_score"] if pooled_conditional else None
        ),
        "conditional_hit_rate": (
            pooled_conditional_accuracy["accuracy"]
            if pooled_conditional_accuracy else None
        ),
        "conditional_accuracy_answers_included": len(
            conditional_accuracy_answers
        ),
        "conditional_accuracy_missing_instances_are_failures": True,
        "e2e_accuracy": pooled_e2e["accuracy"] if pooled_e2e else None,
        "e2e_answers_included": len(e2e_answers),
        "e2e_three_run_equivalent_weight_per_answer": (
            3.0 / len(e2e_answers) if e2e_answers else None
        ),
        "per_answer": per_answer,
        "excluded_answers": excluded,
        "pooled_conditional_counts": pooled_conditional,
        "pooled_conditional_accuracy_counts": pooled_conditional_accuracy,
        "pooled_e2e_counts": pooled_e2e,
    }


def pct(value: Optional[float]) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}"


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fields = [
        "tau", "retriever", "k", "subset_n", "model", "answers_included",
        "three_run_equivalent_weight_per_answer", "answers_excluded",
        "conditional_precision", "conditional_recall",
        "conditional_f1", "conditional_hit_rate", "e2e_accuracy",
        "conditional_accuracy_answers_included",
        "conditional_accuracy_missing_instances_are_failures",
        "e2e_answers_included", "e2e_three_run_equivalent_weight_per_answer",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})


def print_summary(records: List[Dict[str, Any]]) -> None:
    for tau in (3, 0):
        for retriever in ("bm25", "codebert"):
            print(f"\n=== {retriever.upper()} conditional metrics (tau={tau}) ===")
            for k in KS:
                print(f"k={k}, n={EXPECTED_N[retriever][k]}")
                for record in records:
                    if (
                        record["tau"] != tau
                        or record["retriever"] != retriever
                        or record["k"] != k
                    ):
                        continue
                    print(
                        f"  {record['model']:<20} runs={record['answers_included']} "
                        f"excluded={record['answers_excluded']} "
                        f"P={pct(record['conditional_precision'])} "
                        f"R={pct(record['conditional_recall'])} "
                        f"F1={pct(record['conditional_f1'])} "
                        f"CondAcc={pct(record['conditional_hit_rate'])} "
                        f"E2E={pct(record['e2e_accuracy'])}"
                    )


def latex_pair(records: List[Dict[str, Any]], retriever: str, k: int, model: str, metric: str) -> str:
    values = []
    for tau in (3, 0):
        match = next(
            row
            for row in records
            if row["tau"] == tau
            and row["retriever"] == retriever
            and row["k"] == k
            and row["model"] == model
        )
        values.append(pct(match[metric]))
    return f"{values[0]}/{values[1]}"


def write_latex(path: Path, records: List[Dict[str, Any]]) -> None:
    models = list(MODEL_DIRS)
    display = {"bm25": "BM25", "codebert": "CodeBERT"}
    metric_rows = (
        ("Precision (\\%)", "conditional_precision"),
        ("Recall (\\%)", "conditional_recall"),
        ("Cond. F1 (\\%)", "conditional_f1"),
        ("Cond. Acc. (\\%)", "conditional_hit_rate"),
        ("E2E Acc. (\\%)", "e2e_accuracy"),
    )
    lines = [
        "\\begin{table*}[t]",
        "\\caption{LLM performance with varying context size ($k$) for BM25 and CodeBERT ($\\tau=3$ / $\\tau=0$). Conditional Precision, Recall, F1, and Accuracy are computed on fixed retrieval-successful subsets: BM25 $n=\\{40,66,86,97,113\\}$ and CodeBERT $n=\\{31,53,73,85,93\\}$ for $k=\\{5,10,20,30,35\\}$. Conditional Accuracy retains these fixed denominators for incomplete runs and counts missing instances as failures. Precision, Recall, and F1 use complete runs where available; missing values are supplemented from the legacy table. For $k>5$, E2E Accuracy uses the same successful-localization numerator as Conditional Accuracy and the fixed full-set denominator of 453.}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\multicolumn{2}{c}{\\textbf{LLM}} & " + " & ".join(f"\\textbf{{{m}}}" for m in models) + " \\\\",
        "\\midrule",
    ]
    blocks = [(retriever, k) for retriever in ("bm25", "codebert") for k in KS]
    for block_index, (retriever, k) in enumerate(blocks):
        for row_index, (label, metric) in enumerate(metric_rows):
            first = f"\\multirow{{5}}{{*}}{{\\textbf{{{display[retriever]} Top-{k}}}}}" if row_index == 0 else ""
            cells = [latex_pair(records, retriever, k, model, metric) for model in models]
            lines.append(f"{first} & \\textbf{{{label}}} & " + " & ".join(cells) + " \\\\")
        if block_index != len(blocks) - 1:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:table7}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute fixed-subset conditional metrics for the BM25/CodeBERT table"
    )
    parser.add_argument(
        "--tau",
        type=int,
        nargs="+",
        default=[3, 0],
        choices=(0, 3),
        help="Line windows to calculate; default emits the table pair 3 0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "table7_conditional_metrics.json",
    )
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    retrieval_sources: Dict[str, Any] = {}
    for retriever in ("bm25", "codebert"):
        retrieval_sources[retriever] = {}
        for k in KS:
            success_ids, reports = build_shared_success_set(retriever, k)
            retrieval_sources[retriever][str(k)] = {
                "expected_n": EXPECTED_N[retriever][k],
                "derived_n": len(success_ids),
                "fixids": sorted(success_ids),
                "sources": reports,
            }
            for tau in args.tau:
                for model in MODEL_DIRS:
                    record = evaluate_model(model, retriever, k, success_ids, tau)
                    record["tau"] = tau
                    records.append(record)

    payload = {
        "line_window_tau": args.tau,
        "full_dataset_n": FULL_N,
        "caption_subset_sizes": EXPECTED_N,
        "policy": (
            "Conditional metrics use the exact shared retrieval-success fixID set. "
            "Conditional Precision, Recall, and F1 exclude answers missing a "
            "successful fixID. Conditional Accuracy includes incomplete answers "
            "with the declared subset n as denominator, counting missing instances "
            "as failures. For k>5, E2E Accuracy uses the same numerator and "
            "denominator 453. Top-5 retains the published Table 1 computation."
        ),
        "retrieval_success_sets": retrieval_sources,
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    write_csv(csv_path, records)
    tex_path = args.output.with_suffix(".tex")
    write_latex(tex_path, records)
    print_summary(records)
    print(f"\nWrote {args.output}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
