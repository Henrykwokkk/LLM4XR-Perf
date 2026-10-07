"""Generate pooled per-category metrics for Tables 3, 4, and 5.

Tables 3 and 4 use the fixed BM25 and CodeBERT Top-5 retrieval-success
subsets. Table 5 uses all 453 Oracle instances. Precision, recall, and F1 are
calculated from TP/FP/FN counts pooled across answer streams. Category E2E
accuracy uses the full category population from the 453-instance benchmark,
so retrieval failures remain failures in its denominator.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from Table_1_evaluation_three_answers import finalize_counts, score_one_instance
from Table_7_recompute_conditional_metrics import (
    FULL_N,
    MODEL_DIRS,
    answer_streams,
    build_shared_success_set,
    load_jsonl,
    result_files,
)

ROOT = Path(__file__).resolve().parent.parent
TAUS = (3, 0)
ORACLE_PATTERN = re.compile(r"_oracle_10(?:_n\d+)?\.json$")


def oracle_files(directory: Path) -> List[Path]:
    return sorted(
        path
        for path in directory.glob("*.json")
        if ORACLE_PATTERN.search(path.name)
        and not path.name.endswith(".progress.json")
    )


def benchmark_categories() -> Tuple[Dict[str, str], Dict[str, int]]:
    """Return canonical fixID labels and full 453-instance category sizes."""
    candidates = [
        path
        for dirs in MODEL_DIRS.values()
        for path in oracle_files(dirs["bm25"])
    ]
    for path in candidates:
        rows, _ = load_jsonl(path)
        if len(rows) != FULL_N:
            continue
        labels = {
            fixid: str(row.get("label_tag") or "unknown_label")
            for fixid, row in rows.items()
        }
        counts: Dict[str, int] = defaultdict(int)
        for label in labels.values():
            counts[label] += 1
        return labels, dict(counts)
    raise RuntimeError("No complete 453-instance Oracle file found")


def setting_files(model: str, retriever: str) -> List[Path]:
    directory = MODEL_DIRS[model]["bm25" if retriever == "oracle" else retriever]
    if retriever == "oracle":
        return oracle_files(directory)
    return result_files(retriever, 5, directory)


def collect_streams(
    model: str, retriever: str
) -> Iterable[Tuple[str, Path, Dict[str, Dict[str, Any]]]]:
    for path in setting_files(model, retriever):
        rows, _ = load_jsonl(path)
        yield from answer_streams(path, rows)


def pooled_category_metrics(
    streams: List[Tuple[str, Path, Dict[str, Dict[str, Any]]]],
    allowed_ids: Set[str],
    labels: Dict[str, str],
    full_category_sizes: Dict[str, int],
    tau: int,
) -> Dict[str, Dict[str, Any]]:
    """Pool integer counts, retaining fixed conditional and E2E populations."""
    pooled: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0, "hit_num": 0}
    )
    # Keep zero-success categories in the output (for example BM25 HM and
    # CodeBERT ORU), because the manuscript reports explicit all-zero rows.
    categories = sorted(full_category_sizes)

    for inference_key, _, rows in streams:
        for fixid in allowed_ids:
            # A missing row is an E2E and conditional-accuracy miss. It adds no
            # line predictions or ground-truth counts to pooled P/R/F1.
            row = rows.get(fixid)
            if row is None:
                continue
            tp, fp, fn, hit = score_one_instance(
                row, row.get(inference_key), tau
            )
            counts = pooled[labels[fixid]]
            counts["TP"] += tp
            counts["FP"] += fp
            counts["FN"] += fn
            counts["hit_num"] += int(hit)

    answers = len(streams)
    results: Dict[str, Dict[str, Any]] = {}
    for category in categories:
        counts = pooled[category]
        conditional_n = sum(1 for fixid in allowed_ids if labels[fixid] == category)
        line_metrics = finalize_counts(
            counts["TP"], counts["FP"], counts["FN"],
            counts["hit_num"], conditional_n * answers,
        )
        e2e_denominator = full_category_sizes[category] * answers
        e2e_accuracy = (
            counts["hit_num"] / e2e_denominator if e2e_denominator else 0.0
        )
        results[category] = {
            **line_metrics,
            "conditional_n": conditional_n,
            "full_category_n": full_category_sizes[category],
            "e2e_accuracy": e2e_accuracy,
            "answers_pooled": answers,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pooled per-category metrics for Tables 3-5"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation_results_by_tag.csv",
    )
    args = parser.parse_args()

    labels, full_category_sizes = benchmark_categories()
    if len(labels) != FULL_N:
        raise RuntimeError(f"Expected {FULL_N} benchmark labels, found {len(labels)}")

    success_sets = {
        "bm25": build_shared_success_set("bm25", 5)[0],
        "codebert": build_shared_success_set("codebert", 5)[0],
        "oracle": set(labels),
    }
    records: List[Dict[str, Any]] = []
    for model in MODEL_DIRS:
        for retriever in ("bm25", "codebert", "oracle"):
            streams = list(collect_streams(model, retriever))
            if not streams:
                raise RuntimeError(f"No answer streams for {model} {retriever}")
            for tau in TAUS:
                metrics = pooled_category_metrics(
                    streams,
                    success_sets[retriever],
                    labels,
                    full_category_sizes,
                    tau,
                )
                for category, values in metrics.items():
                    records.append(
                        {
                            "Model": model,
                            "Retriever": retriever,
                            "tau": tau,
                            "Tag": category,
                            "subset_n": values["conditional_n"],
                            "full_category_n": values["full_category_n"],
                            "answers_pooled": values["answers_pooled"],
                            "TP": values["TP"],
                            "FP": values["FP"],
                            "FN": values["FN"],
                            "Precision": values["precision"],
                            "Recall": values["recall"],
                            "F1": values["f1_score"],
                            "E2E_Accuracy": values["e2e_accuracy"],
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
