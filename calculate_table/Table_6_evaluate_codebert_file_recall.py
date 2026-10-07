"""
File-level embedding retrieval metrics (w/ and w/o identifier_split).

Uses caches under:
  data/embedding_cache/                                      -> contents_only
  data/embedding_cache_contents_only_with_identifier_split/  -> identifier_split

Metric style matches eval_bm25_file_recall.py:
  per-instance TP/FP/FN, then mean over instances.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Iterable

from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_KS = (5, 10, 20, 30, 35)
DEFAULT_INPUT = PROJECT_ROOT / "data/github_commit_request_output_with_issue_final_labeled.json"
DEFAULT_TOKEN = os.environ.get("GITHUB_TOKEN")
DEFAULT_ENCODINGS = (
    "contents_only",
    "contents_only_with_identifier_split",
)
DEFAULT_MODEL = "microsoft/codebert-base"


def get_oracle_cs_paths(original_data: dict) -> list[str]:
    """Same oracle .cs paths as inference.py base_commit_files."""
    paths: list[str] = []
    seen = set()
    for file_info in original_data.get("context_info", {}).get("files", []):
        path = normalize_path(file_info.get("path"))
        if not path or not is_cs(path) or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def retrieve_like_inference(
    original_data: dict,
    retrieval_output_dir: Path,
    k: int,
    tokens,
    document_encoding: str,
    model_name: str = DEFAULT_MODEL,
    cache_only: bool = True,
) -> list[str]:
    """Mirror inference embedding branch: encode -> CodeBERT cosine -> top-k."""
    return retrieve_topk_by_embedding(
        original_data=original_data,
        retrieval_output_dir=retrieval_output_dir,
        k=k,
        query=build_problem_statement(original_data),
        instance_id=original_data.get("fixID"),
        tokens=tokens,
        model_name=model_name,
        cache_only=cache_only,
        document_encoding=document_encoding,
        return_ranked_paths=True,
    )


def file_level_tp_fp_fn(
    retrieved: Iterable[str], oracle: Iterable[str]
) -> tuple[int, int, int]:
    oracle_set = set(oracle)
    retrieved_set = set(retrieved)
    tp = len(retrieved_set & oracle_set)
    fp = len(retrieved_set - oracle_set)
    fn = len(oracle_set - retrieved_set)
    return tp, fp, fn


def instance_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    hit = 1.0 if tp > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "hit": hit,
        "f1_score": f1_score,
    }


def mean_or_zero(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def short_name(encoding: str) -> str:
    if encoding == "contents_only":
        return "emb_wo_split"
    if encoding == "contents_only_with_identifier_split":
        return "emb_w_split"
    return encoding


def print_table(summary: dict, ks: list[int], encodings: list[str]) -> None:
    headers = [
        "variant",
        *[f"R(k={k})" for k in ks],
        *[f"Acc(k={k})" for k in ks],
        *[f"P(k={k})" for k in ks],
        *[f"F1(k={k})" for k in ks],
    ]
    rows = []
    for encoding in encodings:
        m = summary["by_encoding"][encoding]["by_k"]
        row = [short_name(encoding)]
        for k in ks:
            row.append(f"{m[str(k)]['recall']:.4f}")
        for k in ks:
            row.append(f"{m[str(k)]['accuracy']:.4f}")
        for k in ks:
            row.append(f"{m[str(k)]['precision']:.4f}")
        for k in ks:
            row.append(f"{m[str(k)]['f1_score']:.4f}")
        rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print("\n" + "=" * 80)
    print("File-level metrics: embedding cosine top-K (w/ vs w/o identifier_split)")
    print("=" * 80)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))
    print("=" * 80)
    print(f"Instances evaluated: {summary['num_instances']}")
    print(f"Skipped (no oracle .cs files): {summary['num_skipped_no_oracle']}")


def eval_encoding(
    instances: list[dict],
    encoding: str,
    ks: list[int],
    retrieval_dir: Path,
    tokens,
    model_name: str,
    cache_only: bool,
) -> tuple[dict, list[dict], int]:
    metric_lists = {
        str(k): {"recall": [], "precision": [], "hit": [], "f1_score": []} for k in ks
    }
    per_instance = []
    num_skipped_no_oracle = 0

    for instance in tqdm(instances, desc=f"embedding+{encoding}", unit="instance"):
        oracle_paths = get_oracle_cs_paths(instance)
        if not oracle_paths:
            num_skipped_no_oracle += 1
            continue

        instance_id = instance.get("fixID")
        record = {
            "fixID": instance_id,
            "oracle_paths": oracle_paths,
            "retrieved_by_k": {},
            "by_k": {},
        }

        try:
            all_retrieved = retrieve_like_inference(
                original_data=instance,
                retrieval_output_dir=retrieval_dir,
                k=max(ks),
                tokens=tokens,
                document_encoding=encoding,
                model_name=model_name,
                cache_only=cache_only,
            )
        except Exception as e:
            logger.error("Retrieval failed for {} ({}): {}", instance_id, encoding, e)
            all_retrieved = []

        for k in ks:
            retrieved = all_retrieved[:k] if all_retrieved else []
            tp, fp, fn = file_level_tp_fp_fn(retrieved, oracle_paths)
            m = instance_metrics(tp, fp, fn)
            for key in ("recall", "precision", "hit", "f1_score"):
                metric_lists[str(k)][key].append(m[key])
            record["retrieved_by_k"][str(k)] = retrieved
            record["by_k"][str(k)] = m

        per_instance.append(record)

    by_k = {
        str(k): {
            "recall": mean_or_zero(metric_lists[str(k)]["recall"]),
            "precision": mean_or_zero(metric_lists[str(k)]["precision"]),
            "accuracy": mean_or_zero(metric_lists[str(k)]["hit"]),
            "f1_score": mean_or_zero(metric_lists[str(k)]["f1_score"]),
        }
        for k in ks
    }
    encoding_summary = {
        "encoding": encoding,
        "model_name": model_name,
        "pipeline": f"{encoding} -> CodeBERT cosine -> top-K",
        "by_k": by_k,
    }
    return encoding_summary, per_instance, num_skipped_no_oracle


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass

    parser = argparse.ArgumentParser(
        description=(
            "File-level recall for embedding retrieval "
            "(contents_only vs contents_only_with_identifier_split)"
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--retrieval-dir",
        default=PROJECT_ROOT / "data",
        help="Parent of embedding_cache / embedding_cache_<encoding>",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=list(DEFAULT_ENCODINGS),
        help="Document encodings to evaluate",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--cache-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require precomputed embedding cache (default: True)",
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "data/embedding_file_recall_comparison.json",
    )
    args = parser.parse_args()

    # Delay project imports so ``--help`` works before optional retrieval and
    # machine-learning dependencies are installed.
    global build_problem_statement, is_cs, normalize_path
    global retrieve_topk_by_embedding
    from inference import build_problem_statement, is_cs, normalize_path
    from utils.embedding_retrieval import retrieve_topk_by_embedding

    ks = sorted(set(args.ks))
    encodings = list(args.encodings)
    input_path = Path(args.input)
    retrieval_dir = Path(args.retrieval_dir)
    output_path = Path(args.output)
    tokens = [args.token] if args.token else None

    with open(input_path, "r", encoding="utf-8") as f:
        instances = [json.loads(line) for line in f if line.strip()]

    if args.start_index:
        instances = instances[args.start_index :]
    if args.limit is not None:
        instances = instances[: args.limit]

    by_encoding = {}
    per_instance_by_encoding = {}
    num_skipped_no_oracle = 0
    num_instances = 0

    for encoding in encodings:
        encoding_summary, per_instance, skipped = eval_encoding(
            instances=instances,
            encoding=encoding,
            ks=ks,
            retrieval_dir=retrieval_dir,
            tokens=tokens,
            model_name=args.model_name,
            cache_only=args.cache_only,
        )
        by_encoding[encoding] = encoding_summary
        per_instance_by_encoding[encoding] = per_instance
        num_skipped_no_oracle = skipped
        num_instances = len(per_instance)

    summary = {
        "num_instances": num_instances,
        "num_skipped_no_oracle": num_skipped_no_oracle,
        "ks": ks,
        "retrieval": "embedding retrieve_topk (w/ and w/o identifier_split)",
        "metric_style": "per-instance TP/(TP+FN) like evaluation.py, then mean",
        "by_encoding": by_encoding,
    }

    payload = {
        "summary": summary,
        "per_instance_by_encoding": per_instance_by_encoding,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print_table(summary, ks, encodings)
    print("\n---- Mean over instances ----")
    for encoding in encodings:
        print(f"\n[{short_name(encoding)}] {encoding}")
        for k in ks:
            m = summary["by_encoding"][encoding]["by_k"][str(k)]
            print(
                f"  k={k}: R={m['recall']:.4f} Acc={m['accuracy']:.4f} "
                f"P={m['precision']:.4f} F1={m['f1_score']:.4f}"
            )
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
