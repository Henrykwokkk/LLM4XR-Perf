"""Recompute table abstention with shared denoms and ground_truth_results GT."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

INFERENCE_KEYS = (
    "inference_results",
    "inference_results_req_1",
    "inference_results_req_2",
    "inference_results_req_3",
)

FULL_N = 453
EXPECTED_SUBSET_N = {"BM25 Top-5": 40, "CodeBERT Top-5": 31, "Oracle": 453}

BM25_MODELS = {
    "gemini3-pro": Path("data/gemini3-pro"),
    "qwen3": Path("data/qwen3"),
    "deepseek-v3.2": Path("data/deepseek-v3.2"),
    "claude-sonnet-4-5": Path("data/claude-sonnet-4-5"),
    "gpt-4o": Path("data/gpt-4o"),
    "gpt-5.2": Path("data/gpt-5.2"),
}

EMB_MODELS = {
    "gemini3-pro": Path("data/embedding/gemini-3-pro"),
    "qwen3": Path("data/embedding/qwen3"),
    "deepseek-v3.2": Path("data/embedding/deepseek-3-2"),
    "claude-sonnet-4-5": Path("data/embedding/claude-sonnet-4-5"),
    "gpt-4o": Path("data/embedding/gpt-4o"),
    "gpt-5.2": Path("data/embedding/gpt-5.2"),
}


def normalize_path(p: Optional[str]) -> Optional[str]:
    if not p or p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def gt_paths(instance: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for loc in instance.get("ground_truth_results") or []:
        path = normalize_path(loc.get("path") if isinstance(loc, dict) else loc)
        if path:
            out.add(path)
    return out


def retrieved_paths(instance: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for raw in instance.get("context_files_paths") or []:
        path = normalize_path(raw)
        if path:
            out.add(path)
    return out


def is_hit(instance: Dict[str, Any]) -> bool:
    gt = gt_paths(instance)
    if not gt:
        return False
    return bool(retrieved_paths(instance) & gt)


def is_empty(results: Any) -> bool:
    return results is None or (isinstance(results, list) and len(results) == 0)


def load_instances(path: Path) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            fid = obj.get("fixID")
            if fid is None:
                continue
            by_id[str(fid)] = obj
    if bad:
        print(f"  [warn] {path.name}: skipped {bad} bad JSON line(s)")
    return by_id


def detect_keys(sample: Dict[str, Any]) -> List[str]:
    found = [k for k in INFERENCE_KEYS if k in sample]
    return found if found else ["inference_results"]


def is_n_suffix(name: str) -> bool:
    return bool(re.search(r"_n\d+\.json$", name))


def glob_configs(model_dir: Path, kind: str, k: Optional[int]) -> List[Path]:
    if not model_dir.is_dir():
        return []
    out = []
    for p in sorted(model_dir.glob("*.json")):
        if p.name.endswith(".progress.json"):
            continue
        name = p.name
        if kind == "bm25" and k is not None:
            if f"_bm25_{k}." in name or name.endswith(f"_bm25_{k}.json") or f"_bm25_{k}_n" in name:
                out.append(p)
        elif kind == "embedding" and k is not None:
            if f"_embedding_{k}." in name or f"_embedding_{k}_n" in name:
                out.append(p)
        elif kind == "oracle":
            if "_oracle_" in name:
                out.append(p)
    return out


def collect_answers(files: List[Path]) -> List[Tuple[Path, str, Dict[str, Dict[str, Any]]]]:
    answers = []
    for path in files:
        by_id = load_instances(path)
        if not by_id:
            continue
        sample = next(iter(by_id.values()))
        for key in detect_keys(sample):
            answers.append((path, key, by_id))
    return answers


def shared_success_from_complete(files: List[Path], *, oracle_all: bool) -> Tuple[Set[str], List[str]]:
    reports = []
    sets: List[Set[str]] = []
    for path in files:
        by_id = load_instances(path)
        n = len(by_id)
        if n != FULL_N:
            reports.append(f"{path.parent.name}/{path.name}: n={n} SKIP (not 453)")
            continue
        if oracle_all:
            s = set(by_id.keys())
        else:
            s = {fid for fid, inst in by_id.items() if is_hit(inst)}
        sets.append(s)
        reports.append(f"{path.parent.name}/{path.name}: n={n} hits={len(s)}")
    if not sets:
        return set(), reports
    # Retrieval is deterministic; disagreement invalidates the fixed subset.
    first = sets[0]
    if all(s == first for s in sets):
        return first, reports
    raise RuntimeError(
        "Retrieval-success sets disagree: "
        f"sizes={[len(s) for s in sets]}"
    )


def rate_for_model(
    answers: List[Tuple[Path, str, Dict[str, Dict[str, Any]]]],
    success: Set[str],
) -> Dict[str, Any]:
    denom = len(success)
    per = []
    for path, key, by_id in answers:
        covered = success & set(by_id.keys())
        empty = sum(1 for fid in covered if is_empty(by_id[fid].get(key)))
        rate = empty / denom if denom else 0.0
        per.append(
            {
                "file": path.name,
                "key": key,
                "n": len(by_id),
                "covered": len(covered),
                "empty": empty,
                "rate": rate,
            }
        )
    rates = [p["rate"] for p in per]
    empties = [p["empty"] for p in per]
    return {
        "answers": len(per),
        "per": per,
        "empty_mean": statistics.mean(empties) if empties else 0.0,
        "rate_mean": statistics.mean(rates) if rates else 0.0,
        "rate_std": statistics.pstdev(rates) if len(rates) > 1 else 0.0,
        "rate_pct": 100.0 * (statistics.mean(rates) if rates else 0.0),
        "n_values": sorted({p["n"] for p in per}),
        "covered_values": sorted({p["covered"] for p in per}),
    }


def main() -> None:
    rows_spec = [
        ("BM25 Top-5", "bm25", 5, False, BM25_MODELS),
        ("CodeBERT Top-5", "embedding", 5, False, EMB_MODELS),
        ("Oracle", "oracle", None, True, BM25_MODELS),
    ]

    table: Dict[str, Dict[str, float]] = {}
    payload: Dict[str, Any] = {}

    for label, kind, k, oracle_all, model_dirs in rows_spec:
        print("\n" + "=" * 80)
        print(label)
        all_files = []
        for d in model_dirs.values():
            all_files.extend(glob_configs(d, kind, k))
        # Hit set from complete canonical files only (exclude _n* retrieval variants).
        # Embedding _n3 files still share the same retrieved context, so keep them.
        if kind == "embedding":
            hit_files = all_files
        else:
            hit_files = [p for p in all_files if not is_n_suffix(p.name)]
        success, reports = shared_success_from_complete(hit_files, oracle_all=oracle_all)
        expected_n = EXPECTED_SUBSET_N[label]
        if len(success) != expected_n:
            raise RuntimeError(
                f"{label} retrieval subset has n={len(success)}; expected {expected_n}"
            )
        print(" complete-file hit reports:")
        for r in reports:
            print("  ", r)
        print(f" SHARED DENOM n={len(success)}")
        payload[label] = {"shared_n": len(success), "reports": reports, "models": {}}
        table[label] = {}
        for mname, d in model_dirs.items():
            files = glob_configs(d, kind, k)
            answers = collect_answers(files)
            stats = rate_for_model(answers, success)
            full = [p for p in stats["per"] if p["covered"] == len(success)]
            if full:
                empty_full = [p["empty"] for p in full]
                rates_full = [p["rate"] for p in full]
                pooled_empty = sum(empty_full)
                pooled_denominator = len(success) * len(full)
                stats["full_answers"] = len(full)
                stats["pooled_empty"] = pooled_empty
                stats["pooled_denominator"] = pooled_denominator
                stats["full_rate_mean"] = (
                    pooled_empty / pooled_denominator if pooled_denominator else 0.0
                )
                stats["full_rate_std"] = statistics.pstdev(rates_full) if len(full) > 1 else 0.0
                stats["full_empty_mean"] = statistics.mean(empty_full)
                stats["full_rate_pct"] = 100.0 * stats["full_rate_mean"]
            else:
                raise RuntimeError(
                    f"No complete answer stream for {label} / {mname}"
                )
            table[label][mname] = stats["full_rate_pct"]
            payload[label]["models"][mname] = stats
            print(
                f"  {mname:20s} all={stats['answers']} full={stats['full_answers']} "
                f"n={stats['n_values']} covered={stats['covered_values']} "
                f"empty_all={stats['empty_mean']:.2f} rate_all={stats['rate_pct']:.2f}% | "
                f"empty_full={stats['full_empty_mean']:.2f} rate_full={stats['full_rate_pct']:.2f}%"
            )
            for p in stats["per"]:
                print(
                    f"      {p['file'][-40:]:40s} {p['key']:24s} "
                    f"n={p['n']:3d} cov={p['covered']:3d} empty={p['empty']:3d} "
                    f"rate={100*p['rate']:.2f}%"
                )

    order = [
        "gemini3-pro",
        "qwen3",
        "deepseek-v3.2",
        "claude-sonnet-4-5",
        "gpt-4o",
        "gpt-5.2",
    ]
    print("\n" + "=" * 80)
    print("TABLE percents (mean over runs, shared denom)")
    print(f"{'setting':<22}" + "".join(f"{m[:12]:>14}" for m in order))
    for label, _, _, _, _ in rows_spec:
        vals = "".join(f"{table[label].get(m, float('nan')):14.2f}" for m in order)
        print(f"{label:<22}" + vals)

    out = Path("data/abstention_rate_table_shared_gtr.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
