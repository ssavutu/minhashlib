import argparse
import bz2
import csv
import gzip
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from datasketch import MinHash

try:
    from sklearn.datasets import fetch_20newsgroups
except ImportError:  # pragma: no cover
    fetch_20newsgroups = None

sys.path.append(str(Path(__file__).resolve().parents[1]))
from minhashlib import DiffChecker


@dataclass
class PairEvalConfig:
    random_pairs: int = 3000
    threshold: float = 0.5
    topk: int = 10
    num_queries: int = 200


@dataclass
class RunConfig:
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    k: int
    num_perm: int
    wiki_dump_path: Path | None
    local_docs: Path | None
    max_docs: int
    min_chars: int
    pair_cfg: PairEvalConfig
    out_dir: Path
    p_values: tuple[int, ...]
    scaling_num_docs: tuple[int, ...]
    scaling_num_perm: tuple[int, ...]
    scaling_doc_length: tuple[int, ...]
    include_scaling: bool


def normalize_text(doc: str) -> str:
    return " ".join(doc.split())


def shingle_strings(doc: str, k: int) -> set[str]:
    if len(doc) < k:
        return set()
    return {doc[i : i + k] for i in range(len(doc) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def make_random_docs(num_docs: int, doc_length: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz     "
    return [
        "".join(rng.choice(alphabet) for _ in range(doc_length))
        for _ in range(num_docs)
    ]


def make_near_duplicates(
    docs: list[str], seed: int, pairs: int, edit_rates: tuple[float, ...] = (0.02, 0.05, 0.10)
) -> tuple[list[str], list[tuple[int, int]]]:
    rng = random.Random(seed + 99)
    out_docs = list(docs)
    out_pairs: list[tuple[int, int]] = []
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    for i in range(pairs):
        base_idx = rng.randrange(len(docs))
        rate = edit_rates[i % len(edit_rates)]
        chars = list(docs[base_idx])
        num_edits = max(1, int(len(chars) * rate))
        for _ in range(num_edits):
            op = rng.choice(("insert", "delete", "replace", "swap"))
            if op == "insert":
                pos = rng.randrange(len(chars) + 1)
                chars.insert(pos, rng.choice(alphabet))
            elif op == "delete" and chars:
                chars.pop(rng.randrange(len(chars)))
            elif op == "replace" and chars:
                chars[rng.randrange(len(chars))] = rng.choice(alphabet)
            elif op == "swap" and len(chars) >= 2:
                pos = rng.randrange(len(chars) - 1)
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        out_docs.append("".join(chars))
        out_pairs.append((base_idx, len(out_docs) - 1))
    return out_docs, out_pairs


def load_20newsgroups(max_docs: int, min_chars: int, k: int) -> list[str]:
    if fetch_20newsgroups is None:
        raise RuntimeError("scikit-learn is not installed; cannot load 20 Newsgroups")
    data_home = Path(__file__).resolve().parents[1] / ".cache" / "scikit_learn_data"
    data_home.mkdir(parents=True, exist_ok=True)
    ds = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        data_home=str(data_home),
    )
    docs = [normalize_text(x) for x in ds.data]
    docs = [d for d in docs if len(d) >= max(min_chars, k)]
    return docs[:max_docs]


def load_ag_news(max_docs: int, min_chars: int, k: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets package is not installed; cannot load AG News") from exc
    ds = load_dataset("ag_news", split="train")
    docs = [normalize_text(x["text"]) for x in ds]
    docs = [d for d in docs if len(d) >= max(min_chars, k)]
    return docs[:max_docs]


def open_dump(path: Path):
    if path.suffixes and path.suffixes[-1] == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    if path.suffixes and path.suffixes[-1] == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def normalize_wiki_text(text: str) -> str:
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"==+[^=]+==+", " ", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_wikipedia_texts(path: Path):
    with open_dump(path) as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if elem.tag.endswith("text") and elem.text:
                yield elem.text
            elem.clear()


def load_wikipedia(max_docs: int, min_chars: int, k: int, dump_path: Path) -> list[str]:
    if not dump_path.exists():
        raise RuntimeError(f"Wikipedia dump path not found: {dump_path}")
    docs: list[str] = []
    for text in iter_wikipedia_texts(dump_path):
        cleaned = normalize_wiki_text(text)
        if len(cleaned) >= max(min_chars, k):
            docs.append(cleaned)
        if len(docs) >= max_docs:
            break
    return docs


def load_local_docs(path: Path, max_docs: int, min_chars: int, k: int) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"Local docs path not found: {path}")
    docs: list[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = normalize_text(str(item.get("text", "")))
                if len(text) >= max(min_chars, k):
                    docs.append(text)
                if len(docs) >= max_docs:
                    break
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = normalize_text(line.strip())
                if len(text) >= max(min_chars, k):
                    docs.append(text)
                if len(docs) >= max_docs:
                    break
    return docs


def build_datasketch_signatures(docs: list[str], shingle_sets: list[set[str]], num_perm: int, seed: int):
    signatures = []
    start = time.perf_counter()
    for shingles in shingle_sets:
        m = MinHash(num_perm=num_perm, seed=seed)
        for sh in shingles:
            m.update(sh.encode("utf-8"))
        signatures.append(m)
    build_s = time.perf_counter() - start
    return signatures, build_s


def build_diffchecker_signatures(docs: list[str], k: int, num_perm: int, seed: int, p: int):
    checker = DiffChecker(p=p, k=k, num_perm=num_perm, seed=seed)
    signatures = []
    start = time.perf_counter()
    for doc in docs:
        signatures.append(checker.generateSignature(doc))
    build_s = time.perf_counter() - start
    return checker, signatures, build_s


def sample_random_pairs(n_docs: int, n_pairs: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(n_docs), rng.randrange(n_docs)) for _ in range(n_pairs)]


def eval_pair_scores_diffchecker(checker: DiffChecker, signatures, pairs: list[tuple[int, int]]):
    start = time.perf_counter()
    scores = [checker.checkJaccardSignatureSimilarity(signatures[i], signatures[j]) for i, j in pairs]
    return scores, time.perf_counter() - start


def eval_pair_scores_datasketch(signatures, pairs: list[tuple[int, int]]):
    start = time.perf_counter()
    scores = [signatures[i].jaccard(signatures[j]) for i, j in pairs]
    return scores, time.perf_counter() - start


def compute_mae(pred: list[float], exact: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(pred, exact)) / len(pred)


def precision_recall_f1_at_threshold(pred: list[float], exact: list[float], threshold: float):
    tp = fp = fn = 0
    for p, e in zip(pred, exact):
        pred_pos = p >= threshold
        true_pos = e >= threshold
        if pred_pos and true_pos:
            tp += 1
        elif pred_pos and not true_pos:
            fp += 1
        elif (not pred_pos) and true_pos:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def topk_metrics(
    exact_scores_by_query: list[list[float]],
    approx_scores_by_query: list[list[float]],
    topk: int,
) -> tuple[float, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    for exact_scores, approx_scores in zip(exact_scores_by_query, approx_scores_by_query):
        exact_rank = sorted(range(len(exact_scores)), key=lambda i: exact_scores[i], reverse=True)
        approx_rank = sorted(range(len(approx_scores)), key=lambda i: approx_scores[i], reverse=True)
        exact_top = set(exact_rank[:topk])
        approx_top = set(approx_rank[:topk])
        hit = len(exact_top & approx_top)
        precisions.append(hit / topk)
        recalls.append(hit / topk)
    return statistics.fmean(precisions), statistics.fmean(recalls)


def retrieval_eval(
    docs: list[str],
    shingle_sets: list[set[str]],
    method: str,
    method_obj: Any,
    signatures: list[Any],
    topk: int,
    num_queries: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = random.Random(seed + 13)
    query_indices = [rng.randrange(len(docs)) for _ in range(min(num_queries, len(docs)))]
    exact_scores_by_query: list[list[float]] = []
    approx_scores_by_query: list[list[float]] = []
    start = time.perf_counter()
    for q in query_indices:
        exact_scores: list[float] = []
        approx_scores: list[float] = []
        for j in range(len(docs)):
            if q == j:
                continue
            exact_scores.append(jaccard(shingle_sets[q], shingle_sets[j]))
            if method == "diffchecker":
                approx_scores.append(method_obj.checkJaccardSignatureSimilarity(signatures[q], signatures[j]))
            else:
                approx_scores.append(signatures[q].jaccard(signatures[j]))
        exact_scores_by_query.append(exact_scores)
        approx_scores_by_query.append(approx_scores)
    elapsed = time.perf_counter() - start
    p_at_k, r_at_k = topk_metrics(exact_scores_by_query, approx_scores_by_query, topk=topk)
    return p_at_k, r_at_k, elapsed


def summarize_numeric(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        stdev = 0.0
    else:
        stdev = statistics.stdev(values)
    ci95 = 1.96 * stdev / math.sqrt(len(values)) if values else 0.0
    return {"mean": mean, "stdev": stdev, "ci95": ci95}


def get_peak_rss_mb() -> float:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("darwin"):
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return 0.0


def run_single_case(
    dataset_name: str,
    scenario_name: str,
    docs: list[str],
    cfg: RunConfig,
    seed: int,
    p: int,
) -> list[dict[str, Any]]:
    shingle_sets = [shingle_strings(d, cfg.k) for d in docs]
    pairs = sample_random_pairs(len(docs), cfg.pair_cfg.random_pairs, seed + 7)
    exact_pair = [jaccard(shingle_sets[i], shingle_sets[j]) for i, j in pairs]

    tracemalloc.start()
    dc_checker, dc_sigs, dc_build_s = build_diffchecker_signatures(
        docs=docs, k=cfg.k, num_perm=cfg.num_perm, seed=seed, p=p
    )
    dc_current_mem, dc_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    ds_sigs, ds_build_s = build_datasketch_signatures(
        docs=docs, shingle_sets=shingle_sets, num_perm=cfg.num_perm, seed=seed
    )
    ds_current_mem, ds_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    dc_pair_scores, dc_pair_eval_s = eval_pair_scores_diffchecker(dc_checker, dc_sigs, pairs)
    ds_pair_scores, ds_pair_eval_s = eval_pair_scores_datasketch(ds_sigs, pairs)

    dc_mae = compute_mae(dc_pair_scores, exact_pair)
    ds_mae = compute_mae(ds_pair_scores, exact_pair)

    dc_p, dc_r, dc_f1 = precision_recall_f1_at_threshold(
        dc_pair_scores, exact_pair, cfg.pair_cfg.threshold
    )
    ds_p, ds_r, ds_f1 = precision_recall_f1_at_threshold(
        ds_pair_scores, exact_pair, cfg.pair_cfg.threshold
    )

    dc_p_at_k, dc_r_at_k, dc_retrieval_s = retrieval_eval(
        docs=docs,
        shingle_sets=shingle_sets,
        method="diffchecker",
        method_obj=dc_checker,
        signatures=dc_sigs,
        topk=cfg.pair_cfg.topk,
        num_queries=cfg.pair_cfg.num_queries,
        seed=seed,
    )
    ds_p_at_k, ds_r_at_k, ds_retrieval_s = retrieval_eval(
        docs=docs,
        shingle_sets=shingle_sets,
        method="datasketch",
        method_obj=None,
        signatures=ds_sigs,
        topk=cfg.pair_cfg.topk,
        num_queries=cfg.pair_cfg.num_queries,
        seed=seed,
    )

    rss_mb = get_peak_rss_mb()
    docs_per_second_dc = len(docs) / dc_build_s if dc_build_s else 0.0
    docs_per_second_ds = len(docs) / ds_build_s if ds_build_s else 0.0

    per_sig_dc = dc_peak_mem / max(len(docs), 1)
    per_sig_ds = ds_peak_mem / max(len(docs), 1)

    common = {
        "dataset": dataset_name,
        "scenario": scenario_name,
        "seed": seed,
        "num_docs": len(docs),
        "k": cfg.k,
        "num_perm": cfg.num_perm,
        "pairs": cfg.pair_cfg.random_pairs,
        "threshold": cfg.pair_cfg.threshold,
        "topk": cfg.pair_cfg.topk,
        "num_queries": cfg.pair_cfg.num_queries,
        "p_value": p,
        "peak_rss_mb": rss_mb,
    }

    dc_row = {
        **common,
        "system": "diffchecker",
        "build_seconds": dc_build_s,
        "pair_eval_seconds": dc_pair_eval_s,
        "retrieval_seconds": dc_retrieval_s,
        "docs_per_second": docs_per_second_dc,
        "mae": dc_mae,
        "precision_threshold": dc_p,
        "recall_threshold": dc_r,
        "f1_threshold": dc_f1,
        "precision_at_k": dc_p_at_k,
        "recall_at_k": dc_r_at_k,
        "peak_alloc_bytes": dc_peak_mem,
        "bytes_per_signature": per_sig_dc,
        "tracemalloc_current_bytes": dc_current_mem,
    }
    ds_row = {
        **common,
        "system": "datasketch",
        "build_seconds": ds_build_s,
        "pair_eval_seconds": ds_pair_eval_s,
        "retrieval_seconds": ds_retrieval_s,
        "docs_per_second": docs_per_second_ds,
        "mae": ds_mae,
        "precision_threshold": ds_p,
        "recall_threshold": ds_r,
        "f1_threshold": ds_f1,
        "precision_at_k": ds_p_at_k,
        "recall_at_k": ds_r_at_k,
        "peak_alloc_bytes": ds_peak_mem,
        "bytes_per_signature": per_sig_ds,
        "tracemalloc_current_bytes": ds_current_mem,
    }
    return [dc_row, ds_row]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["dataset"], r["scenario"], r["system"], r["p_value"])
        groups.setdefault(key, []).append(r)

    metrics = [
        "build_seconds",
        "pair_eval_seconds",
        "retrieval_seconds",
        "docs_per_second",
        "mae",
        "precision_threshold",
        "recall_threshold",
        "f1_threshold",
        "precision_at_k",
        "recall_at_k",
        "peak_alloc_bytes",
        "bytes_per_signature",
    ]

    out: list[dict[str, Any]] = []
    for (dataset, scenario, system, p_value), rs in groups.items():
        row: dict[str, Any] = {
            "dataset": dataset,
            "scenario": scenario,
            "system": system,
            "p_value": p_value,
            "runs": len(rs),
            "num_docs": rs[0]["num_docs"],
            "num_perm": rs[0]["num_perm"],
            "k": rs[0]["k"],
            "pairs": rs[0]["pairs"],
            "topk": rs[0]["topk"],
            "num_queries": rs[0]["num_queries"],
            "threshold": rs[0]["threshold"],
        }
        for m in metrics:
            vals = [float(r[m]) for r in rs]
            stats = summarize_numeric(vals)
            row[f"{m}_mean"] = stats["mean"]
            row[f"{m}_stdev"] = stats["stdev"]
            row[f"{m}_ci95"] = stats["ci95"]
        out.append(row)
    return out


def save_rows_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(name: str, cfg: RunConfig, seed: int) -> list[str]:
    if name == "synthetic":
        return make_random_docs(cfg.max_docs, doc_length=800, seed=seed)
    if name == "20newsgroups":
        return load_20newsgroups(cfg.max_docs, cfg.min_chars, cfg.k)
    if name == "ag_news":
        return load_ag_news(cfg.max_docs, cfg.min_chars, cfg.k)
    if name == "wikipedia":
        if cfg.wiki_dump_path is None:
            raise RuntimeError("wikipedia selected but --wiki-dump-path not provided")
        return load_wikipedia(cfg.max_docs, cfg.min_chars, cfg.k, cfg.wiki_dump_path)
    if name == "local":
        if cfg.local_docs is None:
            raise RuntimeError("local selected but --local-docs not provided")
        return load_local_docs(cfg.local_docs, cfg.max_docs, cfg.min_chars, cfg.k)
    raise RuntimeError(f"Unknown dataset: {name}")


def run_scaling(cfg: RunConfig, out_rows: list[dict[str, Any]]):
    base_seed = cfg.seeds[0]
    for n_docs in cfg.scaling_num_docs:
        docs = make_random_docs(n_docs, doc_length=800, seed=base_seed)
        scfg = RunConfig(
            datasets=cfg.datasets,
            seeds=(base_seed,),
            k=cfg.k,
            num_perm=cfg.num_perm,
            wiki_dump_path=cfg.wiki_dump_path,
            local_docs=cfg.local_docs,
            max_docs=n_docs,
            min_chars=cfg.min_chars,
            pair_cfg=cfg.pair_cfg,
            out_dir=cfg.out_dir,
            p_values=cfg.p_values,
            scaling_num_docs=cfg.scaling_num_docs,
            scaling_num_perm=cfg.scaling_num_perm,
            scaling_doc_length=cfg.scaling_doc_length,
            include_scaling=cfg.include_scaling,
        )
        rows = run_single_case(
            dataset_name="scaling_synthetic",
            scenario_name=f"vary_num_docs_{n_docs}",
            docs=docs,
            cfg=scfg,
            seed=base_seed,
            p=cfg.p_values[0],
        )
        out_rows.extend(rows)

    for num_perm in cfg.scaling_num_perm:
        docs = make_random_docs(cfg.max_docs, doc_length=800, seed=base_seed)
        scfg = RunConfig(
            datasets=cfg.datasets,
            seeds=(base_seed,),
            k=cfg.k,
            num_perm=num_perm,
            wiki_dump_path=cfg.wiki_dump_path,
            local_docs=cfg.local_docs,
            max_docs=cfg.max_docs,
            min_chars=cfg.min_chars,
            pair_cfg=cfg.pair_cfg,
            out_dir=cfg.out_dir,
            p_values=cfg.p_values,
            scaling_num_docs=cfg.scaling_num_docs,
            scaling_num_perm=cfg.scaling_num_perm,
            scaling_doc_length=cfg.scaling_doc_length,
            include_scaling=cfg.include_scaling,
        )
        rows = run_single_case(
            dataset_name="scaling_synthetic",
            scenario_name=f"vary_num_perm_{num_perm}",
            docs=docs,
            cfg=scfg,
            seed=base_seed,
            p=cfg.p_values[0],
        )
        out_rows.extend(rows)

    for doc_len in cfg.scaling_doc_length:
        docs = make_random_docs(cfg.max_docs, doc_length=doc_len, seed=base_seed)
        scfg = RunConfig(
            datasets=cfg.datasets,
            seeds=(base_seed,),
            k=cfg.k,
            num_perm=cfg.num_perm,
            wiki_dump_path=cfg.wiki_dump_path,
            local_docs=cfg.local_docs,
            max_docs=cfg.max_docs,
            min_chars=cfg.min_chars,
            pair_cfg=cfg.pair_cfg,
            out_dir=cfg.out_dir,
            p_values=cfg.p_values,
            scaling_num_docs=cfg.scaling_num_docs,
            scaling_num_perm=cfg.scaling_num_perm,
            scaling_doc_length=cfg.scaling_doc_length,
            include_scaling=cfg.include_scaling,
        )
        rows = run_single_case(
            dataset_name="scaling_synthetic",
            scenario_name=f"vary_doc_length_{doc_len}",
            docs=docs,
            cfg=scfg,
            seed=base_seed,
            p=cfg.p_values[0],
        )
        out_rows.extend(rows)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark suite for DiffChecker vs datasketch."
    )
    parser.add_argument(
        "--datasets",
        default="synthetic,20newsgroups,wikipedia",
        help="Comma-separated: synthetic,20newsgroups,ag_news,wikipedia,local",
    )
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated integer seeds")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--p-values", default="2147483647,3037000493")
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--random-pairs", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--wiki-dump-path", type=Path, default=None)
    parser.add_argument("--local-docs", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_outputs"))
    parser.add_argument(
        "--include-scaling", action="store_true", help="Run synthetic scaling sweeps"
    )
    parser.add_argument("--scaling-num-docs", default="500,1000,2000")
    parser.add_argument("--scaling-num-perm", default="64,128,256")
    parser.add_argument("--scaling-doc-length", default="400,800,1600")

    args = parser.parse_args()
    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    p_values = tuple(int(x.strip()) for x in args.p_values.split(",") if x.strip())
    scaling_num_docs = tuple(int(x.strip()) for x in args.scaling_num_docs.split(",") if x.strip())
    scaling_num_perm = tuple(int(x.strip()) for x in args.scaling_num_perm.split(",") if x.strip())
    scaling_doc_length = tuple(
        int(x.strip()) for x in args.scaling_doc_length.split(",") if x.strip()
    )
    pair_cfg = PairEvalConfig(
        random_pairs=args.random_pairs,
        threshold=args.threshold,
        topk=args.topk,
        num_queries=args.num_queries,
    )
    return RunConfig(
        datasets=datasets,
        seeds=seeds,
        k=args.k,
        num_perm=args.num_perm,
        wiki_dump_path=args.wiki_dump_path,
        local_docs=args.local_docs,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        pair_cfg=pair_cfg,
        out_dir=args.out_dir,
        p_values=p_values,
        scaling_num_docs=scaling_num_docs,
        scaling_num_perm=scaling_num_perm,
        scaling_doc_length=scaling_doc_length,
        include_scaling=args.include_scaling,
    )


def main():
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "config": {
            "datasets": cfg.datasets,
            "seeds": cfg.seeds,
            "k": cfg.k,
            "num_perm": cfg.num_perm,
            "p_values": cfg.p_values,
            "max_docs": cfg.max_docs,
            "min_chars": cfg.min_chars,
            "pairs": cfg.pair_cfg.random_pairs,
            "threshold": cfg.pair_cfg.threshold,
            "topk": cfg.pair_cfg.topk,
            "num_queries": cfg.pair_cfg.num_queries,
            "wiki_dump_path": str(cfg.wiki_dump_path) if cfg.wiki_dump_path else None,
            "local_docs": str(cfg.local_docs) if cfg.local_docs else None,
        },
    }

    raw_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for dataset_name in cfg.datasets:
        for seed in cfg.seeds:
            try:
                base_docs = load_dataset(dataset_name, cfg, seed)
            except Exception as exc:
                skipped.append({"dataset": dataset_name, "seed": str(seed), "reason": str(exc)})
                continue
            if len(base_docs) < 2:
                skipped.append(
                    {"dataset": dataset_name, "seed": str(seed), "reason": "insufficient docs"}
                )
                continue

            for p in cfg.p_values:
                random_rows = run_single_case(
                    dataset_name=dataset_name,
                    scenario_name="random_pairs",
                    docs=base_docs,
                    cfg=cfg,
                    seed=seed,
                    p=p,
                )
                raw_rows.extend(random_rows)

                near_docs, _ = make_near_duplicates(
                    base_docs, seed=seed, pairs=min(cfg.pair_cfg.random_pairs // 3, 1000)
                )
                near_rows = run_single_case(
                    dataset_name=dataset_name,
                    scenario_name="near_duplicates",
                    docs=near_docs,
                    cfg=cfg,
                    seed=seed,
                    p=p,
                )
                raw_rows.extend(near_rows)

                print(
                    f"[done] dataset={dataset_name} seed={seed} p={p} "
                    f"random_docs={len(base_docs)} near_docs={len(near_docs)}"
                )

    if cfg.include_scaling:
        run_scaling(cfg, raw_rows)

    agg_rows = aggregate_rows(raw_rows)

    raw_json = cfg.out_dir / "raw_runs.json"
    agg_json = cfg.out_dir / "summary_stats.json"
    raw_csv = cfg.out_dir / "raw_runs.csv"
    agg_csv = cfg.out_dir / "summary_stats.csv"
    meta_json = cfg.out_dir / "run_metadata.json"
    skipped_json = cfg.out_dir / "skipped_runs.json"

    raw_json.write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")
    agg_json.write_text(json.dumps(agg_rows, indent=2), encoding="utf-8")
    meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    skipped_json.write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    save_rows_csv(raw_csv, raw_rows)
    save_rows_csv(agg_csv, agg_rows)

    print(f"\nWrote raw runs to: {raw_json}")
    print(f"Wrote summary to: {agg_json}")
    print(f"Wrote metadata to: {meta_json}")
    print(f"Wrote skipped list to: {skipped_json}")
    if skipped:
        print("\nSkipped datasets/seeds:")
        for item in skipped:
            print(f"- dataset={item['dataset']} seed={item['seed']} reason={item['reason']}")


if __name__ == "__main__":
    main()
