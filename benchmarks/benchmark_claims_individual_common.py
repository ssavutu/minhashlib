import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_claims_suite import (
    PairEvalConfig,
    RunConfig,
    aggregate_rows,
    load_dataset,
    make_near_duplicates,
    run_scaling,
    run_single_case,
    save_rows_csv,
)


def add_common_args(parser: argparse.ArgumentParser):
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
    parser.add_argument("--scaling-num-docs", default="500,1000,2000")
    parser.add_argument("--scaling-num-perm", default="64,128,256")
    parser.add_argument("--scaling-doc-length", default="400,800,1600")


def build_cfg(args: argparse.Namespace, include_scaling: bool = False) -> RunConfig:
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
        include_scaling=include_scaling,
    )


def run_rows(cfg: RunConfig, include_near_duplicates: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
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
                skipped.append({"dataset": dataset_name, "seed": str(seed), "reason": "insufficient docs"})
                continue
            for p in cfg.p_values:
                raw_rows.extend(
                    run_single_case(
                        dataset_name=dataset_name,
                        scenario_name="random_pairs",
                        docs=base_docs,
                        cfg=cfg,
                        seed=seed,
                        p=p,
                    )
                )
                if include_near_duplicates:
                    near_docs, _ = make_near_duplicates(
                        base_docs, seed=seed, pairs=min(cfg.pair_cfg.random_pairs // 3, 1000)
                    )
                    raw_rows.extend(
                        run_single_case(
                            dataset_name=dataset_name,
                            scenario_name="near_duplicates",
                            docs=near_docs,
                            cfg=cfg,
                            seed=seed,
                            p=p,
                        )
                    )
                print(f"[done] dataset={dataset_name} seed={seed} p={p}")
    return raw_rows, skipped


def run_scaling_rows(cfg: RunConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_scaling(cfg, rows)
    return rows


def default_metadata(cfg: RunConfig, test_name: str) -> dict[str, Any]:
    return {
        "test_name": test_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "config": {
            "datasets": cfg.datasets,
            "seeds": cfg.seeds,
            "k": cfg.k,
            "num_perm": cfg.num_perm,
            "p_values": cfg.p_values,
            "max_docs": cfg.max_docs,
            "pairs": cfg.pair_cfg.random_pairs,
            "threshold": cfg.pair_cfg.threshold,
            "topk": cfg.pair_cfg.topk,
            "num_queries": cfg.pair_cfg.num_queries,
        },
    }


def write_filtered_outputs(
    cfg: RunConfig,
    test_name: str,
    rows: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    keep_metric_roots: tuple[str, ...],
):
    out_dir = cfg.out_dir / test_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_rows(rows)
    filtered_summary: list[dict[str, Any]] = []
    for row in summary:
        keep = {
            "dataset",
            "scenario",
            "system",
            "p_value",
            "runs",
            "num_docs",
            "num_perm",
            "k",
            "pairs",
            "topk",
            "num_queries",
            "threshold",
        }
        for key in row:
            if any(key.startswith(root) for root in keep_metric_roots):
                keep.add(key)
        filtered_summary.append({k: row[k] for k in row if k in keep})

    (out_dir / "raw_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary_stats.json").write_text(json.dumps(filtered_summary, indent=2), encoding="utf-8")
    (out_dir / "skipped_runs.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(default_metadata(cfg, test_name), indent=2), encoding="utf-8"
    )
    save_rows_csv(out_dir / "raw_runs.csv", rows)
    save_rows_csv(out_dir / "summary_stats.csv", filtered_summary)

    print(f"Wrote {test_name} outputs to: {out_dir}")
