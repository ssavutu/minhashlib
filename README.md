# Minhashlib

This is a minimal implementation of MinHashing as described in Jeffrey Ullman's book *Mining Massive Datasets*.

It can be installed via `pip install minhashlib`

## Current Benchmark Claim

Based on the benchmark outputs in this repository:

- On recent multi-seed CPU synthetic runs (seeds `42-46`), `minhashlib` builds signatures about `~6x` faster than `datasketch`.
- Accuracy is comparable in magnitude (similar MAE scale and matching threshold-based metrics in these runs), though `datasketch` is slightly better on MAE in most synthetic scenarios.
- Memory results are workload-dependent, so this project does not claim universal memory superiority.

In short: this implementation is minimal and fast, with accuracy that is broadly comparable to `datasketch`, but outcomes vary depending on dataset and configuration.

## Benchmark Suite

Use `benchmarks/benchmark_claims_suite.py` to run comprehensive, reproducible benchmarks:

- Multiple datasets (`synthetic`, `20newsgroups`, `wikipedia`, `ag_news`, `local`)
- Multiple seeds with mean/std/95% CI
- Metrics: MAE(Mean Average Error), Precision/Recall/F1 at threshold, Precision@K/Recall@K
- Speed: build/pair-eval/retrieval latency and throughput
- Memory: peak allocation and bytes/signature
- Optional scaling sweeps over docs/number of hashes/doc length

Example (full):

```bash
python3 benchmarks/benchmark_claims_suite.py
  --datasets synthetic,20newsgroups,wikipedia
  --wiki-dump-path data/simplewiki-latest-pages-articles.xml.bz2
  --seeds 42,43,44
  --p-values 2147483647,3037000493
  --max-docs 2000
  --random-pairs 3000
  --num-queries 200
  --include-scaling
```

Example (offline/local corpus only):

```bash
python3 benchmarks/benchmark_claims_suite.py
  --datasets synthetic,local
  --local-docs /path/to/docs.jsonl
  --seeds 42,43,44
```

Outputs are written to `benchmark_outputs/` by default:

- `raw_runs.json` / `raw_runs.csv`
- `summary_stats.json` / `summary_stats.csv`
- `run_metadata.json`
- `skipped_runs.json`

### Benchmark data setup

Pull required benchmark datasets into local project paths:

```bash
python3 scripts/setup_benchmark_data.py
```

This prepares:

- `data/simplewiki-latest-pages-articles.xml.bz2` (for Wikipedia benchmarks)
- `.cache/scikit_learn_data` (for `20newsgroups` benchmarks)

Optional flags:

```bash
python3 scripts/setup_benchmark_data.py --force
python3 scripts/setup_benchmark_data.py --skip-wikipedia
python3 scripts/setup_benchmark_data.py --skip-20newsgroups
```

### Individual Benchmarks

You can run individual benchmarks instead of the full suite:

```bash
# Accuracy-only
python3 benchmarks/benchmark_claims_accuracy.py --datasets synthetic,20newsgroups,wikipedia

# Performance-only
python3 benchmarks/benchmark_claims_performance.py --datasets synthetic,20newsgroups,wikipedia

# Memory-only
python3 benchmarks/benchmark_claims_memory.py --datasets synthetic,20newsgroups,wikipedia

# Scaling-only (synthetic sweeps)
python3 benchmarks/benchmark_claims_scaling.py --datasets synthetic
```

Each individual benchmark writes outputs under `benchmark_outputs/<test_name>/`.
