# Minhashlib

This is a minimal implementation of Minhashing as described in Jeffrey Ullman's book *Mining Massive Datastructures*. It's been benchmarked against datasketch on a CPU and is comparable in speed and accuracy.

## Benchmark Suite

Use `tests/benchmark_claims_suite.py` to run comprehensive, reproducible benchmarks:

- Multiple datasets (`synthetic`, `20newsgroups`, `wikipedia`, `ag_news`, `local`)
- Multiple seeds with mean/std/95% CI
- Metrics: MAE, Precision/Recall/F1 at threshold, Precision@K/Recall@K
- Speed: build/pair-eval/retrieval latency and throughput
- Memory: peak allocation and bytes/signature
- Optional scaling sweeps over docs/`num_perm`/doc length

Example (full):

```bash
python3 tests/benchmark_claims_suite.py \
  --datasets synthetic,20newsgroups,wikipedia \
  --wiki-dump-path data/simplewiki-latest-pages-articles.xml.bz2 \
  --seeds 42,43,44 \
  --p-values 2147483647,3037000493 \
  --max-docs 2000 \
  --random-pairs 3000 \
  --num-queries 200 \
  --include-scaling
```

Example (offline/local corpus only):

```bash
python3 tests/benchmark_claims_suite.py \
  --datasets synthetic,local \
  --local-docs /path/to/docs.jsonl \
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

### Individual benchmark tests

You can run individual tests instead of the full suite:

```bash
# Accuracy-only
python3 tests/benchmark_claims_accuracy.py --datasets synthetic,20newsgroups,wikipedia

# Performance-only
python3 tests/benchmark_claims_performance.py --datasets synthetic,20newsgroups,wikipedia

# Memory-only
python3 tests/benchmark_claims_memory.py --datasets synthetic,20newsgroups,wikipedia

# Scaling-only (synthetic sweeps)
python3 tests/benchmark_claims_scaling.py --datasets synthetic
```

Each individual test writes outputs under `benchmark_outputs/<test_name>/`.
