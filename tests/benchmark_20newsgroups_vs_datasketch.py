import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from urllib.error import URLError

from datasketch import MinHash
from sklearn.datasets import fetch_20newsgroups

sys.path.append(str(Path(__file__).resolve().parents[1]))
from minhashlib import DiffChecker


@dataclass
class BenchmarkConfig:
    max_docs: int = 2000
    k: int = 5
    num_perm: int = 128
    random_pairs: int = 3000
    near_dup_pairs: int = 1000
    edit_rates: tuple[float, ...] = (0.02, 0.05, 0.10)
    data_home: Path = Path(__file__).resolve().parents[1] / ".cache" / "scikit_learn_data"
    local_docs: Path | None = None
    seed: int = 42


def normalize_text(doc: str) -> str:
    return " ".join(doc.split())


def load_docs(cfg: BenchmarkConfig) -> list[str]:
    if cfg.local_docs is not None:
        docs = load_local_docs(cfg.local_docs)
        docs = [doc for doc in docs if len(doc) >= cfg.k]
        return docs[: cfg.max_docs]

    cfg.data_home.mkdir(parents=True, exist_ok=True)
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        data_home=str(cfg.data_home),
    )
    docs = [normalize_text(doc) for doc in dataset.data]
    docs = [doc for doc in docs if len(doc) >= cfg.k]
    return docs[: cfg.max_docs]


def load_local_docs(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Local docs file does not exist: {path}")

    docs: list[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item.get("text", "")
                if text:
                    docs.append(normalize_text(text))
    else:
        with path.open("r", encoding="utf-8") as f:
            docs = [normalize_text(line.strip()) for line in f if line.strip()]
    return docs


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


def build_diffchecker_signatures(docs: list[str], cfg: BenchmarkConfig):
    checker = DiffChecker(k=cfg.k, num_perm=cfg.num_perm, seed=cfg.seed)
    signatures = []
    start = time.perf_counter()
    for doc in docs:
        signatures.append(checker.generateSignature(doc))
    build_seconds = time.perf_counter() - start
    return checker, signatures, build_seconds


def build_datasketch_signatures(docs: list[str], cfg: BenchmarkConfig):
    minhashes = []
    start = time.perf_counter()
    for doc in docs:
        m = MinHash(num_perm=cfg.num_perm, seed=cfg.seed)
        for shingle in shingle_strings(doc, cfg.k):
            m.update(shingle.encode("utf-8"))
        minhashes.append(m)
    build_seconds = time.perf_counter() - start
    return minhashes, build_seconds


def evaluate_pairs(
    checker: DiffChecker,
    signatures,
    minhashes,
    shingle_sets: list[set[str]],
    pairs: list[tuple[int, int]],
):
    start = time.perf_counter()
    dc_scores = [
        checker.checkJaccardSignatureSimilarity(signatures[i], signatures[j])
        for i, j in pairs
    ]
    dc_eval_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ds_scores = [minhashes[i].jaccard(minhashes[j]) for i, j in pairs]
    ds_eval_seconds = time.perf_counter() - start

    exact_scores = [jaccard(shingle_sets[i], shingle_sets[j]) for i, j in pairs]
    dc_mae = sum(abs(a - b) for a, b in zip(dc_scores, exact_scores)) / len(pairs)
    ds_mae = sum(abs(a - b) for a, b in zip(ds_scores, exact_scores)) / len(pairs)
    return dc_eval_seconds, ds_eval_seconds, dc_mae, ds_mae


def make_random_pairs(cfg: BenchmarkConfig, n_docs: int) -> list[tuple[int, int]]:
    rng = random.Random(cfg.seed + 100)
    return [(rng.randrange(n_docs), rng.randrange(n_docs)) for _ in range(cfg.random_pairs)]


def mutate_document(doc: str, rate: float, rng: random.Random) -> str:
    if not doc:
        return doc
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    chars = list(doc)
    num_edits = max(1, int(len(chars) * rate))
    for _ in range(num_edits):
        op = rng.choice(("insert", "delete", "replace", "swap"))
        if op == "insert":
            pos = rng.randrange(len(chars) + 1)
            chars.insert(pos, rng.choice(alphabet))
        elif op == "delete" and chars:
            pos = rng.randrange(len(chars))
            chars.pop(pos)
        elif op == "replace" and chars:
            pos = rng.randrange(len(chars))
            chars[pos] = rng.choice(alphabet)
        elif op == "swap" and len(chars) >= 2:
            pos = rng.randrange(len(chars) - 1)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def make_near_duplicates(cfg: BenchmarkConfig, source_docs: list[str]):
    rng = random.Random(cfg.seed + 200)
    docs = list(source_docs)
    pair_indices = []
    for i in range(cfg.near_dup_pairs):
        base_idx = rng.randrange(len(source_docs))
        rate = cfg.edit_rates[i % len(cfg.edit_rates)]
        mutated = mutate_document(source_docs[base_idx], rate, rng)
        pair_indices.append((base_idx, len(docs)))
        docs.append(mutated)
    return docs, pair_indices


def run_scenario(name: str, docs: list[str], pairs: list[tuple[int, int]], cfg: BenchmarkConfig):
    shingle_sets = [shingle_strings(doc, cfg.k) for doc in docs]
    checker, signatures, dc_build_seconds = build_diffchecker_signatures(docs, cfg)
    minhashes, ds_build_seconds = build_datasketch_signatures(docs, cfg)
    dc_eval_seconds, ds_eval_seconds, dc_mae, ds_mae = evaluate_pairs(
        checker, signatures, minhashes, shingle_sets, pairs
    )

    print(f"Scenario: {name}")
    print(f"Docs={len(docs)}, pairs={len(pairs)}")
    print(f"Build signatures - DiffChecker: {dc_build_seconds:.4f}s")
    print(f"Build signatures - datasketch: {ds_build_seconds:.4f}s")
    print(f"Pair eval      - DiffChecker: {dc_eval_seconds:.4f}s")
    print(f"Pair eval      - datasketch: {ds_eval_seconds:.4f}s")
    print(f"MAE vs exact   - DiffChecker: {dc_mae:.6f}")
    print(f"MAE vs exact   - datasketch: {ds_mae:.6f}")
    print()


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark DiffChecker vs datasketch.MinHash on 20 Newsgroups."
    )
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--random-pairs", type=int, default=3000)
    parser.add_argument("--near-dup-pairs", type=int, default=1000)
    parser.add_argument(
        "--local-docs",
        type=Path,
        default=None,
        help="Optional local .txt (one doc per line) or .jsonl (expects {'text': ...})",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return BenchmarkConfig(
        max_docs=args.max_docs,
        k=args.k,
        num_perm=args.num_perm,
        random_pairs=args.random_pairs,
        near_dup_pairs=args.near_dup_pairs,
        local_docs=args.local_docs,
        seed=args.seed,
    )


def main():
    cfg = parse_args()
    try:
        base_docs = load_docs(cfg)
    except URLError as exc:
        raise SystemExit(
            "Could not download 20 Newsgroups (network error). "
            "Retry with network access, or pass --local-docs <file>."
        ) from exc
    if len(base_docs) < 2:
        raise RuntimeError("Not enough documents from 20 Newsgroups after filtering.")

    random_pairs = make_random_pairs(cfg, len(base_docs))
    near_dup_docs, near_dup_pairs = make_near_duplicates(cfg, base_docs)

    print("Benchmark: DiffChecker vs datasketch.MinHash on 20 Newsgroups")
    print(
        "Config:",
        f"max_docs={cfg.max_docs}, k={cfg.k}, num_perm={cfg.num_perm},",
        f"random_pairs={cfg.random_pairs}, near_dup_pairs={cfg.near_dup_pairs}, seed={cfg.seed}",
    )
    print()

    run_scenario("Random pairs from real corpus", base_docs, random_pairs, cfg)
    run_scenario("Controlled near-duplicates", near_dup_docs, near_dup_pairs, cfg)


if __name__ == "__main__":
    main()
