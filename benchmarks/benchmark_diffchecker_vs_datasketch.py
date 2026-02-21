import random
import string
import time
from dataclasses import dataclass
from pathlib import Path
import sys

from datasketch import MinHash

sys.path.append(str(Path(__file__).resolve().parents[1]))
from minhashlib import DiffChecker


@dataclass
class BenchmarkConfig:
    num_docs: int = 400
    doc_length: int = 800
    k: int = 5
    num_perm: int = 128
    num_pairs: int = 800
    seed: int = 42


def make_docs(cfg: BenchmarkConfig) -> list[str]:
    rng = random.Random(cfg.seed)
    alphabet = string.ascii_lowercase + "     "
    return [
        "".join(rng.choice(alphabet) for _ in range(cfg.doc_length))
        for _ in range(cfg.num_docs)
    ]


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
    checker = DiffChecker(k=cfg.k, num_perm=cfg.num_perm)
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
        m = MinHash(num_perm=cfg.num_perm)
        for shingle in shingle_strings(doc, cfg.k):
            m.update(shingle.encode("utf-8"))
        minhashes.append(m)
    build_seconds = time.perf_counter() - start
    return minhashes, build_seconds


def run_pairwise_eval(
    checker: DiffChecker,
    signatures,
    minhashes,
    shingle_sets: list[set[str]],
    cfg: BenchmarkConfig,
):
    rng = random.Random(cfg.seed + 1)
    pairs = [
        (rng.randrange(cfg.num_docs), rng.randrange(cfg.num_docs))
        for _ in range(cfg.num_pairs)
    ]

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


def main():
    cfg = BenchmarkConfig()
    docs = make_docs(cfg)
    shingle_sets = [shingle_strings(doc, cfg.k) for doc in docs]

    checker, signatures, dc_build_seconds = build_diffchecker_signatures(docs, cfg)
    minhashes, ds_build_seconds = build_datasketch_signatures(docs, cfg)
    dc_eval_seconds, ds_eval_seconds, dc_mae, ds_mae = run_pairwise_eval(
        checker, signatures, minhashes, shingle_sets, cfg
    )

    print("Benchmark: DiffChecker vs datasketch.MinHash")
    print(
        "Config:",
        f"docs={cfg.num_docs}, length={cfg.doc_length}, k={cfg.k},",
        f"num_perm={cfg.num_perm}, pairs={cfg.num_pairs}, seed={cfg.seed}",
    )
    print()
    print(f"Build signatures - DiffChecker: {dc_build_seconds:.4f}s")
    print(f"Build signatures - datasketch: {ds_build_seconds:.4f}s")
    print(f"Pair eval      - DiffChecker: {dc_eval_seconds:.4f}s")
    print(f"Pair eval      - datasketch: {ds_eval_seconds:.4f}s")
    print(f"MAE vs exact   - DiffChecker: {dc_mae:.6f}")
    print(f"MAE vs exact   - datasketch: {ds_mae:.6f}")


if __name__ == "__main__":
    main()
