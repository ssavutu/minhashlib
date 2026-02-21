import numpy as np
from numba import njit
from xxhash import xxh3_64_intdigest
from numpy.typing import NDArray


@njit(cache=True)
def _generate_signature_numba(
    shingles: NDArray[np.int64], hash_params: NDArray[np.int64], p: int
) -> NDArray[np.int64]:
    num_perm = hash_params.shape[0]
    sig = np.empty(num_perm, dtype=np.int64)
    sig[:] = p

    for i in range(shingles.shape[0]):
        shingle = shingles[i]
        for j in range(num_perm):
            val = (hash_params[j, 0] * shingle + hash_params[j, 1]) % p
            if val < sig[j]:
                sig[j] = val

    return sig


class DiffChecker:
    def __init__(self, p=3037000493, k=2, num_perm=128, seed: int | None = 42):
        self.p = p
        self.k = k
        self.num_perm = num_perm
        self.rng = np.random.default_rng(seed)
        self.signatureMatrix = np.empty((num_perm, 0), dtype=np.int64)
        self.hashParams = self.generateKHashParameters()

    def generateKShingles(self, document: str) -> NDArray[np.int64]:
        if len(document) < self.k:
            return np.array([], dtype=np.int64)

        shingles = np.fromiter(
            (
                xxh3_64_intdigest(document[i : i + self.k].encode("utf-8")) % self.p
                for i in range(len(document) - self.k + 1)
            ),
            dtype=np.int64,
        )

        return np.unique(shingles)

    def generateKHashParameters(self) -> NDArray[np.int64]:
        a = self.rng.integers(1, self.p, size=self.num_perm, dtype=np.int64)
        b = self.rng.integers(0, self.p, size=self.num_perm, dtype=np.int64)
        return np.column_stack((a, b))

    def generateSignature(self, document: str) -> NDArray[np.int64]:
        shingles = self.generateKShingles(document)
        return _generate_signature_numba(shingles, self.hashParams, self.p)

    def appendDocumentAsSignature(self, document: str):
        signature = self.generateSignature(document)
        self.signatureMatrix = np.column_stack([self.signatureMatrix, signature])

    @staticmethod
    def checkJaccardSignatureSimilarity(a: NDArray[np.int64], b: NDArray[np.int64]) -> float:
        return float(np.count_nonzero(a == b)) / float(a.size)
