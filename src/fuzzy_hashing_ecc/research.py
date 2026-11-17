"""Research version of the consensus code, with every design choice adjustable.

docs/SPEC.md fixes the design, and users should use `ConsensusCode` (scheme.py).
This class keeps the alternatives that were compared to make those choices:
matrix families, threshold rules, storage formats, similarity measures,
aggregation rules and extra decoders. They are kept so the experiments behind
SPEC §VI stay reproducible. With its defaults, this class is exactly the SPEC
design.
"""

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import signatures as sg
from ._types import Bits, BoolArray, FloatArray
from .projections import MatrixFamily, bipolar, generate_matrices

type Strategy = Literal["greedy", "parallel_vote", "exhaustive"]

_CHUNK = 4096


@dataclass(frozen=True)
class DecodeResult:
    """Outcome of decoding one block.

    Attributes:
        x: the decoded data block.
        score: its consensus score S(x̂).
        flips: d_H(x̂, x̃), the number of data bits the decoder changed.
        iterations: greedy steps, parallel-vote rounds, or candidates scored.
    """

    x: Bits
    score: float
    flips: int
    iterations: int


@dataclass
class ResearchCode:
    """A systematic block code: n data bits plus J fuzzy-hash signatures.

    The defaults are the SPEC design: n x n Rademacher matrices, the sign rule,
    bitmap storage, agreement similarity, sum aggregation, and proximity weight
    λ = `lam` = 1. With them, exhaustive search over radius n is exact ML
    decoding (SPEC Proposition 1).
    """

    n: int
    jurors: int = 20
    dim: int | None = None
    family: MatrixFamily = "rademacher"
    threshold: sg.Threshold = field(default_factory=sg.Threshold)
    storage: sg.StorageFormat = "bitmap"
    similarity: sg.SimilarityKind = "agreement"
    aggregate: sg.AggregateKind = "sum"
    trim: float = 0.2
    lam: float = 1.0
    seed: int = 0
    strategy: Strategy = "greedy"
    radius: int = 2
    matrices: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 1 or self.jurors < 1:
            raise ValueError("n and jurors must be positive")
        if self.lam < 0:
            raise ValueError("lam must be non-negative")
        self.matrices = generate_matrices(
            self.family, self.n, self.d, self.jurors, self.seed
        )
        sg.signature_bits(self.storage, self.d, self.threshold)  # validates

    # ------------------------------------------------------------------ #
    # Derived parameters
    # ------------------------------------------------------------------ #

    @property
    def d(self) -> int:
        """Projection dimension D (defaults to n)."""
        return self.n if self.dim is None else self.dim

    @property
    def sig_bits(self) -> int:
        """L: stored bits per signature."""
        return sg.signature_bits(self.storage, self.d, self.threshold)

    @property
    def total_bits(self) -> int:
        """L_tot = n + J·L: stored bits per block."""
        return self.n + self.jurors * self.sig_bits

    @property
    def rate(self) -> float:
        return self.n / self.total_bits

    @property
    def overhead(self) -> float:
        """ρ = L_tot / n."""
        return self.total_bits / self.n

    def describe(self) -> str:
        return (
            f"n={self.n} J={self.jurors} D={self.d} {self.family} "
            f"{self.threshold.describe()} {self.storage} {self.similarity} "
            f"{self.aggregate} λ={self.lam:g} (ρ={self.overhead:.1f})"
        )

    # ------------------------------------------------------------------ #
    # Encoding (Defs. 1–2)
    # ------------------------------------------------------------------ #

    def project(self, x: Bits) -> FloatArray:
        """z_i = A_i b(x) for every juror. Shape (..., n) -> (..., J, D)."""
        result: FloatArray = np.einsum("jdk,...k->...jd", self.matrices, bipolar(x))
        return result

    def signatures(self, x: Bits) -> BoolArray:
        """σ_i(x) as masks. Shape (..., n) -> (..., J, D)."""
        return self.threshold.apply(self.project(x))

    def encode(self, x: Bits) -> Bits:
        """Build the stored record E(x). Shape (..., n) -> (..., L_tot)."""
        x = np.asarray(x, dtype=np.uint8)
        if x.shape[-1] != self.n:
            raise ValueError(f"expected blocks of {self.n} bits, got {x.shape[-1]}")
        sig = sg.serialize(self.signatures(x), self.storage, self.threshold)
        sig_flat = sig.reshape(*x.shape[:-1], self.jurors * self.sig_bits)
        return np.concatenate([x, sig_flat], axis=-1).astype(np.uint8)

    def split(self, record: Bits) -> tuple[Bits, BoolArray]:
        """Split a stored record (L_tot,) into x̃ (n,) and stored masks (J, D)."""
        record = np.asarray(record, dtype=np.uint8)
        if record.shape != (self.total_bits,):
            raise ValueError(f"expected a record of {self.total_bits} bits")
        x_read = record[: self.n].copy()
        stored_bits = record[self.n :].reshape(self.jurors, self.sig_bits)
        stored = sg.deserialize(stored_bits, self.storage, self.d, self.threshold)
        return x_read, stored

    # ------------------------------------------------------------------ #
    # Scoring (Def. 4)
    # ------------------------------------------------------------------ #

    def juror_similarities(self, masks: BoolArray, stored: BoolArray) -> FloatArray:
        """Per-juror similarity. masks (..., J, D) vs stored (J, D) -> (..., J)."""
        return sg.similarity(masks, stored, self.similarity)

    def _combine(self, juror_sims: FloatArray, dist: FloatArray) -> FloatArray:
        return sg.aggregate(juror_sims, self.aggregate, self.trim) - self.lam * dist

    def score(self, candidates: Bits, x_read: Bits, stored: BoolArray) -> FloatArray:
        """S(x̂) for every candidate. candidates (C, n) -> (C,)."""
        candidates = np.asarray(candidates, dtype=np.uint8)
        juror = self.juror_similarities(self.signatures(candidates), stored)
        dist = (candidates != x_read).sum(axis=-1).astype(np.float64)
        return self._combine(juror, dist)

    def flip_projections(self, x: Bits, z: FloatArray) -> FloatArray:
        """Projections after flipping each single bit, via the rank-one update.

        A_i b(x ⊕ e_j) = A_i b(x) - 2 b(x)_j A_i[:, j]. Shape (J, D) -> (n, J, D).
        """
        b = bipolar(x)
        cols = np.transpose(self.matrices, (2, 0, 1))  # (n, J, D)
        result: FloatArray = z[None, :, :] - 2.0 * b[:, None, None] * cols
        return result

    def flip_evaluation(
        self, x_hat: Bits, x_read: Bits, stored: BoolArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Score x̂ and all n of its single-flip neighbours.

        Returns (current juror sims (J,), current score, neighbour juror sims
        (n, J), neighbour scores (n,)).
        """
        z = self.project(x_hat)
        cur_juror = self.juror_similarities(self.threshold.apply(z), stored)
        cur_dist = float((x_hat != x_read).sum())
        cur = self._combine(cur_juror, np.array(cur_dist))
        flip_juror = self.juror_similarities(
            self.threshold.apply(self.flip_projections(x_hat, z)), stored
        )
        # Flipping bit j moves x̂ one step away from x̃ if x̂_j == x̃_j, else one closer.
        flip_dist = cur_dist + np.where(x_hat == x_read, 1.0, -1.0)
        return cur_juror, cur, flip_juror, self._combine(flip_juror, flip_dist)

    # ------------------------------------------------------------------ #
    # Decoding (Def. 5)
    # ------------------------------------------------------------------ #

    def decode(
        self,
        record: Bits,
        strategy: Strategy | None = None,
        radius: int | None = None,
    ) -> DecodeResult:
        """Decode one stored record (L_tot,)."""
        strategy = self.strategy if strategy is None else strategy
        radius = self.radius if radius is None else radius
        x_read, stored = self.split(record)
        match strategy:
            case "greedy":
                return self._greedy(x_read, stored)
            case "parallel_vote":
                return self._parallel_vote(x_read, stored)
            case "exhaustive":
                return self._exhaustive(x_read, stored, radius)

    def decode_bits(self, record: Bits) -> Bits:
        """Decode with the configured strategy and return only the data bits."""
        return self.decode(record).x

    def _greedy(self, x_read: Bits, stored: BoolArray) -> DecodeResult:
        x_hat = x_read.copy()
        steps = 0
        _, cur, _, flip_scores = self.flip_evaluation(x_hat, x_read, stored)
        for _ in range(self.n):
            j = int(np.argmax(flip_scores))
            if not flip_scores[j] > cur:
                break
            x_hat[j] ^= 1
            steps += 1
            # Recompute from scratch so no floating-point drift builds up.
            _, cur, _, flip_scores = self.flip_evaluation(x_hat, x_read, stored)
        return DecodeResult(
            x_hat, float(cur), int((x_hat != x_read).sum()), steps
        )

    def _parallel_vote(self, x_read: Bits, stored: BoolArray) -> DecodeResult:
        x_hat = x_read.copy()
        rounds = 0
        for _ in range(self.n):
            cur_juror, _, flip_juror, _ = self.flip_evaluation(x_hat, x_read, stored)
            votes = (flip_juror > cur_juror[None, :]).sum(axis=1)
            to_flip = votes > self.jurors / 2
            if not to_flip.any():
                break
            x_hat ^= to_flip.astype(np.uint8)
            rounds += 1
        final = float(self.score(x_hat[None, :], x_read, stored)[0])
        return DecodeResult(x_hat, final, int((x_hat != x_read).sum()), rounds)

    def _candidates(self, x_read: Bits, radius: int) -> Iterator[Bits]:
        """Yield chunks (C, n) of every x̂ with d_H(x̂, x̃) <= radius."""
        patterns = itertools.chain.from_iterable(
            itertools.combinations(range(self.n), w)
            for w in range(min(radius, self.n) + 1)
        )
        while chunk := list(itertools.islice(patterns, _CHUNK)):
            flips = np.zeros((len(chunk), self.n), dtype=np.uint8)
            for row, idx in enumerate(chunk):
                flips[row, list(idx)] = 1
            yield x_read[None, :] ^ flips

    def _exhaustive(self, x_read: Bits, stored: BoolArray, radius: int) -> DecodeResult:
        best_key: tuple[float, int, tuple[int, ...]] | None = None
        best_x = x_read
        best_score = 0.0
        scored = 0
        for cands in self._candidates(x_read, radius):
            scores = self.score(cands, x_read, stored)
            dist = (cands != x_read).sum(axis=1)
            # Highest score first, then smallest distance, then smallest integer
            # value of x̂ (MSB-first bits). np.lexsort treats its last key as primary.
            keys = [cands[:, k] for k in range(self.n - 1, -1, -1)]
            order = np.lexsort([*keys, dist, -scores])
            i = int(order[0])
            key = (float(scores[i]), -int(dist[i]), tuple(-int(v) for v in cands[i]))
            if best_key is None or key > best_key:
                best_key, best_x, best_score = key, cands[i].copy(), float(scores[i])
            scored += len(cands)
        return DecodeResult(best_x, best_score, int((best_x != x_read).sum()), scored)
