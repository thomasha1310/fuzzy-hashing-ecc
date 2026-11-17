"""Repetition-code baseline, compared at matched storage overhead."""

import math
from dataclasses import dataclass

import numpy as np

from ._types import Bits


@dataclass(frozen=True)
class RepetitionCode:
    """Each data bit is stored r times (r odd) and decoded by majority vote."""

    n: int
    r: int

    def __post_init__(self) -> None:
        if self.r < 1 or self.r % 2 == 0:
            raise ValueError("r must be a positive odd integer")

    @property
    def total_bits(self) -> int:
        return self.n * self.r

    @property
    def rate(self) -> float:
        return 1.0 / self.r

    @property
    def min_distance(self) -> int:
        return self.r

    @property
    def t(self) -> int:
        """Errors per data bit that are guaranteed to be corrected: (r - 1) / 2."""
        return (self.r - 1) // 2

    def encode(self, x: Bits) -> Bits:
        """Shape (..., n) -> (..., n·r). Each bit's r copies are adjacent."""
        return np.repeat(np.asarray(x, dtype=np.uint8), self.r, axis=-1)

    def decode_bits(self, record: Bits) -> Bits:
        groups = np.asarray(record, dtype=np.uint8).reshape(
            *record.shape[:-1], self.n, self.r
        )
        return (groups.sum(axis=-1) > self.t).astype(np.uint8)

    def block_error_rate(self, p: float) -> float:
        """Exact block error rate on a binary symmetric channel with crossover p."""
        pb = sum(
            math.comb(self.r, w) * p**w * (1 - p) ** (self.r - w)
            for w in range(self.t + 1, self.r + 1)
        )
        return 1.0 - (1.0 - pb) ** self.n


def matched_repetition(n: int, total_bits: int) -> RepetitionCode:
    """The largest odd r with n·r <= total_bits. It never uses more storage."""
    r = max(1, total_bits // n)
    if r % 2 == 0:
        r -= 1
    return RepetitionCode(n, r)
