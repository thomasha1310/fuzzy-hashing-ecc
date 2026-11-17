"""Noise models for the storage medium."""

import numpy as np

from ._types import Bits


def bsc(bits: Bits, p: float, rng: np.random.Generator) -> Bits:
    """Binary symmetric channel: flip each bit independently with probability p."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    bits = np.asarray(bits, dtype=np.uint8)
    noise = (rng.random(bits.shape) < p).astype(np.uint8)
    return bits ^ noise


def flip_exactly(bits: Bits, w: int, rng: np.random.Generator) -> Bits:
    """Flip exactly w distinct, uniformly chosen positions of a 1-D bit array."""
    bits = np.asarray(bits, dtype=np.uint8).copy()
    if not 0 <= w <= bits.size:
        raise ValueError(f"cannot flip {w} of {bits.size} bits")
    idx = rng.choice(bits.size, size=w, replace=False)
    bits[idx] ^= 1
    return bits


def destroy_signatures(
    record: Bits, n: int, sig_bits: int, jurors: list[int], rng: np.random.Generator
) -> Bits:
    """Replace the given jurors' stored signatures with uniformly random bits."""
    record = np.asarray(record, dtype=np.uint8).copy()
    for i in jurors:
        start = n + i * sig_bits
        record[start : start + sig_bits] = rng.integers(0, 2, sig_bits, dtype=np.uint8)
    return record
