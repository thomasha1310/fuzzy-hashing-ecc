"""The fuzzy-hash consensus code exactly as docs/SPEC.md defines it.

Blocks are always one byte (n = 8) and decoding is always exact (SPEC §III.4).
Users choose only the number of `jurors` and the `key` (SPEC §I).
"""

from dataclasses import dataclass, field

import numpy as np

from ._types import Bits, BoolArray, Float32Array, FloatArray, IntArray
from .research import DecodeResult, ResearchCode

BLOCK_BITS = 8
"""n: data bits per block. Fixed at one byte."""

_CHUNK = 8192
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.int64)
"""Number of 1 bits in each byte value, so data-bit distances are table lookups."""


def _all_bytes() -> Bits:
    """All 256 possible blocks as rows of bits, most significant bit first."""
    return np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)


@dataclass(frozen=True)
class ConsensusCode:
    """One byte stored with J sign-of-projection signatures (SPEC §II–III).

    Attributes:
        jurors: number of signatures J. Each block is stored in 8(J + 1) bits.
        key: seed for the J random ±1 matrices. Encoder and decoder must agree.
    """

    jurors: int = 20
    key: int = 0
    _impl: ResearchCode = field(init=False, repr=False, compare=False)
    _codebook: Bits = field(init=False, repr=False, compare=False)
    _codebook_pm: Float32Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.jurors < 1:
            raise ValueError("jurors must be at least 1")
        if self.key < 0:
            raise ValueError("key must be a non-negative integer")
        impl = ResearchCode(BLOCK_BITS, self.jurors, family="rademacher", seed=self.key)
        codebook = impl.encode(_all_bytes())  # (256, L): every possible record
        object.__setattr__(self, "_impl", impl)
        object.__setattr__(self, "_codebook", codebook)
        object.__setattr__(
            self, "_codebook_pm", (1.0 - 2.0 * codebook.astype(np.float32)).T.copy()
        )

    # Derived quantities (SPEC §I) ------------------------------------------

    @property
    def n(self) -> int:
        return BLOCK_BITS

    @property
    def total_bits(self) -> int:
        """L = 8(J + 1): stored bits per block."""
        return self._impl.total_bits

    @property
    def overhead(self) -> int:
        """Storage overhead L / n = J + 1."""
        return self.jurors + 1

    @property
    def rate(self) -> float:
        """Code rate R = 1 / (J + 1)."""
        return self._impl.rate

    @property
    def codebook(self) -> Bits:
        """All 256 records E(x), in order of the byte value x. Shape (256, L)."""
        return self._codebook

    def describe(self) -> str:
        return (
            f"8-bit blocks, jurors={self.jurors}, key={self.key} "
            f"(stores {self.overhead}× the data)"
        )

    # Operations (SPEC §III) -----------------------------------------------

    def signatures(self, x: Bits) -> BoolArray:
        """The J signatures σ_i(x). Shape (..., 8) -> (..., J, 8)."""
        return self._impl.signatures(x)

    def encode(self, x: Bits) -> Bits:
        """The stored record E(x) = (x, σ_1(x), …, σ_J(x)). Shape (..., 8) -> (..., L)."""
        return self._impl.encode(x)

    def split(self, record: Bits) -> tuple[Bits, BoolArray]:
        """Split one read-back record into its data bits and its J stored signatures."""
        return self._impl.split(record)

    def score(self, candidates: Bits, record: Bits) -> FloatArray:
        """S(x̂) (SPEC §III.3) of each candidate block against one read-back record."""
        x_read, stored = self._impl.split(record)
        return self._impl.score(candidates, x_read, stored)

    def decode_values(self, records: Bits) -> IntArray:
        """Exact decoding of many records at once. Shape (..., L) -> byte values (...).

        Picks the record in the codebook closest in Hamming distance to what
        was read (SPEC §III.4). Ties go to the candidate that changes fewer
        data bits, then to the smaller byte value.
        """
        records = np.asarray(records, dtype=np.uint8)
        if records.shape[-1] != self.total_bits:
            raise ValueError(f"expected records of {self.total_bits} bits")
        flat = records.reshape(-1, self.total_bits)
        out = np.empty(flat.shape[0], dtype=np.int64)
        size = self.total_bits
        values = np.arange(256, dtype=np.int64)
        for start in range(0, flat.shape[0], _CHUNK):
            chunk = flat[start : start + _CHUNK]
            pm = 1.0 - 2.0 * chunk.astype(np.float32)
            # For ±1 vectors, <u, v> = L - 2·d(u, v); float32 is exact here.
            dist = np.rint((size - pm @ self._codebook_pm) / 2).astype(np.int64)
            read_values = np.packbits(chunk[:, :BLOCK_BITS], axis=1)[:, 0].astype(np.int64)
            data_dist = _POPCOUNT[read_values[:, None] ^ values[None, :]]
            key = (dist * (BLOCK_BITS + 1) + data_dist) * 256 + values
            out[start : start + len(chunk)] = key.argmin(axis=1)
        return out.reshape(records.shape[:-1])

    def decode_bits(self, records: Bits) -> Bits:
        """Decode records and return data bits. Shape (..., L) -> (..., 8)."""
        values = self.decode_values(records).astype(np.uint8)
        return np.unpackbits(values[..., None], axis=-1)

    def decode(self, record: Bits) -> DecodeResult:
        """Decode one record, with its score and the number of data bits changed."""
        record = np.asarray(record, dtype=np.uint8)
        x = self.decode_bits(record)
        score = float(self.score(x[None, :], record)[0])
        flips = int((x != record[:BLOCK_BITS]).sum())
        return DecodeResult(x, score, flips, 256)
