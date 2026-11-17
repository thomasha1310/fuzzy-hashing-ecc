"""Threshold rules, storage formats, similarity measures and aggregation.

A signature is represented internally as a boolean mask over [D] (True means
"active index").
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._types import Bits, BoolArray, FloatArray, IntArray

type ThresholdKind = Literal["sign", "top_s", "level"]
type StorageFormat = Literal["bitmap", "index"]
type SimilarityKind = Literal["agreement", "overlap", "jaccard"]
type AggregateKind = Literal["sum", "median", "trimmed"]


# --------------------------------------------------------------------------- #
# Threshold rules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Threshold:
    """Maps projections z in R^D to signature masks.

    Attributes:
        kind: "sign" (z_r > 0), "top_s" (the s largest entries, ties broken
            toward the smaller index), or "level" (z_r > tau).
        s: sparsity used by "top_s".
        tau: level used by "level".
    """

    kind: ThresholdKind = "sign"
    s: int = 1
    tau: float = 0.0

    def apply(self, z: FloatArray) -> BoolArray:
        """Threshold along the last axis. Shape (..., D) -> (..., D) bool."""
        match self.kind:
            case "sign":
                return z > 0.0
            case "level":
                return z > self.tau
            case "top_s":
                d = z.shape[-1]
                if not 1 <= self.s <= d:
                    raise ValueError(f"top_s needs 1 <= s <= D (s={self.s}, D={d})")
                # A stable sort on -z keeps equal values in index order, so
                # ties go to the smaller index.
                top: IntArray = np.argsort(-z, axis=-1, kind="stable")[..., : self.s]
                mask = np.zeros(z.shape, dtype=np.bool_)
                np.put_along_axis(mask, top, True, axis=-1)
                return mask

    def describe(self) -> str:
        match self.kind:
            case "sign":
                return "sign"
            case "level":
                return f"level(τ={self.tau:g})"
            case "top_s":
                return f"top-{self.s}"


# --------------------------------------------------------------------------- #
# Storage formats
# --------------------------------------------------------------------------- #


def index_width(d: int) -> int:
    """Bits per index in the index-list format: ceil(log2 D)."""
    if d < 2:
        raise ValueError("index-list storage requires D >= 2")
    return math.ceil(math.log2(d))


def signature_bits(fmt: StorageFormat, d: int, threshold: Threshold) -> int:
    """Stored size L of one signature, in bits."""
    if fmt == "bitmap":
        return d
    if threshold.kind != "top_s":
        raise ValueError("index-list storage is only defined for the top_s rule")
    return threshold.s * index_width(d)


def serialize(masks: BoolArray, fmt: StorageFormat, threshold: Threshold) -> Bits:
    """Serialise signature masks of shape (..., D) to bits of shape (..., L)."""
    if fmt == "bitmap":
        return masks.astype(np.uint8)
    d = masks.shape[-1]
    w = index_width(d)
    s = threshold.s
    counts = masks.sum(axis=-1)
    if not np.all(counts == s):
        raise ValueError("index-list storage requires exactly s active indices")
    # Stable argsort of ~mask puts the active indices first, in increasing order.
    idx: IntArray = np.argsort(~masks, axis=-1, kind="stable")[..., :s]
    shifts = np.arange(w - 1, -1, -1, dtype=np.int64)
    bits = (idx[..., None] >> shifts) & 1
    return bits.reshape(*masks.shape[:-1], s * w).astype(np.uint8)


def deserialize(
    bits: Bits, fmt: StorageFormat, d: int, threshold: Threshold
) -> BoolArray:
    """Read (possibly corrupted) stored bits of shape (..., L) back as masks (..., D).

    For index lists, out-of-range indices (>= D) are discarded and duplicates
    are merged.
    """
    if fmt == "bitmap":
        return np.asarray(bits, dtype=np.bool_)
    w = index_width(d)
    s = threshold.s
    groups = np.asarray(bits, dtype=np.int64).reshape(*bits.shape[:-1], s, w)
    weights = 1 << np.arange(w - 1, -1, -1, dtype=np.int64)
    values = (groups * weights).sum(axis=-1)  # (..., s)
    hits = values[..., :, None] == np.arange(d, dtype=np.int64)  # (..., s, D)
    result: BoolArray = hits.any(axis=-2)
    return result


# --------------------------------------------------------------------------- #
# Similarity and aggregation
# --------------------------------------------------------------------------- #


def similarity(u: BoolArray, v: BoolArray, kind: SimilarityKind) -> FloatArray:
    """Similarity of signature masks along the last axis (with broadcasting).

    Note that "overlap" is only meaningful when all signatures have the same
    size, i.e. for top_s. With other rules a superset scores higher than an
    exact match.
    """
    match kind:
        case "agreement":
            d = u.shape[-1]
            mismatches = np.logical_xor(u, v).sum(axis=-1).astype(np.float64)
            agreement: FloatArray = d - mismatches
            return agreement
        case "overlap":
            return np.logical_and(u, v).sum(axis=-1).astype(np.float64)
        case "jaccard":
            inter = np.logical_and(u, v).sum(axis=-1).astype(np.float64)
            union = np.logical_or(u, v).sum(axis=-1).astype(np.float64)
            out: FloatArray = np.ones_like(inter)
            np.divide(inter, union, out=out, where=union > 0)
            return out


def aggregate(sims: FloatArray, kind: AggregateKind, trim: float = 0.2) -> FloatArray:
    """Combine the per-juror similarities along the last (juror) axis.

    "median" and "trimmed" are scaled by J. That makes them robust estimates
    of the sum, so the proximity weight lambda has the same meaning under
    every aggregation rule.
    """
    j = sims.shape[-1]
    match kind:
        case "sum":
            return sims.sum(axis=-1)
        case "median":
            result: FloatArray = j * np.median(sims, axis=-1)
            return result
        case "trimmed":
            if not 0.0 <= trim < 0.5:
                raise ValueError("trim fraction must be in [0, 0.5)")
            k = math.floor(trim * j)
            ordered = np.sort(sims, axis=-1)
            kept = ordered[..., k : j - k]
            trimmed: FloatArray = j * kept.mean(axis=-1)
            return trimmed
