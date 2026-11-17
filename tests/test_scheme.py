"""Tests of the public ConsensusCode, which is exactly the design in SPEC.md."""

import math

import numpy as np
import pytest

from fuzzy_hashing_ecc import BLOCK_BITS, ConsensusCode
from fuzzy_hashing_ecc.channel import bsc, flip_exactly
from fuzzy_hashing_ecc.experiments import (
    all_blocks,
    exact_min_distance,
    prop2_single_flip,
    single_flip_distances,
)
from fuzzy_hashing_ecc.research import ResearchCode
from fuzzy_hashing_ecc.signatures import Threshold

ALL = all_blocks(BLOCK_BITS)  # all 256 bytes as bit rows


def test_defaults_and_derived_quantities() -> None:
    code = ConsensusCode()
    assert (code.n, code.jurors, code.key) == (8, 20, 0)
    assert code.total_bits == 8 * 21
    assert code.overhead == 21
    assert code.rate == pytest.approx(1 / 21)
    assert code.codebook.shape == (256, 168)


def test_uses_rademacher_matrices_from_key() -> None:
    a = ConsensusCode(3, key=7)._impl.matrices
    assert a.shape == (3, 8, 8)
    assert set(np.unique(a)) == {-1.0, 1.0}
    assert np.array_equal(a, ConsensusCode(3, key=7)._impl.matrices)
    assert not np.array_equal(a, ConsensusCode(3, key=8)._impl.matrices)


@pytest.mark.parametrize("kwargs", [{"jurors": 0}, {"key": -1}])
def test_rejects_invalid_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ConsensusCode(**kwargs)


def _signature_bits(code: ResearchCode, bits: list[int]) -> list[int]:
    sig = code.signatures(np.array(bits, dtype=np.uint8))
    return [int(v) for v in sig[0]]


def test_spec_worked_example() -> None:
    """SPEC §II.4: one juror, n = 4."""
    a = np.array([[1, -1, 1, 1], [-1, -1, 1, -1], [1, 1, 1, -1], [-1, 1, 1, 1]], dtype=float)
    code = ResearchCode(4, 1)
    code.matrices = a[None, :, :]
    z = code.project(np.array([1, 0, 1, 0], dtype=np.uint8))[0]
    assert z.tolist() == [-2, -2, -2, 2]
    assert _signature_bits(code, [1, 0, 1, 0]) == [0, 0, 0, 1]
    assert _signature_bits(code, [1, 0, 1, 1]) == [0, 0, 0, 0]  # the juror notices
    assert _signature_bits(code, [1, 0, 0, 0]) == [0, 0, 0, 1]  # the juror misses it


def test_codebook_matches_encode() -> None:
    code = ConsensusCode(5, key=2)
    assert np.array_equal(code.codebook, code.encode(ALL))
    assert np.array_equal(code.codebook[:, :8], ALL)  # systematic


def test_noiseless_and_single_flip_all_bytes() -> None:
    code = ConsensusCode(10, key=3)
    records = code.encode(ALL)
    assert np.array_equal(code.decode_bits(records), ALL)
    for j in range(code.total_bits):  # one flip anywhere in the record
        hit = records.copy()
        hit[:, j] ^= 1
        assert np.array_equal(code.decode_bits(hit), ALL)


def test_batch_decoding_matches_single_and_exhaustive_search() -> None:
    code = ConsensusCode(6, key=4)
    rng = np.random.default_rng(0)
    received = bsc(code.encode(ALL[rng.integers(0, 256, 300)]), 0.2, rng)
    batch = code.decode_bits(received)
    for rec, x in zip(received, batch):
        assert np.array_equal(code.decode(rec).x, x)
        assert np.array_equal(code._impl.decode(rec, "exhaustive", radius=8).x, x)


def test_score_identity_and_proposition_1() -> None:
    """S(x̂) = J·n − d(E(x̂), read), and decoding returns the closest record."""
    code = ConsensusCode(6)
    rng = np.random.default_rng(2)
    for _ in range(20):
        x = rng.integers(0, 2, 8, dtype=np.uint8)
        received = flip_exactly(code.encode(x), 10, rng)
        dists = (code.codebook != received).sum(axis=1)
        assert np.allclose(code.score(ALL, received), code.jurors * 8 - dists)
        assert (code.encode(code.decode_bits(received)) != received).sum() == dists.min()


def test_corrects_up_to_t_errors() -> None:
    code = ConsensusCode(20, key=1)
    t = (exact_min_distance(code._impl) - 1) // 2
    assert t >= 10
    rng = np.random.default_rng(3)
    for _ in range(50):
        x = rng.integers(0, 2, 8, dtype=np.uint8)
        assert np.array_equal(code.decode_bits(flip_exactly(code.encode(x), t, rng)), x)


@pytest.mark.parametrize("n", [3, 4, 5, 8])
def test_proposition_2_exact_by_enumeration(n: int) -> None:
    """Averaging over all 2^n possible ±1 rows gives exactly p_n, for every x and j."""
    rows = 1.0 - 2.0 * all_blocks(n).astype(float)
    p_n = math.comb(n - 1, n // 2) / 2 ** (n - 1)
    for x in all_blocks(n)[:: max(1, 2**n // 8)]:
        b = 1.0 - 2.0 * x
        for j in range(n):
            b2 = b.copy()
            b2[j] *= -1
            changed = ((rows @ b) > 0) != ((rows @ b2) > 0)
            assert changed.mean() == pytest.approx(p_n)
    assert prop2_single_flip(n) == pytest.approx(n * p_n)


def test_proposition_2_monte_carlo_total() -> None:
    code = ConsensusCode(400, key=11)
    changed = single_flip_distances(code._impl, 30, np.random.default_rng(1)) - 1
    assert changed.mean() / 400 == pytest.approx(prop2_single_flip(8), rel=0.02)


@pytest.mark.parametrize("key", range(4))
def test_proposition_3_ceiling_spec_design(key: int) -> None:
    code = ConsensusCode(20, key=key)
    assert exact_min_distance(code._impl) <= 1 + code.jurors * math.sqrt(8)


@pytest.mark.parametrize(
    "variant",
    [
        ResearchCode(8, 5, family="orthogonal", seed=1),
        ResearchCode(8, 5, threshold=Threshold("level", tau=2.0), seed=2),
        ResearchCode(8, 5, dim=12, seed=3),
    ],
    ids=["orthogonal", "level", "D>n"],
)
def test_proposition_3_average_bound_any_matrices(variant: ResearchCode) -> None:
    """The proof's key step: the exact average single-flip distance is <= 1 + J·D/√n."""
    base = variant.encode(ALL)
    eye = np.eye(8, dtype=np.uint8)
    flipped = variant.encode(ALL[:, None, :] ^ eye[None, :, :])
    average = (flipped != base[:, None, :]).sum(axis=-1).mean()
    assert average <= 1 + variant.jurors * variant.d / math.sqrt(8)
