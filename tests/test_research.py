import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fuzzy_hashing_ecc.channel import flip_exactly
from fuzzy_hashing_ecc.experiments import (
    all_blocks,
    exact_min_distance,
    gv_distance,
    haar_single_flip,
    single_flip_distances,
)
from fuzzy_hashing_ecc.research import ResearchCode, Strategy
from fuzzy_hashing_ecc.signatures import Threshold

"""Tests of the research variants. The SPEC design itself is tested in test_scheme.py."""

CONFIGS = [
    ResearchCode(10),
    ResearchCode(10, family="orthogonal"),
    ResearchCode(10, threshold=Threshold("top_s", s=3)),
    ResearchCode(10, threshold=Threshold("top_s", s=3), storage="index", similarity="overlap"),
    ResearchCode(10, similarity="jaccard", aggregate="median"),
]
STRATEGIES: list[Strategy] = ["greedy", "parallel_vote", "exhaustive"]


def test_dimensions() -> None:
    code = ResearchCode(16)
    assert code.sig_bits == 16
    assert code.total_bits == 16 + 20 * 16
    assert code.overhead == pytest.approx(21.0)
    x = np.zeros(16, dtype=np.uint8)
    rec = code.encode(x)
    assert rec.shape == (code.total_bits,)
    assert np.array_equal(rec[:16], x)  # the code is systematic


@pytest.mark.parametrize("code", CONFIGS, ids=lambda c: c.describe())
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_noiseless_round_trip(code: ResearchCode, strategy: Strategy) -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.integers(0, 2, code.n, dtype=np.uint8)
        result = code.decode(code.encode(x), strategy)
        assert np.array_equal(result.x, x)
        assert result.flips == 0


@pytest.mark.parametrize("code", CONFIGS[:3], ids=lambda c: c.describe())
@pytest.mark.parametrize("strategy", ["greedy", "exhaustive"])
def test_single_data_flip_is_corrected(code: ResearchCode, strategy: Strategy) -> None:
    rng = np.random.default_rng(1)
    for j in range(code.n):
        x = rng.integers(0, 2, code.n, dtype=np.uint8)
        rec = code.encode(x)
        rec[j] ^= 1
        assert np.array_equal(code.decode(rec, strategy, radius=1).x, x)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**8 - 1))
def test_score_is_maximised_at_true_block(value: int) -> None:
    code = ResearchCode(8, 6)
    x = all_blocks(8)[value]
    x_read, stored = code.split(code.encode(x))
    scores = code.score(all_blocks(8), x_read, stored)
    assert int(np.argmax(scores)) == value
    assert np.sum(scores == scores.max()) == 1


def test_exhaustive_full_radius_is_minimum_distance_decoding() -> None:
    """SPEC Proposition 1: S(x̂) = J·D - d_H(E(x̂), received)."""
    code = ResearchCode(8, 6)
    rng = np.random.default_rng(2)
    everything = code.encode(all_blocks(8))
    for _ in range(20):
        x = rng.integers(0, 2, 8, dtype=np.uint8)
        received = flip_exactly(code.encode(x), 6, rng)
        x_read, stored = code.split(received)
        scores = code.score(all_blocks(8), x_read, stored)
        dists = (everything != received).sum(axis=1)
        assert np.allclose(scores, code.jurors * code.d - dists)
        result = code.decode(received, "exhaustive", radius=8)
        assert (code.encode(result.x) != received).sum() == dists.min()


def test_errors_within_t_are_corrected_by_ml() -> None:
    code = ResearchCode(8, 6)
    t = (exact_min_distance(code) - 1) // 2
    assert t >= 1
    rng = np.random.default_rng(3)
    for _ in range(30):
        x = rng.integers(0, 2, 8, dtype=np.uint8)
        received = flip_exactly(code.encode(x), t, rng)
        assert np.array_equal(code.decode(received, "exhaustive", radius=8).x, x)


def test_exact_min_distance_bounded_by_single_flip_sample() -> None:
    code = ResearchCode(8, 4)
    flips = single_flip_distances(code, 50, np.random.default_rng(0))
    assert exact_min_distance(code) <= int(flips.min())


def test_haar_single_flip_formula_monte_carlo() -> None:
    n, trials = 16, 3000
    code = ResearchCode(n, trials, family="orthogonal", seed=5)  # many Haar matrices
    flips = single_flip_distances(code, 4, np.random.default_rng(5)) - 1
    per_juror = flips.mean() / trials
    assert per_juror == pytest.approx(haar_single_flip(n, n), rel=0.03)


def test_gv_distance_small_cases() -> None:
    # [7, 4]: Varshamov guarantees d = 3 (achieved by the Hamming code).
    assert gv_distance(7, 4) == 3
    # [n, 1]: the condition holds up to d = n (the repetition code).
    assert gv_distance(9, 1) == 9
    assert gv_distance(30, 30) == 1
    assert math.comb(5, 2) == 10  # sanity check of the helper used above
