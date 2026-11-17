import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fuzzy_hashing_ecc import signatures as sg
from fuzzy_hashing_ecc.research import ResearchCode


def test_sign_and_level_rules() -> None:
    z = np.array([-1.0, 0.0, 0.5, 2.0])
    assert sg.Threshold("sign").apply(z).tolist() == [False, False, True, True]
    assert sg.Threshold("level", tau=1.0).apply(z).tolist() == [False, False, False, True]


def test_top_s_matches_reference_and_breaks_ties_low() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((50, 12))
    mask = sg.Threshold("top_s", s=3).apply(z)
    for row, m in zip(z, mask):
        assert set(np.flatnonzero(m)) == set(np.argsort(-row)[:3])
    tied = np.array([1.0, 1.0, 1.0, 0.0])
    assert np.flatnonzero(sg.Threshold("top_s", s=2).apply(tied)).tolist() == [0, 1]


@settings(max_examples=50)
@given(st.integers(2, 40), st.data())
def test_index_storage_round_trip(d: int, data: st.DataObject) -> None:
    s = data.draw(st.integers(1, d))
    th = sg.Threshold("top_s", s=s)
    z = np.array(data.draw(st.lists(st.floats(-5, 5), min_size=d, max_size=d)))
    mask = th.apply(z)
    bits = sg.serialize(mask, "index", th)
    assert bits.shape == (sg.signature_bits("index", d, th),)
    assert np.array_equal(sg.deserialize(bits, "index", d, th), mask)


def test_index_storage_discards_out_of_range_and_merges_duplicates() -> None:
    th = sg.Threshold("top_s", s=3)
    # D = 5 needs 3-bit indices. The stored values are 1, 1 and 7: 7 is out of range.
    bits = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    assert np.flatnonzero(sg.deserialize(bits, "index", 5, th)).tolist() == [1]


def test_bitmap_round_trip() -> None:
    mask = np.array([[True, False, True], [False, False, False]])
    th = sg.Threshold()
    assert np.array_equal(sg.deserialize(sg.serialize(mask, "bitmap", th), "bitmap", 3, th), mask)


def test_similarities() -> None:
    u = np.array([True, True, False, False])
    v = np.array([True, False, True, False])
    assert sg.similarity(u, v, "agreement") == 2.0
    assert sg.similarity(u, v, "overlap") == 1.0
    assert sg.similarity(u, v, "jaccard") == pytest.approx(1 / 3)
    empty = np.zeros(4, dtype=np.bool_)
    assert sg.similarity(empty, empty, "jaccard") == 1.0


def test_aggregation() -> None:
    sims = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert sg.aggregate(sims, "sum") == 110.0
    assert sg.aggregate(sims, "median") == 15.0  # J times the median
    assert sg.aggregate(sims, "trimmed", trim=0.2) == pytest.approx(5 * 3.0)


@pytest.mark.parametrize("family", ["orthogonal", "rademacher"])
def test_incremental_flip_matches_recomputation(family: str) -> None:
    code = ResearchCode(12, 5, family=family)  # type: ignore[arg-type]
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 12, dtype=np.uint8)
    fast = code.flip_projections(x, code.project(x))
    slow = code.project(x[None, :] ^ np.eye(12, dtype=np.uint8))
    assert np.allclose(fast, slow, atol=1e-12)
