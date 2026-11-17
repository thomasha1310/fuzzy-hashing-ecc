import numpy as np
import pytest

from fuzzy_hashing_ecc.projections import bipolar, generate_matrices, random_orthogonal


@pytest.mark.parametrize("n", [1, 2, 7, 16, 64])
def test_orthogonal_matrix_is_orthogonal(n: int) -> None:
    q = random_orthogonal(n, np.random.default_rng(n))
    assert np.allclose(q @ q.T, np.eye(n), atol=1e-12)


def test_matrices_regenerate_from_seed() -> None:
    a = generate_matrices("orthogonal", 8, 8, 5, seed=42)
    b = generate_matrices("orthogonal", 8, 8, 5, seed=42)
    c = generate_matrices("orthogonal", 8, 8, 5, seed=43)
    assert a.shape == (5, 8, 8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rademacher_entries_and_shape() -> None:
    m = generate_matrices("rademacher", 6, 10, 3, seed=0)
    assert m.shape == (3, 10, 6)
    assert set(np.unique(m)) <= {-1.0, 1.0}


def test_orthogonal_requires_square() -> None:
    with pytest.raises(ValueError):
        generate_matrices("orthogonal", 8, 9, 1, seed=0)


def test_bipolar_mapping() -> None:
    x = np.array([0, 1, 1, 0], dtype=np.uint8)
    assert bipolar(x).tolist() == [1.0, -1.0, -1.0, 1.0]
