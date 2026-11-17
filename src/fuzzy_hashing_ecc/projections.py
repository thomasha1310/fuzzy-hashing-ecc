"""Projection matrices and the bipolar mapping."""

from typing import Literal

import numpy as np

from ._types import Bits, FloatArray

type MatrixFamily = Literal["orthogonal", "rademacher"]


def bipolar(x: Bits) -> FloatArray:
    """Map bits to the bipolar domain: b(x)_j = (-1)^{x_j} = 1 - 2 x_j."""
    return 1.0 - 2.0 * np.asarray(x, dtype=np.float64)


def random_orthogonal(n: int, rng: np.random.Generator) -> FloatArray:
    """Draw a Haar-distributed matrix from O(n).

    QR-factor a Gaussian matrix, then fix the signs of Q's columns using the
    diagonal of R (Mezzadri 2007). Without that sign fix the distribution is
    not Haar.
    """
    g = rng.standard_normal((n, n))
    q, r = np.linalg.qr(g)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    result: FloatArray = q * signs
    return result


def random_rademacher(n: int, d: int, rng: np.random.Generator) -> FloatArray:
    """Draw a D x n matrix with i.i.d. uniform ±1 entries.

    The values are stored as float64, but every product with a bipolar vector
    is an integer smaller than 2^53. So the arithmetic is exact and gives
    bit-identical results on every platform.
    """
    return rng.choice(np.array([-1.0, 1.0]), size=(d, n))


def generate_matrices(
    family: MatrixFamily, n: int, d: int, jurors: int, seed: int
) -> FloatArray:
    """Regenerate the J projection matrices from the seed. Shape: (J, D, n)."""
    if family == "orthogonal" and d != n:
        raise ValueError(f"orthogonal family requires D == n (got D={d}, n={n})")
    rng = np.random.default_rng(seed)
    if family == "orthogonal":
        mats = [random_orthogonal(n, rng) for _ in range(jurors)]
    else:
        mats = [random_rademacher(n, d, rng) for _ in range(jurors)]
    return np.stack(mats).astype(np.float64)
