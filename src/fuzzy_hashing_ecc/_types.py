"""Shared array type aliases."""

import numpy as np
from numpy.typing import NDArray

type Bits = NDArray[np.uint8]
"""Array of bits, each entry 0 or 1."""

type FloatArray = NDArray[np.float64]
type Float32Array = NDArray[np.float32]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.int64]
