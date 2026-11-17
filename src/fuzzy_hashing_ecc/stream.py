"""Byte-stream framing.

Layout: [length header blocks][data blocks]. Every block is an encoded record
of `total_bits` bits; the concatenation is packed MSB-first into bytes and
zero-padded at the end. The 64-bit big-endian length header is encoded with
the same code as the data, so it is protected in exactly the same way.
"""

import math
from typing import Protocol

import numpy as np

from ._types import Bits

HEADER_BITS = 64


class BlockCode(Protocol):
    """Any code that maps n-bit blocks to total_bits-bit records and back.

    Both methods work on batches: encode maps (..., n) to (..., total_bits),
    and decode_bits maps (..., total_bits) to (..., n).
    """

    @property
    def n(self) -> int: ...

    @property
    def total_bits(self) -> int: ...

    def encode(self, x: Bits) -> Bits: ...

    def decode_bits(self, record: Bits) -> Bits: ...


def _to_blocks(bits: Bits, n: int) -> Bits:
    pad = (-bits.size) % n
    padded = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return padded.reshape(-1, n)


def _header_bits(length: int) -> Bits:
    return np.unpackbits(np.frombuffer(length.to_bytes(8, "big"), dtype=np.uint8))


def encode_bytes(data: bytes, code: BlockCode) -> bytes:
    """Encode arbitrary bytes into a protected byte stream."""
    payload = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    blocks = np.concatenate(
        [_to_blocks(_header_bits(len(data)), code.n), _to_blocks(payload, code.n)]
    )
    records = code.encode(blocks)
    return np.packbits(records.reshape(-1)).tobytes()


def decode_bytes(blob: bytes, code: BlockCode) -> bytes:
    """Decode a (possibly corrupted) stream produced by `encode_bytes`."""
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
    rec = code.total_bits
    n_header = math.ceil(HEADER_BITS / code.n)

    def blocks(first: int, count: int) -> Bits:
        """Decode `count` consecutive blocks in one batch, flattened to bits."""
        if (first + count) * rec > bits.size:
            raise ValueError("stream is truncated")
        records = bits[first * rec : (first + count) * rec].reshape(count, rec)
        return code.decode_bits(records).reshape(-1)

    header = blocks(0, n_header)[:HEADER_BITS]
    length = int.from_bytes(np.packbits(header).tobytes(), "big")
    n_data = math.ceil(8 * length / code.n)
    if (n_header + n_data) * rec > bits.size:
        raise ValueError(f"decoded length {length} exceeds the stream size")
    if n_data == 0:
        return b""
    payload = blocks(n_header, n_data)
    return np.packbits(payload[: 8 * length]).tobytes()
