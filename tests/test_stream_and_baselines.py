from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fuzzy_hashing_ecc import main
from fuzzy_hashing_ecc.baselines import RepetitionCode, matched_repetition
from fuzzy_hashing_ecc.channel import bsc, flip_exactly
from fuzzy_hashing_ecc.scheme import ConsensusCode
from fuzzy_hashing_ecc.stream import decode_bytes, encode_bytes

CODE = ConsensusCode(8)


@pytest.mark.parametrize("data", [b"", b"a", b"hello world", bytes(range(256))])
def test_stream_round_trip(data: bytes) -> None:
    assert decode_bytes(encode_bytes(data, CODE), CODE) == data


@settings(max_examples=25, deadline=None)
@given(st.binary(max_size=40))
def test_stream_round_trip_with_repetition(data: bytes) -> None:
    rep = RepetitionCode(8, 3)
    assert decode_bytes(encode_bytes(data, rep), rep) == data


def test_stream_survives_one_flip_per_block() -> None:
    data = b"radiation-hardened"
    blob = np.unpackbits(np.frombuffer(encode_bytes(data, CODE), dtype=np.uint8))
    rng = np.random.default_rng(0)
    for start in range(0, blob.size - CODE.total_bits + 1, CODE.total_bits):
        blob[start + int(rng.integers(CODE.n))] ^= 1  # hit the data bits
    assert decode_bytes(np.packbits(blob).tobytes(), CODE) == data


def test_truncated_stream_raises() -> None:
    blob = encode_bytes(b"hello", CODE)
    with pytest.raises(ValueError):
        decode_bytes(blob[:10], CODE)


def test_repetition_corrects_up_to_t_per_bit() -> None:
    rep = RepetitionCode(4, 5)
    x = np.array([1, 0, 1, 1], dtype=np.uint8)
    rec = rep.encode(x)
    rec[0:2] ^= 1  # two of bit 0's copies
    rec[5:7] ^= 1  # two of bit 1's copies
    assert np.array_equal(rep.decode_bits(rec), x)
    rec[2] ^= 1  # a third copy of bit 0 is too many
    assert not np.array_equal(rep.decode_bits(rec), x)


def test_repetition_exact_bler_matches_simulation() -> None:
    rep = RepetitionCode(4, 3)
    rng = np.random.default_rng(1)
    x = np.zeros(4, dtype=np.uint8)
    trials = 20000
    errs = sum(
        not np.array_equal(rep.decode_bits(bsc(rep.encode(x), 0.2, rng)), x)
        for _ in range(trials)
    )
    assert errs / trials == pytest.approx(rep.block_error_rate(0.2), abs=0.015)


def test_matched_repetition_never_exceeds_budget() -> None:
    for total in range(16, 400, 7):
        rep = matched_repetition(16, total)
        assert rep.r % 2 == 1
        assert rep.total_bits <= max(total, 16)


def test_flip_exactly_weight() -> None:
    bits = np.zeros(50, dtype=np.uint8)
    assert flip_exactly(bits, 7, np.random.default_rng(0)).sum() == 7


def test_cli_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "in.bin"
    src.write_bytes(b"The quick brown fox jumps over the lazy dog.")
    enc, noisy, dec = tmp_path / "enc", tmp_path / "noisy", tmp_path / "dec"
    main(["encode", str(src), str(enc)])
    main(["noise", str(enc), str(noisy), "--p", "0.05", "--noise-seed", "1"])
    main(["decode", str(noisy), str(dec)])
    assert dec.read_bytes() == src.read_bytes()
