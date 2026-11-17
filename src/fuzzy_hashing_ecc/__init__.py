"""Fuzzy-hash consensus error correction: a proof of concept (see docs/SPEC.md)."""

import argparse
import sys
from pathlib import Path

import numpy as np

from .baselines import RepetitionCode, matched_repetition
from .channel import bsc
from .research import DecodeResult
from .scheme import BLOCK_BITS, ConsensusCode
from .stream import decode_bytes, encode_bytes

__all__ = [
    "BLOCK_BITS",
    "ConsensusCode",
    "DecodeResult",
    "RepetitionCode",
    "bsc",
    "decode_bytes",
    "encode_bytes",
    "main",
    "matched_repetition",
]


def _add_code_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group(
        "code settings (SPEC §I; encode and decode must use the same values)"
    )
    g.add_argument(
        "--jurors", type=int, default=20,
        help="signatures per byte; storage cost is (jurors + 1)x (default 20)",
    )
    g.add_argument("--key", type=int, default=0, help="seed for the secret matrices (default 0)")


def _build_code(a: argparse.Namespace) -> ConsensusCode:
    try:
        return ConsensusCode(jurors=a.jurors, key=a.key)
    except ValueError as err:
        sys.exit(f"error: {err}")


def _cmd_encode(a: argparse.Namespace) -> None:
    code = _build_code(a)
    out = encode_bytes(Path(a.input).read_bytes(), code)
    Path(a.output).write_bytes(out)
    print(f"{code.describe()}\nwrote {len(out)} bytes to {a.output}")


def _cmd_decode(a: argparse.Namespace) -> None:
    code = _build_code(a)
    try:
        out = decode_bytes(Path(a.input).read_bytes(), code)
    except ValueError as err:
        sys.exit(
            f"error: {err}. Check that --jurors and --key match the values "
            "used to encode."
        )
    Path(a.output).write_bytes(out)
    print(f"wrote {len(out)} bytes to {a.output}")


def _cmd_noise(a: argparse.Namespace) -> None:
    data = np.frombuffer(Path(a.input).read_bytes(), dtype=np.uint8)
    bits = np.unpackbits(data)
    noisy = bsc(bits, a.p, np.random.default_rng(a.noise_seed))
    Path(a.output).write_bytes(np.packbits(noisy).tobytes())
    print(f"flipped {int((noisy != bits).sum())} of {bits.size} bits (p={a.p})")


def _cmd_demo(a: argparse.Namespace) -> None:
    from . import experiments

    print(experiments.toy_example())
    if a.toy_only:
        return
    for table in experiments.run_all(quick=True):
        print()
        print(table.render())


def _cmd_experiments(a: argparse.Namespace) -> None:
    from . import experiments as ex

    names = a.only or ["q1", "e1", "e2", "e3", "e4"]
    for name in names:
        match name:
            case "q1":
                print(ex.q1_min_distance(samples=a.samples).render())
            case "e1":
                print(ex.e1_single_flip(samples=a.samples).render())
            case "e2":
                print(ex.e2_errors(trials=a.trials).render())
            case "e3":
                table, curves = ex.e3_bler(trials=a.trials)
                print(table.render())
                if a.plot:
                    try:
                        from .plotting import plot_bler
                    except ImportError:
                        print("matplotlib is not installed: uv sync --extra plot", file=sys.stderr)
                    else:
                        path = plot_bler(curves, Path(a.plot), "E3: block error rate at matched storage overhead")
                        print(f"  plot written to {path}")
            case "e4":
                print(ex.e4_jury(trials=a.trials).render())
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fuzzy-hashing-ecc",
        description="Fuzzy-hash consensus error correction (see docs/SPEC.md).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("encode", help="encode a file into a protected stream")
    p.add_argument("input")
    p.add_argument("output")
    _add_code_args(p)
    p.set_defaults(func=_cmd_encode)

    p = sub.add_parser("decode", help="decode a (possibly corrupted) stream")
    p.add_argument("input")
    p.add_argument("output")
    _add_code_args(p)
    p.set_defaults(func=_cmd_decode)

    p = sub.add_parser("noise", help="flip every bit of a file with probability p")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--p", type=float, required=True)
    p.add_argument("--noise-seed", type=int, default=None)
    p.set_defaults(func=_cmd_noise)

    p = sub.add_parser("demo", help="toy example plus quick versions of every experiment")
    p.add_argument("--toy-only", action="store_true")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("experiments", help="run the SPEC §VI experiments at full size")
    p.add_argument("--only", nargs="+", choices=["q1", "e1", "e2", "e3", "e4"])
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--plot", metavar="PNG", help="write the E3 plot to this path")
    p.set_defaults(func=_cmd_experiments)

    args = parser.parse_args(argv)
    args.func(args)
