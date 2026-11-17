"""Experiments answering the open questions in SPEC §VI (Q1, E1–E4).

Every experiment returns a `Table`, which can be printed. The tables are
reproducible: each one takes an explicit seed for the noise, and the code's
matrices come from the code's own seed.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from ._types import Bits
from .baselines import matched_repetition
from .channel import bsc, destroy_signatures, flip_exactly
from .research import ResearchCode, Strategy
from .signatures import AggregateKind, Threshold

type Cell = str | int | float


@dataclass
class Table:
    """A titled table of results, with optional notes printed beneath it."""

    title: str
    headers: list[str]
    rows: list[list[Cell]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        def fmt(c: Cell) -> str:
            if isinstance(c, float):
                return f"{c:.4g}"
            return str(c)

        cells = [[fmt(c) for c in row] for row in self.rows]
        widths = [
            max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
            for i, h in enumerate(self.headers)
        ]
        line = "  ".join(h.rjust(w) for h, w in zip(self.headers, widths))
        out = [self.title, "=" * len(self.title), line, "-" * len(line)]
        out += ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in cells]
        out += [f"  * {note}" for note in self.notes]
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #


def standard_configs(n: int, jurors: int = 20, seed: int = 0) -> list[ResearchCode]:
    """The configurations compared in Q1 and E1. The first one is the SPEC design."""
    s = max(1, n // 4)
    return [
        ResearchCode(n, jurors, seed=seed),
        ResearchCode(n, jurors, family="orthogonal", seed=seed),
        ResearchCode(
            n, jurors, family="orthogonal", threshold=Threshold("top_s", s=s), seed=seed
        ),
        ResearchCode(
            n,
            jurors,
            family="orthogonal",
            threshold=Threshold("top_s", s=s),
            storage="index",
            similarity="overlap",
            seed=seed,
        ),
    ]


def short_name(code: ResearchCode) -> str:
    name = f"{code.family[:5]}/{code.threshold.describe()}/{code.storage}"
    return f"{name} (spec)" if is_spec_design(code) else name


def is_spec_design(code: ResearchCode) -> bool:
    return (
        code.family == "rademacher"
        and code.threshold.kind == "sign"
        and code.storage == "bitmap"
        and code.d == code.n
    )


def prop2_single_flip(n: int) -> float:
    """SPEC Proposition 2 (Rademacher, sign rule): expected bits changed per juror, n·p_n."""
    return float(n * math.comb(n - 1, n // 2)) / float(2 ** (n - 1))


def haar_single_flip(n: int, d: int) -> float:
    """The same quantity for Haar-orthogonal matrices: D·arccos(1 - 2/n)/π.

    This follows from the SimHash collision probability θ/π (Charikar 2002).
    """
    return d * math.acos(1.0 - 2.0 / n) / math.pi


def ceiling(code: ResearchCode) -> int | None:
    """SPEC Proposition 4: d_min <= 1 + J·D/√n for threshold-type signatures."""
    if code.threshold.kind == "top_s" or code.storage != "bitmap":
        return None
    return math.floor(1 + code.jurors * code.d / math.sqrt(code.n))


def all_blocks(n: int) -> Bits:
    """Every x in {0,1}^n as rows, MSB first. Shape (2^n, n)."""
    vals = np.arange(2**n, dtype=np.int64)[:, None]
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    return ((vals >> shifts) & 1).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Q1: minimum distance
# --------------------------------------------------------------------------- #


def exact_min_distance(code: ResearchCode) -> int:
    """Exact d_min(C) by comparing all 2^n codewords (use only for n <= 12)."""
    if code.n > 12:
        raise ValueError("exact minimum distance is limited to n <= 12")
    pm = 1.0 - 2.0 * code.encode(all_blocks(code.n)).astype(np.float32)
    gram = pm @ pm.T  # entry (a, b) = L_tot - 2 d_H(a, b)
    np.fill_diagonal(gram, -np.inf)
    return int(round((code.total_bits - gram.max()) / 2))


def single_flip_distances(
    code: ResearchCode, samples: int, rng: np.random.Generator
) -> Bits:
    """Record distances d_H(E(x), E(x ⊕ e_j)) for random x and every j.

    Returns a (samples, n) integer array. The minimum is an upper bound on d_min.
    """
    xs = rng.integers(0, 2, (samples, code.n), dtype=np.uint8)
    base = code.encode(xs)  # (S, L)
    eye = np.eye(code.n, dtype=np.uint8)
    flipped = code.encode(xs[:, None, :] ^ eye[None, :, :])  # (S, n, L)
    dist: Bits = (flipped != base[:, None, :]).sum(axis=-1)
    return dist


def gv_distance(length: int, k: int) -> int:
    """Largest d that the Gilbert–Varshamov bound guarantees for a linear [length, k] code.

    Varshamov's condition: a linear [N, k, d] code exists if
    sum_{i=0}^{d-2} C(N-1, i) < 2^{N-k}.
    """
    budget = 2 ** (length - k)
    total = 0
    d = 1
    while d - 1 <= length - 1:
        total += math.comb(length - 1, d - 1)  # adds the i = d - 1 term
        if total >= budget:
            break
        d += 1
    return d


def q1_min_distance(
    ns: Sequence[int] = (8, 10, 12, 16, 32),
    samples: int = 400,
    jurors: int = 20,
    seed: int = 0,
) -> Table:
    table = Table(
        "Q1  Minimum distance vs repetition (same storage) and the Gilbert–Varshamov bound",
        ["n", "config", "L_tot", "ρ", "rep d", "d_min", "ceiling", "GV d", "mean 1-flip dist", "t"],
    )
    rng = np.random.default_rng(seed)
    for n in ns:
        for code in standard_configs(n, jurors):
            rep = matched_repetition(n, code.total_bits)
            flips = single_flip_distances(code, samples, rng)
            if n <= 12:
                dmin = exact_min_distance(code)
                dmin_cell: Cell = dmin
            else:
                dmin = int(flips.min())
                dmin_cell = f"≤{dmin}"
            table.rows.append(
                [
                    n,
                    short_name(code),
                    code.total_bits,
                    code.overhead,
                    rep.min_distance,
                    dmin_cell,
                    cap if (cap := ceiling(code)) is not None else "-",
                    gv_distance(code.total_bits, n),
                    float(flips.mean()),
                    (dmin - 1) // 2,
                ]
            )
    table.notes += [
        "d_min is exact for n <= 12 (all pairs). For larger n it is the minimum over "
        f"{samples}·n sampled single-flip pairs, which is only an upper bound.",
        "ceiling: SPEC Proposition 4, d_min <= 1 + J·D/√n, valid for any matrices "
        "with threshold signatures (not defined for top-s).",
        "rep d = r: the minimum distance of the repetition code with odd r and n·r <= L_tot.",
        "GV d: a linear code with the same length L_tot and n data bits, and at least "
        "this minimum distance, is guaranteed to exist.",
        "t = floor((d_min-1)/2): errors per block that are guaranteed to be corrected "
        "under ML decoding.",
    ]
    return table


# --------------------------------------------------------------------------- #
# E1: single-flip detectability
# --------------------------------------------------------------------------- #


def e1_single_flip(
    ns: Sequence[int] = (8, 16, 32, 64),
    samples: int = 400,
    jurors: int = 20,
    seed: int = 1,
) -> Table:
    table = Table(
        "E1  Signature bits changed by flipping one data bit",
        ["n", "config", "per juror", "predicted", "total (J)", "5th pct", "P(total=0)"],
    )
    rng = np.random.default_rng(seed)
    for n in ns:
        for code in standard_configs(n, jurors):
            sig_change = single_flip_distances(code, samples, rng) - 1
            prediction: Cell = "-"
            if is_spec_design(code):
                prediction = prop2_single_flip(n)
            elif code.family == "orthogonal" and code.threshold.kind == "sign":
                prediction = haar_single_flip(n, code.d)
            table.rows.append(
                [
                    n,
                    short_name(code),
                    float(sig_change.mean()) / jurors,
                    prediction,
                    float(sig_change.mean()),
                    float(np.percentile(sig_change, 5)),
                    float((sig_change == 0).mean()),
                ]
            )
    table.notes += [
        "predicted (spec design): SPEC Proposition 2, n·C(n-1, ⌊n/2⌋)/2^(n-1).",
        "predicted (orthogonal/sign): the SimHash formula D·arccos(1-2/n)/π.",
    ]
    return table


# --------------------------------------------------------------------------- #
# E2: correction rate vs number of data errors
# --------------------------------------------------------------------------- #

type Decoder = tuple[str, Callable[[Bits], Bits]]


def consensus_decoders(code: ResearchCode, exhaustive_radius: int) -> list[Decoder]:
    def make(strategy: Strategy, radius: int) -> Callable[[Bits], Bits]:
        return lambda rec: code.decode(rec, strategy, radius).x

    decoders: list[Decoder] = [
        ("greedy", make("greedy", 0)),
        ("parallel", make("parallel_vote", 0)),
    ]
    label = "exact (spec)" if exhaustive_radius >= code.n else f"exh(r={exhaustive_radius})"
    decoders.append((label, make("exhaustive", exhaustive_radius)))
    return decoders


def e2_errors(
    n: int = 8,
    ws: Sequence[int] = (0, 1, 2, 3, 4),
    trials: int = 200,
    exhaustive_radius: int = 8,
    seed: int = 2,
) -> Table:
    code = ResearchCode(n)
    decoders = consensus_decoders(code, exhaustive_radius)
    headers = ["w"]
    for name, _ in decoders:
        headers += [f"{name} clean", f"{name} noisy"]
    table = Table(
        f"E2  Fraction of blocks decoded correctly with exactly w data-bit errors ({code.describe()})",
        headers,
    )
    rng = np.random.default_rng(seed)
    for w in ws:
        hits = np.zeros((len(decoders), 2))
        for _ in range(trials):
            x = rng.integers(0, 2, n, dtype=np.uint8)
            rec = code.encode(x)
            clean = rec.copy()
            clean[:n] = flip_exactly(rec[:n], w, rng)
            noisy = clean.copy()
            noisy[n:] = bsc(rec[n:], w / n, rng)
            for k, (_, dec) in enumerate(decoders):
                hits[k, 0] += np.array_equal(dec(clean), x)
                hits[k, 1] += np.array_equal(dec(noisy), x)
        rates = (hits / trials).reshape(-1)
        table.rows.append([w, *(float(v) for v in rates)])
    table.notes += [
        "clean: signatures intact. noisy: signature bits also flipped at rate w/n "
        "(the same bit-error rate as the data).",
        "greedy and parallel are research decoders. With ±1 matrices, pairs of data "
        "errors often cancel inside a projection, so fixing one error at a time gives "
        "greedy no signal (SPEC §VI).",
        f"{trials} trials per row.",
    ]
    return table


# --------------------------------------------------------------------------- #
# E3: block error rate on a BSC, matched overhead
# --------------------------------------------------------------------------- #


@dataclass
class BlerCurve:
    label: str
    ps: list[float]
    bler: list[float]


def e3_bler(
    n: int = 8,
    ps: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    trials: int = 300,
    seed: int = 3,
) -> tuple[Table, list[BlerCurve]]:
    code = ResearchCode(n)
    rep = matched_repetition(n, code.total_bits)
    decoders = consensus_decoders(code, n if n <= 12 else 3)
    decoders.append((f"repetition r={rep.r}", rep.decode_bits))
    curves = [BlerCurve(name, list(ps), []) for name, _ in decoders]
    analytic = BlerCurve(f"repetition r={rep.r} (exact)", list(ps), [])
    rng = np.random.default_rng(seed)
    for p in ps:
        errors = np.zeros(len(decoders))
        for _ in range(trials):
            x = rng.integers(0, 2, n, dtype=np.uint8)
            for k, (name, dec) in enumerate(decoders):
                rec = rep.encode(x) if name.startswith("repetition") else code.encode(x)
                errors[k] += not np.array_equal(dec(bsc(rec, p, rng)), x)
        for k, curve in enumerate(curves):
            curve.bler.append(float(errors[k] / trials))
        analytic.bler.append(rep.block_error_rate(p))
    curves.append(analytic)
    table = Table(
        f"E3  Block error rate on a binary symmetric channel at matched overhead ({code.describe()}; "
        f"repetition uses {rep.total_bits} ≤ {code.total_bits} bits)",
        ["p", *(c.label for c in curves)],
        [[p, *(c.bler[i] for c in curves)] for i, p in enumerate(ps)],
    )
    table.notes += [
        "Every stored bit (data and signatures) passes through the same BSC.",
        f"{trials} trials per point; with 0 errors observed, the 95% upper bound is ≈ {3 / trials:.3g}.",
    ]
    return table, curves


# --------------------------------------------------------------------------- #
# E4: jury size and juror loss
# --------------------------------------------------------------------------- #


def e4_jury(
    n: int = 8,
    jury_sizes: Sequence[int] = (1, 5, 10, 20, 40),
    w: int = 2,
    trials: int = 200,
    seed: int = 4,
) -> Table:
    aggs: list[AggregateKind] = ["sum", "median", "trimmed"]
    headers = ["J", "ρ", "destroyed"]
    headers += [f"{a}" + (" (spec)" if a == "sum" else "") for a in aggs]
    table = Table(
        f"E4  Correct decodes with w={w} data errors, as jurors are added or destroyed "
        f"(exact search, n={n})",
        headers,
    )
    rng = np.random.default_rng(seed)
    for j in jury_sizes:
        codes = [replace(ResearchCode(n, j), aggregate=a) for a in aggs]
        for destroyed in sorted({0, j // 4}):
            hits = np.zeros(len(aggs))
            for _ in range(trials):
                x = rng.integers(0, 2, n, dtype=np.uint8)
                rec = codes[0].encode(x)
                rec[:n] = flip_exactly(rec[:n], w, rng)
                victims = list(rng.choice(j, size=destroyed, replace=False))
                rec = destroy_signatures(rec, n, codes[0].sig_bits, victims, rng)
                for k, code in enumerate(codes):
                    hits[k] += np.array_equal(code.decode(rec, "exhaustive", n).x, x)
            table.rows.append(
                [j, codes[0].overhead, destroyed, *(float(h / trials) for h in hits)]
            )
    table.notes.append(
        "destroyed: that many jurors' stored signatures are replaced with random bits."
    )
    return table


# --------------------------------------------------------------------------- #
# Toy example
# --------------------------------------------------------------------------- #


def toy_example(jurors: int = 20, seed: int = 7) -> str:
    """One byte with one flipped bit, showing every bit's juror votes."""
    from .scheme import ConsensusCode

    rng = np.random.default_rng(seed)
    public = ConsensusCode(jurors)
    code = public._impl
    n = public.n
    x = rng.integers(0, 2, n, dtype=np.uint8)
    rec = code.encode(x)
    bad = int(rng.integers(n))
    corrupted = rec.copy()
    corrupted[bad] ^= 1
    x_read, stored = code.split(corrupted)
    cur_juror, cur, flip_juror, flip_scores = code.flip_evaluation(
        x_read, x_read, stored
    )
    votes = (flip_juror > cur_juror[None, :]).sum(axis=1)
    against = (flip_juror < cur_juror[None, :]).sum(axis=1)
    result = public.decode(corrupted)

    def show(bits: Bits) -> str:
        return "".join(str(int(b)) for b in bits)

    lines = [
        f"Toy example: one byte, {jurors} jurors, one bit flipped by 'radiation'",
        "=" * 66,
        f"code:      {public.describe()}",
        f"original:  {show(x)}",
        f"read back: {show(x_read)}   (bit {bad} flipped)",
        f"score of read-back block S = {float(cur):.0f} (max possible {jurors * code.d})",
        "",
        "bit  jurors for flipping  jurors against  score after flip",
    ]
    for j in range(n):
        mark = "  <- corrupted bit" if j == bad else ""
        lines.append(
            f"{j:>3}  {int(votes[j]):>19}  {int(against[j]):>14}  {float(flip_scores[j]):>16.0f}{mark}"
        )
    lines += [
        "",
        f"exact decode:  {show(result.x)}   flips={result.flips}  "
        f"{'CORRECT' if np.array_equal(result.x, x) else 'WRONG'}",
    ]
    return "\n".join(lines)


def run_all(quick: bool = False) -> list[Table]:
    """Run every experiment. `quick` shrinks trial counts for a fast demo."""
    t = 60 if quick else 300
    s = 100 if quick else 400
    return [
        q1_min_distance(samples=s),
        e1_single_flip(samples=s),
        e2_errors(trials=t),
        e3_bler(trials=t)[0],
        e4_jury(trials=t),
    ]
