# Fuzzy Hashing ECC

A proof of concept for **error correction by consensus against fuzzy hashes**. Each byte of data is stored unchanged, next to $J$ (default 20) locality-sensitive hashes of it: the signs of random ±1 projections, as in SimHash. To repair a corrupted byte, the decoder picks the byte whose data and hashes agree best with what was read. Each hash acts as a *juror* that supports the bytes it recognises.

[`SPEC.md`](SPEC.md) has the full definition, three propositions with proofs, and the reasoning behind each design choice.

## Usage

```sh
uv sync --extra plot     # numpy, plus matplotlib for the optional plot

# Protect a file, damage it, recover it
uv run fuzzy-hashing-ecc encode photo.jpg photo.ecc               # 21× larger
uv run fuzzy-hashing-ecc noise  photo.ecc photo.bad --p 0.05      # flip 5% of all bits
uv run fuzzy-hashing-ecc decode photo.bad photo.out.jpg
cmp photo.jpg photo.out.jpg && echo "recovered exactly"
```

There are only two settings, and `encode` and `decode` must use the same values:

| Flag | Default | Meaning |
|---|---|---|
| `--jurors` | 20 | signatures per byte; storage cost is (jurors + 1)× |
| `--key` | 0 | seed for the secret matrices |

```sh
uv run fuzzy-hashing-ecc demo --toy-only     # one byte, one flipped bit, every juror's vote
uv run fuzzy-hashing-ecc experiments --plot results/e3_bler.png   # all results, ~15 s
uv run pytest && uv run mypy                 # tests and strict type check
```

From Python:

```python
from fuzzy_hashing_ecc import ConsensusCode, encode_bytes, decode_bytes

code = ConsensusCode(jurors=20, key=0)
blob = encode_bytes(b"hello", code)
assert decode_bytes(blob, code) == b"hello"
```

| Module | Contents |
|---|---|
| `scheme.py` | `ConsensusCode`: the design in SPEC.md, with fast exact decoding of whole files |
| `stream.py`, `channel.py` | File framing with a protected length header; bit-flip noise models |
| `baselines.py` | Repetition code with the same storage cost, for comparison |
| `research.py`, `projections.py`, `signatures.py` | The alternatives compared before fixing the design (matrix families, threshold rules, decoders) |
| `experiments.py`, `plotting.py` | The experiments behind SPEC §IV–VI |

## Results

The full output is in [`results/experiments.txt`](results/experiments.txt). All numbers are for the default 20 jurors, which store **21×** the original data.

**1. Every 168-bit record survives 12–13 flipped bits anywhere,** guaranteed. The minimum distance is 26–27, depending on the key. A repetition code using the same storage (21 copies of each bit) has minimum distance 21.

**2. It beats that repetition code in practice too** (E3). At a bit-flip rate of 25%, it loses 0.3% of bytes while repetition loses 5%. At 30%, the figures are 6.7% against 19%.

![Block error rate against bit-flip probability](results/e3_bler.png)

**3. The theory matches the measurements.** Proposition 2 gives an exact formula for how many signature bits react to one flipped data bit; the measurements agree to within 1% (E1). Proposition 3 proves a ceiling on the minimum distance for *any* choice of matrices, and every configuration tested stays under it (Q1).

**4. It is still far from a *good* code** (Q1). The Gilbert–Varshamov bound guarantees that some linear code with the same size reaches minimum distance 69, against 27 here, and Proposition 3 shows the gap grows with block length. The reason is that fuzzy hashes are designed to keep similar inputs similar, which is the opposite of what error correction needs.

**5. Losing jurors barely matters** (E4). Replacing a quarter of the stored signatures with random bits has no measurable effect once $J \ge 10$.

## Open directions

- **Noise-tolerant matching instead of correction.** Fuzzy hashes are weak as an error-correcting code but well suited to recognising noisy inputs, as in biometric templates or approximate authentication.
- **Security of the key.** How hard is it to recover the secret matrices from observed signatures, especially if noise is added deliberately? This connects to lattice-based cryptography.
- **Choosing good keys.** The minimum distance varies with the key. Is there a fast way to find good keys, or a structured family of matrices with a guaranteed larger distance?
