# Error Correction with Fuzzy-Hash Signatures: Results

*Thomas Ha · November 2026 · full specification and proofs in `SPEC.md`, code and data in this repository*

## Question

Can random-projection "fuzzy hashes" serve as an error-correcting code for stored data that suffers random bit flips? Professor Hoffstein suggested the idea and expected it to be infeasible. This project builds it, measures it, and identifies exactly why it falls short.

## The scheme

Each data byte $x$ is stored unchanged, followed by $J$ signatures $\sigma_i(x) = [A_i\,b(x) > 0]$. Here $b(x) \in \{\pm1\}^8$ is the byte as signs, and the $A_i$ are secret random $\pm1$ matrices of size $8 \times 8$, generated from a key. Decoding chooses the byte whose full record (data plus signatures) is closest in Hamming distance to what was read. Each signature acts as a "juror" voting for the bytes it recognises. Storage cost is $(J+1)\times$.

## Results

**Proposition 1.** The decoder is maximum-likelihood on the binary symmetric channel: the consensus score equals $8J - d_H(E(\hat x), \text{received})$.

**Proposition 2 (exact sensitivity).** For $\pm1$ matrices, flipping one data bit changes each signature bit independently with probability
$$p_n = \binom{n-1}{\lfloor n/2\rfloor}\big/2^{n-1} \approx \sqrt{2/(\pi n)},$$
which equals the influence of a single input on the majority function. Measurements match to within 1%.

**Proposition 3 (ceiling).** For *any* choice of matrices, $d_{\min} \le 1 + J\sqrt{n}$. Each signature bit is a linear threshold function, so it is unate, and its total influence is $\sum_k|\hat f(k)| \le \sqrt n$ by Cauchy–Schwarz and Parseval. The minimum distance is at most the average distance between neighbouring codewords, which gives the bound. As a fraction of the record length, $d_{\min}$ therefore shrinks like $1/\sqrt n$, while good codes keep a constant fraction.

**Measured performance**, compared with other codes at the same storage cost:

| Setting ($n = 8$) | This scheme | Comparison |
|---|---|---|
| $J = 20$ (21× storage): minimum distance | 26–27 (exact; varies with key) | repetition 21; ceiling 57; Gilbert–Varshamov guarantee 69 |
| $J = 20$: bytes lost at 30% bit-flip rate | 6.7% | repetition: 19% |
| $J = 2$ (3× storage): minimum distance | 1–2 | TMR: 3 |
| $J = 2$: bytes lost at 0.1% bit-flip rate | 2.9 × 10⁻⁴ (best of 200 keys) | TMR: 2.4 × 10⁻⁵ |

With many jurors the scheme beats plain repetition, but it reaches less than half the distance a good code of the same size is guaranteed. With few jurors it loses even to triple modular redundancy.

## Conclusion

The idea is infeasible as an error-correcting code, for a structural reason. Fuzzy hashes are built to keep nearby inputs nearby, while an error-correcting code needs nearby inputs to map to far-apart codewords. Proposition 3 turns this tension into a bound that no choice of matrices can escape. Replacing the threshold with parity checks would remove the limit, but the result would be an LDPC code with Gallager's bit-flipping decoder, an existing construction.

## Open questions

1. **Noise-tolerant authentication.** The same locality is useful for matching noisy inputs, such as biometric templates. With real-valued sign projections the secret matrices are easy to learn, since each signature bit is a halfspace. Could a keyed, noise-tolerant hash be based on a rounding-modulo-$q$ variant, similar to *learning with rounding*, so that recovering the key is as hard as a lattice problem?
2. **Key selection.** $d_{\min}$ varies with the key. Is there a structured family of matrices that provably attains a larger minimum distance, within the ceiling?
