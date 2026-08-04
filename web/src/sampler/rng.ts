// Mulberry32: tiny deterministic PRNG with 32 bits of state.
//
// Used in place of NumPy's PCG64 for the JS port. Stochastic parity
// only — we don't reproduce NumPy traces bit-for-bit, but we do want
// determinism within a session (same seed → same trajectory) and
// reproducibility in tests.
//
// References: https://gist.github.com/tommyettinger/46a3f6d8b53a8b3a5746
// — public-domain reference impl.

export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed | 0;
  }

  /** Uniform float in [0, 1). */
  random(): number {
    let a = (this.state = (this.state + 0x6d2b79f5) | 0);
    a = Math.imul(a ^ (a >>> 15), a | 1);
    a ^= a + Math.imul(a ^ (a >>> 7), a | 61);
    return ((a ^ (a >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform integer in [0, n). */
  integers(n: number): number {
    return Math.floor(this.random() * n);
  }

  /** Fisher-Yates shuffle, returning a new array (does not mutate). */
  permutation<T>(arr: readonly T[]): T[] {
    const out = arr.slice();
    for (let i = out.length - 1; i > 0; i--) {
      const j = this.integers(i + 1);
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

}
