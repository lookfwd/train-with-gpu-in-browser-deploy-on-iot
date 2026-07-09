// dsp.js — FFT, signal synthesis, and peak-picking. Pure functions, no dependencies.
//
// This is the "figure it out" side of the demo: the browser only ever sees the mixed
// signal, runs an FFT on it to discover the two dominant frequencies, and then
// synthesizes its own labeled training data at those frequencies. Nothing here uses
// any ground truth that would come from the device.

// In-place iterative radix-2 FFT. re/im are Float64Array of length N (a power of two).
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

// Magnitude spectrum (first N/2 bins) of a real signal.
export function magnitudeSpectrum(x) {
  const n = x.length;
  const re = Float64Array.from(x);
  const im = new Float64Array(n);
  fft(re, im);
  const mag = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k++) mag[k] = Math.hypot(re[k], im[k]) / n;
  return mag;
}

// Identify the two dominant frequencies (Hz) from a magnitude spectrum.
// Returns [f1, f2] sorted ascending, enforcing a minimum bin separation.
export function identifyTwoFreqs(mag, fs, minSepBins = 3) {
  const nBins = mag.length; // = N/2
  const cands = [];
  for (let k = 2; k < nBins - 1; k++) {
    if (mag[k] > mag[k - 1] && mag[k] >= mag[k + 1]) cands.push([mag[k], k]);
  }
  cands.sort((a, b) => b[0] - a[0]);
  const picked = [];
  for (const [, k] of cands) {
    if (picked.every((p) => Math.abs(p - k) >= minSepBins)) picked.push(k);
    if (picked.length === 2) break;
  }
  while (picked.length < 2) picked.push((picked[0] || 4) + minSepBins * (picked.length + 1));
  const binHz = fs / (nBins * 2);
  return picked.map((k) => k * binHz).sort((a, b) => a - b);
}

// One sinusoid component: A*sin(2*pi*f*n/fs + phase), length n.
export function sinComponent(n, f, fs, amp, phase) {
  const y = new Float32Array(n);
  const w = (2 * Math.PI * f) / fs;
  for (let i = 0; i < n; i++) y[i] = amp * Math.sin(w * i + phase);
  return y;
}

function gaussian() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// A mixed window: sum of two components + gaussian noise. Returns {mix, c1, c2}.
// c1/c2 are the clean components — used only as training labels / sim-mode ground truth.
export function makeMixture(n, f1, f2, fs, opts = {}) {
  const a1 = opts.a1 ?? 0.4 + 0.6 * Math.random();
  const a2 = opts.a2 ?? 0.4 + 0.6 * Math.random();
  const p1 = opts.p1 ?? 2 * Math.PI * Math.random();
  const p2 = opts.p2 ?? 2 * Math.PI * Math.random();
  const noise = opts.noise ?? 0.05;
  const c1 = sinComponent(n, f1, fs, a1, p1);
  const c2 = sinComponent(n, f2, fs, a2, p2);
  const mix = new Float32Array(n);
  for (let i = 0; i < n; i++) mix[i] = c1[i] + c2[i] + noise * gaussian();
  return { mix, c1, c2 };
}

// Pick two random, well-separated frequencies inside [lo, hi] Hz.
export function randomFreqPair(lo, hi, minSep = 30) {
  for (let tries = 0; tries < 100; tries++) {
    const f1 = lo + Math.random() * (hi - lo);
    const f2 = lo + Math.random() * (hi - lo);
    if (Math.abs(f1 - f2) >= minSep) return [f1, f2].sort((a, b) => a - b);
  }
  return [lo, lo + minSep];
}
