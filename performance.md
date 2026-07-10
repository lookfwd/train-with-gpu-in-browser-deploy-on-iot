# Performance — TFLite Micro interpreter tax on ESP32-S3

On-device measurements of what TensorFlow Lite Micro actually costs per inference on an
ESP32-S3, and what that means for a tiny per-sample linear pipeline (the motivating case: a
16-channel EEG ICA/FIR datapath). Every number here was measured on hardware with the
`model-builder/bench.py` harness; nothing is extrapolated. Where a causal claim is made it was
either measured directly or cross-checked against published sources (listed at the end), and two
earlier wrong claims are called out in [Corrections](#corrections).

## TL;DR

- **Per-Invoke floor ≈ 96 µs.** Irreducible interpreter + tensor bookkeeping. Nothing on this
  chip Invokes faster, no matter how little the model does.
- **The real 16-ch EEG pipeline (640 MAC, float) runs in ~333 µs/sample.**
- **Cost is dominated by per-*output-element* work in the *float* reference kernels**
  (~2.2 µs/output for `CONV_2D`) — **not** by MAC count and **not** by interpreter dispatch
  (both small).
- **int8 makes a tiny pipeline *slower*** (EEG 333 → 603 µs). The `QUANTIZE`/`DEQUANTIZE`
  boundary is never ESP-NN-accelerated and costs more than the tiny int8 core saves. **Stay
  float** at this scale.
- **Hand-coded ESP-DSP (~10–25 µs) beats TFLM by ~20×**, but TFLM float is still inside a
  per-sample real-time budget at typical EEG rates (≤ 1 kHz).
- **Holding the FIR history *inside* the model does not help** — the data-in copy isn't part of
  the measured inference time, and an in-model shift-register adds ops that cost more than the
  copy it removes.

## Setup & method

- **Board:** ESP32-S3-DevKitC-1 @ 240 MHz. ESP-IDF framework, `esp-tflite-micro` (ESP-NN
  enabled for int8). 96 KB tensor arena.
- **What's measured:** the firmware loads a raw `.tflite` pushed over MQTT (`model/flatbuffer`),
  runs it on the simulated ADC stream, and reports:
  - `invoke_us` — `esp_timer` around `Invoke()` **only**.
  - `load_us` — interpreter build + `AllocateTensors()`.
- **Harness:** `model-builder/bench.py` builds a spread of Keras models, converts each (float and
  int8 with a representative dataset), **verifies every op is in the device resolver** before
  pushing (so nothing fails to load mid-run), pushes each model, and collects 40 `invoke_us`
  samples. `python bench.py build` then `python bench.py measure`.
- **All models have float32 I/O** so the firmware is unchanged across the sweep; int8 models wrap
  an int8 core with `QUANTIZE`/`DEQUANTIZE`.
- **Timer scope matters:** the firmware copies inputs *before* starting the timer, so `invoke_us`
  excludes the data-in copy by construction (see [the stateful question](#the-store-history-inside-the-model-question)).
- **min vs median:** the inference task is preempted by the WiFi/MQTT stack during streaming,
  which inflates wall-clock `invoke_us`. The **minimum** over 40 samples ≈ true uninterrupted
  compute; the median reflects this demo's own background load (a dedicated deployment wouldn't
  stream). Mins reproduce across firmware rebuilds (e.g. `passthrough` = 238 µs in two independent
  runs). **Numbers below are the min unless stated.**
- **Two firmware I/O modes** were used. Initially the firmware hard-coded 256-in/512-out, so every
  model was padded to fit — this inflated results with pad cost. It was then made **size-aware**
  (copy/read clamped to the model's real tensor size; see `esp-project/src/main.cpp`), which let
  the pipeline be measured with **native I/O** and no pad shim.

## Measured data

### Native I/O — the real numbers (size-aware firmware)

| model | I/O (floats) | ops | MAC | `invoke_us` (min) | note |
|---|---|---|---|---|---|
| `pass_tiny16` | 16 → 16 | 1 | 0 | **96** | true per-Invoke floor (one `RESHAPE`) |
| `fc16_float` | 16 → 16 | 3 | 256 | 169 | one `FULLY_CONNECTED` |
| `fc16_int8` | 16 → 16 | 5 | 256 | 298 | +129 µs = `QUANTIZE`/`DEQUANTIZE` alone |
| **`eeg_native`** | 128 → 16 | 5 | 640 | **333** | **real EEG pipeline, float** |
| `eeg_nat_int8` | 128 → 16 | 7 | 640 | 603 | int8 = 1.8× slower |
| `passthrough` | 256 → 512 | 3 | 0 | 238 | old "floor" — mostly `PAD` |
| `eeg_float` | 256 → 512 | 6 | 784 | 456 | old EEG, inflated ~123 µs by the pad shim |
| `eeg_int8` | 256 → 512 | 8 | 784 | 1086 | |

`eeg_native` = `RESHAPE → DEPTHWISE_CONV_2D (16-ch × 8-tap FIR) → FULLY_CONNECTED (W, 16×16) →
FULLY_CONNECTED (Winv, 16×16) → RESHAPE`; the per-component gains fold into `W`. 640 MAC exactly
matches the spec (128 FIR + 256 + 256).

`load_us` was 1.1–3.0 ms for every model — rebuilding the interpreter and re-`AllocateTensors` on a
model swap is cheap.

### Op-count and compute ladders (fixed 256/512 firmware, clean medians)

Built to separate interpreter dispatch from kernel compute. Each `chain_Nop` is
`RESHAPE + N×CONV_2D + RESHAPE` with near-zero compute per conv.

| model | ops | MAC | `invoke_us` (med) |
|---|---|---|---|
| `fc_min` (512 MAC, **2** output elems) | 4 | 512 | 270 |
| `floor_1op` (512 MAC, **512** output elems) | 3 | 512 | 1431 |
| `chain_2op` | 4 | 768 | 2026 |
| `chain_4op` | 6 | 1280 | 3043 |
| `chain_8op` | 10 | 2304 | 5410 |
| `chain_16op` | 18 | 4352 | 9905 |
| `compute_mid` | 3 | 33 280 | 19 807 |
| `compute_big` (= cnn3) | 5 | 337 920 | 85 685 |
| `floor_int8` | 5 | 512 | 1696 |

## Where the time goes

**1. Per-Invoke floor ≈ 96 µs.** `pass_tiny16` (16→16, a single reshape) can't be undercut. This
is the fixed interpreter + tensor-management cost for float32 on this port/arena.

**2. Cost scales with output *elements*, not MACs.** `fc_min` and `floor_1op` have the *same 512
MACs*, but `fc_min` emits 2 output elements (270 µs) and `floor_1op` emits 512 (1431 µs) — 5.3×
for identical arithmetic. The `chain` slope is ~563 µs per added `Conv1D(1,k=1)`, and each such
conv emits 256 output elements ⇒ **~2.2 µs per output element** for the float `CONV_2D` kernel.

**3. Why float is slow: it is unaccelerated scalar C++.** ESP-NN — Espressif's SIMD kernel library
— accelerates **int8 only**; it has no float path. TFLM's float kernels are the reference
implementations, [documented as "not performant… designed for readability rather than
performance."](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/optimized_kernel_implementations.md)
So every float conv/matmul here runs plain scalar, hence the ~2.2 µs/output.

**4. Interpreter dispatch is small.** The `chain` slope is conv *compute* (256 output elements ×
2.2 µs), not dispatch; adding ops that don't add output elements adds little. This matches TFLM's
own paper, which measures interpreter overhead at **< 0.1 % for large models and ~3–4 % for a
small one (Google Hotword)** — a fraction of compute, not a fixed millisecond cost.

**5. The 256/512 `PAD` shim was a measurement artifact.** `passthrough` (238 µs) − `pass_tiny16`
(96 µs) = 142 µs is the pad-to-512 plus the larger reshapes moving ~500 floats. The real EEG
pipeline without that shim (`eeg_native`, 333 µs) is ~123 µs faster than the shimmed `eeg_float`
(456 µs).

## int8: when it helps vs. when it hurts

int8 quantization is a large win for **big convolution models** — ESP-NN's SIMD int8 kernels give
order-of-magnitude speedups (Espressif's person-detection demo: [2300 ms → 54 ms with
ESP-NN](https://components.espressif.com/components/espressif/esp-tflite-micro); an earlier `wide`
model here went 177 → 6 ms).

For a **tiny pipeline it is a net loss** — measured here, not assumed:

| | float | int8 | Δ |
|---|---|---|---|
| `fc16` (256 MAC) | 169 µs | 298 µs | **+129 µs** |
| `eeg` (640 MAC) | 333 µs | 603 µs | **1.8× slower** |

**Mechanism:** ESP-NN does not accelerate `QUANTIZE`/`DEQUANTIZE` — those float↔int8 boundary
conversions always run scalar. A large model amortises that fixed boundary cost over a big
accelerated int8 core; a ~640-MAC pipeline has too little core to pay it back. int8 being *slower*
than float for small graphs is a [well-documented, ecosystem-wide
effect](https://github.com/tensorflow/tensorflow/issues/40183), not an ESP32 quirk.

**Rule of thumb:** int8 wins only when the accelerated int8 core is large enough to dwarf the fixed
`QUANTIZE`/`DEQUANTIZE` boundary. Below that, stay float.

## The "store history inside the model" question

Idea: keep the FIR's past samples in a shift-register *inside* the model, so each Invoke is fed
only the new samples instead of the whole window — to avoid "paying the overhead of data-in per
iteration." **It does not help, and slightly hurts.** Three measured reasons:

1. **The data-in copy is not in the inference time.** The firmware copies inputs *then* starts the
   timer, so `invoke_us` never included it — and it's a ~0.5 µs `memcpy` of 128 floats regardless.
2. **The FIR reads 128 values either way** (16 ch × 8 taps = 128 MACs). Internal storage does not
   reduce the arithmetic.
3. **In-model buffer manipulation has its own cost.** The `slice_256in` probe adds a single
   `STRIDED_SLICE` over a 256-vector and costs ~180 µs. A ring-buffer needs
   `READ_VARIABLE` + shift/concat + `ASSIGN_VARIABLE` every Invoke (TFLM *does* support persistent
   state via resource variables — the streaming-keyword-spotter pattern), and those ops add more
   than the ~0.5 µs feed they remove.

The levers that actually reduce per-sample cost are the opposite end: **fewer output elements**
(the pad cost 142 µs), **stay float** (int8 costs ~2×), and **block N samples per Invoke** to
amortise the 96 µs floor over N.

## Decision: TFLM interpreter vs. hand-coded ESP-DSP (16-ch EEG)

Hand-coded cost for ~640 MAC is a few µs — [ESP-DSP primitives](https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/esp-dsp-benchmarks.html)
run `dsps_fir_f32`/`dsps_dotprod_f32` at ~1.7 cycles/tap, i.e. ~10–25 µs all-in for the FIR + two
16×16 matrix-vector products.

| implementation | per sample | @250 Hz | @500 Hz | @1 kHz |
|---|---|---|---|---|
| hand-coded ESP-DSP (~15 µs) | ~15 µs | 0.4 % | 0.7 % | 1.5 % |
| **TFLM float** (`eeg_native`) | **333 µs** | **8 %** | 17 % | 33 % |

*(CPU % of one 240 MHz core, per-sample.)*

**Verdict:** hand-coded DSP wins ~20× on raw CPU and always will — the float kernels are
unaccelerated scalar and there's a fixed ~96 µs per-Invoke floor no shrinking beats. But TFLM
float is a non-issue at clinical 250 Hz (8 %), workable at 500 Hz, and heavy but usable at 1 kHz
(33 %). Choose by what you value:

- **Take TFLM** for the trainable/deployable story this demo is built around — browser-trained ICA
  + FIR, real-tooling `.tflite` conversion, OTA model push, retrain-on-drift — if you can spare
  ~10–35 % of a core.
- **Hand-code it** if CPU/power is tight, the sample rate is > 1 kHz, or you want that core back.
- **Either way, stay float**, and if you need TFLM headroom, block multiple samples per Invoke
  rather than reaching for int8 or in-model state.

## Corrections

Two claims made *before* measuring were wrong and are retracted here, because the process that
caught them is part of the finding:

- **"~15 ms fixed per-Invoke overhead"** — wrong. It was a straight-line-fit artifact across models
  with different per-MAC efficiency, disproved by the measured 6 ms int8 `wide` (which can't sit on
  top of a 15 ms floor) and by TFLM's own 3–4 % overhead figure.
- **"the ~240 µs floor is interpreter dispatch"** — wrong. Direct measurement (`pass_tiny16` 96 µs
  vs `passthrough` 238 µs) shows it was mostly the `PAD` op moving elements; the true floor is
  96 µs and dispatch is a small fraction of it. Flagged by an adversarial review of the verdict
  before it shipped, then confirmed on-device.

## Reproduce

```bash
cd model-builder
./.venv/bin/python bench.py build      # build + convert + verify device-valid ops
./.venv/bin/python bench.py measure    # push each to the board, collect load_us / invoke_us
# subset: ./.venv/bin/python bench.py measure eeg_native eeg_nat_int8 pass_tiny16
```

Requires the broker at `192.168.86.50` and the board running the firmware in
`esp-project/` (size-aware I/O). Results are written to `model-builder/bench/results.json`.

## Sources

- [TensorFlow Lite Micro: Embedded ML on TinyML Systems (arXiv 2010.08678)](https://arxiv.org/pdf/2010.08678) — interpreter overhead < 0.1 % (large) / 3–4 % (small).
- [tflite-micro — reference kernels "not performant by design"](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/optimized_kernel_implementations.md)
- [espressif/esp-nn README](https://github.com/espressif/esp-nn/blob/master/README.md) — int8-only acceleration; person-detection 2300 → 54 ms.
- [int8 slower than float32 (tensorflow#40183)](https://github.com/tensorflow/tensorflow/issues/40183), [model-optimization#599](https://github.com/tensorflow/model-optimization/issues/599) — boundary-cost effect is ecosystem-wide.
- [ESP-DSP benchmarks](https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/esp-dsp-benchmarks.html) — `dsps_fir_f32` / `dsps_dotprod_f32` cycle counts.
