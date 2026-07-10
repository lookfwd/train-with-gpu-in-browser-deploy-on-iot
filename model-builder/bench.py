#!/usr/bin/env python3
"""On-device TFLite-Micro *interpreter tax* benchmark.

Question it answers: for a tiny linear pipeline (e.g. a 16-ch EEG ICA/FIR datapath,
~640 MACs over ~4 ops), what does the TFLM interpreter cost vs hand-coded ESP-DSP?
The published "<0.1%-4% overhead" figures come from big-convolution models where a huge
kernel amortises the dispatch. A tiny pipeline is the OPPOSITE regime, so those numbers
don't transfer. We measure the actual thing on the actual silicon.

Every model has float32 I/O, input [1,256,1] -> output [1,256,2], matching the firmware's
fixed `memcpy(in->data.f, x, 256*4)` write and `out->data.f[0..511]` read exactly, so
nothing overflows the arena.

The zoo isolates the two components of the tax:
  * per-Invoke FLOOR      -> floor_1op (one op, ~512 MAC)
  * per-op DISPATCH slope -> chain_{2,4,8,16}op  (N tiny ops, ~256 MAC each; slope = cost/op)
  * where COMPUTE takes over -> compute_mid (33k MAC), compute_big / cnn3 (338k MAC, anchors
    to the already-measured ~90 ms)
  * the REAL EEG pipeline  -> eeg_float (DWConv FIR + FC W + Mul gains + FC Winv)
  * does int8 help or HURT here -> eeg_int8, floor_int8 (QUANTIZE/DEQUANTIZE wrap cost)

Usage:
  python bench.py build     # build + convert + desktop-validate + count post-convert ops
  python bench.py measure   # push each to the board over MQTT, record load_us / invoke_us
"""
import os, sys, json, time, statistics
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Keras 2 -> reliable tf.lite conversion

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(HERE, "bench")
MANIFEST = os.path.join(BENCH_DIR, "manifest.json")
RESULTS = os.path.join(BENCH_DIR, "results.json")

BROKER, PORT = "192.168.86.50", 1883
T_MODEL, T_STATUS, T_INFER = "model/flatbuffer", "status/device", "infer/device"

N = 256  # firmware input/output length (fixed)

# The device's MicroMutableOpResolver (main.cpp:100-113). Any op outside this set fails to
# load on the board, so build() asserts every model stays within it.
SUPPORTED = {
    "CONV_2D", "QUANTIZE", "DEQUANTIZE", "EXPAND_DIMS", "RESHAPE", "RELU",
    "SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND", "PAD", "DEPTHWISE_CONV_2D",
    "ADD", "MUL", "FULLY_CONNECTED", "STRIDED_SLICE",
}


# ----------------------------------------------------------------------------- models
# Each builder returns a Keras model  (1,256,1) -> (1,256,2). batch_size is pinned to 1 so
# every Reshape stays STATIC -- an unknown batch dim makes the converter emit SHAPE/PACK to
# compute the target shape at runtime, and those ops are not in the device resolver.
def _models():
    import numpy as np
    import tensorflow as tf
    K = tf.keras

    def _in():
        return K.Input(batch_size=1, shape=(N, 1))

    def passthrough():
        """No matmul at all: RESHAPE + PAD + RESHAPE. Isolates the per-Invoke fixed cost
        (interpreter walk + tensor bookkeeping + moving <=512 elements)."""
        inp = _in()
        h = K.layers.Reshape((N,))(inp)
        h = K.layers.Lambda(lambda t: tf.pad(t, [[0, 0], [0, N]]))(h)   # [1,256]->[1,512]
        h = K.layers.Reshape((N, 2))(h)
        return K.Model(inp, h)

    def fc_min():
        """One tiny FC with just 2 output elements (512 MAC). Same MACs as floor_1op's
        Conv1D but 256x fewer output elements -> proves the conv cost is per-output-element,
        not per-MAC."""
        inp = _in()
        h = K.layers.Reshape((N,))(inp)
        h = K.layers.Dense(2)(h)                                        # 256->2 : 512 MAC
        h = K.layers.Lambda(lambda t: tf.pad(t, [[0, 0], [0, 2 * N - 2]]))(h)
        h = K.layers.Reshape((N, 2))(h)
        return K.Model(inp, h)

    # --- native-I/O probes (need the size-aware firmware) ------------------------------
    def pass_tiny16():
        """16 in -> 16 out, RESHAPE only. True per-Invoke floor with minimal I/O (no pad)."""
        inp = K.Input(batch_size=1, shape=(16, 1))
        h = K.layers.Reshape((8, 2))(inp)
        return K.Model(inp, h)

    def slice_256to16():
        """256 fed in, sliced to 16, 16 out. Same Invoke work as pass_tiny16 but 16x the input
        tensor -> if invoke_us matches pass_tiny16, input size is irrelevant to inference time
        (the firmware copies inputs BEFORE starting the timer). Answers the stateful-FIR question."""
        inp = K.Input(batch_size=1, shape=(256, 1))
        h = K.layers.Reshape((256,))(inp)
        h = K.layers.Lambda(lambda t: t[:, :16])(h)  # STRIDED_SLICE first 16
        h = K.layers.Reshape((8, 2))(h)
        return K.Model(inp, h)

    def fc16(bias=True):
        """One 16->16 FC (256 MAC). float vs int8 versions isolate the QUANTIZE/DEQUANTIZE
        boundary cost from the conv-kernel story."""
        inp = K.Input(batch_size=1, shape=(16, 1))
        h = K.layers.Reshape((16,))(inp)
        h = K.layers.Dense(16, use_bias=bias)(h)
        h = K.layers.Reshape((8, 2))(h)
        return K.Model(inp, h)

    def eeg_native():
        """The REAL EEG datapath with native I/O: 128 in (16 ch x 8 taps) -> 16 out, NO pad/shim.
        16 ch x 8-tap FIR (DepthwiseConv2D) + W(16x16) + gains + Winv(16x16) = 640 MAC, matching
        the original spec exactly."""
        g = tf.constant(np.linspace(0.5, 1.5, 16, dtype="float32"))
        inp = K.Input(batch_size=1, shape=(128, 1))   # 16 ch * 8 taps
        h = K.layers.Reshape((1, 8, 16))(inp)          # H=1, W=8 taps, C=16 ch
        h = K.layers.DepthwiseConv2D((1, 8), padding="valid")(h)  # FIR -> (1,1,16) : 128 MAC
        h = K.layers.Reshape((16,))(h)
        h = K.layers.Dense(16, name="W")(h)            # 256 MAC
        h = K.layers.Lambda(lambda t: t * g, name="gains")(h)
        h = K.layers.Dense(16, name="Winv")(h)         # 256 MAC
        h = K.layers.Reshape((8, 2))(h)                # 16 -> (8,2), no pad
        return K.Model(inp, h)

    def chain(nfiller):
        """nfiller x Conv1D(1,k=1) then Conv1D(2,k=1). Each Conv1D lowers to
        EXPAND_DIMS+CONV_2D+RESHAPE (~256 MAC, near-zero compute) -> the invoke_us slope
        across nfiller is the per-op dispatch cost."""
        inp = _in()
        h = inp
        for i in range(nfiller):
            h = K.layers.Conv1D(1, 1, padding="same", name=f"f{i}")(h)
        h = K.layers.Conv1D(2, 1, padding="same", name="out")(h)
        return K.Model(inp, h)

    def compute_mid():
        inp = _in()
        h = K.layers.Conv1D(2, 65, padding="same")(inp)  # 256*2*65 = 33,280 MAC
        return K.Model(inp, h)

    def compute_big():  # == the cnn3 arch already measured (~90 ms float)
        inp = _in()
        h = K.layers.Conv1D(8, 15, padding="same", activation="relu")(inp)
        h = K.layers.Conv1D(8, 15, padding="same", activation="relu")(h)
        h = K.layers.Conv1D(2, 15, padding="same")(h)  # ~338k MAC
        return K.Model(inp, h)

    def eeg():
        """The real EEG datapath, one sample: 16 ch x 16-tap FIR (one filter, per channel)
        -> W (16x16) -> per-component gains -> Winv (16x16). ~784 MAC over 4 compute ops.
        Bracketed by free reshape/pad so I/O stays [1,256,1]->[1,256,2]."""
        g = tf.constant(np.linspace(0.5, 1.5, 16, dtype="float32"))
        inp = _in()                                            # 256 = 16 ch * 16 taps
        h = K.layers.Reshape((1, 16, 16))(inp)                 # H=1, W=16 taps, C=16 ch
        h = K.layers.DepthwiseConv2D((1, 16), padding="valid")(h)  # FIR -> (1,1,16) : 256 MAC
        h = K.layers.Reshape((16,))(h)
        h = K.layers.Dense(16, name="W")(h)                    # unmixing : 256 MAC
        h = K.layers.Lambda(lambda t: t * g, name="gains")(h)  # per-component amplitude -> MUL
        h = K.layers.Dense(16, name="Winv")(h)                 # remixing : 256 MAC
        h = K.layers.Lambda(lambda t: tf.pad(t, [[0, 0], [0, 496]]), name="pad")(h)  # ->512
        h = K.layers.Reshape((N, 2))(h)
        return K.Model(inp, h)

    # name -> (builder, quantize, intended_macs, note)
    return [
        # native-I/O probes (size-aware firmware)
        ("pass_tiny16", pass_tiny16(), False,      0, "16->16 reshape: true floor, no shim"),
        ("slice_256in", slice_256to16(),False,     0, "256 fed in, 16 out: input-size independence"),
        ("fc16_float",  fc16(),        False,    256, "one 16->16 FC, float"),
        ("fc16_int8",   fc16(),        True,     256, "one 16->16 FC, int8 (boundary cost)"),
        ("eeg_native",  eeg_native(),  False,    640, "REAL EEG, native 128->16, NO pad (float)"),
        ("eeg_nat_int8",eeg_native(),  True,     640, "REAL EEG, native 128->16, NO pad (int8)"),
        # original 256/512-shim probes (kept for cross-check)
        ("passthrough", passthrough(), False,      0, "no matmul -> pure per-Invoke floor"),
        ("fc_min",      fc_min(),      False,    512, "1 tiny FC (2 outputs), 512 MAC"),
        ("floor_1op",   chain(0),      False,    512, "Conv1D 512 MAC over 512 output elems"),
        ("chain_2op",   chain(1),      False,    768, "2 ops"),
        ("chain_4op",   chain(3),      False,   1280, "4 ops -> EEG-analog op count"),
        ("chain_8op",   chain(7),      False,   2304, "8 ops"),
        ("chain_16op",  chain(15),     False,   4352, "16 ops -> dispatch slope"),
        ("compute_mid", compute_mid(), False,  33280, "1 op, 33k MAC -> compute starts to bite"),
        ("compute_big", compute_big(), False, 337920, "cnn3, 338k MAC -> anchors to ~90 ms"),
        ("eeg_float",   eeg(),         False,    784, "REAL EEG pipeline (float)"),
        ("eeg_int8",    eeg(),         True,     784, "REAL EEG pipeline (int8+ESP-NN)"),
        ("floor_int8",  chain(0),      True,     512, "int8 floor -> QUANTIZE/DEQUANTIZE tax"),
    ]


def _rep_dataset(shape):
    import numpy as np
    rng = np.random.default_rng(0)
    def gen():
        for _ in range(120):
            yield [rng.standard_normal(shape).astype("float32")]
    return gen


def _convert(model, quantize):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        shape = tuple(int(d) if d is not None else 1 for d in model.inputs[0].shape)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = _rep_dataset(shape)
        # inference_input/output_type left FLOAT32 -> int8 core wrapped by QUANTIZE/DEQUANTIZE
    return conv.convert()


def _desktop_check(tflite_bytes):
    """Allocate + invoke on the desktop interpreter; return (out_shape, [op_names]).
    Uses the BUILTIN_REF resolver so no XNNPACK DELEGATE fuses ops away -- the op list is
    then exactly what TFLM will dispatch on the device."""
    import numpy as np
    import tensorflow as tf
    interp = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interp.allocate_tensors()
    ind, outd = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(ind["index"], np.random.randn(*ind["shape"]).astype("float32"))
    interp.invoke()
    out = interp.get_tensor(outd["index"])
    try:
        ops = [d["op_name"] for d in interp._get_ops_details()]
    except Exception:
        ops = None
    return tuple(out.shape), ops


def build():
    os.makedirs(BENCH_DIR, exist_ok=True)
    manifest = []
    print(f"{'name':<12} {'q':<4} {'MAC':>7} {'bytes':>7} {'ops':>4}  out_shape  op_sequence")
    print("-" * 100)
    for name, model, q, macs, note in _models():
        tfl = _convert(model, q)
        path = os.path.join(BENCH_DIR, f"{name}.tflite")
        with open(path, "wb") as fh:
            fh.write(tfl)
        out_shape, ops = _desktop_check(tfl)
        # firmware reads out as [1, M, 2] interleaved (M<=N); input must be <=N floats so the
        # size-aware memcpy fills it without overflow.
        assert out_shape[0] == 1 and out_shape[2] == 2 and out_shape[1] <= N, \
            f"{name}: output {out_shape} not [1,<= {N},2]"
        unsupported = sorted(set(ops or []) - SUPPORTED)
        if unsupported:
            print(f"  !! {name}: UNSUPPORTED on device: {unsupported}  (would load_error)")
        n_ops = len(ops) if ops else -1
        manifest.append({
            "name": name, "file": path, "quantize": q, "macs": macs, "note": note,
            "tflite_bytes": len(tfl), "n_ops": n_ops, "ops": ops,
        })
        seq = ",".join(ops) if ops else "?"
        print(f"{name:<12} {str(q):<4} {macs:>7} {len(tfl):>7} {n_ops:>4}  {str(out_shape):<10} {seq}")
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest)} models + manifest to {BENCH_DIR}")


# ----------------------------------------------------------------------------- measure
def measure(only=None):
    import paho.mqtt.client as mqtt
    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    if only:
        manifest = [m for m in manifest if m["name"] in only]

    TARGET, DISCARD = 40, 3          # collect 40 invoke_us after discarding 3 warmup/stale
    LOAD_TO, COLLECT_TO = 25.0, 45.0  # seconds
    LOAD_RETRIES = 2                 # re-publish if model_loaded is lost (qos1, but be safe)

    state = {"loaded": None, "error": None, "samples": [], "collecting": False}

    def on_connect(c, u, flags, rc, props=None):
        c.subscribe([(T_STATUS, 1), (T_INFER, 1)])
        print(f"connected to {BROKER}:{PORT}, subscribed")

    def on_message(c, u, msg):
        try:
            d = json.loads(msg.payload.decode())
        except Exception:
            return
        if msg.topic == T_STATUS and d.get("event") == "model_loaded":
            state["loaded"] = d
        elif msg.topic == T_STATUS and d.get("event") == "load_error":
            state["error"] = d
        elif msg.topic == T_INFER and state["collecting"] and "invoke_us" in d:
            state["samples"].append(int(d["invoke_us"]))

    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    cli.on_connect, cli.on_message = on_connect, on_message
    deadline = time.time() + 90  # tolerate the board/broker still coming up
    while True:
        try:
            cli.connect(BROKER, PORT, keepalive=30)
            break
        except OSError as e:
            if time.time() > deadline:
                print(f"cannot reach broker {BROKER}:{PORT} ({e}). Is it up?")
                sys.exit(2)
            print(f"  broker not up yet ({e}); retrying...")
            time.sleep(3)
    cli.loop_start()

    results = []
    for m in manifest:
        name = m["name"]
        with open(m["file"], "rb") as fh:
            blob = fh.read()
        state.update(loaded=None, error=None, samples=[], collecting=False)

        for attempt in range(LOAD_RETRIES):
            cli.publish(T_MODEL, blob, qos=1)
            t0 = time.time()
            while state["loaded"] is None and state["error"] is None and time.time() - t0 < LOAD_TO:
                time.sleep(0.02)
            if state["loaded"] is not None or state["error"] is not None:
                break
            print(f"{name:<12} no model_loaded in {LOAD_TO:.0f}s, re-publishing "
                  f"(attempt {attempt + 2}/{LOAD_RETRIES})")
        if state["error"] is not None:
            print(f"{name:<12} LOAD_ERROR (arena too small?) {state['error']}")
            results.append({**m, "load_error": True})
            continue
        if state["loaded"] is None:
            print(f"{name:<12} TIMEOUT waiting for model_loaded")
            results.append({**m, "timeout": True})
            continue

        load_us = state["loaded"].get("load_us")
        arena = state["loaded"].get("arena_bytes")
        # discard a few windows (warmup / any in-flight old-model sample), then collect TARGET
        state["samples"] = []
        state["collecting"] = True
        time.sleep(0.25 * (DISCARD + 1))
        state["samples"] = []           # drop the warmup ones
        t0 = time.time()
        while len(state["samples"]) < TARGET and time.time() - t0 < COLLECT_TO:
            time.sleep(0.05)
        state["collecting"] = False
        s = sorted(state["samples"])
        if not s:
            print(f"{name:<12} NO invoke samples (is the board streaming?)")
            results.append({**m, "load_us": load_us, "arena_bytes": arena, "invoke": None})
            continue

        def pct(p):
            return s[min(len(s) - 1, int(p * len(s)))]
        stat = {"n": len(s), "med": int(statistics.median(s)), "min": s[0],
                "p10": pct(0.10), "p90": pct(0.90)}
        results.append({**m, "load_us": load_us, "arena_bytes": arena, "invoke": stat})
        print(f"{name:<12} load={load_us:>7}us arena={arena:>6}B  "
              f"invoke med={stat['med']:>7}us [p10={stat['p10']} p90={stat['p90']} "
              f"min={stat['min']} n={stat['n']}]")

    cli.loop_stop()
    cli.disconnect()
    with open(RESULTS, "w") as fh:
        json.dump(results, fh, indent=2)
    _print_table(results)
    print(f"\nwrote results to {RESULTS}")


def _print_table(results):
    print("\n=== TFLM interpreter tax on ESP32-S3 (240 MHz), float32 I/O ===")
    print(f"{'model':<12} {'q':<5} {'ops':>4} {'MAC':>7} {'load_us':>8} {'invoke_us(med)':>15} {'us/MAC':>8}")
    print("-" * 70)
    for r in results:
        if not r.get("invoke"):
            continue
        med = r["invoke"]["med"]
        upm = med / r["macs"] if r["macs"] else 0
        print(f"{r['name']:<12} {str(r['quantize']):<5} {r['n_ops']:>4} {r['macs']:>7} "
              f"{r.get('load_us','?'):>8} {med:>15} {upm:>8.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "measure":
        measure(only=sys.argv[2:] or None)
    else:
        print(__doc__)
        sys.exit(1)
