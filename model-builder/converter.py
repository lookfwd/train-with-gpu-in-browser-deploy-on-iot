"""Pure weights -> .tflite conversion.

Shared by all three front-ends: the MQTT builder (builder.py), the standalone HTTPS
server (server.py), and the AWS Lambda handler (lambda_function.py). No MQTT, no HTTP,
no I/O beyond loading the shared architecture spec.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Keras 2, for reliable tf.lite conversion

import time
import numpy as np
import tensorflow as tf

from architectures import load_spec, get_arch, build_model

_SPEC = None


def spec():
    global _SPEC
    if _SPEC is None:
        _SPEC = load_spec()
    return _SPEC


def build_from_weights(arch_id, weights):
    """Rebuild the Keras model for arch_id and load the browser's trained weights into it."""
    s = spec()
    model = build_model(get_arch(s, arch_id), s["input_len"], s["channels_out"])
    arrays = [np.asarray(w["data"], dtype=np.float32).reshape(w["shape"]) for w in weights]
    expected = [tuple(w.shape) for w in model.get_weights()]
    got = [tuple(a.shape) for a in arrays]
    if expected != got:
        raise ValueError(f"weight shape mismatch for {arch_id}: expected {expected}, got {got}")
    model.set_weights(arrays)
    return model


def _self_validate(model, tflite_bytes, input_len):
    x = np.random.randn(1, input_len, 1).astype(np.float32)
    y_keras = model(x, training=False).numpy()
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    interp.set_tensor(in_d["index"], x)
    interp.invoke()
    y_tfl = interp.get_tensor(out_d["index"])
    return float(np.max(np.abs(y_keras - y_tfl)))


def _representative_dataset(freqs, count=150):
    """Windows resembling the training inputs, for int8 activation calibration.
    Frequency-matched when the two tones are given; otherwise sweeps the whole band."""
    s = spec()
    N, fs = s["input_len"], s["fs_hz"]
    n = np.arange(N)
    rng = np.random.default_rng(0)

    def one():
        f1, f2 = (freqs if freqs and len(freqs) == 2 else rng.uniform(40, 200, 2))
        a1, a2 = rng.uniform(0.4, 1.0, 2)
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        x = a1 * np.sin(2 * np.pi * f1 * n / fs + p1) + a2 * np.sin(2 * np.pi * f2 * n / fs + p2)
        x = x + 0.05 * rng.standard_normal(N)
        return x.astype(np.float32).reshape(1, N, 1)

    def gen():
        for _ in range(count):
            yield [one()]
    return gen


def convert_weights(arch_id, weights, quantize=False, freqs=None):
    """arch_id + browser weights -> (tflite_bytes, meta).

    quantize=True -> full int8 (weights + activations) but with float32 I/O, so the firmware is
    unchanged while the S3's ESP-NN SIMD kernels run. Needs `freqs` (the two tones) to calibrate.
    meta = {arch, tflite_bytes, convert_ms, max_abs_err, quantized}.
    """
    t0 = time.time()
    model = build_from_weights(arch_id, weights)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = _representative_dataset(freqs)
        # inference_input/output_type left FLOAT32 -> QUANTIZE/DEQUANTIZE wrap the int8 core.
    tflite_bytes = conv.convert()
    err = _self_validate(model, tflite_bytes, spec()["input_len"])
    meta = {
        "arch": arch_id,
        "tflite_bytes": len(tflite_bytes),
        "convert_ms": int((time.time() - t0) * 1000),
        "max_abs_err": err,
        "quantized": bool(quantize),
    }
    return tflite_bytes, meta
