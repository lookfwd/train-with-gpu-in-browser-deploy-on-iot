"""Build a Keras model for one of the shared architectures.

This is the Python twin of the browser's TF.js model builder. BOTH read
`shared/architectures.json`, so the layer order and weight shapes match exactly and
the raw weights the browser trains can be dropped straight into the Keras model with
`set_weights(...)`.

We force legacy Keras 2 (`tf-keras`) because the classic `tf.lite.TFLiteConverter`
path is far more reliable there than with Keras 3 (the default in TF >= 2.16).
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # must be set before importing TF

import json
import tensorflow as tf  # noqa: E402

_ACTIVATIONS = {"linear": None, "relu": "relu", "tanh": "tanh"}

def _find_spec():
    # Local dev: ../web/architectures.json. Lambda container: co-located copy. Env override wins.
    candidates = [
        os.environ.get("AIEDGE_SPEC_PATH"),
        os.path.join(os.path.dirname(__file__), "..", "web", "architectures.json"),
        os.path.join(os.path.dirname(__file__), "architectures.json"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("architectures.json not found in: " + ", ".join(str(c) for c in candidates))


def load_spec(path=None):
    with open(path or _find_spec()) as f:
        return json.load(f)


def get_arch(spec, arch_id):
    for a in spec["architectures"]:
        if a["id"] == arch_id:
            return a
    raise KeyError(f"unknown arch_id {arch_id!r}")


def build_model(arch, input_len, channels_out):
    """A stack of 'same'-padded Conv1D layers: [input_len, 1] -> [input_len, channels_out]."""
    k = tf.keras
    inp = k.Input(shape=(input_len, 1), name="x")
    h = inp
    for i, layer in enumerate(arch["layers"]):
        if layer["type"] != "conv1d":
            raise ValueError(f"unsupported layer type: {layer['type']!r}")
        h = k.layers.Conv1D(
            filters=layer["filters"],
            kernel_size=layer["kernel"],
            dilation_rate=layer.get("dilation", 1),
            padding="same",
            activation=_ACTIVATIONS.get(layer.get("activation", "linear"), layer.get("activation")),
            name=f"conv{i}",
        )(h)
    model = k.Model(inp, h, name=arch["id"])
    assert model.output_shape[-1] == channels_out, (model.output_shape, channels_out)
    return model
