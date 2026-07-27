"""PennyLane VQC layers used by BottleneckVQC-UNet."""
from __future__ import annotations

import math

import pennylane as qml
import tensorflow as tf

def build_cond_encoder_vqc(
    cond_dim: int = 3,
    cond_emb_dim: int = 16,
    n_qubits: int = 5,
    n_layers: int = 2,
) -> tf.keras.Model:
    """
    Condition encoder (VQC): cond -> Dense(cond_dim->n_qubits,tanh) -> *pi -> VQC -> Dense -> cond_emb
    Exposed separately to export cond_emb and analyze angular smoothness.
    """
    cond_in = tf.keras.Input(shape=(cond_dim,), name="cond")
    q_inputs = tf.keras.layers.Dense(n_qubits, activation="tanh", name="cond_to_qubits")(cond_in)
    q_inputs = tf.keras.layers.Lambda(lambda x: x * math.pi, name="cond_to_qubits_scale")(q_inputs)
    qlayer = build_vqc_layer(n_qubits=n_qubits, n_layers=n_layers)
    q_out = qlayer(q_inputs)
    q_out = tf.keras.layers.Activation(tf.keras.activations.linear, name="q_out_linear")(q_out)
    cond_emb = tf.keras.layers.Dense(cond_emb_dim, activation="relu", name="cond_emb_from_vqc")(q_out)
    return tf.keras.Model(inputs=cond_in, outputs=cond_emb, name="CondEncoderVQC")


class _CompatKerasLayer(tf.keras.layers.Layer):
    """
    Lightweight KerasLayer-compatible wrapper for PennyLane >=0.42,
    where qml.qnn.KerasLayer was removed.
    """

    def __init__(self, qnode, weight_shapes, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.qnode = qnode
        self.weight_shapes = weight_shapes
        self.output_dim = output_dim
        self._weight_names = []
        self._weight_vars = {}

    def build(self, input_shape):
        for name, shape in self.weight_shapes.items():
            w = self.add_weight(
                name=name,
                shape=shape,
                initializer="random_normal",
                trainable=True,
            )
            self._weight_names.append(name)
            self._weight_vars[name] = w
        super().build(input_shape)

    def _ensure_tensor(self, res):
        if isinstance(res, (list, tuple)):
            return tf.stack(res, axis=-1)
        return res

    def call(self, inputs):
        weights = {name: self._weight_vars[name] for name in self._weight_names}

        def _call_one(x):
            return self._ensure_tensor(self.qnode(x, **weights))

        if len(inputs.shape) == 1:
            out = _call_one(inputs)
        else:
            out = tf.vectorized_map(_call_one, inputs)
        return tf.cast(out, tf.float32)


def _make_keras_layer(qnode, weight_shapes, output_dim: int):
    # Prefer native qml.qnn.KerasLayer when available (<=0.41.x).
    try:
        return qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=output_dim)
    except AttributeError:
        return _CompatKerasLayer(qnode, weight_shapes, output_dim=output_dim)


def build_vqc_layer(n_qubits: int = 5, n_layers: int = 2) -> tf.keras.layers.Layer:
    """
    PennyLane VQC used in the bottleneck:
      H_layer -> AngleEmbedding(Y) -> [ entangling_layer + Rot ] * n_layers
    Outputs PauliZ expectation per qubit (n_qubits dims).
    """
    # Prefer TF device interface to avoid complex128->float32 warnings
    try:
        dev = qml.device("default.qubit.tf", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)

    def H_layer():
        for idx in range(n_qubits):
            qml.Hadamard(wires=idx)

    def Data_AngleEmbedding_layer(inputs):
        qml.templates.AngleEmbedding(inputs, rotation="Y", wires=range(n_qubits))

    def ROT_layer(w):
        for i in range(n_qubits):
            # Avoid iterating over a symbolic tf.Tensor with unpacking
            qml.Rot(w[i, 0], w[i, 1], w[i, 2], wires=i)

    def entangling_layer():
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])

    @qml.qnode(dev, interface="tf")
    def qnode(inputs, weights_1):
        H_layer()
        Data_AngleEmbedding_layer(inputs)
        for k in range(n_layers):
            entangling_layer()
            ROT_layer(weights_1[k])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {"weights_1": (n_layers, n_qubits, 3)}
    return _make_keras_layer(qnode, weight_shapes, output_dim=n_qubits)

