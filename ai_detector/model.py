"""Lightweight GRU classifier architecture.

Sized deliberately small (single GRU layer, 16 units) so that after INT8
post-training quantization the model comfortably fits ARM Cortex-M4 class
budgets (typically <=256KB flash / <=64KB RAM for the ML workload, alongside
the crypto and QKD-interface firmware).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_gru_model(
    window_size: int,
    n_features: int,
    gru_units: int = 32,
    dense_units: int = 16,
    unroll: bool = False,
    n_classes: int | None = None,
) -> keras.Model:
    """`unroll=True` statically unrolls the GRU's time loop. Training is
    equally correct either way, but TFLite's converter cannot lower the
    dynamic TensorListReserve op used by a rolled RNN, so the model must be
    rebuilt with `unroll=True` before conversion (see quantize.py).

    `n_classes` controls the output head:
      - None (default) — binary intrusion detector: single sigmoid output,
        exactly the original architecture used by the escalate/de-escalate
        switch controller. Unchanged from before this parameter existed.
      - An int >= 2 — multi-class attack-*type* classifier: softmax output
        over that many classes (e.g. benign/eavesdrop/jamming/pns), trained
        with sparse categorical crossentropy. This is used only for the
        additive attack-type tagging + fleet correlation, never for the
        core escalate/de-escalate decision.
    """
    inputs = keras.Input(shape=(window_size, n_features), name="qber_window")
    x = layers.GRU(gru_units, name="gru", unroll=unroll)(inputs)
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)

    if n_classes is None:
        outputs = layers.Dense(1, activation="sigmoid", name="intrusion_prob")(x)
        model = keras.Model(inputs, outputs, name="qsafe_gru_detector")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
    else:
        outputs = layers.Dense(n_classes, activation="softmax", name="attack_type_probs")(x)
        model = keras.Model(inputs, outputs, name="qsafe_gru_attack_type_classifier")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
    return model


def count_params(model: keras.Model) -> int:
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))
