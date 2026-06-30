"""
Experimental Pre-Ictal Risk Model
==================================
Sequence model retained for future-work risk-ranking experiments:

1. Temporal Fusion Transformer (TFT) - Google's SOTA for time series
2. Multi-scale Temporal Convolutional Network (MS-TCN)
3. Self-Attention with Relative Position Encoding
4. Uncertainty Quantification via Monte Carlo Dropout
5. Multi-horizon Prediction Heads

This model scores pre-ictal risk windows 5-30 minutes before annotated onset.
It is not a clinical warning system.

Model components:
- Dual-branch architecture (watch-only vs full sensors)
- Multi-scale temporal feature extraction
- Attention-based temporal modeling
- Calibrated confidence scores for research operating-point review
- Explainable attention maps for interpretability
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from typing import Tuple, Optional, List, Dict


class PositionalEncoding(layers.Layer):
    """Learnable positional encoding for temporal sequences."""

    def __init__(self, max_len: int = 1000, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(self.max_len, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class GatedLinearUnit(layers.Layer):
    """Gated Linear Unit for temporal gating."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.dense = layers.Dense(self.units * 2)

    def call(self, x):
        y = self.dense(x)
        return y[..., : self.units] * tf.sigmoid(y[..., self.units :])

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class GatedResidualNetwork(layers.Layer):
    """
    Gated Residual Network (GRN) from Temporal Fusion Transformer.
    Enables flexible nonlinear processing with skip connections.
    """

    def __init__(self, units: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense1 = layers.Dense(self.units, activation="elu")
        self.dense2 = layers.Dense(self.units)
        self.glu = GatedLinearUnit(self.units)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.norm = layers.LayerNormalization()

        if input_shape[-1] != self.units:
            self.skip_proj = layers.Dense(self.units)
        else:
            self.skip_proj = None

    def call(self, x, training=False):
        skip = self.skip_proj(x) if self.skip_proj else x

        h = self.dense1(x)
        h = self.dropout(h, training=training)
        h = self.dense2(h)
        h = self.dropout(h, training=training)
        h = self.glu(h)

        return self.norm(skip + h)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout_rate": self.dropout_rate})
        return config


class TemporalConvBlock(layers.Layer):
    """
    Temporal Convolutional Block with dilated convolutions.
    Captures multi-scale temporal patterns.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(
            self.filters,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="causal",
            activation="relu",
        )
        self.conv2 = layers.Conv1D(
            self.filters,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="causal",
            activation="relu",
        )
        self.dropout = layers.Dropout(self.dropout_rate)
        self.norm = layers.LayerNormalization()

        if input_shape[-1] != self.filters:
            self.skip_conv = layers.Conv1D(self.filters, 1)
        else:
            self.skip_conv = None

    def call(self, x, training=False):
        skip = self.skip_conv(x) if self.skip_conv else x

        h = self.conv1(x)
        h = self.dropout(h, training=training)
        h = self.conv2(h)
        h = self.dropout(h, training=training)

        return self.norm(skip + h)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class MultiScaleTCN(layers.Layer):
    """
    Multi-scale Temporal Convolutional Network.
    Parallel TCN branches with different dilation rates.
    """

    def __init__(
        self,
        filters: int = 64,
        kernel_size: int = 3,
        dilation_rates: List[int] = None,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates or [1, 2, 4, 8]
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.tcn_blocks = []
        for d in self.dilation_rates:
            block = TemporalConvBlock(
                self.filters, self.kernel_size, d, self.dropout_rate
            )
            self.tcn_blocks.append(block)
        self.combine = layers.Dense(self.filters)

    def call(self, x, training=False):
        outputs = []
        for block in self.tcn_blocks:
            outputs.append(block(x, training=training))
        combined = tf.concat(outputs, axis=-1)
        return self.combine(combined)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rates": self.dilation_rates,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class InterpretableMultiHeadAttention(layers.Layer):
    """
    Multi-Head Attention with interpretable attention weights.
    Returns attention weights for explainability.
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
        )
        self.norm = layers.LayerNormalization()

    def call(self, x, training=False, return_attention=False):
        if return_attention:
            attn_output, attn_weights = self.mha(
                x, x, return_attention_scores=True, training=training
            )
            return self.norm(x + attn_output), attn_weights
        else:
            attn_output = self.mha(x, x, training=training)
            return self.norm(x + attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class VariableSelectionNetwork(layers.Layer):
    """
    Variable Selection Network from TFT.
    Learns to weight importance of different input features.
    """

    def __init__(
        self, num_features: int, units: int, dropout_rate: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        self.grn = GatedResidualNetwork(self.units, self.dropout_rate)
        self.weight_dense = layers.Dense(self.num_features)

    def call(self, x, training=False):

        context = tf.reduce_mean(x, axis=1)
        context = tf.expand_dims(context, 1)
        context = tf.tile(context, [1, tf.shape(x)[1], 1])

        combined = tf.concat([x, context], axis=-1)

        grn_out = self.grn(combined, training=training)
        weights = tf.nn.softmax(self.weight_dense(grn_out), axis=-1)

        return x * weights, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "units": self.units,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class SeizurePredictionTransformer(Model):
    """
    State-of-the-Art Seizure Prediction Model.

    Architecture:
    1. Variable Selection Network (feature importance)
    2. Multi-scale TCN (temporal feature extraction)
    3. Positional Encoding
    4. Transformer Encoder (self-attention)
    5. Multi-horizon Prediction Heads
    6. Uncertainty Quantification (MC Dropout)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 3,
        tcn_filters: int = 64,
        dropout_rate: float = 0.2,
        prediction_horizons: List[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dropout_rate = dropout_rate
        self.tcn_filters = tcn_filters
        self.prediction_horizons = prediction_horizons or [300, 900, 1800]

        self.variable_selection = VariableSelectionNetwork(
            input_dim, d_model, dropout_rate
        )

        self.multi_scale_tcn = MultiScaleTCN(
            filters=tcn_filters,
            kernel_size=3,
            dilation_rates=[1, 2, 4, 8, 16],
            dropout_rate=dropout_rate,
        )

        self.input_projection = layers.Dense(d_model)

        self.pos_encoding = PositionalEncoding(max_len=1000, d_model=d_model)

        self.attention_layers = []
        self.grn_layers = []
        for i in range(num_encoder_layers):
            self.attention_layers.append(
                InterpretableMultiHeadAttention(
                    d_model, num_heads, dropout_rate, name=f"attention_{i}"
                )
            )
            self.grn_layers.append(
                GatedResidualNetwork(d_model, dropout_rate, name=f"grn_{i}")
            )
        self.global_pool = layers.GlobalAveragePooling1D()

        self.prediction_heads = {}
        for horizon in self.prediction_horizons:
            self.prediction_heads[horizon] = keras.Sequential(
                [
                    layers.Dense(64, activation="relu"),
                    layers.Dropout(dropout_rate),
                    layers.Dense(32, activation="relu"),
                    layers.Dropout(dropout_rate),
                    layers.Dense(1, activation="sigmoid"),
                ],
                name=f"head_{horizon}",
            )
        self.shared_dense = keras.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(128, activation="relu"),
            ]
        )

    def call(self, inputs, training=False, return_attention=False):
        x = inputs

        x, var_weights = self.variable_selection(x, training=training)

        x = self.multi_scale_tcn(x, training=training)

        x = self.input_projection(x)

        x = self.pos_encoding(x)

        attention_weights = []
        for i in range(len(self.attention_layers)):
            if return_attention:
                x, attn = self.attention_layers[i](
                    x, training=training, return_attention=True
                )
                attention_weights.append(attn)
            else:
                x = self.attention_layers[i](x, training=training)
            x = self.grn_layers[i](x, training=training)
        x = self.global_pool(x)

        shared_features = self.shared_dense(x, training=training)

        outputs = {}
        for horizon, head in self.prediction_heads.items():
            outputs[horizon] = head(shared_features, training=training)
        if return_attention:
            return outputs, attention_weights, var_weights
        return outputs

    def predict_with_uncertainty(
        self, inputs, n_samples: int = 30, horizon: int = 900
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout for uncertainty quantification.

        Returns:
            mean_prediction: Mean prediction
            std_prediction: Uncertainty (standard deviation)
            confidence: Calibrated confidence score
        """
        predictions = []

        for _ in range(n_samples):
            pred = self(inputs, training=True)
            predictions.append(pred[horizon].numpy())
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        confidence = mean_pred * (1 - std_pred)

        return mean_pred, std_pred, confidence

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "tcn_filters": getattr(self, "tcn_filters", None),
                "dropout_rate": self.dropout_rate,
                "prediction_horizons": list(self.prediction_horizons),
            }
        )
        return config


def build_seizure_prediction_model(
    input_shape: Tuple[int, int],
    prediction_horizons: List[int] = None,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout_rate: float = 0.2,
) -> Model:
    """
    Build the seizure prediction model.

    Args:
        input_shape: (sequence_length, features)
        prediction_horizons: List of prediction horizons in seconds
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    horizons = prediction_horizons or [300, 900, 1800]

    model = SeizurePredictionTransformer(
        input_dim=input_shape[-1],
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        dropout_rate=dropout_rate,
        prediction_horizons=horizons,
    )

    sample_input = tf.zeros((1,) + tuple(input_shape), dtype=tf.float32)
    model(sample_input, training=False)

    return model


def compile_prediction_model(
    model: Model, learning_rate: float = 1e-4, horizons: List[int] = None
) -> Model:
    """Compile the multi-output prediction model."""
    horizons = horizons or [300, 900, 1800]

    losses = {h: "binary_crossentropy" for h in horizons}

    metrics = {
        h: [
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
        for h in horizons
    }

    loss_weights = {300: 0.2, 900: 0.5, 1800: 0.3}

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-5
        ),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    return model


def build_simple_prediction_model(
    input_shape: Tuple[int, int],
    d_model: int = 64,
    dropout_rate: float = 0.2,
) -> Model:
    """
    Simplified prediction model for single horizon.
    Uses same architecture but single output.
    """
    inputs = keras.Input(shape=input_shape)

    x = MultiScaleTCN(filters=64, dilation_rates=[1, 2, 4, 8])(inputs)

    x = layers.Dense(d_model)(x)
    x = PositionalEncoding(max_len=1000, d_model=d_model)(x)

    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=d_model // 4)(x, x)
        x = layers.LayerNormalization()(x + attn)
        x = GatedResidualNetwork(d_model, dropout_rate)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="SeizurePredictionSimple")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model
