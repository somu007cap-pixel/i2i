"""
Phase 3: TSMixer Model with Modality Dropout
=============================================
Dual-Branch TSMixer architecture prepared for compact TensorFlow Lite export.

Features:
- Dual input branches (Primary Watch + Secondary Sensors)
- TSMixer blocks (MLP-Mixer for time series)
- ReLU activation (integer quantization friendly)
- Modality Dropout (forces learning from primary-only baseline)
- Designed for TensorFlow Lite conversion and software-based edge feasibility review
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional

tf.get_logger().setLevel("ERROR")


def edge_safe_model_enabled() -> bool:
    return os.environ.get("DETECTION_EDGE_SAFE_MODEL", "1") == "1"


@keras.utils.register_keras_serializable(package="seizure_detection")
class SensitivitySpecificityWeightedBinaryCrossentropy(keras.losses.Loss):
    """Binary cross-entropy with separate sensitivity and specificity terms."""

    def __init__(
        self,
        sensitivity_weight: float = 0.90,
        specificity_weight: float = 0.10,
        name: str = "sensitivity_specificity_weighted_bce",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sensitivity_weight = float(sensitivity_weight)
        self.specificity_weight = float(specificity_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(
            y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon()
        )
        positive_loss = -y_true * tf.math.log(y_pred)
        negative_loss = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        positive_count = tf.reduce_sum(y_true) + keras.backend.epsilon()
        negative_count = tf.reduce_sum(1.0 - y_true) + keras.backend.epsilon()
        positive_term = tf.reduce_sum(positive_loss) / positive_count
        negative_term = tf.reduce_sum(negative_loss) / negative_count
        return (
            self.sensitivity_weight * positive_term
            + self.specificity_weight * negative_term
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sensitivity_weight": self.sensitivity_weight,
                "specificity_weight": self.specificity_weight,
            }
        )
        return config


class MLPBlock(layers.Layer):
    """
    MLP Block for TSMixer.

    Uses ReLU activation to keep the model friendly to common quantization paths.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        use_layer_norm: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = (
            not edge_safe_model_enabled() if use_layer_norm is None else use_layer_norm
        )

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        self.dense1 = layers.Dense(self.hidden_dim, activation="relu")
        self.dropout = layers.Dropout(self.dropout_rate)
        self.dense2 = layers.Dense(feature_dim)
        if self.use_layer_norm:
            self.norm = layers.LayerNormalization()

    def call(self, x, training=False):
        residual = x
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = x + residual
        return self.norm(x) if self.use_layer_norm else x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
                "use_layer_norm": self.use_layer_norm,
            }
        )
        return config


class TSMixerBlock(layers.Layer):
    """
    TSMixer Block: Time-mixing + Feature-mixing.

    Applies MLP-Mixer style processing:
    1. Time mixing: MLP across time dimension
    2. Feature mixing: MLP across feature dimension
    """

    def __init__(
        self,
        time_hidden_dim: int = 64,
        feature_hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.time_hidden_dim = time_hidden_dim
        self.feature_hidden_dim = feature_hidden_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        time_dim = input_shape[1]
        feature_dim = input_shape[2]

        self.time_mlp = MLPBlock(self.time_hidden_dim, self.dropout_rate)

        self.feature_mlp = MLPBlock(self.feature_hidden_dim, self.dropout_rate)

    def call(self, x, training=False):

        x_t = tf.transpose(x, perm=[0, 2, 1])
        x_t = self.time_mlp(x_t, training=training)
        x = tf.transpose(x_t, perm=[0, 2, 1])

        x = self.feature_mlp(x, training=training)

        return x


class ChannelIndependentTimeMixer(layers.Layer):
    """Time mixer applied independently to each sensor channel."""

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        use_layer_norm: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = (
            not edge_safe_model_enabled() if use_layer_norm is None else use_layer_norm
        )

    def build(self, input_shape):
        self.time_dim = int(input_shape[1])
        self.feature_dim = int(input_shape[2])
        self.dense1 = layers.Dense(self.hidden_dim, activation="relu")
        self.dropout = layers.Dropout(self.dropout_rate)
        self.dense2 = layers.Dense(self.time_dim)
        if self.use_layer_norm:
            self.norm = layers.LayerNormalization()

    def call(self, inputs, training=False):
        residual = inputs
        x = tf.transpose(inputs, perm=[0, 2, 1])
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [-1, self.time_dim])
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = tf.reshape(x, [batch_size, self.feature_dim, self.time_dim])
        x = tf.transpose(x, perm=[0, 2, 1])
        x = x + residual
        return self.norm(x) if self.use_layer_norm else x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
                "use_layer_norm": self.use_layer_norm,
            }
        )
        return config


class ChannelIndependentTSMixerBlock(layers.Layer):
    """TSMixer block that delays cross-channel mixing until after time mixing."""

    def __init__(
        self,
        time_hidden_dim: int = 64,
        feature_hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        use_layer_norm: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.time_hidden_dim = time_hidden_dim
        self.feature_hidden_dim = feature_hidden_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = (
            not edge_safe_model_enabled() if use_layer_norm is None else use_layer_norm
        )

    def build(self, input_shape):
        self.time_mixer = ChannelIndependentTimeMixer(
            self.time_hidden_dim, self.dropout_rate
        )
        self.feature_mlp = MLPBlock(self.feature_hidden_dim, self.dropout_rate)

    def call(self, inputs, training=False):
        x = self.time_mixer(inputs, training=training)
        return self.feature_mlp(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "time_hidden_dim": self.time_hidden_dim,
                "feature_hidden_dim": self.feature_hidden_dim,
                "dropout_rate": self.dropout_rate,
                "use_layer_norm": self.use_layer_norm,
            }
        )
        return config


class GatedTimeMixerBlock(layers.Layer):
    """Time mixer with a learned gate for slower physiological booster signals."""

    def __init__(
        self,
        time_hidden_dim: int = 64,
        feature_hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        use_layer_norm: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.time_hidden_dim = time_hidden_dim
        self.feature_hidden_dim = feature_hidden_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = (
            not edge_safe_model_enabled() if use_layer_norm is None else use_layer_norm
        )

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.candidate = TSMixerBlock(
            self.time_hidden_dim, self.feature_hidden_dim, self.dropout_rate
        )
        self.gate_dense = layers.Dense(feature_dim, activation="sigmoid")
        if self.use_layer_norm:
            self.norm = layers.LayerNormalization()

    def call(self, inputs, training=False):
        candidate = self.candidate(inputs, training=training)
        pooled = tf.reduce_mean(inputs, axis=1)
        gate = tf.expand_dims(self.gate_dense(pooled), axis=1)
        x = inputs + gate * candidate
        return self.norm(x) if self.use_layer_norm else x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "time_hidden_dim": self.time_hidden_dim,
                "feature_hidden_dim": self.feature_hidden_dim,
                "dropout_rate": self.dropout_rate,
                "use_layer_norm": self.use_layer_norm,
            }
        )
        return config


class GatedBoosterFusion(layers.Layer):
    """Use booster features as a gate over the baseline representation."""

    def __init__(self, hidden_dim: int, use_layer_norm: Optional[bool] = None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.use_layer_norm = (
            not edge_safe_model_enabled() if use_layer_norm is None else use_layer_norm
        )

    def build(self, input_shape):
        self.baseline_proj = layers.Dense(self.hidden_dim, activation="relu")
        self.booster_proj = layers.Dense(self.hidden_dim, activation="relu")
        self.gate_proj = layers.Dense(self.hidden_dim, activation="sigmoid")
        self.out_proj = layers.Dense(self.hidden_dim, activation="relu")
        if self.use_layer_norm:
            self.norm = layers.LayerNormalization()

    def call(self, inputs):
        baseline, booster = inputs
        baseline_x = self.baseline_proj(baseline)
        booster_x = self.booster_proj(booster)
        gate = self.gate_proj(booster)
        fused = baseline_x * (1.0 + gate) + booster_x * gate
        x = self.out_proj(fused)
        return self.norm(x) if self.use_layer_norm else x

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_dim": self.hidden_dim, "use_layer_norm": self.use_layer_norm}
        )
        return config


class SummaryStats(layers.Layer):
    """Compact temporal summary features for a sequence tensor."""

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1)
        variance = tf.reduce_mean(tf.square(inputs - tf.expand_dims(mean, axis=1)), axis=1)
        std = tf.sqrt(tf.maximum(variance, 1e-6))
        minimum = tf.reduce_min(inputs, axis=1)
        maximum = tf.reduce_max(inputs, axis=1)
        delta = inputs[:, -1, :] - inputs[:, 0, :]
        return tf.concat([mean, std, minimum, maximum, delta], axis=-1)


class SecondaryFeatureSelector(layers.Layer):
    """Serializable selector for secondary feature channels."""

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = [int(idx) for idx in indices]

    def call(self, inputs):
        if not self.indices:
            return inputs[:, :, :0]
        return tf.gather(inputs, self.indices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"indices": self.indices})
        return config


class ChannelPatchEmbedding(layers.Layer):
    """
    Convert raw time samples into per-channel temporal patches.

    The same small projection is applied to each sensor channel independently.
    This reduces the mixer sequence length and avoids early contamination across
    heterogeneous biosignals.
    """

    def __init__(
        self,
        patch_size: int = 8,
        embed_dim: int = 8,
        use_layer_norm: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.use_layer_norm = (
            not edge_safe_model_enabled() if use_layer_norm is None else use_layer_norm
        )

    def build(self, input_shape):
        self.time_dim = int(input_shape[1])
        self.feature_dim = int(input_shape[2])
        self.num_patches = max(1, self.time_dim // self.patch_size)
        self.trim_length = self.num_patches * self.patch_size
        self.patch_proj = layers.Dense(self.embed_dim, activation="relu")
        if self.use_layer_norm:
            self.norm = layers.LayerNormalization()

    def call(self, inputs):
        x = inputs[:, : self.trim_length, :]
        batch_size = tf.shape(x)[0]
        x = tf.reshape(
            x,
            [
                batch_size,
                self.num_patches,
                self.patch_size,
                self.feature_dim,
            ],
        )
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        x = tf.reshape(x, [-1, self.patch_size])
        x = self.patch_proj(x)
        x = tf.reshape(
            x,
            [
                batch_size,
                self.num_patches,
                self.feature_dim * self.embed_dim,
            ],
        )
        return self.norm(x) if self.use_layer_norm else x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "use_layer_norm": self.use_layer_norm,
            }
        )
        return config


class ModalityDropout(layers.Layer):
    """
    Modality Dropout Layer.

    During training, randomly zeros out entire secondary branch with probability p.
    This forces the model to learn a baseline using only the primary branch.

    During inference, this layer is a no-op (or can be used to simulate "watch-only" mode).
    """

    def __init__(self, dropout_prob: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.dropout_prob = dropout_prob

    def call(self, inputs, training=False):
        if not training:
            return inputs
        batch_size = tf.shape(inputs)[0]
        mask = tf.random.uniform([batch_size, 1, 1]) > self.dropout_prob
        mask = tf.cast(mask, inputs.dtype)

        return inputs * mask

    def get_config(self):
        config = super().get_config()
        config.update({"dropout_prob": self.dropout_prob})
        return config


def build_tsmixer_branch(
    input_shape: Tuple[int, int],
    n_blocks: int = 2,
    time_hidden_dim: int = 32,
    feature_hidden_dim: int = 32,
    dropout_rate: float = 0.1,
    name: str = "branch",
) -> Tuple[layers.Input, layers.Layer]:
    """
    Build a single TSMixer branch.

    Args:
        input_shape: (window_size, n_features)
        n_blocks: Number of TSMixer blocks
        time_hidden_dim: Hidden dim for time mixing MLP
        feature_hidden_dim: Hidden dim for feature mixing MLP
        dropout_rate: Dropout rate
        name: Branch name

    Returns:
        (input_layer, output_tensor)
    """
    inputs = layers.Input(shape=input_shape, name=f"{name}_input")
    x = inputs

    x = layers.Dense(feature_hidden_dim, activation="relu", name=f"{name}_proj")(x)

    for i in range(n_blocks):
        x = TSMixerBlock(
            time_hidden_dim=time_hidden_dim,
            feature_hidden_dim=feature_hidden_dim,
            dropout_rate=dropout_rate,
            name=f"{name}_tsmixer_{i}",
        )(x)
    return inputs, x


def build_channel_independent_branch(
    inputs,
    n_blocks: int,
    time_hidden_dim: int,
    feature_hidden_dim: int,
    dropout_rate: float,
    name: str,
    patch_size: int = 8,
    patch_embed_dim: int = 8,
):
    if patch_size > 1:
        x = ChannelPatchEmbedding(
            patch_size=patch_size,
            embed_dim=patch_embed_dim,
            name=f"{name}_patch_embedding",
        )(inputs)
    else:
        x = inputs
    x = layers.Dense(feature_hidden_dim, activation="relu", name=f"{name}_proj")(x)
    for i in range(n_blocks):
        x = ChannelIndependentTSMixerBlock(
            time_hidden_dim=time_hidden_dim,
            feature_hidden_dim=feature_hidden_dim,
            dropout_rate=dropout_rate,
            name=f"{name}_channel_independent_tsmixer_{i}",
        )(x)
    return x


def build_gated_booster_branch(
    inputs,
    n_blocks: int,
    time_hidden_dim: int,
    feature_hidden_dim: int,
    dropout_rate: float,
    name: str,
    patch_size: int = 8,
    patch_embed_dim: int = 8,
):
    if patch_size > 1:
        x = ChannelPatchEmbedding(
            patch_size=patch_size,
            embed_dim=patch_embed_dim,
            name=f"{name}_patch_embedding",
        )(inputs)
    else:
        x = inputs
    x = layers.Dense(feature_hidden_dim, activation="relu", name=f"{name}_proj")(x)
    for i in range(n_blocks):
        x = GatedTimeMixerBlock(
            time_hidden_dim=time_hidden_dim,
            feature_hidden_dim=feature_hidden_dim,
            dropout_rate=dropout_rate,
            name=f"{name}_gated_tsmixer_{i}",
        )(x)
    return x


def temporal_pooling(x, name: str):
    avg = layers.GlobalAveragePooling1D(name=f"{name}_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name=f"{name}_max_pool")(x)
    return layers.Concatenate(name=f"{name}_temporal_pool")([avg, max_pool])


def feature_indices(names):
    mapping = {"BVP": 0, "HR": 1, "EDA": 2, "TEMP": 3}
    return [mapping[name] for name in names if name in mapping]


def select_secondary_features(secondary_input, feature_names, name):
    indices = feature_indices(feature_names)
    return SecondaryFeatureSelector(indices, name=f"{name}_secondary_features")(
        secondary_input
    )


def build_dual_branch_tsmixer(
    primary_shape: Tuple[int, int] = (32, 3),
    secondary_shape: Tuple[int, int] = (32, 4),
    n_blocks: int = 2,
    hidden_dim: int = 32,
    modality_dropout_prob: float = 0.3,
    output_dim: int = 1,
    baseline_secondary_features: Tuple[str, ...] = ("EDA", "TEMP"),
    booster_secondary_features: Tuple[str, ...] = ("BVP", "HR"),
    patch_size: int = 8,
    patch_embed_dim: int = 8,
) -> Model:
    """
    Build the Dual-Branch TSMixer model for seizure detection.

    Architecture:
    - Primary Branch (Watch): ACC_x, ACC_y, ACC_z -> TSMixer -> embedding
    - Secondary Branch (Add-on): PPG outputs (BVP, HR) plus EDA, TEMP -> TSMixer -> embedding
    - Fusion: Concatenate embeddings -> MLP -> sigmoid output

    Args:
        primary_shape: Input shape for primary (watch) branch
        secondary_shape: Input shape for secondary (sensors) branch
        n_blocks: Number of TSMixer blocks per branch
        hidden_dim: Hidden dimension for TSMixer
        modality_dropout_prob: Probability of dropping secondary modality during training
        output_dim: Output dimension (1 for binary classification)

    Returns:
        Keras Model with two inputs and one output
    """

    primary_input = layers.Input(shape=primary_shape, name="primary_input")
    secondary_input = layers.Input(shape=secondary_shape, name="secondary_input")

    baseline_secondary = select_secondary_features(
        secondary_input, baseline_secondary_features, "baseline"
    )
    baseline_input = layers.Concatenate(name="baseline_stream_input")(
        [primary_input, baseline_secondary]
    )
    primary_out = build_channel_independent_branch(
        inputs=baseline_input,
        n_blocks=n_blocks,
        time_hidden_dim=hidden_dim * 2,
        feature_hidden_dim=hidden_dim,
        dropout_rate=0.1,
        name="baseline",
        patch_size=patch_size,
        patch_embed_dim=patch_embed_dim,
    )

    booster_secondary = select_secondary_features(
        secondary_input, booster_secondary_features, "booster"
    )
    booster_secondary = ModalityDropout(dropout_prob=modality_dropout_prob)(
        booster_secondary
    )
    secondary_x = build_gated_booster_branch(
        inputs=booster_secondary,
        n_blocks=n_blocks,
        time_hidden_dim=hidden_dim * 2,
        feature_hidden_dim=hidden_dim,
        dropout_rate=0.1,
        name="booster",
        patch_size=patch_size,
        patch_embed_dim=patch_embed_dim,
    )

    fused_sequence = layers.Concatenate(name="sequence_fusion")(
        [primary_out, secondary_x]
    )
    fused_sequence = layers.Dense(hidden_dim, activation="relu", name="fusion_proj")(
        fused_sequence
    )
    fusion_blocks = int(os.environ.get("DETECTION_FUSION_TSMIXER_BLOCKS", "2"))
    for i in range(fusion_blocks):
        fused_sequence = TSMixerBlock(
            time_hidden_dim=hidden_dim * 2,
            feature_hidden_dim=hidden_dim,
            dropout_rate=0.1,
            name=f"fusion_tsmixer_{i}",
        )(fused_sequence)

    primary_pooled = temporal_pooling(primary_out, "primary")
    secondary_pooled = temporal_pooling(secondary_x, "secondary")
    fusion_pooled = temporal_pooling(fused_sequence, "fusion")
    use_summary_stats = os.environ.get("DETECTION_EDGE_SAFE_MODEL", "1") != "1"
    if use_summary_stats:
        primary_stats = SummaryStats(name="primary_summary_stats")(baseline_input)
        secondary_stats = SummaryStats(name="secondary_summary_stats")(booster_secondary)
    else:
        primary_stats = layers.GlobalAveragePooling1D(name="primary_edge_avg")(
            baseline_input
        )
        secondary_stats = layers.GlobalAveragePooling1D(name="secondary_edge_avg")(
            booster_secondary
        )

    baseline_embedding = layers.Concatenate(name="baseline_embedding")(
        [primary_pooled, primary_stats]
    )
    booster_embedding = layers.Concatenate(name="booster_embedding")(
        [secondary_pooled, fusion_pooled, secondary_stats]
    )
    fused = GatedBoosterFusion(hidden_dim * 2, name="gated_booster_fusion")(
        [baseline_embedding, booster_embedding]
    )
    if not edge_safe_model_enabled():
        fused = layers.LayerNormalization(name="fusion_norm")(fused)

    x = layers.Dense(hidden_dim * 2, activation="relu", name="fc1")(fused)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(hidden_dim, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(max(hidden_dim // 2, 8), activation="relu", name="fc3")(x)
    output = layers.Dense(output_dim, activation="sigmoid", name="output")(x)

    model = Model(
        inputs=[primary_input, secondary_input],
        outputs=output,
        name="PatchedDualStreamTSMixer",
    )

    return model


def compile_model(model: Model, learning_rate: float = 1e-3) -> Model:
    """Compile the model with appropriate loss and metrics."""
    loss_name = os.environ.get("DETECTION_LOSS", "focal").strip().lower()
    if loss_name == "focal":
        loss = keras.losses.BinaryFocalCrossentropy(
            gamma=float(os.environ.get("DETECTION_FOCAL_GAMMA", "2.0")),
            apply_class_balancing=False,
        )
    elif loss_name in {"sswce", "sensitivity_specificity"}:
        loss = SensitivitySpecificityWeightedBinaryCrossentropy(
            sensitivity_weight=float(
                os.environ.get("DETECTION_SENSITIVITY_WEIGHT", "0.90")
            ),
            specificity_weight=float(
                os.environ.get("DETECTION_SPECIFICITY_WEIGHT", "0.10")
            ),
        )
    else:
        loss = "binary_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def count_trainable_parameters(model: Model) -> int:
    """Return the number of trainable parameters in a Keras model."""
    return int(np.sum([np.prod(weight.shape) for weight in model.trainable_weights]))


def model_size_profile(model: Model) -> dict:
    """Small footprint summary used by edge/TinyML experiment reports."""
    total = int(model.count_params())
    trainable = count_trainable_parameters(model)
    return {
        "model_name": model.name,
        "parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": int(total - trainable),
        "under_100k_parameters": bool(total < 100_000),
        "under_1m_parameters": bool(total < 1_000_000),
    }


def get_model_summary(model: Model) -> str:
    """Get model summary as string."""
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)
