"""Custom tf.keras layers."""
import tensorflow as tf


class StaggeredConv1D(tf.keras.layers.Layer):
    """Layers which use two CNN to scan alternately."""

    def __init__(self, *args, **kwargs):
        """Initialize underlying layers."""
        super().__init__()
        self.conv1d_a = tf.keras.layers.Conv1D(*args, padding='same', **kwargs)
        self.conv1d_b = tf.keras.layers.Conv1D(*args, padding='same', **kwargs)
        self.n_filters = self.conv1d_a.filters

    def call(self, inputs, training=None, mask=None):
        """Forward pass."""
        input_shape = tf.keras.backend.int_shape(inputs)
        output_shape = (-1, input_shape[1], self.n_filters)
        conv_a = self.conv1d_a(inputs)[:, ::2, :]
        conv_b = self.conv1d_b(inputs)[:, 1::2, :]
        # Currently not support sequence with odd length
        staggered = tf.keras.backend.reshape(
            tf.keras.backend.stack([conv_a, conv_b], axis=-2), output_shape)
        return staggered

    def compute_mask(self, inputs, mask=None):
        """Compute mask for sequence."""
        return mask
