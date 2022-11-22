import tensorflow as tf

class MaskedConv1D(tf.keras.layers.Conv1D):
    def compute_mask(self, inputs, mask=None):
        return mask
