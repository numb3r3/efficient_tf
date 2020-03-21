import tensorflow as tf

from ..ops import regularizer_ops
from ..ops import noise_ops

def _to_array(value, size, default_val=None):
    if value is None:
        value = [default_val] * size
    elif not isinstance(value, list):
        value = [value] * size
    return value

def dense_layers(tensor,
                 units,
                 activation=tf.nn.relu,
                 linear_top_layer=False,
                 drop_rates=None,
                 drop_type="regular",
                 batch_norm=False,
                 training=True,
                 name=None,
                 reuse=None,
                 **kwargs):
    """Builds a stack of fully connected layers with optional dropout."""
    
    with tf.variable_scope(name, default_name="dense_layers", reuse=reuse):
        drop_rates = _to_array(drop_rates, len(units), 0.)
        kernel_initializer = tf.glorot_uniform_initializer()

        for i, (size, drp) in enumerate(zip(units, drop_rates)):
            if i == len(units) - 1 and linear_top_layer:
                activation = None
            with tf.variable_scope("dense_block_%d" % i):
                tensor = noise_ops.dropout(
                    tensor, drp, training=training, type=drop_type)

                tensor = tf.layers.dense(
                    tensor,
                    size,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    **kwargs)
                # tensor = tf.keras.layers.Dense(
                #     size,
                #     use_bias=False,
                #     kernel_initializer=kernel_initializer,
                #     kernel_regularizer=kernel_regularizer,
                #     **kwargs).apply(tensor)
                if batch_norm:
                    tensor = tf.layers.batch_normalization(
                        tensor, training=training)
                if activation:
                    tensor = activation(tensor)
        return tensor

