import functools
import tensorflow as tf

from ..ops import regularizer_ops
from ..ops import noise_ops


def _to_array(value, size, default_val=None):
    if value is None:
        value = [default_val] * size
    elif not isinstance(value, list):
        value = [value] * size
    return value



def _merge_dicts(dict, *others):
    dict = dict.copy()
    for other in others:
        dict.update(other)
    return dict


def conv_layers(tensor,
                filters,
                kernels,
                strides=None,
                pool_sizes=None,
                pool_strides=None,
                padding="same",
                activation=tf.nn.relu,
                linear_top_layer=False,
                drop_rates=None,
                drop_type="regular",
                conv_method="conv",
                pool_method="conv",
                pool_activation=None,
                dilations=None,
                batch_norm=False,
                training=False,
                weight_decay=0.0,
                weight_regularizer="l2",
                **kwargs):
    """Builds a stack of convolutional layers with dropout and max pooling."""
    if not filters:
        return tensor

    kernels = _to_array(kernels, len(filters), 1)
    pool_sizes = _to_array(pool_sizes, len(filters), 1)
    pool_strides = _to_array(pool_strides, len(filters), 1)
    strides = _to_array(strides, len(filters), 1)
    drop_rates = _to_array(drop_rates, len(filters), 0.)
    dilations = _to_array(dilations, len(filters), 1)
    conv_method = _to_array(conv_method, len(filters), "conv")
    pool_method = _to_array(pool_method, len(filters), "conv")

    kernel_initializer = tf.glorot_uniform_initializer()
    kernel_regularizer = regularizer_ops.weight_regularizer(
        weight_decay, weight_regularizer)

    conv = {
        "conv":
        functools.partial(
            tf.keras.layers.Conv2D,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer),
        "transposed":
        functools.partial(
            tf.keras.layers.Conv2DTranspose,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer),
        "separable":
        functools.partial(
            tf.keras.layers.SeparableConv2D,
            depthwise_initializer=kernel_initializer,
            pointwise_initializer=kernel_initializer,
            depthwise_regularizer=kernel_regularizer,
            pointwise_regularizer=kernel_regularizer),
    }

    for i, (fs, ks, ss, pz, pr, drp, dl, cm, pm) in enumerate(
            zip(filters, kernels, strides, pool_sizes, pool_strides,
                drop_rates, dilations, conv_method, pool_method)):

        with tf.variable_scope("conv_block_%d" % i):
            if i == len(filters) - 1 and linear_top_layer:
                activation = None
                pool_activation = None
            tensor = noise_ops.dropout(
                tensor, drp, training=training, type=drop_type)
            if dl > 1:
                conv_kwargs = _merge_dicts(kwargs, {"dilation_rate": dl})
            else:
                conv_kwargs = kwargs

            tensor = conv[cm](
                filters=fs,
                kernel_size=ks,
                strides=ss,
                padding=padding,
                use_bias=False,
                name="conv2d",
                **conv_kwargs).apply(tensor)
            if activation:
                if batch_norm:
                    tensor = tf.layers.batch_normalization(
                        tensor, training=training)
                tensor = activation(tensor)
            if pz > 1:
                if pm == "max":
                    tensor = tf.keras.layers.MaxPool2D(
                        pz, pr, padding, name="max_pool").apply(tensor)
                elif pm == "std":
                    tensor = tf.space_to_depth(
                        tensor, pz, name="space_to_depth")
                elif pm == "dts":
                    tensor = tf.depth_to_space(
                        tensor, pz, name="depth_to_space")
                else:
                    tensor = conv["conv"](
                        fs,
                        pz,
                        pr,
                        padding,
                        use_bias=False,
                        name="strided_conv2d",
                        **kwargs).apply(tensor)
                    if pool_activation:
                        if batch_norm:
                            tensor = tf.layers.batch_normalization(
                                tensor, training=training)
                        tensor = pool_activation(tensor)
    return tensor
