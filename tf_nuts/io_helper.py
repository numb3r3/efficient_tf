import tensorflow as tf


def make_input_fn(dataset, split, mode, params,
                  shuffle_batches=10,
                  num_parallel_batches=5,
                  num_parallel_calls=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def _parse_record(record):
        return dataset.parse(mode, params, record)

    def _input_fn(params):
        with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
            d = dataset.read(split, params)

            if mode == tf.estimator.ModeKeys.TRAIN:
                d = d.shuffle(params['batch_size'] * shuffle_batches)

            d = d.apply(
                tf.data.experimental.map_and_batch(
                    lambda record: _parse_record(record),
                    batch_size=params['batch_size'],
                    num_parallel_batches=num_parallel_batches,
                    num_parallel_calls=num_parallel_calls,
                    drop_remainder=False))

            return d

    return _input_fn


def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Dequantize the feature from the byte format to the float format.
    Args:
      feat_vector: the input 1-d vector.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.
    Returns:
      A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def quantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Quantize the feature from the float format to the byte format.
    Args:
      feat_vector: the input 1-d vector.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.
    Returns:
      A byte vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return (feat_vector - bias) / scalar


def encode(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Encode feature list."""
    feat_vector[feat_vector >= 2] = 2
    feat_vector[feat_vector <= -2] = -2
    feat_vector = quantize(feat_vector, max_quantized_value,
                           min_quantized_value)
    feat_list = []
    for idx in range(feat_vector.shape[0]):
        feat_list.append(feat_vector[idx, :].astype(np.uint8).tobytes())
    return feat_list


def resize_axis(tensor, axis, new_size, fill_value=0):
    """Truncate or pad a tensor to new_size on on a given axis.
    Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
    size increases, the padding will be performed at the end, using fill_value.
    Args:
      tensor: The tensor to be resized.
      axis: An integer representing the dimension to be sliced.
      new_size: An integer or 0d tensor representing the new value for
        tensor.shape[axis].
      fill_value: Value to use to fill any new entries in the tensor. Will be
        cast to the type of tensor.
    Returns:
      The resized tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.unstack(tf.shape(tensor))

    pad_shape = shape[:]
    pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

    shape[axis] = tf.minimum(shape[axis], new_size)
    shape = tf.stack(shape)

    resized = tf.concat([
        tf.slice(tensor, tf.zeros_like(shape), shape),
        tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
    ], axis)

    # Update shape.
    new_shape = tensor.get_shape().as_list()    # A copy is being made.
    new_shape[axis] = new_size
    resized.set_shape(new_shape)
    return resized
