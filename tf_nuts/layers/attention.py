import tensorflow as tf

from .. import ops


def scaled_dot_attention(q, k, v, mask,
                         name='scaled_dot_attention',
                         reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention += (mask * -1e9)

        weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(weights, v)

        return output, weights


def multihead_attention(query, memory, query_mask, memory_mask,
                        num_units=None,
                        num_heads=8,
                        training=True,
                        dropout_rate=.0,
                        name="multihead_attn",
                        reuse=None):
    """Appllies multihead attention.

    Args:
      query: A tensor with shape of [B, T_q, D].
      memory: A tensor with shape of [B, T_m, D].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      name: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (B, T_q, D)

    """
    if num_units is None:
        num_units = memory.get_shape().as_list()[-1]

    depth = (num_units // num_heads)

    with tf.variable_scope(name, reuse=reuse):
        # Linear projections
        Q = tf.layers.dense(
            query, num_units,
            use_bias=False,
            activation=tf.nn.relu,
            name="Q")
        K = tf.layers.dense(
            memory, num_units,
            use_bias=False,
            activation=tf.nn.relu,
            name="K")
        V = tf.layers.dense(
            memory, num_units,
            use_bias=False,
            activation=tf.nn.relu,
            name="V")

        # Split the matrix to multiple heads and then concat to have a larger
        Q_split = tf.concat(
            tf.split(Q, num_heads,
                     axis=2),
            axis=0)    # (h*B, T_q, D/h)
        K_split = tf.concat(
            tf.split(K, num_heads,
                     axis=2),
            axis=0)    # (h*B, T_m, D/h)
        V_split = tf.concat(
            tf.split(V, num_heads,
                     axis=2),
            axis=0)    # (h*B, T_m, D/h)

        # Scaled dot product
        outputs = tf.matmul(Q_split, tf.transpose(
            K_split, [0, 2, 1]))    # (h*B, T_q, T_m)

        # Scale by sqrt(dimention)
        outputs = outputs / tf.sqrt(tf.cast(depth, tf.float32))

        # Memory masking
        tiled_memory_mask = tf.tile(memory_mask, [num_heads, 1])    # (h*B, T_m)
        tiled_memory_mask = tf.tile(
            tf.expand_dims(tiled_memory_mask, 1), [1, tf.shape(query)[1], 1])    # (h*B, T_q, T_m)

        outputs = tf.multiply(outputs, tiled_memory_mask) - (1.0 - tiled_memory_mask) * (
            1e30)
        # paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        # outputs = tf.where(tf.equal(mask1, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)    # (h*B, T_q, T_m)

        # Query masking
        titled_query_mask = tf.tile(query_mask, [num_heads, 1])    # (h*B, T_q)
        titled_query_mask = tf.tile(
            tf.expand_dims(titled_query_mask, -1), [1, 1, tf.shape(memory_mask)[1]])    # (h*B, T_q, T_m)
        outputs *= titled_query_mask    # (h*B, T_q, T_m)

        # Dropouts
        outputs = tf.layers.dropout(
            outputs,
            rate=dropout_rate,
            training=training)

        # Weighted sum
        outputs = tf.matmul(outputs, V_split)    # ( h*B, T_q, D/h)

        # Restore shape
        outputs = tf.concat(
            tf.split(outputs, num_heads,
                     axis=0),
            axis=2)    # (B, T_q, D)

        # Residual connection
        if query.get_shape().as_list()[-1] == num_units:
            outputs += query

        # Layernorm
        outputs = ops.layer_norm(outputs, name="layer_norm")

        return outputs


def self_attention(input, input_mask,
                   num_units=None,
                   num_heads=8,
                   training=True,
                   dropout_rate=.0,
                   name="self_attention",
                   reuse=None):

    return multihead_attention(
        input, input, input_mask, input_mask,
        num_units=num_units,
        num_heads=num_heads,
        training=training,
        dropout_rate=dropout_rate,
        name=name,
        reuse=reuse)
