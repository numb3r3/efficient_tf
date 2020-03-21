import numpy as np
import tensorflow as tf


def netvlad_layer(input_x, max_samples, cluster_size: int = 128, training: bool =True, name: str=None, reuse: bool =None):
    """Implementation from https://github.com/antoine77340/LOUPE/blob/master/loupe.py

    Args:
        reshaped_input: If your input is in that form:
            'batch_size' x 'max_samples' x D
        It should be reshaped in the following form:
            'batch_size*max_samples' x D
        by performing:
            reshaped_input = tf.reshape(input, [-1, D])
    Returns:
        vlad: the pooled vector of size: 'batch_size' x 'num_units'
    """
    with tf.variable_scope(name, default_name="netvlad_layer", reuse=reuse):
        # short cuts variables
        L = max_samples
        D = input_x.shape.as_list()[-1]
        K = cluster_size

        reshaped_input = tf.reshape(input_x, [-1, D])
        
        # soft-assignment.
        cluster_weights = tf.get_variable("cluster_weights",
                                          [D, K],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / np.math.sqrt(D)))  # D x K

        activation = tf.matmul(reshaped_input, cluster_weights)  # (B x L) x K
        cluster_biases = tf.get_variable("cluster_biases",
                                         [K],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / np.math.sqrt(D)))  # K
        activation += cluster_biases  # (B x L) x K
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, L, K])  # B x L x K

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)  # B x 1 x K

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, D, K],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / np.math.sqrt(D)))  # 1 x D x K

        a = tf.multiply(a_sum, cluster_weights2)  # B x D x K

        activation = tf.transpose(activation, perm=[0, 2, 1])  # B x K x L

        # reshaped_input = tf.reshape(reshaped_input, [-1, L, D])  # B x L x D

        vlad = tf.matmul(activation, input_x)  # B x K x D

        vlad = tf.transpose(vlad, perm=[0, 2, 1])  # B x D x K
        vlad = tf.subtract(vlad, a)  # B x D x K
        
        # vlad = tf.nn.l2_normalize(vlad, 1)

        # vlad = tf.reshape(vlad, [-1, K*D])  # B x (D x K)
        # vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad


