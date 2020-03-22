import tensorflow as tf
from tensorflow.contrib import autograph


@autograph.convert()
def stop_grad(global_step, tensor, freeze_step):

    if freeze_step > 0:
        if global_step <= freeze_step:
            tensor = tf.stop_gradient(tensor)

    return tensor


@autograph.convert()
def filter_loss(loss, features, problem):

    if tf.reduce_mean(features['%s_loss_multiplier' % problem]) == 0:
        return_loss = 0.0
    else:
        return_loss = loss

    return return_loss