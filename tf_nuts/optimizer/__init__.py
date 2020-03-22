
"""Optimization ops."""

import collections
import tensorflow as tf
import numpy as np

from .adam import AdamWeightDecayOptimizer


def create_optimizer(params):
    return {
        "Adagrad": tf.train.AdagradOptimizer,
        "Adam": tf.train.AdamOptimizer,
        "Ftrl": tf.train.FtrlOptimizer,
        "Momentum": lambda lr: tf.train.MomentumOptimizer(lr, params['momentum']),
        "RMSProp": tf.train.RMSPropOptimizer,
        "SGD": tf.train.GradientDescentOptimizer,
        'AdamWeightedDecay': lambda lr: AdamWeightDecayOptimizer(lr, params['weight_decay_rate'], exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]),
    }[params['optimizer']](params['learning_rate'])


def exponential_decay(learning_rate, step, decay_steps=20000, decay_rate=0.5):
    """Exponential decay.

    When training a model, it is often recommended to lower the learning rate as
    the training progresses.  This function applies an exponential decay function
    to a provided initial learning rate.

    The function returns the decayed learning rate.  It is computed as:

    ```python
    decayed_learning_rate = learning_rate *
                            decay_rate ^ (global_step / decay_steps)
    ```
    """

    learning_rate *= decay_rate**(step // decay_steps)
    return learning_rate


def cyclic_decay(learning_rate, step, decay_steps=1000, decay_rate=0.1):
    """Cyclic decay."""
    min_learning_rate = learning_rate * decay_rate
    cycle = tf.cos(tf.math.mod(step * np.pi / decay_steps, np.pi)) * 0.5 + 0.5
    learning_rate = (
        (learning_rate - min_learning_rate) * cycle + min_learning_rate)
    return learning_rate


def get_learning_rate(params):
    step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)

    learning_rate = params['learning_rate']

    if params['warmup_steps']:
        # Apply inverse decay if global steps less than warmup steps.
        # Inspired by "Attention Is All You Need"
        # https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
        #
        # This corresponds to increasing the learning rate linearly
        # for the first warmup_steps training steps, and decreasing it
        # thereafter proportionally to the inverse square root of the step number.
        #
        # When step < warmup_steps,
        #   learing_rate *= warmup_factor ** (warmup_steps - step)
        #
        # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
        # warmup_factor = tf.exp(tf.math.log(0.01) / params.warmup_steps)
        # inv_decay = warmup_factor**(tf.cast(params.warmup_steps - step,
        #                                     tf.float32))

        # # learning_rate = tf.cond(
        #     step < params.warmup_steps,
        #     lambda: inv_decay * learning_rate,
        #     lambda: learning_rate,
        #     name="lr_warmup_cond")

        warmup_steps = tf.cast(params['warmup_steps'], tf.float32)
        warmup_percent_done = step / warmup_steps
        warmup_learning_rate = learning_rate * warmup_percent_done

        learning_rate = tf.cond(
            step < params['warmup_steps'],
            lambda: warmup_learning_rate,
            lambda: learning_rate,
            name="lr_warmup_cond")

        # learning_rate *= tf.minimum(1., (step + 1.0) / params.warmup_steps)
        step = tf.maximum(0., step - params['warmup_steps'])

    if params['constant_steps']:
        step = tf.maximum(0., step - params['constant_steps'])

    if params['exponential_decay_rate'] < 1:
        learning_rate = exponential_decay(
            learning_rate=learning_rate,
            step=step,
            decay_steps=params['exponential_decay_steps'],
            decay_rate=params['exponential_decay_rate'])

    # if params.lr_decay_method == 'exponential':
    #     learning_rate = exponential_decay(
    #         learning_rate=learning_rate,
    #         step=step,
    #         decay_steps=params.decay_steps,
    #         decay_rate=params.decay_rate)
    # elif params.lr_decay_method == 'cyclic':
    #     learning_rate = cyclic_decay(
    #         learning_rate=learning_rate,
    #         step=step,
    #         decay_steps=params.decay_steps,
    #         decay_rate=params.decay_rate)
    # elif params.lr_decay_method == 'polynomial':
    #     learning_rate = tf.train.polynomial_decay(
    #         learning_rate,
    #         step,
    #         params.decay_steps,
    #         end_learning_rate=0.0,
    #         power=1.0,
    #         cycle=False)

    if params['cycle_decay_rate'] < 1:
        learning_rate = cyclic_decay(
            learning_rate=learning_rate,
            step=step,
            decay_steps=params['cycle_decay_steps'],
            decay_rate=params['cycle_decay_rate'])

    return learning_rate


def average_grads_and_vars(tower_grads_and_vars):
    """Compute average gradients and variables."""
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0) / len(grad_and_vars)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        average_grads_and_vars = []
        for grad_and_vars in zip(*tower_grads_and_vars):
            if grad_and_vars[0][0] is None:
                grad = None
            elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
                grad = average_sparse(grad_and_vars)
            else:
                grad = average_dense(grad_and_vars)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads_and_vars.append(grad_and_var)
        return average_grads_and_vars


def make_parallel(model_fn, features, labels, mode, params, num_gpus):
    with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
        split_features = {
            k: tf.split(v, num_gpus) for k, v in features.items()
        }
        split_labels = {k: tf.split(v, num_gpus) for k, v in labels.items()}

    predictions = collections.defaultdict(list)
    losses = []
    tower_grads_and_vars = []

    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.name_scope("tower_%d" % i) as name_scope:
                with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                    device_features = {
                        k: v[i] for k, v in split_features.items()
                    }
                    device_labels = {k: v[i] for k, v in split_labels.items()}

                    device_predictions, device_loss, device_metrics = model_fn(
                        device_features, device_labels, mode, params)

                    if i == 0:
                        eval_metrics = device_metrics
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                       name_scope)

                        reg_losses = tf.get_collection(
                            tf.GraphKeys.REGULARIZATION_LOSSES, name_scope)

                    for k, v in device_predictions.items():
                        predictions[k].append(v)

                    if device_loss is not None:
                        losses.append(device_loss)

                        device_all_vars = tf.trainable_variables()
                        device_grads = tf.gradients(
                            device_loss, device_all_vars)
                        device_grads_and_vars = list(
                            zip(device_grads, device_all_vars))

                        tower_grads_and_vars.append(device_grads_and_vars)

    for k, v in predictions.items():
        predictions[k] = tf.concat(v, axis=0)

    return predictions, losses, reg_losses, update_ops, eval_metrics, tower_grads_and_vars


def make_model_fn(model_fn,
                  restore_fn=None,
                  num_gpus=None,
                  weight_averaging_decay=None):
    """Build the model function."""

    def _model_fn_wpp(features, labels, mode, params):
        predictions, loss, eval_metrics = model_fn(
            features, labels, mode, params)

        if restore_fn:
            restore_fn()

        return predictions, loss, eval_metrics

    def _model_fn(features, labels, mode, params):
        loss = None
        if num_gpus and num_gpus > 1:
            predictions, losses, reg_losses, update_ops, eval_metrics, tower_grads_and_vars = make_parallel(
                _model_fn_wpp, features, labels, mode, params, num_gpus)
            grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
            grads, tvars = zip(*grads_and_vars)
            # loss = tf.add_n(losses) / len(losses)
        else:
            predictions, loss, eval_metrics = _model_fn_wpp(
                features, labels, mode, params)
            losses = [loss]
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)

        if losses:
            loss = tf.add_n(losses) / len(losses)
            tf.summary.scalar("loss/main", loss)

            if reg_losses:
                loss += tf.add_n(reg_losses)
                tf.summary.scalar("loss/regularization", tf.add_n(reg_losses))

        global_step = tf.train.get_or_create_global_step()

        learning_rate = get_learning_rate(params)
        tf.summary.scalar("learning_rate", learning_rate)

        opt = create_optimizer(params)

        # if weight_averaging_decay is not None:
        #     ema = tf.train.ExponentialMovingAverage(
        #         decay=weight_averaging_decay)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Batch normalization requires UPDATE_OPS to be added as a dependency to
            # the train operation.
            with tf.control_dependencies(update_ops):
                clipped, _ = tf.clip_by_global_norm(
                    grads, params['gradient_max_norm'])
                train_op = opt.apply_gradients(
                    zip(clipped, tvars), global_step=global_step)

        else:
            train_op = None

            # if ema is not None:
            #     saver = tf.train.Saver(ema.variables_to_restore())
            #     tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metrics,
            scaffold=None)

    return _model_fn
