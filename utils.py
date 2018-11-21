import tensorflow as tf
import math

def minimize_clipped(optimizer, value, max_gradient_norm, var=None):
    global_step = tf.train.get_global_step()
    if var is None:
        var = tf.trainable_variables()
    gradients = tf.gradients(value, var)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    grad_dict = dict(zip(var, clipped_gradients))
    op = optimizer.apply_gradients(zip(clipped_gradients, var), global_step=global_step)
    return op, grad_dict
