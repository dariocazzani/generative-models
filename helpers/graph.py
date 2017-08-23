import tensorflow as tf

def get_variables(shape, scope):
    xavier = tf.contrib.layers.xavier_initializer()
    const = tf.constant_initializer(0.1)
    W = tf.get_variable('weight_{}'.format(scope), shape, initializer=xavier)
    b = tf.get_variable('bias_{}'.format(scope), shape[-1], initializer=const)
    return W, b

def linear(_input, output_dim, scope=None):
    with tf.variable_scope(scope, reuse=None):
        shape = [int(_input.get_shape()[1]), output_dim]
        W, b = get_variables(shape, scope)
        return tf.matmul(_input, W) + b

def AdamOptimizer(loss, lr, beta1):
    clip_grad = False
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
    grads_and_vars = optimizer.compute_gradients(loss)
    if clip_grad:
        grads_and_vars = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads_and_vars]
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op, grads_and_vars
