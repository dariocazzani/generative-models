#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Authors:    Dario Cazzani
"""
import sys
sys.path.append('../')
from config import set_config
from helpers.misc import check_tf_version, extend_options

import subprocess
import tensorflow as tf
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
input_dim = mnist.train.images.shape[1]
hidden_layer1 = 1000
hidden_layer2 = 1000

def generate_image_grid(sess, decoder_input, op):
    x_points = np.arange(0, 2, 1.5).astype(np.float32)
    y_points = np.arange(0, 2, 1.5).astype(np.float32)

    nx, ny = len(x_points), len(y_points)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
        z = np.reshape(z, (1, 2))
        x = sess.run(op, feed_dict={decoder_input: z})
        ax = plt.subplot(g)
        img = np.array(x.tolist()).reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.show()

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

# The autoencoder network
def encoder(x, reuse=False):
    """
    Encode part of the autoencoder
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_linear_1 = tf.nn.relu(linear(x, hidden_layer1, 'e_linear_1'))
        e_linear_2 = tf.nn.relu(linear(e_linear_1, hidden_layer2, 'e_linear_2'))
        latent_variable = linear(e_linear_2, options.z_dim, 'e_latent_variable')
        return latent_variable


def decoder(z, reuse=False):
    """
    Decoder part of the autoencoder
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_linear_1 = tf.nn.relu(linear(z, hidden_layer2, 'd_linear_1'))
        d_linear_2 = tf.nn.relu(linear(d_linear_1, hidden_layer1, 'd_linear_2'))
        output = tf.nn.sigmoid(linear(d_linear_2, input_dim, 'd_output'))
        return output


def train(options):
    # Placeholders for input data and the targets
    with tf.name_scope('Input'):
        x_input = tf.placeholder(dtype=tf.float32, shape=[options.batch_size, input_dim], name='Input')
        input_images = tf.reshape(x_input, [-1, 28, 28, 1])

    with tf.name_scope('Target'):
        x_target = tf.placeholder(dtype=tf.float32, shape=[options.batch_size, input_dim], name='Target')
    with tf.name_scope('Decoder_input'):
        decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, options.z_dim], name='Decoder_input')

    with tf.name_scope('Autoencoder'):
        with tf.variable_scope(tf.get_variable_scope()):
            encoder_output = encoder(x_input)
            decoder_output = decoder(encoder_output)
            generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

    with tf.variable_scope(tf.get_variable_scope()):
        decoder_image = decoder(decoder_input, reuse=True)

    # Loss - MSE
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Optimizer
    train_op, grads_and_vars = AdamOptimizer(loss, options.learning_rate, options.beta1)

    # Visualization
    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)

    for grad, var in grads_and_vars:
        tf.summary.histogram(var.name + '/gradient', grad)

    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    step = 0
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if not options.run_inference:
            try:
                writer = tf.summary.FileWriter(logdir=options.tensorboard_path, graph=sess.graph)
                for i in range(options.epochs):
                    n_batches = int(mnist.train.num_examples / options.batch_size)
                    for b in range(n_batches):
                        batch_x, _ = mnist.train.next_batch(options.batch_size)
                        sess.run(train_op, feed_dict={x_input: batch_x, x_target: batch_x})
                        if b % 50 == 0:
                            batch_loss, summary = sess.run([loss, summary_op], feed_dict={x_input: batch_x, x_target: batch_x})
                            writer.add_summary(summary, global_step=step)
                            print("Loss: {}".format(batch_loss))
                            print("Epoch: {}, iteration: {}".format(i, b))
                            with open(options.logs_path + '/log.txt', 'a') as log:
                                log.write("Epoch: {}, iteration: {}\n".format(i, b))
                                log.write("Loss: {}\n".format(batch_loss))
                        step += 1
                    saver.save(sess, save_path=options.checkpoints_path, global_step=step)
                print("Model Trained!")
                print("Tensorboard Path: {}".format(options.tensorboard_path))
                print("Log Path: {}".format(options.logs_path + '/log.txt'))
                print("Saved Model Path: {}".format(options.checkpoints_path))
            except KeyboardInterrupt:
                print('Stopping training...')
                print("Saved Model Path: {}".format(options.checkpoints_path))
                saver.save(sess, save_path=options.checkpoints_path, global_step=step)
        else:
            print('Restoring latest saved TensorFlow model...')
            dir_path = os.path.dirname(os.path.realpath(__file__))
            cur_dir = dir_path.split('/')[-1]
            experiments = glob.glob(os.path.join(options.MAIN_PATH, cur_dir) + '/*')
            sorted_experiments = sorted(experiments)
            if len(experiments) > 0:
                saver.restore(sess, tf.train.latest_checkpoint(os.path.join(experiments[-1], 'checkpoints')))
                generate_image_grid(sess, decoder_input, op=decoder_image)
            else:
                print('No checkpoint found at {}'.format(os.path.join(options.MAIN_PATH, cur_dir)))

if __name__ == '__main__':
    check_tf_version()
    parser = set_config()
    (options, args) = parser.parse_args()
    if not options.run_inference:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cur_dir = dir_path.split('/')[-1]
        options = extend_options(parser, cur_dir)

    train(options)
