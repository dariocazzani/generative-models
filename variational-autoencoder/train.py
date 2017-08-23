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

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
input_dim = mnist.train.images.shape[1]
hidden_layer1 = 1000
hidden_layer2 = 1000

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

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

# Q(z|X)
def encoder(X, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    e_linear_1 = tf.nn.relu(linear(X, hidden_layer1, 'e_linear_1'))
    e_linear_2 = tf.nn.relu(linear(e_linear_1, hidden_layer2, 'e_linear_2'))
    z_mu = linear(e_linear_2, options.z_dim, 'z_mu')
    z_logvar = linear(e_linear_2, options.z_dim, 'z_logvar')
    return z_mu, z_logvar

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

# P(X|z)
def decoder(z, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    d_linear_1 = tf.nn.relu(linear(z, hidden_layer2, 'd_linear_1'))
    d_linear_2 = tf.nn.relu(linear(d_linear_1, hidden_layer1, 'd_linear_2'))
    logits = linear(d_linear_2, input_dim, 'logits')
    prob = tf.nn.sigmoid(logits)
    return prob, logits


# =============================== TRAINING ====================================

def train(options):
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, input_dim])

    with tf.name_scope('Latent_variable'):
        z = tf.placeholder(tf.float32, shape=[None, options.z_dim])

    with tf.variable_scope(tf.get_variable_scope()):
        z_mu, z_logvar = encoder(X)
        z_sample = sample_z(z_mu, z_logvar)
        _, logits = decoder(z_sample)

    with tf.variable_scope(tf.get_variable_scope()):
        # Sampling from random z
        X_samples, _ = decoder(z, reuse=True)


    # E[log P(X|z)]
    recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
    # VAE loss
    vae_loss = tf.reduce_mean(recon_loss + kl_loss)

    # Optimizer
    train_op, grads_and_vars = AdamOptimizer(vae_loss, options.learning_rate, options.beta1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    for it in range(1000000):
        X_mb, _ = mnist.train.next_batch(options.batch_size)

        _, loss = sess.run([train_op, vae_loss], feed_dict={X: X_mb})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('Loss: {:.4}'. format(loss))
            print()

            samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, options.z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

if __name__ == '__main__':
    check_tf_version()
    parser = set_config()
    (options, args) = parser.parse_args()
    if not options.run_inference:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cur_dir = dir_path.split('/')[-1]
        options = extend_options(parser, cur_dir)

    train(options)
