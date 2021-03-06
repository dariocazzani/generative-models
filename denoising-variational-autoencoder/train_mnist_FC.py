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
from helpers.graph import get_variables, linear, AdamOptimizer
from helpers.display import plot

import subprocess
import tensorflow as tf
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
mnist = input_data.read_data_sets('../Data', one_hot=True)

# Parameters
input_dim = mnist.train.images.shape[1]
hidden_layer1 = 1000
hidden_layer2 = 1000
noise_length = int(input_dim / 5.)

# The autoencoder network
def encoder(x):
    e_linear_1 = tf.nn.relu(linear(x, hidden_layer1, 'e_linear_1'))
    e_linear_2 = tf.nn.relu(linear(e_linear_1, hidden_layer2, 'e_linear_2'))
    z_mu = linear(e_linear_2, options.z_dim, 'z_mu')
    z_logvar = linear(e_linear_2, options.z_dim, 'z_logvar')
    return z_mu, z_logvar

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

def decoder(z):
    d_linear_1 = tf.nn.relu(linear(z, hidden_layer2, 'd_linear_1'))
    d_linear_2 = tf.nn.relu(linear(d_linear_1, hidden_layer1, 'd_linear_2'))
    logits = linear(d_linear_2, input_dim, 'logits')
    prob = tf.nn.sigmoid(logits)
    return prob, logits

def train(options):
    # Placeholders for input data and the targets
    with tf.name_scope('Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[options.batch_size, input_dim], name='Input')
        X_noisy = tf.placeholder(dtype=tf.float32, shape=[options.batch_size, input_dim], name='Input_noisy')
        input_images = tf.reshape(X_noisy, [-1, 28, 28, 1])

    with tf.name_scope('Latent_variable'):
        z = tf.placeholder(dtype=tf.float32, shape=[None, options.z_dim], name='Latent_variable')

    with tf.variable_scope('Encoder'):
        z_mu, z_logvar = encoder(X_noisy)

    with tf.variable_scope('Decoder') as scope:
        z_sample = sample_z(z_mu, z_logvar)
        decoder_output, logits = decoder(z_sample)
        generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])
        scope.reuse_variables()
        X_samples, _ = decoder(z)

    with tf.name_scope('Loss'):
        # E[log P(X|z)]
        reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        # VAE loss
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    # Optimizer
    train_op, grads_and_vars = AdamOptimizer(vae_loss, options.learning_rate, options.beta1)

    # Visualization
    tf.summary.scalar(name='Loss', tensor=vae_loss)
    tf.summary.histogram(name='Sampled variable', values=z_sample)

    for grad, var in grads_and_vars:
        tf.summary.histogram('Gradients/' + var.name, grad)
        tf.summary.histogram('Values/' + var.name, var)

    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    step = 0
    init = tf.global_variables_initializer()
    n_batches = int(mnist.train.num_examples / options.batch_size)
    with tf.Session() as sess:
        sess.run(init)
        if not options.run_inference:
            try:
                writer = tf.summary.FileWriter(logdir=options.tensorboard_path, graph=sess.graph)
                for epoch in range(options.epochs):
                    for iteration in range(n_batches):
                        batch_x, _ = mnist.train.next_batch(options.batch_size)

                        # generate mask for noisy batch
                        mask = np.ones(input_dim)
                        idx = np.random.randint(input_dim - noise_length)
                        mask[idx:idx+noise_length] = 0.
                        mask = np.tile(mask, (options.batch_size, 1))
                        noisy_batch = batch_x * mask
                        # Train
                        sess.run(train_op, feed_dict={X: batch_x, X_noisy: noisy_batch})

                        if iteration % 50 == 0:
                            summary, batch_loss = sess.run([summary_op, vae_loss], feed_dict={X: batch_x, X_noisy: noisy_batch})
                            writer.add_summary(summary, global_step=step)
                            print("Epoch: {} - Iteration {} - Loss: {:.4f}\n".format(epoch, iteration, batch_loss))

                        step += 1
                    saver.save(sess, save_path=options.checkpoints_path, global_step=step)
                print("Model Trained!")

            except KeyboardInterrupt:
                print('Stopping training...')
                saver.save(sess, save_path=options.checkpoints_path, global_step=step)
        else:
            print('Restoring latest saved TensorFlow model...')
            dir_path = os.path.dirname(os.path.realpath(__file__))
            cur_dir = dir_path.split('/')[-1]
            script_name = os.path.basename(__file__).split('.')[0]
            experiments = glob.glob(os.path.join(options.MAIN_PATH, cur_dir) + '/{}*'.format(script_name))
            experiments.sort(key=lambda x: os.path.getmtime(x))
            if len(experiments) > 0:
                print('Restoring: {}'.format(experiments[-1]))
                saver.restore(sess, tf.train.latest_checkpoint(os.path.join(experiments[-1], 'checkpoints')))
                plot(sess, z, X_samples, num_images=15, height=28, width=28)

            else:
                print('No checkpoint found at {}'.format(os.path.join(options.MAIN_PATH, cur_dir)))

if __name__ == '__main__':
    check_tf_version()
    parser = set_config()
    (options, args) = parser.parse_args()
    if not options.run_inference:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cur_dir = dir_path.split('/')[-1]
        script_name = os.path.basename(__file__).split('.')[0]
        options = extend_options(parser, cur_dir, script_name)

    train(options)
