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

def plot(sess, z, X_samples, num_images):
    samples = []
    n = 15
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            samples.append(sess.run(X_samples, feed_dict={z: z_sample}))

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(num_images, num_images)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

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

def train(options):
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, input_dim])
        input_images = tf.reshape(X, [-1, 28, 28, 1])

    with tf.name_scope('Latent_variable'):
        z = tf.placeholder(tf.float32, shape=[None, options.z_dim])

    with tf.name_scope('Autoencoder'):
        with tf.variable_scope(tf.get_variable_scope()):
            z_mu, z_logvar = encoder(X)
            z_sample = sample_z(z_mu, z_logvar)
            decoder_output, logits = decoder(z_sample)
            generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

    with tf.variable_scope(tf.get_variable_scope()):
        # Sampling from random z
        X_samples, _ = decoder(z, reuse=True)

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
    tf.summary.histogram(name='Encoder z_mu', values=z_mu)
    tf.summary.histogram(name='Encoder z_logvar', values=z_logvar)
    tf.summary.histogram(name='Sampled variable', values=z_sample)

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
                if not os.path.exists('out/'):
                    os.makedirs('out/')
                for epoch in range(options.epochs):
                    n_batches = int(mnist.train.num_examples / options.batch_size)
                    for iteration in range(n_batches):
                        batch_x, _ = mnist.train.next_batch(options.batch_size)

                        # Train
                        sess.run(train_op, feed_dict={X: batch_x})

                        if iteration % 50 == 0:
                            summary, batch_loss = sess.run([summary_op, vae_loss], feed_dict={X: batch_x})
                            writer.add_summary(summary, global_step=step)
                            print("Loss: {}".format(batch_loss))
                            print("Epoch: {}, iteration: {}".format(epoch, iteration))

                            with open(options.logs_path + '/log.txt', 'a') as log:
                                log.write("Epoch: {}, iteration: {}\n".format(epoch, iteration))
                                log.write("Loss: {}\n".format(batch_loss))
                            if options.save_plots:
                                fig = plot(sess, z, X_samples, num_images=15)
                                plt.savefig('out/{}.png'.format(str(step).zfill(8)), bbox_inches='tight')
                                plt.close(fig)

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
                saver.restore(sess, tf.train.latest_checkpoint(os.path.join(sorted_experiments[-1], 'checkpoints')))
                fig = plot(sess, z, X_samples, num_images=15)
                plt.show()
                # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)
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
