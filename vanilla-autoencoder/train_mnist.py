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
    grid_x = np.linspace(-2, 2, num_images)
    grid_y = np.linspace(-2, 2, num_images)
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

# The autoencoder network
def encoder(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_linear_1 = tf.nn.relu(linear(x, hidden_layer1, 'e_linear_1'))
        e_linear_2 = tf.nn.relu(linear(e_linear_1, hidden_layer2, 'e_linear_2'))
        latent_variable = linear(e_linear_2, options.z_dim, 'e_latent_variable')
        return latent_variable


def decoder(z, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_linear_1 = tf.nn.relu(linear(z, hidden_layer2, 'd_linear_1'))
        d_linear_2 = tf.nn.relu(linear(d_linear_1, hidden_layer1, 'd_linear_2'))
        logits = linear(d_linear_2, input_dim, 'logits')
        prob = tf.nn.sigmoid(logits)
        return prob


def train(options):
    # Placeholders for input data and the targets
    with tf.name_scope('Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[options.batch_size, input_dim], name='Input')
        input_images = tf.reshape(X, [-1, 28, 28, 1])

    with tf.name_scope('Latent_variable'):
        z = tf.placeholder(dtype=tf.float32, shape=[None, options.z_dim], name='Latent_variable')

    with tf.name_scope('Autoencoder'):
        with tf.variable_scope(tf.get_variable_scope()):
            encoder_output = encoder(X)
            decoder_output = decoder(encoder_output)
            generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

    with tf.variable_scope(tf.get_variable_scope()):
        X_samples = decoder(z, reuse=True)

    # Loss - MSE
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.square(X - decoder_output))

    # Optimizer
    train_op, grads_and_vars = AdamOptimizer(loss, options.learning_rate, options.beta1)

    # Visualization
    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
    tf.summary.histogram(name='Latent_variable', values=encoder_output)

    for grad, var in grads_and_vars:
        tf.summary.histogram(var.name + '/gradient', grad)
        tf.summary.histogram(var.name + '/value', var)

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
                for i in range(options.epochs):
                    n_batches = int(mnist.train.num_examples / options.batch_size)
                    for b in range(n_batches):
                        batch_x, _ = mnist.train.next_batch(options.batch_size)
                        sess.run(train_op, feed_dict={X: batch_x})
                        if b % 50 == 0:
                            batch_loss, summary = sess.run([loss, summary_op], feed_dict={X: batch_x})
                            writer.add_summary(summary, global_step=step)
                            print("Loss: {}".format(batch_loss))
                            print("Epoch: {}, iteration: {}".format(i, b))

                            with open(options.logs_path + '/log.txt', 'a') as log:
                                log.write("Epoch: {}, iteration: {}\n".format(i, b))
                                log.write("Loss: {}\n".format(batch_loss))
                            if options.save_plots:
                                fig = plot(sess, z, X_samples, num_images=50)
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
            script_name = os.path.basename(__file__).split('.')[0]
            experiments = glob.glob(os.path.join(options.MAIN_PATH, cur_dir) + '/{}*'.format(script_name))
            experiments.sort(key=lambda x: os.path.getmtime(x))
            if len(experiments) > 0:
                print('Restoring: {}'.format(experiments[-1]))
                saver.restore(sess, tf.train.latest_checkpoint(os.path.join(experiments[-1], 'checkpoints')))
                fig = plot(sess, z, X_samples, num_images=50)
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
