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
from helpers.graph import get_variables, linear, AdamOptimizer, clip_weights
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
label_dim = mnist.train.labels.shape[1]
hidden_layer1 = 1000
hidden_layer2 = 1000
discriminator_hidden_layer = 512
clipping_parameter = 0.01
n_critic = 5
lambda_reconstruction = 1.E2

# The autoencoder network
def encoder(x, c):
    with tf.name_scope('Condition_input'):
        inputs = tf.concat(axis=1, values=[x, c])

    e_linear_1 = tf.nn.relu(linear(inputs, hidden_layer1, 'e_linear_1'))
    e_linear_2 = tf.nn.relu(linear(e_linear_1, hidden_layer2, 'e_linear_2'))
    z_mu = linear(e_linear_2, options.z_dim, 'z_mu')
    z_logvar = linear(e_linear_2, options.z_dim, 'z_logvar')
    return z_mu, z_logvar

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

def decoder(z, c):
    with tf.name_scope('Condition_latent_variable'):
        inputs = tf.concat(axis=1, values=[z, c])

    d_linear_1 = tf.nn.relu(linear(inputs, hidden_layer2, 'd_linear_1'))
    d_linear_2 = tf.nn.relu(linear(d_linear_1, hidden_layer1, 'd_linear_2'))
    logits = linear(d_linear_2, input_dim, 'd_logits')
    prob = tf.nn.sigmoid(logits)
    return prob, logits

def discriminator(x_hat, c):
    with tf.name_scope('Condition_latent_variable'):
        inputs = tf.concat(axis=1, values=[x_hat, c])

    dis_linear_1 = tf.nn.relu(linear(inputs, discriminator_hidden_layer, 'dis_linear_1'))
    dis_linear_2 = tf.nn.relu(linear(dis_linear_1, discriminator_hidden_layer, 'dis_linear_2'))
    dis_linear_3 = tf.nn.relu(linear(dis_linear_2, discriminator_hidden_layer, 'dis_linear_3'))
    logits = linear(dis_linear_3, 1, 'dis_logits')
    prob = tf.nn.sigmoid(logits)
    return prob

def train(options):
    # Placeholders for input data and the targets
    with tf.name_scope('Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[options.batch_size, input_dim], name='Input')
        input_images = tf.reshape(X, [-1, 28, 28, 1])

    with tf.name_scope('Condition'):
        c = tf.placeholder(tf.float32, shape=[None, label_dim])

    with tf.name_scope('Latent_variable'):
        z = tf.placeholder(dtype=tf.float32, shape=[None, options.z_dim], name='Latent_variable')

    with tf.variable_scope('Encoder'):
        z_mu, z_logvar = encoder(X, c)
        z_sample = sample_z(z_mu, z_logvar)

    with tf.variable_scope('Decoder') as scope:
        decoder_prob, decoder_logits = decoder(z_sample, c)
        generated_images = tf.reshape(decoder_prob, [-1, 28, 28, 1])
        scope.reuse_variables()
        X_samples, _ = decoder(z, c)

    with tf.variable_scope('Discriminator') as scope:
        D_real = discriminator(X, c)
        scope.reuse_variables()
        D_fake = discriminator(decoder_prob, c)

    # Loss - E[log P(X|z)]
    with tf.name_scope('Loss'):
        with tf.name_scope('Reconstruction_Loss'):
            loss_reconstr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_logits, labels=X))
        with tf.name_scope('KL_loss'):
            kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        with tf.name_scope('Discriminator_loss'):
            loss_discriminator = (tf.reduce_mean(D_fake) - tf.reduce_mean(D_real))
        with tf.name_scope('Autoencoder_loss'):
            loss_autoencoder = -tf.reduce_mean(D_fake) + lambda_reconstruction * loss_reconstr + tf.reduce_mean(kl_loss)

    with tf.variable_scope('Discriminator_Accuracy'):
        accuracy_real = tf.reduce_mean(tf.cast(tf.greater_equal(D_real, 0.5), tf.float16))
        accuracy_fake = tf.reduce_mean(tf.cast(tf.less(D_fake, 0.5), tf.float16))
        accuracy_tot = (accuracy_real + accuracy_fake) / 2

    vars = tf.trainable_variables()
    enc_params = [v for v in vars if v.name.startswith('Encoder/')]
    dec_params = [v for v in vars if v.name.startswith('Decoder/')]
    dis_params = [v for v in vars if v.name.startswith('Discriminator/')]
    dis_weights = [w for w in dis_params if 'weight' in w.name]

    clipped_weights = clip_weights(dis_weights, clipping_parameter, 'clip_weights')

    with tf.name_scope('Optimizer'):
        train_op_Dis, Dis_grads_and_vars = AdamOptimizer(loss_discriminator, options.learning_rate, options.beta1, var_list=dis_params)
        train_op_autoenc, Autoenc_grads_and_vars = AdamOptimizer(loss_autoencoder, options.learning_rate, options.beta1, var_list=enc_params+dec_params)

    # Visualization
    tf.summary.scalar(name='Discriminator_loss', tensor=loss_discriminator)
    tf.summary.scalar(name='Autoencoder_loss', tensor=loss_autoencoder)
    tf.summary.scalar(name='Discriminator_Accuracy', tensor=accuracy_tot)
    tf.summary.histogram(name='Sampled variable', values=z_sample)

    for grad, var in Autoenc_grads_and_vars + Dis_grads_and_vars:
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
                        # update discriminator
                        for _ in range(n_critic):
                            batch_x, batch_y = mnist.train.next_batch(options.batch_size)
                            sess.run(train_op_Dis, feed_dict={X: batch_x, c: batch_y})

                            _ = sess.run(clipped_weights)

                        # update autoencoder/generator parameters
                        batch_x, batch_y = mnist.train.next_batch(options.batch_size)
                        sess.run(train_op_autoenc, feed_dict={X: batch_x, c: batch_y})

                        if iteration % 50 == 0:
                            summary, ld, la, acc = sess.run(
                                    [summary_op, loss_discriminator, loss_autoencoder, accuracy_tot],
                                    feed_dict={X: batch_x, c: batch_y})
                            writer.add_summary(summary, global_step=step)
                            print("Epoch: {} - Iteration {} - Autoencoder loss: {:.4f}".format(epoch, iteration, la))
                            print("Discriminator loss: {} - Discriminator accuracy: {:.4f}%\n".format(ld, acc*100))

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
                plot(sess, z, X_samples, num_images=25, height=28, width=28, condition=True, c=c)

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
