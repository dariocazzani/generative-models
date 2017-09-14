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
hidden_layer1 = 1000
hidden_layer2 = 1000
discriminator_X_hidden_layer = 512
discriminator_z_hidden_layer = 128
clipping_parameter = 0.01
n_critic = 5
lambda_reconstruction = 1.E2

# The autoencoder network
def encoder(x):
    e_linear_1 = tf.nn.relu(linear(x, hidden_layer1, 'e_linear_1'))
    e_linear_2 = tf.nn.relu(linear(e_linear_1, hidden_layer2, 'e_linear_2'))
    latent_variable = linear(e_linear_2, options.z_dim, 'e_latent_variable')
    return latent_variable


def decoder(z):
    d_linear_1 = tf.nn.relu(linear(z, hidden_layer2, 'd_linear_1'))
    d_linear_2 = tf.nn.relu(linear(d_linear_1, hidden_layer1, 'd_linear_2'))
    logits = linear(d_linear_2, input_dim, 'd_logits')
    prob = tf.nn.sigmoid(logits)
    return prob, logits

def discriminator_x(x_hat):
    dis_linear_1 = tf.nn.relu(linear(x_hat, discriminator_X_hidden_layer, 'dis_x_linear_1'))
    dis_linear_2 = tf.nn.relu(linear(dis_linear_1, discriminator_X_hidden_layer, 'dis_x_linear_2'))
    dis_linear_3 = tf.nn.relu(linear(dis_linear_2, discriminator_X_hidden_layer, 'dis_x_linear_3'))
    logits = linear(dis_linear_3, 1, 'dis_x_logits')
    prob = tf.nn.sigmoid(logits)
    return prob

def discriminator_z(z):
    dis_linear_1 = tf.nn.relu(linear(z, discriminator_z_hidden_layer, 'dis_z_linear_1'))
    dis_linear_2 = tf.nn.relu(linear(dis_linear_1, discriminator_z_hidden_layer, 'dis_z_linear_2'))
    logits = linear(dis_linear_2, 1, 'dis_z_logits')
    prob = tf.nn.sigmoid(logits)
    return prob

def train(options):
    # Placeholders for input data and the targets
    with tf.name_scope('Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[options.batch_size, input_dim], name='Input')
        input_images = tf.reshape(X, [-1, 28, 28, 1])

    with tf.name_scope('Latent_variable'):
        z = tf.placeholder(dtype=tf.float32, shape=[None, options.z_dim], name='Latent_variable')

    with tf.variable_scope('Encoder'):
        encoder_output = encoder(X)

    with tf.variable_scope('Decoder') as scope:
        decoder_prob, decoder_logits = decoder(encoder_output)
        generated_images = tf.reshape(decoder_prob, [-1, 28, 28, 1])
        scope.reuse_variables()
        X_samples, _ = decoder(z)

    with tf.variable_scope('Discriminator_X') as scope:
        D_X_real = discriminator_x(X)
        scope.reuse_variables()
        D_X_fake = discriminator_x(decoder_prob)

    with tf.variable_scope('Discriminator_z') as scope:
        D_z_real = discriminator_z(z)
        scope.reuse_variables()
        D_z_fake = discriminator_z(encoder_output)


    # Loss - E[log P(X|z)]
    with tf.name_scope('Loss'):
        with tf.name_scope('Reconstruction_Loss'):
            loss_reconstr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_logits, labels=X))
        with tf.name_scope('Discriminator_X_loss'):
            loss_discriminator_X = (tf.reduce_mean(D_X_fake) - tf.reduce_mean(D_X_real))
        with tf.name_scope('Discriminator_z_loss'):
            loss_discriminator_z = (tf.reduce_mean(D_z_fake) - tf.reduce_mean(D_z_real))
        with tf.name_scope('Encoder_loss'):
            loss_encoder =  -tf.reduce_mean(D_z_fake)
        with tf.name_scope('Autoencoder_loss'):
            loss_autoencoder = -tf.reduce_mean(D_X_fake) + lambda_reconstruction * loss_reconstr

    with tf.variable_scope('Discriminator_X_Accuracy'):
        accuracy_real_X = tf.reduce_mean(tf.cast(tf.greater_equal(D_X_real, 0.5), tf.float16))
        accuracy_fake_X = tf.reduce_mean(tf.cast(tf.less(D_X_fake, 0.5), tf.float16))
        accuracy_tot_X = (accuracy_real_X + accuracy_fake_X) / 2

    with tf.variable_scope('Discriminator_z_Accuracy'):
        accuracy_real_z = tf.reduce_mean(tf.cast(tf.greater_equal(D_z_real, 0.5), tf.float16))
        accuracy_fake_z = tf.reduce_mean(tf.cast(tf.less(D_z_fake, 0.5), tf.float16))
        accuracy_tot_z = (accuracy_real_z + accuracy_fake_z) / 2

    vars = tf.trainable_variables()
    enc_params = [v for v in vars if v.name.startswith('Encoder/')]
    dec_params = [v for v in vars if v.name.startswith('Decoder/')]
    dis_z_params = [v for v in vars if v.name.startswith('Discriminator_z/')]
    dis_X_params = [v for v in vars if v.name.startswith('Discriminator_X/')]
    dis_z_weights = [w for w in dis_z_params if 'weight' in w.name]
    dis_X_weights = [w for w in dis_X_params if 'weight' in w.name]

    clipped_z_weights = clip_weights(dis_z_weights, clipping_parameter, 'clip_weights')
    clipped_X_weights = clip_weights(dis_X_weights, clipping_parameter, 'clip_weights')

    with tf.name_scope('Optimizer'):
        train_op_Dis_z, Dis_z_grads_and_vars = AdamOptimizer(loss_discriminator_z, options.learning_rate, options.beta1, var_list=dis_z_params)
        train_op_Dis_X, Dis_X_grads_and_vars = AdamOptimizer(loss_discriminator_X, options.learning_rate, options.beta1, var_list=dis_X_params)
        train_op_autoenc, Autoenc_grads_and_vars = AdamOptimizer(loss_autoencoder, options.learning_rate, options.beta1, var_list=enc_params+dec_params)
        train_op_Enc, Enc_grads_and_vars = AdamOptimizer(loss_encoder, options.learning_rate/10., options.beta1, var_list=enc_params)

    # Visualization
    tf.summary.scalar(name='Discriminator_X_loss', tensor=loss_discriminator_X)
    tf.summary.scalar(name='Discriminator_z_loss', tensor=loss_discriminator_z)
    tf.summary.scalar(name='Autoencoder_loss', tensor=loss_autoencoder)
    tf.summary.scalar(name='Encoder_loss', tensor=loss_encoder)
    tf.summary.scalar(name='Discriminator_X_Accuracy', tensor=accuracy_tot_X)
    tf.summary.scalar(name='Discriminator_z_Accuracy', tensor=accuracy_tot_z)
    tf.summary.histogram(name='Latent_variable', values=encoder_output)

    for grad, var in Autoenc_grads_and_vars + Dis_z_grads_and_vars + Dis_X_grads_and_vars:
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
                        # update discriminator X
                        for _ in range(n_critic):
                            batch_x, _ = mnist.train.next_batch(options.batch_size)
                            sess.run(train_op_Dis_X, feed_dict={X: batch_x})

                            _ = sess.run(clipped_X_weights)
                        # update discriminator z
                        for _ in range(n_critic):
                            batch_x, _ = mnist.train.next_batch(options.batch_size)
                            batch_z = np.random.randn(options.batch_size, options.z_dim)
                            sess.run(train_op_Dis_z, feed_dict={X: batch_x, z: batch_z})

                            _ = sess.run(clipped_z_weights)

                        # update autoencoder/generator parameters
                        batch_x, _ = mnist.train.next_batch(options.batch_size)
                        batch_z = np.random.randn(options.batch_size, options.z_dim)
                        sess.run(train_op_autoenc, feed_dict={X: batch_x})
                        sess.run(train_op_Enc, feed_dict={X: batch_x})

                        if iteration % 50 == 0:
                            summary, ld_X, ld_z, la, le, acc_X, acc_z = sess.run(
                                    [summary_op, loss_discriminator_X, loss_discriminator_z, loss_autoencoder, loss_encoder, accuracy_tot_X, accuracy_tot_z],
                                    feed_dict={X: batch_x, z: batch_z})
                            writer.add_summary(summary, global_step=step)
                            print("Epoch: {} - Iteration {} - Loss Discriminator X: {:.4f} - Loss Discriminator z: {:.4f}".format(epoch, iteration, ld_X, ld_z))
                            print("Autoencoder Loss: {:.4f} - Encoder Loss: {:.4f}".format(la, le))
                            print("Discriminator X accuracy: {:.2f}% - Discriminator z accuracy: {:.2f}%\n".format(acc_X*100, acc_z*100))

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
